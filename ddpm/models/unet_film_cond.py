import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================
# EMA
# =============================
class EMA:
    def __init__(self, beta: float):
        self.beta = beta
        self.step = 0

    @torch.no_grad()
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            ma_params.copy_(self.update_average(ma_params, current_params))

    def update_average(self, old, new):
        return old * self.beta + (1.0 - self.beta) * new

    @torch.no_grad()
    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
        else:
            self.update_model_average(ema_model, model)
        self.step += 1

    @torch.no_grad()
    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


# =============================
# FiLM
# =============================
class FiLM(nn.Module):
    def __init__(self, emb_dim: int, target_ch: int):
        super().__init__()
        self.to_scale = nn.Linear(emb_dim, target_ch)
        self.to_shift = nn.Linear(emb_dim, target_ch)

    def forward(self, emb):
        scale = self.to_scale(emb)
        shift = self.to_shift(emb)
        return scale, shift


# =============================
# Double Conv
# =============================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False, groups=8):
        super().__init__()
        self.residual = residual

        if mid_channels is None:
            mid_channels = out_channels

        if residual and in_channels != out_channels:
            raise ValueError(
                f"Residual DoubleConv requires in_channels == out_channels, "
                f"got {in_channels} and {out_channels}"
            )

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
        )

    def forward(self, x):
        out = self.double_conv(x)
        if self.residual:
            out = x + out
        return F.gelu(out)


# =============================
# Down block with FiLM
# =============================
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=128, groups=8):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True, groups=groups),
            DoubleConv(in_channels, out_channels, groups=groups),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.film = FiLM(emb_dim, out_channels)

    @staticmethod
    def apply_film(x, scale, shift):
        b, c, _, _ = x.shape
        return x * (1.0 + scale.view(b, c, 1, 1)) + shift.view(b, c, 1, 1)

    def forward(self, x, emb):
        x = self.maxpool_conv(x)
        e = self.emb_layer(emb)
        s, b = self.film(e)
        x = self.apply_film(x, s, b)
        return x


# =============================
# Up block with FiLM
# =============================
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=128, groups=8):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.dc1 = DoubleConv(in_channels, in_channels, residual=True, groups=groups)
        self.dc2 = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2, groups=groups)

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.film1 = FiLM(emb_dim, in_channels)
        self.film2 = FiLM(emb_dim, out_channels)

    @staticmethod
    def apply_film(x, scale, shift):
        b, c, _, _ = x.shape
        return x * (1.0 + scale.view(b, c, 1, 1)) + shift.view(b, c, 1, 1)

    def forward(self, x, skip_x, emb):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)

        e = self.emb_layer(emb)

        x = self.dc1(x)
        s1, b1 = self.film1(e)
        x = self.apply_film(x, s1, b1)

        x = self.dc2(x)
        s2, b2 = self.film2(e)
        x = self.apply_film(x, s2, b2)

        return x


# =============================
# Self-Attention
# =============================
class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels

        self.ln1 = nn.LayerNorm(channels)
        self.mha = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)

        self.ff = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_seq = x.view(b, c, h * w).transpose(1, 2)   # (B, HW, C)

        x_ln = self.ln1(x_seq)
        attn_out, _ = self.mha(x_ln, x_ln, x_ln)
        x_seq = x_seq + attn_out
        x_seq = x_seq + self.ff(x_seq)

        return x_seq.transpose(1, 2).view(b, c, h, w)


# =============================
# U-Net with stronger conditioning
# =============================
class Unet(nn.Module):
    def __init__(
        self,
        c_in=1,
        c_out=1,
        time_dim=128,
        num_classes=None,
        cond_continuous=False,
        groups=8,
    ):
        super().__init__()

        self.time_dim = time_dim
        self.cond_continuous = cond_continuous

        # -------- input --------
        self.inc = DoubleConv(c_in, 32, groups=groups)

        # -------- time embedding MLP --------
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # -------- label embedding --------
        if cond_continuous:
            self.label_emb = nn.Linear(1, time_dim)
        elif num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)
        else:
            self.label_emb = None

        # -------- encoder --------
        self.down1 = Down(32, 64, emb_dim=time_dim, groups=groups)
        self.sa1 = SelfAttention(64, num_heads=4)

        self.down2 = Down(64, 128, emb_dim=time_dim, groups=groups)
        self.down3 = Down(128, 128, emb_dim=time_dim, groups=groups)

        # -------- bottleneck --------
        self.bot1 = DoubleConv(128, 128, groups=groups)
        self.bot2 = DoubleConv(128, 128, groups=groups)

        # -------- decoder --------
        self.up1 = Up(256, 64, emb_dim=time_dim, groups=groups)
        self.up2 = Up(128, 32, emb_dim=time_dim, groups=groups)
        self.sa2 = SelfAttention(32, num_heads=4)
        self.up3 = Up(64, 32, emb_dim=time_dim, groups=groups)

        # -------- output --------
        self.outc = nn.Conv2d(32, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        """
        t: shape (B, 1)
        returns: shape (B, channels)
        """
        device = t.device
        half = channels // 2

        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, half, device=device).float() / channels)
        )

        sinusoid_inp = t * inv_freq[None, :]
        pos_enc = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)

        if channels % 2 != 0:
            pos_enc = F.pad(pos_enc, (0, 1))

        return pos_enc

    def build_conditioning(self, t, y=None):
        """
        Build conditioning embedding from time and optional label.
        """
        t = t.unsqueeze(-1).float()                    # (B,) -> (B,1)
        t_emb = self.pos_encoding(t, self.time_dim)   # (B, time_dim)
        t_emb = self.time_mlp(t_emb)                  # learned time embedding

        if self.label_emb is None or y is None:
            return t_emb

        if self.cond_continuous:
            if y.dim() == 1:
                y = y.unsqueeze(-1)
            y_emb = self.label_emb(y.float())
        else:
            y_emb = self.label_emb(y.long())

        return t_emb + y_emb

    def forward(self, x, t, y=None):
        emb = self.build_conditioning(t, y)

        x1 = self.inc(x)

        x2 = self.down1(x1, emb)
        x2 = self.sa1(x2)

        x3 = self.down2(x2, emb)
        x4 = self.down3(x3, emb)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)

        x = self.up1(x4, x3, emb)
        x = self.up2(x, x2, emb)
        x = self.sa2(x)
        x = self.up3(x, x1, emb)

        return self.outc(x)