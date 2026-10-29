import os
from pathlib import Path
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from typing import Optional
import torch
import torch.nn.functional as F
from datetime import datetime


def get_timestamp():
    from datetime import datetime
    return datetime.now().strftime("%m%d_%H%M")


def setup_logging(results_dir: str, run_name: str):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{run_name}_{timestamp}"

    base_dir = os.path.join(results_dir, run_name)

    paths = {
        "base": base_dir,
        "models": os.path.join(base_dir, "models"),
        "logs": os.path.join(base_dir, "logs"),
        "samples": os.path.join(base_dir, "samples"),
        "test": os.path.join(base_dir, "test"),
        "noising_test": os.path.join(base_dir, "noising_test"),
    }

    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    return paths


def transformation(delta, method="asinh", alpha=1.0, eps=1e-6):
    d = np.asarray(delta, dtype=np.float64)

    if method == "asinh":
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0 for asinh. Got alpha={alpha}")
        return np.arcsinh(d / alpha)

    if method == "log10":
        return np.log10(np.maximum(1.0 + d, eps))

    if method == "ln":
        return np.log(np.maximum(1.0 + d, eps))

    raise ValueError(f"Unknown method='{method}'. Use 'asinh', 'log10', or 'ln'.")


def norm_stats_slabs_filename(method, alpha, eps, grid, thick):
    alpha = float(alpha)
    eps = float(eps)
    return (
        f"norm_stats_{method}"
        f"_a{alpha:.6g}"
        f"_e{eps:.0e}"
        f"_g{grid}"
        f"_th{thick}.npz"
    )


def normalize_delta(
    delta: np.ndarray,
    mu: float,
    sigma: float,
    method: str = "asinh",
    alpha: float = 1.0,
    eps: float = 1e-6,
) -> np.ndarray:
    s = transformation(delta, method=method, alpha=alpha, eps=eps)
    y = (s - mu) / (sigma + 1e-12)
    return y.astype(np.float32, copy=False)


def denormalize_delta(
    y: np.ndarray,
    mu: float,
    sigma: float,
    method: str = "asinh",
    alpha: float = 1.0,
    eps: float = 1e-6,
) -> np.ndarray:
    s = y * sigma + mu

    if method == "asinh":
        delta = alpha * np.sinh(s)
    elif method == "log10":
        delta = np.power(10.0, s) - 1.0
    elif method == "ln":
        delta = np.exp(s) - 1.0
    else:
        raise ValueError(f"Unknown method '{method}'")

    return delta.astype(np.float32, copy=False)


def save_images(images: torch.Tensor, path: str, epoch: Optional[int], name: str) -> None:
    images = images.detach().cpu()
    files_npy = os.path.join(path, "files_npy")
    os.makedirs(files_npy, exist_ok=True)

    num_images = images.size(0)
    fig, axs = plt.subplots(1, num_images, figsize=(8 * num_images, 6))
    if num_images == 1:
        axs = [axs]

    for i, ax in enumerate(axs):
        array = images[i].numpy()
        suffix = f"_{epoch}_{i}" if epoch is not None else f"_{i}"
        np.save(os.path.join(files_npy, f"{name}{suffix}.npy"), array)

        if array.ndim == 2:
            img_vis = array
        elif array.ndim == 3:
            img_vis = array.mean(axis=0)
        else:
            raise ValueError(f"Unexpected shape: {array.shape}")

        ax.imshow(img_vis, cmap="viridis")
        ax.axis("off")

    plt.tight_layout()

    file_suffix = f"_{epoch}" if epoch is not None else ""
    plt.savefig(os.path.join(path, f"{name}{file_suffix}.jpg"))
    plt.close(fig)

    np.save(os.path.join(files_npy, f"{name}{file_suffix}.npy"), images.numpy())


# ===========================================================================
# Spectral losses
#
# All spectral losses follow the same signature:
#   (x_pred, x_true, ..., reduce="mean"|"per_sample") -> scalar or (B,)
#
# Variants:
#   fourier_amplitude_loss   — MSE on log-amplitudes per mode (legacy)
#   spectral_l1_loss         — L1 on amplitude (paper Eq. 14), optionally
#                              coupled with phase (paper Eq. 15)
#   power_spectrum_loss      — radially binned P(k) MSE (cosmology-native)
# ===========================================================================


def fourier_amplitude_loss(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    norm: str = "ortho",
    mode: str = "logamp",
    eps: float = 1e-8,
    reduce: str = "mean",
    k_weight_power: float = 0.0,
) -> torch.Tensor:
    """Per-mode MSE on log-amplitudes.  Legacy / ablation baseline."""
    if x_pred.shape != x_true.shape:
        raise ValueError(f"Shape mismatch: {x_pred.shape} vs {x_true.shape}")

    x_pred = x_pred.float()
    x_true = x_true.float()

    F_pred = torch.fft.rfft2(x_pred, dim=(-2, -1), norm=norm)
    F_true = torch.fft.rfft2(x_true, dim=(-2, -1), norm=norm)

    amp_pred = torch.abs(F_pred)
    amp_true = torch.abs(F_true)

    if mode == "amp":
        per_mode = (amp_pred - amp_true) ** 2
    elif mode == "logamp":
        per_mode = (torch.log(amp_pred + eps) - torch.log(amp_true + eps)) ** 2
    else:
        raise ValueError(f"Unknown mode='{mode}'.")

    if k_weight_power != 0.0:
        H, W = x_pred.shape[-2], x_pred.shape[-1]
        ky = torch.fft.fftfreq(H, d=1.0, device=x_pred.device).to(per_mode.dtype)
        kx = torch.fft.rfftfreq(W, d=1.0, device=x_pred.device).to(per_mode.dtype)
        k_grid = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
        k_max = k_grid.max().clamp_min(1e-12)
        w_k = (k_grid / k_max).pow(k_weight_power).view(1, 1, *k_grid.shape)
        per_mode = per_mode * w_k

    if reduce == "mean":
        return per_mode.mean()
    if reduce == "per_sample":
        return per_mode.mean(dim=(1, 2, 3))
    raise ValueError(f"Unknown reduce='{reduce}'.")


# def spectral_l1_loss(
#     x_pred: torch.Tensor,
#     x_true: torch.Tensor,
#     use_phase: bool = False,
#     norm: str = "ortho",
#     reduce: str = "mean",
# ) -> torch.Tensor:
#     """
#     L1 spectral loss from Chandran et al. 2026 (arXiv:2603.02447).

#     Amplitude only (Eq. 14):
#         L_A = || |F(x_pred)| - |F(x_true)| ||_1

#     Amplitude + phase (Eq. 15):
#         L_AP = || (|F(x_pred)| - |F(x_true)|) * (1 + |angle(F_pred) - angle(F_true)|) ||_1

#     Phase is weighted by amplitude difference so that phase errors in
#     low-amplitude modes (which are physically meaningless) are suppressed.

#     Using L1 instead of L2 in Fourier space *breaks Parseval equivalence*,
#     providing a spectral inductive bias that L2 cannot (Parseval: L2 in
#     Fourier = L2 in pixel space, so MSE in Fourier adds nothing new).

#     Args:
#         x_pred, x_true : (B, C, H, W)
#         use_phase       : if True, use amplitude+phase coupling (Eq. 15)
#         norm            : FFT normalization
#         reduce          : "mean" (scalar) or "per_sample" ((B,))

#     Returns:
#         scalar or (B,) tensor.
#     """
#     if x_pred.shape != x_true.shape:
#         raise ValueError(f"Shape mismatch: {x_pred.shape} vs {x_true.shape}")

#     x_pred = x_pred.float()
#     x_true = x_true.float()

#     F_pred = torch.fft.rfft2(x_pred, dim=(-2, -1), norm=norm)
#     F_true = torch.fft.rfft2(x_true, dim=(-2, -1), norm=norm)

#     amp_diff = torch.abs(F_pred).sub(torch.abs(F_true)).abs()  # |A_pred - A_true|

#     if use_phase:
#         # Phase difference, wrapped to [-pi, pi]
#         phase_diff = torch.angle(F_pred) - torch.angle(F_true)
#         phase_diff = (phase_diff + math.pi) % (2 * math.pi) - math.pi
#         per_mode = amp_diff * (1.0 + phase_diff.abs())
#     else:
#         per_mode = amp_diff

#     if reduce == "mean":
#         return per_mode.mean()
#     if reduce == "per_sample":
#         return per_mode.mean(dim=(1, 2, 3))
#     raise ValueError(f"Unknown reduce='{reduce}'.")

def spectral_l1_loss(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    use_phase: bool = False,
    norm: str = "ortho",
    reduce: str = "mean",
    k_weight_power: float = 0.0,
    log_amp: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    L1 spectral loss.

    Amplitude:
        L_A = || |F(x_pred)| - |F(x_true)| ||_1

    Log-amplitude optional:
        L_A = || log(|F(x_pred)|+eps) - log(|F(x_true)|+eps) ||_1

    Amplitude + phase:
        L_AP = L_A * (1 + |wrapped phase difference|)
    """
    if x_pred.shape != x_true.shape:
        raise ValueError(f"Shape mismatch: {x_pred.shape} vs {x_true.shape}")

    x_pred = x_pred.float()
    x_true = x_true.float()

    F_pred = torch.fft.rfft2(x_pred, dim=(-2, -1), norm=norm)
    F_true = torch.fft.rfft2(x_true, dim=(-2, -1), norm=norm)

    amp_pred = torch.abs(F_pred)
    amp_true = torch.abs(F_true)

    if log_amp:
        per_mode = torch.abs(torch.log(amp_pred + eps) - torch.log(amp_true + eps))
    else:
        per_mode = torch.abs(amp_pred - amp_true)

    if use_phase:
        phase_diff = torch.angle(F_pred) - torch.angle(F_true)
        phase_diff = (phase_diff + math.pi) % (2.0 * math.pi) - math.pi
        per_mode = per_mode * (1.0 + phase_diff.abs())

    if k_weight_power != 0.0:
        H, W = x_pred.shape[-2], x_pred.shape[-1]
        ky = torch.fft.fftfreq(H, d=1.0, device=x_pred.device).to(per_mode.dtype)
        kx = torch.fft.rfftfreq(W, d=1.0, device=x_pred.device).to(per_mode.dtype)
        k_grid = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)

        k_max = k_grid.max().clamp_min(1e-12)
        k_safe = k_grid.clamp_min(1e-6)

        w_k = (k_safe / k_max).pow(k_weight_power)
        w_k = w_k.view(1, 1, *k_grid.shape)

        per_mode = per_mode * w_k

    if reduce == "mean":
        return per_mode.mean()

    if reduce == "per_sample":
        return per_mode.mean(dim=(1, 2, 3))

    raise ValueError(f"Unknown reduce='{reduce}'.")


def power_spectrum_loss(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    n_bins: int = 64,
    bin_spacing: str = "log",
    use_log_pk: bool = True,
    eps: float = 1e-12,
    reduce: str = "mean",
    k_weight_power: float = 0.0,
    norm: str = "ortho",
) -> torch.Tensor:
    """
    Radially-binned 2D power spectrum loss.

    Operates in normalized space (same space the model trains in).
    P_delta(k) is evaluated offline after denormalization.

    Args:
        x_pred, x_true : (B, C, H, W)
        n_bins          : radial k-bins (default 64)
        bin_spacing     : "log" or "linear"
        use_log_pk      : MSE on log P(k) if True (relative error, recommended)
        eps             : log stabilizer
        reduce          : "mean" or "per_sample"
        k_weight_power  : 0=uniform; >0 = emphasize high-k
        norm            : FFT normalization
    """
    if x_pred.shape != x_true.shape:
        raise ValueError(f"Shape mismatch: {x_pred.shape} vs {x_true.shape}")
    if x_pred.ndim != 4:
        raise ValueError(f"Expected (B,C,H,W), got {x_pred.shape}")

    x_pred = x_pred.float()
    x_true = x_true.float()

    B, C, H, W = x_pred.shape
    device = x_pred.device

    F_pred = torch.fft.rfft2(x_pred, dim=(-2, -1), norm=norm)
    F_true = torch.fft.rfft2(x_true, dim=(-2, -1), norm=norm)

    pk_pred_modes = F_pred.real ** 2 + F_pred.imag ** 2
    pk_true_modes = F_true.real ** 2 + F_true.imag ** 2

    ky = torch.fft.fftfreq(H, d=1.0, device=device)
    kx = torch.fft.rfftfreq(W, d=1.0, device=device)
    k_grid = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    k_flat = k_grid.flatten()
    M = k_flat.numel()

    nonzero_mask = (k_flat > 0)

    k_max_grid = k_flat.max().item()
    if bin_spacing == "log":
        k_min_pos = k_flat[nonzero_mask].min().item()
        bin_edges = torch.logspace(
            math.log10(k_min_pos), math.log10(k_max_grid), n_bins + 1, device=device,
        )
    elif bin_spacing == "linear":
        bin_edges = torch.linspace(0.0, k_max_grid, n_bins + 1, device=device)
    else:
        raise ValueError(f"Unknown bin_spacing='{bin_spacing}'.")

    bin_idx = torch.bucketize(k_flat, bin_edges, right=False) - 1
    bin_idx = bin_idx.clamp(0, n_bins - 1)

    mode_weight = nonzero_mask.to(pk_pred_modes.dtype)

    pk_pred_flat = pk_pred_modes.view(B * C, M) * mode_weight[None, :]
    pk_true_flat = pk_true_modes.view(B * C, M) * mode_weight[None, :]

    pk_pred_bins = torch.zeros(B * C, n_bins, device=device, dtype=pk_pred_flat.dtype)
    pk_true_bins = torch.zeros(B * C, n_bins, device=device, dtype=pk_true_flat.dtype)
    counts = torch.zeros(n_bins, device=device, dtype=pk_pred_flat.dtype)

    bin_idx_2d = bin_idx.unsqueeze(0).expand(B * C, -1)
    pk_pred_bins.scatter_add_(1, bin_idx_2d, pk_pred_flat)
    pk_true_bins.scatter_add_(1, bin_idx_2d, pk_true_flat)
    counts.scatter_add_(0, bin_idx, mode_weight)

    counts_safe = counts.clamp_min(1.0)
    pk_pred_avg = pk_pred_bins / counts_safe[None, :]
    pk_true_avg = pk_true_bins / counts_safe[None, :]

    valid_bin = (counts > 0.5)

    if use_log_pk:
        diff = torch.log(pk_pred_avg + eps) - torch.log(pk_true_avg + eps)
    else:
        diff = pk_pred_avg - pk_true_avg
    per_bin_sq = diff ** 2
    
    valid_w = valid_bin.to(per_bin_sq.dtype)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_centers = bin_centers.to(device=device, dtype=per_bin_sq.dtype)

    if k_weight_power != 0.0:
        k_max_eff = bin_centers.max().clamp_min(1e-12)

        # Important for negative k_weight_power:
        # avoids exploding weights at very small k.
        bin_centers_safe = bin_centers.clamp_min(1e-6)

        w_k = (bin_centers_safe / k_max_eff).pow(k_weight_power)
    else:
        w_k = torch.ones(n_bins, device=device, dtype=per_bin_sq.dtype)

    # keep only valid bins
    w = w_k * valid_w

    # normalize weights so the loss is a true weighted average over bins
    w = w / w.sum().clamp_min(1e-12)

    per_sc = (per_bin_sq * w[None, :]).sum(dim=1)

    # valid_w = valid_bin.to(per_bin_sq.dtype)
    # if k_weight_power != 0.0:
    #     bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    #     k_max_eff = bin_centers.max().clamp_min(1e-12)
    #     w_k = (bin_centers / k_max_eff).pow(k_weight_power)
    # else:
    #     w_k = torch.ones(n_bins, device=device, dtype=per_bin_sq.dtype)

    # w = w_k * valid_w
    # n_valid = valid_w.sum().clamp_min(1.0)
    # w = w * (n_valid / w.sum().clamp_min(1e-12))

    # per_sc = (per_bin_sq * w[None, :]).sum(dim=1) / n_valid

    if reduce == "per_sample":
        return per_sc.view(B, C).mean(dim=1)
    if reduce == "mean":
        return per_sc.mean()
    raise ValueError(f"Unknown reduce='{reduce}'.")


# =========================================================
# Data processing utilities
# =========================================================

class RunningPercentiles:
    def __init__(self, max_samples=2_000_000, seed=42):
        self.max_samples = max_samples
        self.rng = np.random.default_rng(seed)
        self.buffer = np.empty(max_samples, dtype=np.float32)
        self.n_seen = 0
        self.n_stored = 0

    def update(self, x):
        x = x.ravel()
        for val in x:
            self.n_seen += 1
            if self.n_stored < self.max_samples:
                self.buffer[self.n_stored] = val
                self.n_stored += 1
            else:
                j = self.rng.integers(0, self.n_seen)
                if j < self.max_samples:
                    self.buffer[j] = val

    def compute(self, percentiles=(0.1, 1, 5, 50, 95, 99, 99.9)):
        data = self.buffer[:self.n_stored]
        values = np.percentile(data, percentiles)
        return {p: v for p, v in zip(percentiles, values)}

    def plot_histogram(self, bins=200, log_y=True, title="Histogram", save_path=None):
        data = self.buffer[:self.n_stored]
        plt.figure(figsize=(8, 5))
        plt.hist(data, bins=bins)
        if log_y:
            plt.yscale("log")
        plt.title(title)
        plt.xlabel("Value")
        plt.ylabel("Count")
        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()