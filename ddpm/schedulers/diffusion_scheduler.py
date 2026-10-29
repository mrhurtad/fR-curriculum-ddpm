import math
import torch


class NoiseScheduler2D:
    """
    2D diffusion noise scheduler for DDPM-style training/sampling.

    Supports:
      - linear / cosine beta schedules
      - q(x_t | x_0)
      - timestep sampling strategies
      - v-prediction sampling

    Expected tensor shapes:
      x: (B, C, H, W)
      t: (B,) long
    """

    def __init__(
        self,
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        image_size=256,
        in_channels=5,
        schedule_type="linear",
        device="cuda",
        t_normalize=False,
        prediction_type="v",  # "v" or "eps"
    ):
        self.num_timesteps = int(num_timesteps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.image_size = int(image_size)
        self.in_channels = int(in_channels)
        self.schedule_type = str(schedule_type).lower()
        self.device = device
        self.t_normalize = bool(t_normalize)
        self.prediction_type = str(prediction_type).lower()

        if self.prediction_type not in {"v", "eps"}:
            raise ValueError("prediction_type must be 'v' or 'eps'.")

        # --- beta schedule ---
        if self.schedule_type == "linear":
            beta = torch.linspace(
                self.beta_start,
                self.beta_end,
                self.num_timesteps,
                dtype=torch.float32,
                device=self.device,
            )
        elif self.schedule_type == "cosine":
            beta = self._cosine_schedule(self.num_timesteps, s=0.008)
        else:
            raise ValueError("Unsupported schedule_type. Choose from ['linear', 'cosine'].")

        self.beta = torch.clamp(beta, min=1e-8, max=0.999)

        # --- alphas ---
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.alpha_hat = torch.clamp(self.alpha_hat, min=1e-12, max=1.0)

        self.alpha_hat_prev = torch.cat(
            [
                torch.ones(1, device=self.device, dtype=self.alpha_hat.dtype),
                self.alpha_hat[:-1],
            ],
            dim=0,
        )

    # ------------------------------------------------------------
    # schedules
    # ------------------------------------------------------------
    def _cosine_schedule(self, num_timesteps, s=0.008):
        T = int(num_timesteps)
        t = torch.linspace(0, T, T + 1, dtype=torch.float32, device=self.device) / T
        f_t = torch.cos(((t + s) / (1.0 + s)) * (math.pi / 2.0)) ** 2
        alpha_hat = f_t / f_t[0]
        beta = 1.0 - (alpha_hat[1:] / alpha_hat[:-1])
        return beta

    # ------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------
    def _t_in(self, t_long: torch.Tensor) -> torch.Tensor:
        t_in = t_long.float()
        if self.t_normalize:
            t_in = t_in / (self.num_timesteps - 1)
        return t_in

    def _expand(self, vals: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return vals.view(-1, *([1] * (x.ndim - 1)))

    def _alpha_hat_t(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self._expand(self.alpha_hat[t], x)

    def _alpha_t(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self._expand(self.alpha[t], x)

    def _beta_t(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self._expand(self.beta[t], x)

    def _alpha_hat_prev_t(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self._expand(self.alpha_hat_prev[t], x)

    # ------------------------------------------------------------
    # q-sample
    # ------------------------------------------------------------
    def noise_images(self, x, t):
        """
        Sample q(x_t | x_0):

            x_t = sqrt(alpha_hat_t) * x_0 + sqrt(1-alpha_hat_t) * eps

        Returns:
            x_t, eps
        """
        alpha_hat_t = self._alpha_hat_t(t, x)
        alpha_hat_t = torch.clamp(alpha_hat_t, min=1e-12, max=1.0)

        eps = torch.randn_like(x)

        x_t = (
            torch.sqrt(alpha_hat_t) * x
            + torch.sqrt(torch.clamp(1.0 - alpha_hat_t, min=1e-12)) * eps
        )

        return x_t, eps

    # ------------------------------------------------------------
    # timestep sampling
    # ------------------------------------------------------------
    def sample_timesteps(self, n):
        return torch.randint(
            0,
            self.num_timesteps,
            (n,),
            device=self.device,
            dtype=torch.long,
        )

    def sample_timesteps_stratified(self, B, split=0.5):
        T = self.num_timesteps
        split_t = int(split * T)

        n_low = (B + 1) // 2
        n_high = B - n_low

        t_low = torch.randint(0, split_t, (n_low,), device=self.device)
        t_high = torch.randint(split_t, T, (n_high,), device=self.device)

        t = torch.cat([t_low, t_high], dim=0)
        t = t[torch.randperm(B, device=self.device)]

        return t.long()

    def sample_timesteps_mixed(self, B, hi_start=500, p_hi=0.5):
        T = self.num_timesteps
        hi_start = int(max(0, min(hi_start, T - 1)))

        u = torch.rand(B, device=self.device)
        t = torch.empty(B, device=self.device, dtype=torch.long)

        hi_mask = u < p_hi
        lo_mask = ~hi_mask

        if lo_mask.any():
            t[lo_mask] = torch.randint(
                0,
                T,
                (int(lo_mask.sum().item()),),
                device=self.device,
            )

        if hi_mask.any():
            t[hi_mask] = torch.randint(
                hi_start,
                T,
                (int(hi_mask.sum().item()),),
                device=self.device,
            )

        return t.long()

    def sample_timesteps_3bins(
        self,
        B,
        low_end=200,
        mid_end=700,
        frac_low=0.25,
        frac_mid=0.50,
    ):
        T = self.num_timesteps

        low_end = int(max(1, min(low_end, T - 2)))
        mid_end = int(max(low_end + 1, min(mid_end, T - 1)))

        frac_high = 1.0 - frac_low - frac_mid

        if frac_high < 0:
            raise ValueError("frac_low + frac_mid must be <= 1.0")

        n_low = int(round(B * frac_low))
        n_mid = int(round(B * frac_mid))
        n_high = B - n_low - n_mid

        if B >= 3:
            if n_low == 0:
                n_low = 1
            if n_mid == 0:
                n_mid = 1
            if n_high == 0:
                n_high = 1

            total = n_low + n_mid + n_high

            while total > B:
                if n_mid >= n_low and n_mid >= n_high and n_mid > 1:
                    n_mid -= 1
                elif n_high >= n_low and n_high > 1:
                    n_high -= 1
                elif n_low > 1:
                    n_low -= 1

                total = n_low + n_mid + n_high

            if total < B:
                n_mid += B - total

        t_low = torch.randint(0, low_end, (n_low,), device=self.device)
        t_mid = torch.randint(low_end, mid_end, (n_mid,), device=self.device)
        t_high = torch.randint(mid_end, T, (n_high,), device=self.device)

        t = torch.cat([t_low, t_mid, t_high], dim=0)
        t = t[torch.randperm(B, device=self.device)]

        return t.long()

    # ------------------------------------------------------------
    # v-pred helpers
    # ------------------------------------------------------------
    def get_v_target(self, x0, eps, t):
        """
        v = sqrt(alpha_hat_t) * eps - sqrt(1-alpha_hat_t) * x0
        """
        alpha_hat_t = self._alpha_hat_t(t, x0)
        alpha_hat_t = torch.clamp(alpha_hat_t, min=1e-12, max=1.0)

        sqrt_a = torch.sqrt(alpha_hat_t)
        sqrt_1ma = torch.sqrt(torch.clamp(1.0 - alpha_hat_t, min=1e-12))

        return sqrt_a * eps - sqrt_1ma * x0

    def v_to_x0(self, x_t, t, v):
        """
        Convert v-prediction to x0 estimate:

            x0_hat = sqrt(alpha_hat_t) * x_t
                   - sqrt(1-alpha_hat_t) * v
        """
        alpha_hat_t = self._alpha_hat_t(t, x_t)
        alpha_hat_t = torch.clamp(alpha_hat_t, min=1e-12, max=1.0)

        sqrt_a = torch.sqrt(alpha_hat_t)
        sqrt_1ma = torch.sqrt(torch.clamp(1.0 - alpha_hat_t, min=1e-12))

        x0_hat = sqrt_a * x_t - sqrt_1ma * v

        return x0_hat

    def v_to_eps(self, x_t, t, v):
        """
        Convert v-prediction to epsilon estimate:

            eps_hat = sqrt(1-alpha_hat_t) * x_t
                    + sqrt(alpha_hat_t) * v
        """
        alpha_hat_t = self._alpha_hat_t(t, x_t)
        alpha_hat_t = torch.clamp(alpha_hat_t, min=1e-12, max=1.0)

        sqrt_a = torch.sqrt(alpha_hat_t)
        sqrt_1ma = torch.sqrt(torch.clamp(1.0 - alpha_hat_t, min=1e-12))

        eps_hat = sqrt_1ma * x_t + sqrt_a * v

        return eps_hat

    def eps_to_x0(self, x_t, t, eps):
        """
        Convert epsilon-prediction to x0 estimate:

            x0_hat = (x_t - sqrt(1-alpha_hat_t) * eps) / sqrt(alpha_hat_t)
        """
        alpha_hat_t = self._alpha_hat_t(t, x_t)
        alpha_hat_t = torch.clamp(alpha_hat_t, min=1e-12, max=1.0)

        sqrt_a = torch.sqrt(alpha_hat_t)
        sqrt_1ma = torch.sqrt(torch.clamp(1.0 - alpha_hat_t, min=1e-12))

        x0_hat = (x_t - sqrt_1ma * eps) / sqrt_a

        return x0_hat

    # ------------------------------------------------------------
    # model prediction wrapper
    # ------------------------------------------------------------
    def predict_x0_and_eps(self, model, x, t, labels=None, cfg_scale=0.0):
        """
        Runs the model and returns:

            x0_pred, eps_hat, raw_model_output

        If prediction_type == "v":
            raw output is v_theta.

        If prediction_type == "eps":
            raw output is eps_theta.
        """
        t_in = self._t_in(t)

        if labels is not None:
            pred_cond = model(x, t_in, labels).float()
        else:
            pred_cond = model(x, t_in, None).float()

        pred = pred_cond

        if cfg_scale > 0.0 and labels is not None:
            pred_uncond = model(x, t_in, None).float()
            pred = torch.lerp(pred_uncond, pred_cond, cfg_scale)

        if self.prediction_type == "v":
            v_hat = pred
            x0_pred = self.v_to_x0(x, t, v_hat)
            eps_hat = self.v_to_eps(x, t, v_hat)
            return x0_pred, eps_hat, v_hat

        eps_hat = pred
        x0_pred = self.eps_to_x0(x, t, eps_hat)

        return x0_pred, eps_hat, eps_hat

    # ------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------
    def sample(
        self,
        model,
        n,
        labels=None,
        cfg_scale=0.0,
        clamp_x0=True,
        x0_min=-10.0,  # unclipped training data reach z ~ 8
        x0_max=10.0,
    ):
        """
        DDPM ancestral sampling.

        If prediction_type == "v", model output is interpreted as v_theta.
        If prediction_type == "eps", model output is interpreted as eps_theta.
        """
        with torch.no_grad():
            S = self.image_size

            x = torch.randn(
                (n, self.in_channels, S, S),
                device=self.device,
                dtype=torch.float32,
            )

            if labels is not None:
                labels = labels.to(self.device)

            for i in reversed(range(self.num_timesteps)):
                t = torch.full(
                    (n,),
                    i,
                    device=self.device,
                    dtype=torch.long,
                )

                x0_pred, eps_hat, raw_pred = self.predict_x0_and_eps(
                    model=model,
                    x=x,
                    t=t,
                    labels=labels,
                    cfg_scale=cfg_scale,
                )

                if clamp_x0:
                    x0_pred = torch.clamp(x0_pred, x0_min, x0_max)

                alpha_t = self._alpha_t(t, x)
                beta_t = self._beta_t(t, x)
                alpha_hat_t = self._alpha_hat_t(t, x)
                alpha_hat_prev_t = self._alpha_hat_prev_t(t, x)

                one_minus_alpha_hat_t = torch.clamp(
                    1.0 - alpha_hat_t,
                    min=1e-12,
                )

                # Posterior mean:
                #
                # q(x_{t-1} | x_t, x0_pred)
                #
                # mu = coef1 * x0_pred + coef2 * x_t
                coef1 = (
                    torch.sqrt(alpha_hat_prev_t)
                    * beta_t
                    / one_minus_alpha_hat_t
                )

                coef2 = (
                    torch.sqrt(alpha_t)
                    * (1.0 - alpha_hat_prev_t)
                    / one_minus_alpha_hat_t
                )

                mu = coef1 * x0_pred + coef2 * x

                if i > 0:
                    beta_tilde = (
                        beta_t
                        * (1.0 - alpha_hat_prev_t)
                        / one_minus_alpha_hat_t
                    )

                    beta_tilde = torch.clamp(beta_tilde, min=1e-20)

                    x = mu + torch.sqrt(beta_tilde) * torch.randn_like(x)
                else:
                    x = mu

        return x