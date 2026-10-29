import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_pk_2d(field_2d, n_bins=64):
    """Compute radially-binned 2D power spectrum."""
    field_2d = np.asarray(field_2d)

    H, W = field_2d.shape
    F = np.fft.rfft2(field_2d, norm="ortho")
    power = np.abs(F) ** 2

    ky = np.fft.fftfreq(H, d=1.0)
    kx = np.fft.rfftfreq(W, d=1.0)
    k_grid = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)

    k_flat = k_grid.flatten()
    power_flat = power.flatten()

    mask = k_flat > 0
    k_flat = k_flat[mask]
    power_flat = power_flat[mask]

    k_min = k_flat.min()
    k_max = k_flat.max()

    bin_edges = np.logspace(np.log10(k_min), np.log10(k_max), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    pk = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)

    bin_idx = np.digitize(k_flat, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    for i in range(n_bins):
        m = bin_idx == i
        if m.sum() > 0:
            pk[i] = power_flat[m].mean()
            counts[i] = m.sum()

    valid = counts > 0

    return bin_centers[valid], pk[valid], counts[valid]


def _to_numpy_field(x):
    """
    Accepts:
      torch.Tensor or np.ndarray
      shapes:
        (B,C,H,W), (C,H,W), (H,W)

    Returns:
      np.ndarray with shape (C,H,W)
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    else:
        x = np.asarray(x)

    if x.ndim == 4:
        x = x[0]
    elif x.ndim == 2:
        x = x[None, :, :]
    elif x.ndim == 3:
        pass
    else:
        raise ValueError(f"Expected (B,C,H,W), (C,H,W), or (H,W), got {x.shape}")

    return x


def make_pk_comparison_figure(
    gen,
    orig,
    channel=None,
    n_bins=64,
    title_gen="Generated (EMA)",
    hist_bins=200,
):
    """
    Creates the same 2x2 figure as your pk_comparison.py, but from tensors/arrays.

    Args:
        gen:  torch.Tensor or np.ndarray, shape (B,C,H,W), (C,H,W), or (H,W)
        orig: torch.Tensor or np.ndarray, shape (B,C,H,W), (C,H,W), or (H,W)

    Returns:
        fig, metrics
    """
    gen = _to_numpy_field(gen)
    orig = _to_numpy_field(orig)

    if gen.ndim == 2:
        gen = gen[None, :, :]
    if orig.ndim == 2:
        orig = orig[None, :, :]

    if gen.shape != orig.shape:
        raise ValueError(f"Shape mismatch: gen={gen.shape}, orig={orig.shape}")

    C = gen.shape[0]
    ch = channel if channel is not None else C // 2
    ch = int(np.clip(ch, 0, C - 1))

    # ---- Compute P(k) selected channel ----
    k_orig, pk_orig, _ = compute_pk_2d(orig[ch], n_bins=n_bins)
    k_gen, pk_gen, _ = compute_pk_2d(gen[ch], n_bins=n_bins)

    pk_gen_interp = np.interp(k_orig, k_gen, pk_gen)
    ratio = pk_gen_interp / (pk_orig + 1e-12)

    # ---- All-channel average ----
    pk_orig_all, pk_gen_all = [], []

    for c in range(C):
        _, pk_o, _ = compute_pk_2d(orig[c], n_bins=n_bins)
        _, pk_g, _ = compute_pk_2d(gen[c], n_bins=n_bins)

        # Keep consistent length by interpolating generated to selected k_orig
        # This follows your script logic but makes it safer.
        k_o, pk_o, _ = compute_pk_2d(orig[c], n_bins=n_bins)
        k_g, pk_g, _ = compute_pk_2d(gen[c], n_bins=n_bins)
        pk_g_interp = np.interp(k_o, k_g, pk_g)

        pk_orig_all.append(pk_o)
        pk_gen_all.append(pk_g_interp)

    # If lengths differ due to valid-bin behavior, truncate to common length
    min_len = min(len(p) for p in pk_orig_all + pk_gen_all)
    pk_orig_all = np.stack([p[:min_len] for p in pk_orig_all], axis=0)
    pk_gen_all = np.stack([p[:min_len] for p in pk_gen_all], axis=0)

    k_avg = k_orig[:min_len]
    pk_orig_avg = np.mean(pk_orig_all, axis=0)
    pk_gen_avg = np.mean(pk_gen_all, axis=0)
    ratio_avg = pk_gen_avg / (pk_orig_avg + 1e-12)

    # ---- Plot ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    ax = axes[0, 0]
    ax.loglog(k_orig, pk_orig, "b-", lw=2, label="Original", alpha=0.8)
    ax.loglog(k_gen, pk_gen, "r--", lw=2, label=title_gen, alpha=0.8)
    ax.set_xlabel("k (cycles/pixel)", fontsize=12)
    ax.set_ylabel("P(k)", fontsize=12)
    ax.set_title(f"Power Spectrum — channel {ch}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.semilogx(k_orig, ratio, "k-", lw=2)
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
    ax.fill_between(k_orig, 0.95, 1.05, alpha=0.15, color="green", label="±5%")
    ax.fill_between(k_orig, 0.90, 1.10, alpha=0.1, color="orange", label="±10%")
    ax.set_xlabel("k (cycles/pixel)", fontsize=12)
    ax.set_ylabel("P_gen(k) / P_orig(k)", fontsize=12)
    ax.set_title(f"P(k) Ratio — channel {ch}", fontsize=13)
    ax.set_ylim(0, max(3, np.nanmax(ratio) * 1.1))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.hist(
        orig[ch].flatten(),
        bins=hist_bins,
        alpha=0.6,
        label="Original",
        density=True,
        color="blue",
    )
    ax.hist(
        gen[ch].flatten(),
        bins=hist_bins,
        alpha=0.6,
        label="Generated",
        density=True,
        color="red",
    )
    ax.set_xlabel("pixel value (normalized space)", fontsize=12)
    ax.set_ylabel("density", fontsize=12)
    ax.set_title(f"Pixel Value Distribution — channel {ch}", fontsize=13)
    ax.legend(fontsize=11)

    vmin = min(orig[ch].min(), gen[ch].min()) - 0.5
    vmax = max(orig[ch].max(), gen[ch].max()) + 0.5
    ax.set_xlim(vmin, vmax)

    ax = axes[1, 1]
    ax.semilogx(k_avg, ratio_avg, "k-", lw=2, label=f"mean over {C} channels")
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
    ax.fill_between(k_avg, 0.95, 1.05, alpha=0.15, color="green", label="±5%")
    ax.fill_between(k_avg, 0.90, 1.10, alpha=0.1, color="orange", label="±10%")
    ax.set_xlabel("k (cycles/pixel)", fontsize=12)
    ax.set_ylabel("P_gen(k) / P_orig(k)", fontsize=12)
    ax.set_title(f"P(k) Ratio — average over {C} channels", fontsize=13)
    ax.set_ylim(0, max(3, np.nanmax(ratio_avg) * 1.1))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    metrics = {
        "channel": ch,
        "pk_ratio_min": float(np.nanmin(ratio)),
        "pk_ratio_max": float(np.nanmax(ratio)),
        "pk_ratio_mean": float(np.nanmean(ratio)),
        "pk_ratio_median": float(np.nanmedian(ratio)),
        "pk_ratio_avg_min": float(np.nanmin(ratio_avg)),
        "pk_ratio_avg_max": float(np.nanmax(ratio_avg)),
        "pk_ratio_avg_mean": float(np.nanmean(ratio_avg)),
        "pk_ratio_avg_median": float(np.nanmedian(ratio_avg)),
        "orig_mean": float(orig.mean()),
        "orig_std": float(orig.std()),
        "orig_min": float(orig.min()),
        "orig_max": float(orig.max()),
        "gen_mean": float(gen.mean()),
        "gen_std": float(gen.std()),
        "gen_min": float(gen.min()),
        "gen_max": float(gen.max()),
    }

    return fig, metrics


def log_pk_comparison_to_wandb(
    gen,
    orig,
    epoch,
    step=None,
    channel=None,
    n_bins=64,
    tag="EMA",
):
    """
    Logs the pk_comparison panel to wandb.

    Requires wandb.init(...) to already be called.
    """
    import wandb

    fig, metrics = make_pk_comparison_figure(
        gen=gen,
        orig=orig,
        channel=channel,
        n_bins=n_bins,
        title_gen=f"Generated ({tag})",
    )

    log_dict = {
        f"diagnostics/{tag}_pk_comparison": wandb.Image(fig),
        "epoch": epoch,
    }

    for k, v in metrics.items():
        log_dict[f"diagnostics/{tag}_{k}"] = v

    wandb.log(log_dict, step=step)
    plt.close(fig)

    return metrics