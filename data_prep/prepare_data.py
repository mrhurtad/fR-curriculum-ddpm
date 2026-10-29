"""
Build the 2D overdensity slabs used for training from the MG-Quijote snapshots.

For every realization the dark-matter particles are assigned to a slab_size^3 grid (CIC),
converted to delta = rho / <rho> - 1 and averaged into slabs along the three axes.
Realizations are split 85/15 into train/val, and the normalization statistics (MU, SIGMA and
the percentiles of the z-scored field) are computed from the training slabs only.

Usage:
    python data_prep/prepare_data.py --config configs/ddpm_config_256.yml \
        --snapshots_root /path/to/MG-Quijote/Snapshots --n_realizations 22
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import readgadget
import MAS_library as MASL

from ddpm.configs import yadoc
from ddpm.utils.utils import transformation, norm_stats_slabs_filename, RunningPercentiles


def iter_slabs_from_delta3d(delta3d, thick=1):
    """
    Yield 2D slabs obtained by averaging a 3D delta field along each axis.

    Parameters
    ----------
    delta3d : np.ndarray
        3D overdensity field of shape (N, N, N).
    thick : int
        Slab thickness in voxels.

    Yields
    ------
    ax : int
        Axis along which the slab is taken (0, 1, 2).
    i : int
        Slab index along that axis.
    d2d : np.ndarray
        2D slab of shape (N, N), dtype float32.
    """
    n = delta3d.shape[0]
    n_full = n // thick
    leftover = n - n_full * thick

    def make_slab(ax, lo, hi):
        slc = [slice(None)] * 3
        slc[ax] = slice(lo, hi)
        d2d = delta3d[tuple(slc)].mean(axis=ax, dtype=np.float64)
        return d2d.astype(np.float32, copy=False)

    for ax in (0, 1, 2):
        for i in range(n_full):
            lo, hi = i * thick, (i + 1) * thick
            yield ax, i, make_slab(ax, lo, hi)

        if leftover > 0:
            lo, hi = n_full * thick, n
            yield ax, n_full, make_slab(ax, lo, hi)


def process_snapshots(
    root,
    masses=("fR_p", "fR_pp", "fR_ppp", "fR_pppp"),
    realizations=range(0, 100),
    snapdir="snapdir_004",
    snap="snap_004",
    slab_size=256,
    BoxSize=1000.0,
    MAS="CIC",
    thick=1,
    norm_method="asinh",
    alpha=1.0,
    eps=1e-6,
    data_dir="data/fR_slabs",
):
    """
    Build 2D delta slabs from snapshots and compute global normalization stats
    from TRAIN slabs only.

    IMPORTANT:
    The final slab size is controlled directly by the MAS grid used to build rho3d.
    That means:
        rho.shape = (slab_size, slab_size, slab_size)

    Output structure:
        data_dir/
            train/<mass>/<real>/slab_ax0_0000.npy
            val/<mass>/<real>/slab_ax0_0000.npy
            norm_stats_....npz

    Percentiles + histogram are computed AFTER z-scoring.
    """
    print("Start converting snapshots to 2D delta slabs (.npy)...")
    print("Normalization stats will be computed from TRAIN slabs only.")
    print("Histogram / percentiles will be computed AFTER z-scoring.")
    print(f"Using MAS grid = slab_size = {slab_size}")

    os.makedirs(data_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Split realizations into train / val
    # ------------------------------------------------------------
    realizations = np.array(sorted(realizations))
    rng = np.random.default_rng(42)
    perm = rng.permutation(realizations)

    n = len(perm)
    n_train = int(0.85 * n)

    train_reals = set(perm[:n_train].tolist())
    val_reals = set(perm[n_train:].tolist())

    print("Split realizations:")
    print("  train:", sorted(train_reals))
    print("  val  :", sorted(val_reals))

    # ------------------------------------------------------------
    # Train-only stats accumulator
    # ------------------------------------------------------------
    sum_s = 0.0
    sum_sq_s = 0.0
    total_voxels = 0
    total_slabs = 0
    train_slab_paths = []

    def split_name(real_id):
        return "train" if real_id in train_reals else "val"

    # ------------------------------------------------------------
    # Main loop: save slabs + compute MU/SIGMA in transformed space
    # ------------------------------------------------------------
    for mass in masses:
        for real in realizations:
            split = split_name(real)
            snap_path = os.path.join(root, mass, str(real), snapdir, snap)

            if not (os.path.exists(snap_path + ".hdf5") or os.path.exists(snap_path + ".0.hdf5")):
                print(f"Skipping {snap_path} (not found)")
                continue

            print(f"Processing: split={split} | Mass={mass} | Realization={real}")

            # --------------------------------------------------------
            # particles -> rho3d on the FINAL target grid
            # --------------------------------------------------------
            pos = readgadget.read_block(snap_path, "POS ", ptype=[1]) / 1e3  # kpc/h -> Mpc/h
            pos = pos.astype(np.float32, copy=False)
            pos = pos % BoxSize

            rho = np.zeros((slab_size, slab_size, slab_size), dtype=np.float32)
            MASL.MA(pos, rho, BoxSize, MAS)

            rho_mean = np.mean(rho, dtype=np.float64)
            if rho_mean <= 0.0 or not np.isfinite(rho_mean):
                raise RuntimeError(
                    f"Bad rho_mean={rho_mean} for {snap_path}. Check BoxSize/units/POS."
                )

            # --------------------------------------------------------
            # rho3d -> delta3d
            # --------------------------------------------------------
            delta = (rho / rho_mean) - 1.0
            delta = np.maximum(delta, -1.0 + 1e-6).astype(np.float32, copy=False)

            # --------------------------------------------------------
            # Output dir
            # --------------------------------------------------------
            out_dir = os.path.join(data_dir, split, mass, str(real))
            os.makedirs(out_dir, exist_ok=True)

            # --------------------------------------------------------
            # delta3d -> slabs, save delta slabs
            # --------------------------------------------------------
            n_slabs_this_box = 0
            for ax, i, d2d in iter_slabs_from_delta3d(delta, thick=thick):
                slab_file = os.path.join(out_dir, f"{mass}_r{real:03d}_ax{ax}_slab{i:04d}.npy")
                np.save(slab_file, d2d, allow_pickle=False)
                n_slabs_this_box += 1

                if split == "train":
                    s2d = transformation(d2d, method=norm_method, alpha=alpha, eps=eps)

                    sum_s += s2d.sum(dtype=np.float64)
                    sum_sq_s += np.square(s2d, dtype=np.float64).sum(dtype=np.float64)
                    total_voxels += s2d.size
                    total_slabs += 1
                    train_slab_paths.append(slab_file)

            print(f"  saved slabs: {n_slabs_this_box}")

    # ------------------------------------------------------------
    # Finalize MU / SIGMA
    # ------------------------------------------------------------
    if total_voxels <= 0:
        raise RuntimeError("No voxels processed for train stats. Check paths / snapshots.")

    mu_global = sum_s / total_voxels
    var = (sum_sq_s / total_voxels) - mu_global * mu_global
    var = max(var, 0.0)
    sigma_global = np.sqrt(var)

    if sigma_global <= 0.0 or not np.isfinite(sigma_global):
        raise RuntimeError(f"Bad sigma_global={sigma_global}")

    # ------------------------------------------------------------
    # PASS 2: percentiles / histogram AFTER z-score
    # ------------------------------------------------------------
    rp = RunningPercentiles()

    print("\nStarting second pass on saved TRAIN slabs for z-scored histogram/percentiles...")
    for i, slab_file in enumerate(train_slab_paths):
        d2d = np.load(slab_file, allow_pickle=False).astype(np.float32, copy=False)
        s2d = transformation(d2d, method=norm_method, alpha=alpha, eps=eps)
        z2d = (s2d - mu_global) / sigma_global
        rp.update(z2d)

        if (i + 1) % 500 == 0 or (i + 1) == len(train_slab_paths):
            print(f"  processed {i+1}/{len(train_slab_paths)} train slabs")

    percentiles = rp.compute()

    print("\nPercentiles (train, z-scored transformed space):")
    for k, v in percentiles.items():
        print(f"  p{k:>5}: {v:.6f}")

    hist_path = os.path.join(data_dir, "hist_train_zscored_transformed.png")
    rp.plot_histogram(
        bins=200,
        log_y=True,
        title="Histogram of z-scored transformed train slabs",
        save_path=hist_path,
    )

    print(f"Histogram saved to: {hist_path}")

    # ------------------------------------------------------------
    # Save stats
    # ------------------------------------------------------------
    name_stats = norm_stats_slabs_filename(
        method=norm_method,
        alpha=alpha,
        eps=eps,
        grid=slab_size,
        thick=thick,
    )

    stats_file = os.path.join(data_dir, name_stats)
    np.savez(
        stats_file,
        MU=mu_global,
        SIGMA=sigma_global,
        P001_Z=percentiles[0.1],
        P1_Z=percentiles[1],
        P5_Z=percentiles[5],
        P50_Z=percentiles[50],
        P95_Z=percentiles[95],
        P99_Z=percentiles[99],
        P999_Z=percentiles[99.9],
        METHOD=norm_method,
        ALPHA=alpha,
        EPS=eps,
        GRID=slab_size,
        SLAB_SIZE=slab_size,
        THICK=thick,
        BOXSIZE=BoxSize,
        MAS=MAS,
        TOTAL_SLABS=total_slabs,
        TOTAL_VOXELS=total_voxels,
        SPLIT="train/val",
        HIST_SPACE="z_scored_transformed_space",
    )

    print("\n" + "=" * 60)
    print("Processing finished ok!")
    print(f"Stats saved in: {stats_file}")
    print(f"Train MU: {mu_global:.6f}")
    print(f"Train SIGMA: {sigma_global:.6f}")
    print(f"Transform: {norm_method} | alpha={alpha} | eps={eps}")
    print(f"Slab size: {slab_size}")
    print(f"Slab thickness: {thick}")
    print(f"Total train slabs: {total_slabs}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="training config (data_params are read from it)")
    parser.add_argument("--snapshots_root", required=True, help="<root>/<class>/<realization>/snapdir_004/")
    parser.add_argument("--n_realizations", type=int, default=22)
    args = parser.parse_args()

    params = yadoc.parameters(args.config)

    process_snapshots(
        root=args.snapshots_root,
        masses=("fR_p", "fR_pp", "fR_ppp", "fR_pppp"),
        realizations=range(args.n_realizations),
        snapdir="snapdir_004",
        snap="snap_004",
        slab_size=256,                      # native grid; lower curriculum stages use random crops
        BoxSize=params.data_params.box_size,
        MAS="CIC",
        thick=params.data_params.slab_thickness,
        norm_method=params.data_params.norm_method,
        alpha=params.data_params.alpha,
        eps=params.data_params.eps,
        data_dir=params.data_params.dataset_path,
    )
