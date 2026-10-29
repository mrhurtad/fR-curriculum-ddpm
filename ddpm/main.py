import argparse
import os
import json
import socket
from datetime import datetime
import shutil

from ddpm.configs import yadoc

from . import train
from . import test
from ddpm.utils.utils import setup_logging


def launch():
    parser = argparse.ArgumentParser(description="Arguments for image generation")
    parser.add_argument(
        "--config",
        dest="config_path",
        required=True,
        help="Path to the configuration file (YAML)",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "test"],
        default="train",
        help="Mode to run: 'train' or 'test'",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to a trained model checkpoint for testing (only used in test mode)",
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a FULL training checkpoint (e.g. .../models/latest.pt) to resume training",
    )

    parser.add_argument(
        "--init_from",
        type=str,
        default=None,
        help="Path to a checkpoint or weights file used only to initialize a NEW training run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    parser.add_argument(
        "--dist_backend",
        type=str,
        default="nccl",
        help="Distributed backend (default: nccl)",
    )

    # ---- Test-specific arguments ----
    parser.add_argument("--n_per_class", type=int, default=200)
    parser.add_argument("--sample_bs", type=int, default=4)
    
    args = parser.parse_args()

    # ---- Load YAML config once (DotConfig) ----
    params = yadoc.parameters(args.config_path)

    # ---- Resolve logging using Slurm env vars (fallback to YAML) ----
    # NOTE: params is DotConfig -> attribute access
    results_dir = os.getenv("RESULTS_DIR", params.logging.results_dir)
    run_name = os.getenv("RUN_NAME", params.logging.run_name)

    # ---- Create run folders ----
    paths = setup_logging(results_dir, run_name)

    rank = int(os.getenv("RANK", os.getenv("SLURM_PROCID", "0")))
    if rank == 0:
        # 1) Snapshot the YAML file as-is (source of truth)
        shutil.copy2(args.config_path, os.path.join(paths["base"], "config_snapshot.yml"))

        # 2) Compact summary (read DotConfig by attributes)
        model = params.model_params
        diff = params.diffusion_params
        train_cfg = params.train_params
        data = params.data_params

        world_size = int(os.getenv("WORLD_SIZE", "1"))
        batch_per_gpu = int(train_cfg.batch_size)
        accum_steps = int(train_cfg.accum_steps)
        global_batch = batch_per_gpu * world_size
        eff_batch_per_update = global_batch * accum_steps

        summary = {
            "timestamp": datetime.now().isoformat(),
            "host": socket.gethostname(),
            "slurm": {
                "job_id": os.getenv("SLURM_JOB_ID"),
                "job_name": os.getenv("SLURM_JOB_NAME"),
                "partition": os.getenv("SLURM_JOB_PARTITION"),
                "nodelist": os.getenv("SLURM_JOB_NODELIST"),
            },
            "logging": {
                "results_dir": results_dir,
                "run_name": run_name,
                "base_dir": paths["base"],
            },
            "model_params": {
                "in_channels": model.in_channels,
                "out_channels": model.out_channels,
            },
            "diffusion_params": {
                "num_timesteps": diff.num_timesteps,
                "beta_start": diff.beta_start,
                "beta_end": diff.beta_end,
                "schedule_type": diff.schedule_type,
            },
            "train_params": {
                "batch_size": train_cfg.batch_size,
                "num_epochs": train_cfg.num_epochs,
                "lr": train_cfg.lr,
                "accum_steps": train_cfg.accum_steps,
                "save_every_n_epochs": train_cfg.save_every_n_epochs,
                "use_amp": train_cfg.use_amp,
            },
            "data_params": {
                "dataset_path": data.dataset_path,
                "slab_size": data.slab_size,
                "slab_thickness": data.slab_thickness,
                "norm_method": data.norm_method,
                "alpha": data.alpha,
                "eps": data.eps,
                "num_workers": data.num_workers,
            },
            "ddp": {
                "world_size": world_size,
                "batch_per_gpu": batch_per_gpu,
                "accum_steps": accum_steps,
                "global_batch": global_batch,
                "effective_batch_per_update": eff_batch_per_update,
            },
        }

        with open(os.path.join(paths["base"], "run_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"🗂️ Logging base dir: {paths['base']}")

    # Dispatch
    if args.mode == "train":
        train.node_worker(args, paths=paths)
    else:
        test.node_worker(args, paths=paths)


if __name__ == "__main__":
    launch()