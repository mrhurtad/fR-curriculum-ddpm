# ddpm/test.py
import os
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.multiprocessing import spawn

from ddpm.configs import yadoc
from ddpm.models.unet_film_cond import Unet
from ddpm.schedulers.diffusion_scheduler import NoiseScheduler2D
from ddpm.utils.utils import setup_logging, save_images


def dist_init(rank: int, args):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

    dist.init_process_group(
        backend=args.dist_backend,
        init_method="env://",
        world_size=args.world_size,
        rank=rank,
    )

    dist.barrier(device_ids=[torch.cuda.current_device()])


def _load_checkpoint_flexible(ckpt_path: str, map_location="cpu"):
    # ckpt = torch.load(ckpt_path, map_location=map_location)
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=True)

    # case 1: full training checkpoint
    if isinstance(ckpt, dict) and (
        "model_state_dict" in ckpt or "ema_state_dict" in ckpt
    ):
        return ckpt

    # case 2: plain state_dict
    return {"model_state_dict": ckpt}


def node_worker(args, paths=None):
    args.nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
    args.gpus_per_node = torch.cuda.device_count()

    if args.gpus_per_node < 1:
        raise RuntimeError("No GPUs found.")

    args.world_size = args.nodes * args.gpus_per_node

    if paths is None:
        raise RuntimeError("paths is None. main must pass paths from setup_logging().")

    args.run_dir = paths["base"]
    spawn(gpu_worker, nprocs=args.gpus_per_node, args=(args, paths))


def gpu_worker(local_rank: int, args, paths):
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    node = int(os.environ.get("SLURM_NODEID", 0))
    rank = node * args.gpus_per_node + local_rank
    is_main = (rank == 0)

    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    dist_init(rank, args)

    params = yadoc.parameters(args.config_path)
    diffusion_config = params.diffusion_params
    model_config = params.model_params
    data_config = params.data_params

    # -----------------------------------------------------
    # Model (same as train)
    # -----------------------------------------------------
    model = Unet(
        c_in=model_config.in_channels,
        c_out=model_config.out_channels,
        time_dim=model_config.time_dim,
        num_classes=4,
        cond_continuous=False,
        groups=model_config.groups,
    ).to(device).eval()

    # -----------------------------------------------------
    # Diffusion (same as train)
    # -----------------------------------------------------
    diffusion = NoiseScheduler2D(
        num_timesteps=diffusion_config.num_timesteps,
        beta_start=diffusion_config.beta_start,
        beta_end=diffusion_config.beta_end,
        image_size=model_config.img_size,
        in_channels=model_config.in_channels,
        schedule_type=diffusion_config.schedule_type,
        t_normalize=getattr(diffusion_config, "t_normalize", False),
        device=device,
    )

    # -----------------------------------------------------
    # Load checkpoint
    # -----------------------------------------------------
    if args.ckpt is None:
        raise ValueError("In test mode you must pass --ckpt")

    ckpt = _load_checkpoint_flexible(args.ckpt, map_location="cpu")

    use_ema = bool(getattr(args, "use_ema", False))
    if use_ema and "ema_state_dict" in ckpt:
        state = ckpt["ema_state_dict"]
        loaded_what = "ema_state_dict"
    else:
        state = ckpt["model_state_dict"]
        loaded_what = "model_state_dict"

    missing, unexpected = model.load_state_dict(state, strict=False)

    if is_main:
        print(f"[TEST] Loaded ckpt: {args.ckpt}")
        print(f"[TEST] using: {loaded_what}")
        print(f"[TEST] missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) > 0:
            print("  missing[:5]   =", missing[:5])
        if len(unexpected) > 0:
            print("  unexpected[:5]=", unexpected[:5])

    # -----------------------------------------------------
    # Sampling plan
    # -----------------------------------------------------
    n_per_class = int(getattr(args, "n_per_class", 200))
    sample_bs = int(getattr(args, "sample_bs", 4))

    idx_to_name = {
        0: "fR_p",
        1: "fR_pp",
        2: "fR_ppp",
        3: "fR_pppp",
    }
    classes = [0, 1, 2, 3]

    out_root = Path(paths["test"])
    out_root.mkdir(parents=True, exist_ok=True)

    world = dist.get_world_size()

    if is_main:
        print(f"[TEST] world_size={world} sample_bs={sample_bs} n_per_class={n_per_class}")
        print(f"[TEST] output: {out_root}")

    out_rank = out_root / f"rank_{rank:02d}"
    out_rank.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # Generate
    # -----------------------------------------------------
    for cls in classes:
        class_name = idx_to_name[cls]
        tag = f"class_{cls}_{class_name}"
        cls_dir = out_rank / tag
        cls_dir.mkdir(parents=True, exist_ok=True)

        # rank-sharded indices
        my_indices = list(range(rank, n_per_class, world))

        for start in range(0, len(my_indices), sample_bs):
            chunk = my_indices[start:start + sample_bs]
            b = len(chunk)

            y = torch.full((b,), int(cls), device=device, dtype=torch.long)

            with torch.no_grad():
                x = diffusion.sample(model=model, n=b, labels=y, cfg_scale=0.0)

            # save each sample individually as .npy
            for j, global_i in enumerate(chunk):
                npy_path = cls_dir / f"sample_{global_i:05d}.npy"
                np.save(npy_path, x[j].detach().cpu().numpy())

            # optional preview jpg for the first batch of each class on rank 0
            if is_main and start == 0:
                if is_main and start == 0:
                    save_images(
                        images=x.detach().cpu(),
                        path=str(out_root),
                        epoch=0,
                        name=f"preview_{tag}",
                    )

            del x, y
            torch.cuda.empty_cache()

        if is_main:
            print(f"[TEST] finished cls={cls} ({class_name})")

    dist.barrier()

    if is_main:
        print("[TEST] done. Each rank wrote to:", out_root)

    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--dist_backend", type=str, default="nccl")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--n_per_class", type=int, default=200)
    parser.add_argument("--sample_bs", type=int, default=4)
    parser.add_argument("--use_ema", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    # same logging infra style as train
    paths = setup_logging(
        run_name="test_samples",
        config_path=args.config_path,
        create_dirs=True,
    )

    node_worker(args, paths)


if __name__ == "__main__":
    main()