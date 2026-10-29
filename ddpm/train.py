import logging
import os
import sys
import random
import copy
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import optim
from torch.multiprocessing import spawn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, distributed
from torch.utils.tensorboard import SummaryWriter

import wandb

from ddpm.configs import yadoc
from ddpm.dataset.fields import create_data_loaders
from ddpm.models.unet_film_cond import Unet, EMA
from ddpm.schedulers.diffusion_scheduler import NoiseScheduler2D
from ddpm.utils.utils import (
    setup_logging,
    save_images,
    fourier_amplitude_loss,
    spectral_l1_loss,
    power_spectrum_loss,
)
from ddpm.utils.diagnostics import log_pk_comparison_to_wandb

logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)

# =========================================================
# DDP init
# =========================================================
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
    if rank == 0:
        print(f"✅ Rank {rank} initialized (dist_init)")


def node_worker(args, paths=None):
    args.nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
    args.gpus_per_node = torch.cuda.device_count()

    if args.gpus_per_node < 1:
        raise RuntimeError("🚨 No GPUs found on node! Check CUDA setup.")

    args.world_size = args.nodes * args.gpus_per_node

    if paths is None:
        raise RuntimeError("paths is None. main.py must pass paths from setup_logging().")

    args.run_dir = paths["base"]

    spawn(gpu_worker, nprocs=args.gpus_per_node, args=(args, paths))


# =========================================================
# Worker
# =========================================================
def gpu_worker(local_rank: int, args, paths):
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    node = int(os.environ.get("SLURM_NODEID", 0))
    rank = node * args.gpus_per_node + local_rank
    is_main = rank == 0

    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    dist_init(rank, args)

    # -----------------------------------------------------
    # Load params inside worker
    # -----------------------------------------------------
    params = yadoc.parameters(args.config_path)

    diffusion_config = params.diffusion_params
    model_config = params.model_params
    train_config = params.train_params
    data_config = params.data_params

    log_paths = paths

    if is_main:
        print("CONFIG PATH:", args.config_path)
        print("diffusion_config.t_normalize =", diffusion_config.t_normalize)
        print("diffusion_config.schedule_type =", diffusion_config.schedule_type)

    # -----------------------------------------------------
    # Dataset + sampler + loader
    # -----------------------------------------------------
    temp_loader = create_data_loaders(
        directory=os.path.join(data_config.dataset_path, "train"),
        norm_stats_path=data_config.norm_stats_path,
        slab_size=data_config.slab_size,
        batch_size=train_config.batch_size,
        shuffle=False,
        dataset_limit=None,
        half=train_config.half,
        realization_limit=train_config.realization_limit,
        use_percentile_clip=bool(data_config.use_percentile_clip) if "use_percentile_clip" in data_config else False,
    )

    dataset = temp_loader.dataset

    train_sampler = distributed.DistributedSampler(
        dataset,
        num_replicas=args.world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(data_config.num_workers > 0),
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------
    model = Unet(
        c_in=model_config.in_channels,
        c_out=model_config.out_channels,
        time_dim=model_config.time_dim,
        num_classes=4,
        cond_continuous=False,
        groups=model_config.groups,
    ).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # -----------------------------------------------------
    # Pretrained weights for curriculum: 64 -> 128 -> 256
    # This loads ONLY model weights.
    # It is not resume: optimizer, scheduler, epoch, global_step are not loaded.
    # -----------------------------------------------------
    pretrained_path = train_config.pretrained_path

    if pretrained_path is not None:
        if getattr(args, "resume", None):
            raise RuntimeError(
                "Do not use pretrained_path and --resume at the same time. "
                "pretrained_path is for curriculum/fine-tuning; --resume is for continuing the same run."
            )

        if is_main:
            print(f"\n>>> Loading pretrained model weights from:\n{pretrained_path}")

        ckpt_pre = torch.load(
            pretrained_path,
            map_location=device,
            weights_only=False,
        )

        if "model_state_dict" in ckpt_pre:
            state_dict = ckpt_pre["model_state_dict"]
        else:
            state_dict = ckpt_pre

        missing, unexpected = model.module.load_state_dict(state_dict, strict=False)

        if is_main:
            print(">>> Pretrained weights loaded.")
            print(f">>> Missing keys: {missing}")
            print(f">>> Unexpected keys: {unexpected}\n")

        del ckpt_pre
        torch.cuda.empty_cache()
        dist.barrier(device_ids=[torch.cuda.current_device()])

    # -----------------------------------------------------
    # Optimizer / loss
    # -----------------------------------------------------
    optimizer = optim.AdamW(model.parameters(), lr=train_config.lr)
    loss_fn = torch.nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_config.lr,
        steps_per_epoch=len(train_loader),
        epochs=train_config.num_epochs,
        pct_start=0.05,
    )

    # -----------------------------------------------------
    # Diffusion scheduler (2D)
    #
    # IMPORTANT:
    # This training loop uses v-prediction.
    #
    # Forward process:
    #   x_t = sqrt(alpha_hat_t) * x_0
    #       + sqrt(1 - alpha_hat_t) * eps
    #
    # v target:
    #   v = sqrt(alpha_hat_t) * eps
    #     - sqrt(1 - alpha_hat_t) * x_0
    #
    # x0 reconstruction:
    #   x0_pred = sqrt(alpha_hat_t) * x_t
    #           - sqrt(1 - alpha_hat_t) * v_theta
    #
    # IMPORTANT:
    # diffusion.sample() must also be converted to v-prediction.
    # -----------------------------------------------------
    diffusion = NoiseScheduler2D(
        num_timesteps=diffusion_config.num_timesteps,
        beta_start=diffusion_config.beta_start,
        beta_end=diffusion_config.beta_end,
        image_size=model_config.img_size,
        in_channels=model_config.in_channels,
        schedule_type=diffusion_config.schedule_type,
        t_normalize=diffusion_config.t_normalize,
        device=device,
    )

    if is_main:
        print(f"alpha_hat_t range: [{diffusion.alpha_hat.min():.2e}, {diffusion.alpha_hat.max():.6f}]")

    # -----------------------------------------------------
    # AMP
    # -----------------------------------------------------
    use_amp = train_config.use_amp
    amp_dtype = torch.bfloat16
    use_scaler = bool(use_amp and amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # -----------------------------------------------------
    # Logger
    # -----------------------------------------------------
    logger = SummaryWriter(log_paths["logs"]) if is_main else None

    # -----------------------------------------------------
    # Paths / misc
    # -----------------------------------------------------
    model_dir = Path(log_paths["models"])
    model_dir.mkdir(parents=True, exist_ok=True)

    loss_log_path = Path(log_paths["logs"]) / "loss_log.txt"
    save_every_n_epochs = train_config.save_every_n_epochs
    accum_steps = train_config.accum_steps

    lam_fourier = float(train_config.lam_fourier)
    fourier_min_w = float(train_config.fourier_min_w)

    # -----------------------------------------------------
    # Spectral loss dispatch
    # -----------------------------------------------------
    fourier_loss_type = str(train_config.fourier_loss_type).lower()

    if fourier_loss_type == "pk":
        pk_cfg = train_config.pk

        spectral_loss_fn = partial(
            power_spectrum_loss,
            n_bins=int(pk_cfg.n_bins),
            bin_spacing=str(pk_cfg.bin_spacing),
            use_log_pk=bool(pk_cfg.use_log_pk),
            reduce="per_sample",
            k_weight_power=float(pk_cfg.k_weight_power),
        )

    elif fourier_loss_type == "amp":
        amp_cfg = train_config.amp

        spectral_loss_fn = partial(
            fourier_amplitude_loss,
            mode=str(amp_cfg.mode),
            reduce="per_sample",
            k_weight_power=float(amp_cfg.k_weight_power),
        )

    elif fourier_loss_type == "l1":
        l1_cfg = train_config.l1

        spectral_loss_fn = partial(
            spectral_l1_loss,
            use_phase=bool(l1_cfg.use_phase),
            log_amp=bool(l1_cfg.log_amp),
            reduce="per_sample",
            k_weight_power=float(l1_cfg.k_weight_power),
        )

    elif fourier_loss_type == "l1_phase":
        l1p_cfg = train_config.l1_phase

        spectral_loss_fn = partial(
            spectral_l1_loss,
            use_phase=bool(l1p_cfg.use_phase),
            log_amp=bool(l1p_cfg.log_amp),
            reduce="per_sample",
            k_weight_power=float(l1p_cfg.k_weight_power),
        )

    else:
        raise ValueError(
            f"fourier_loss_type must be 'pk', 'amp', 'l1', or 'l1_phase'; got {fourier_loss_type}"
        )

    # -----------------------------------------------------
    # Timestep sampler dispatch
    # -----------------------------------------------------
    t_sampler = train_config.t_sampler

    bin_low_end = train_config.low_end
    bin_mid_end = train_config.mid_end
    bin_frac_low = train_config.frac_low
    bin_frac_mid = train_config.frac_mid

    def sample_t(B):
        if t_sampler == "3bins":
            return diffusion.sample_timesteps_3bins(
                B,
                low_end=bin_low_end,
                mid_end=bin_mid_end,
                frac_low=bin_frac_low,
                frac_mid=bin_frac_mid,
            )
        elif t_sampler == "stratified":
            return diffusion.sample_timesteps_stratified(B)
        elif t_sampler == "mixed":
            return diffusion.sample_timesteps_mixed(B)
        elif t_sampler == "uniform":
            return diffusion.sample_timesteps(B)
        else:
            raise ValueError(f"Unknown t_sampler={t_sampler}")

    # -----------------------------------------------------
    # EMA
    # -----------------------------------------------------
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model.module).to(device).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    # -----------------------------------------------------
    # W&B
    # -----------------------------------------------------
    resume_wandb_id = None

    if is_main and getattr(args, "resume", None):
        ckpt_for_wandb = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )
        resume_wandb_id = ckpt_for_wandb.get("wandb_run_id", None) 

    use_wandb = train_config.use_wandb

    if is_main and use_wandb:
        wandb_project = train_config.wandb_project
        wandb_name = train_config.wandb_name
        wandb_entity = train_config.wandb_entity

        wandb_config = {
            "config_path": args.config_path,
            "prediction_type": "v",
            "lr": train_config.lr,
            "batch_size": train_config.batch_size,
            "num_epochs": train_config.num_epochs,
            "accum_steps": accum_steps,
            "num_timesteps": diffusion_config.num_timesteps,
            "beta_start": diffusion_config.beta_start,
            "beta_end": diffusion_config.beta_end,
            "schedule_type": diffusion_config.schedule_type,
            "t_normalize": diffusion_config.t_normalize,
            "img_size": model_config.img_size,
            "in_channels": model_config.in_channels,
            "out_channels": model_config.out_channels,
            "time_dim": model_config.time_dim,
            "groups": model_config.groups,
            "slab_size": data_config.slab_size,
            "realization_limit": train_config.realization_limit,
            "use_amp": use_amp,
            "amp_dtype": str(amp_dtype),
            "lam_fourier": lam_fourier,
            "fourier_loss_type": fourier_loss_type,
            "fourier_min_w": fourier_min_w,
            "t_sampler": train_config.t_sampler,
            "pretrained_path": train_config.pretrained_path,
        }

        # --------------------------------------------------
        # spectral-loss-specific config
        # --------------------------------------------------
        if fourier_loss_type == "pk":
            wandb_config.update({
                "pk_n_bins": train_config.pk.n_bins,
                "pk_bin_spacing": train_config.pk.bin_spacing,
                "pk_use_log_pk": train_config.pk.use_log_pk,
                "pk_k_weight_power": train_config.pk.k_weight_power,
            })

        elif fourier_loss_type == "amp":
            wandb_config.update({
                "amp_mode": train_config.amp.mode,
                "amp_k_weight_power": train_config.amp.k_weight_power,
            })

        elif fourier_loss_type == "l1":
            wandb_config.update({
                "l1_use_phase": train_config.l1.use_phase,
                "l1_log_amp": train_config.l1.log_amp,
                "l1_k_weight_power": train_config.l1.k_weight_power,
            })

        elif fourier_loss_type == "l1_phase":
            wandb_config.update({
                "l1p_use_phase": train_config.l1_phase.use_phase,
                "l1p_log_amp": train_config.l1_phase.log_amp,
                "l1p_k_weight_power": train_config.l1_phase.k_weight_power,
            })

        wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            name=wandb_name,
            id=resume_wandb_id,
            resume="allow",
            dir=str(log_paths["base"]),
            config=wandb_config,
        )

    # -----------------------------------------------------
    # Resume from checkpoint (if provided)
    # -----------------------------------------------------
    start_epoch = 0
    best_loss = float("inf")
    global_step = 0
    
    resume_path = getattr(args, "resume", "") or ""
    if resume_path:
        if is_main:
            print(f"\n>>> Resuming from {resume_path}")

        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        global_step = ckpt.get("global_step", (int(ckpt["epoch"]) + 1) * len(train_loader))

        # Sanity: prediction type must match current train.py
        ckpt_pred_type = ckpt.get("prediction_type", "epsilon")
        if ckpt_pred_type != "v":
            raise RuntimeError(
                f"Checkpoint was trained with prediction_type='{ckpt_pred_type}' "
                f"but this train.py uses v-prediction. Cannot resume safely."
            )

        # All ranks load model/EMA/optimizer/scheduler/scaler — keeps DDP in sync
        model.module.load_state_dict(ckpt["model_state_dict"])
        ema_model.load_state_dict(ckpt["ema_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        if use_scaler and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        # Restore EMA warmup counter so EMA-decay continues correctly
        if "ema_step" in ckpt:
            ema.step = int(ckpt["ema_step"])

        # RNG states were saved from rank 0 — only rank 0 restores them.
        # Other ranks keep their per-rank seeded RNG.
        if is_main:
            try:
                if "rng_state" in ckpt:
                    torch.set_rng_state(ckpt["rng_state"].cpu())
                if "np_rng_state" in ckpt:
                    np.random.set_state(ckpt["np_rng_state"])
                if "py_rng_state" in ckpt:
                    random.setstate(ckpt["py_rng_state"])
            except Exception as e:
                print(f"[warn] Could not fully restore RNG state: {e}")

        start_epoch = int(ckpt["epoch"]) + 1
        best_loss = float(ckpt.get("best_loss", float("inf")))

        if is_main:
            print(f">>> Resumed: starting at epoch {start_epoch}, best_loss={best_loss:.6f}\n")

        del ckpt
        torch.cuda.empty_cache()
        dist.barrier(device_ids=[torch.cuda.current_device()])

    idx_to_name = {
        0: "fR_p",
        1: "fR_pp",
        2: "fR_ppp",
        3: "fR_pppp",
    }

    # =====================================================
    # Training loop
    # =====================================================
    for epoch in range(start_epoch, train_config.num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)

        running_loss_local = 0.0
        l = len(train_loader)
        save_this_epoch = epoch % save_every_n_epochs == 0

        if is_main:
            print(f"\nStarting epoch {epoch}:")

        optimizer.zero_grad(set_to_none=True)
        first_batch_cpu = None

        for i, batch in enumerate(train_loader):
            if len(batch) == 3:
                images, labels, names = batch
            else:
                images, labels = batch
                names = None

            # ---------------- sanity once ----------------
            if rank == 0 and epoch == start_epoch and i == 0:
                print("\n===== SANITY (rank0, epoch0, iter0) =====")
                print("CONFIG PATH:", args.config_path)
                print("images.shape:", tuple(images.shape), "dtype:", images.dtype)
                print("labels.shape:", tuple(labels.shape), "dtype:", labels.dtype)
                print("labels min/max:", labels.min().item(), labels.max().item())
                print("unique labels:", torch.unique(labels).cpu().tolist())
                print(
                    f"images stats: mean={images.mean():.3f} std={images.std():.3f} "
                    f"min={images.min():.3f} max={images.max():.3f}"
                )

                assert images.ndim == 4, f"Expected (B,C,H,W), got {images.shape}"
                assert images.shape[1] == model_config.in_channels
                assert images.shape[2] == data_config.slab_size
                assert images.shape[3] == data_config.slab_size
                assert labels.dtype == torch.int64
                assert labels.min().item() >= 0 and labels.max().item() < 4

                print(
                    f"prediction_type=v  fourier_loss_type={fourier_loss_type}  "
                    f"lam_fourier={lam_fourier}  fourier_min_w={fourier_min_w}"
                )

                if t_sampler == "3bins":
                    print(
                        f"t_sampler={t_sampler}  low_end={bin_low_end}  mid_end={bin_mid_end}  "
                        f"frac_low={bin_frac_low}  frac_mid={bin_frac_mid}  "
                        f"frac_high={1.0 - bin_frac_low - bin_frac_mid:.2f}"
                    )
                else:
                    print(f"t_sampler={t_sampler}")

                print("===== END SANITY =====\n")

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()

            if is_main and save_this_epoch and i == 0:
                first_batch_cpu = images.detach().cpu()

            # -------------------------------------------------
            # forward / loss
            # -------------------------------------------------
            t = sample_t(images.shape[0])
            t_in = diffusion._t_in(t)

            x_t, noise = diffusion.noise_images(images, t)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                # -------------------------------------------------
                # v target:
                # v = sqrt(alpha_hat_t) * eps - sqrt(1-alpha_hat_t) * x0
                # -------------------------------------------------
                target_v = diffusion.get_v_target(x0=images.float(), eps=noise.float(), t=t)

                # model predicts v, not epsilon
                predicted_v = model(x_t, t_in, labels).float()

                loss_v = loss_fn(predicted_v, target_v)

                # -------------------------------------------------
                # reconstruct x0 from predicted v:
                # x0_pred = sqrt(alpha_hat_t) * x_t
                #         - sqrt(1-alpha_hat_t) * v_theta
                # -------------------------------------------------
                x0_pred = diffusion.v_to_x0(x_t=x_t.float(), t=t, v=predicted_v)

                # spectral loss on reconstructed x0
                loss_fourier_ps = spectral_loss_fn(x0_pred, images.float())  # (B,)

                # SNR-like weighting for spectral loss
                w_t = diffusion.alpha_hat[t].detach().to(loss_fourier_ps.dtype)
                w_t = w_t.clamp_min(fourier_min_w)

                loss_fourier = ((w_t * loss_fourier_ps).sum() / w_t.sum().clamp_min(1e-12))

                loss = (loss_v + lam_fourier * loss_fourier) / accum_steps

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # -------------------------------------------------
            # debug every 100 steps on rank 0
            # -------------------------------------------------
            if rank == 0 and (i % 100 == 0):
                with torch.no_grad():
                    t_cpu = t.detach().cpu()
                    t_in_cpu = t_in.detach().float().cpu()

                    print(
                        f"[debug t] t_normalize={diffusion.t_normalize} "
                        f"t: min={int(t_cpu.min())} mean={t_cpu.float().mean():.1f} max={int(t_cpu.max())} | "
                        f"t_in: min={t_in_cpu.min():.4f} mean={t_in_cpu.mean():.4f} max={t_in_cpu.max():.4f}"
                    )

                    pred = predicted_v.detach().float()
                    tgt = target_v.detach().float()

                    pred_std = pred.std().item()
                    tgt_std = tgt.std().item()
                    corr = torch.mean(pred * tgt).item()
                    corr_norm = corr / (pred_std * tgt_std + 1e-8)

                    loss_print = loss.item() * accum_steps

                    print(
                        f"[epoch {epoch} step {i}] "
                        f"loss={loss_print:.4f} "
                        f"loss_v={loss_v.item():.4f} "
                        f"pred_v_std={pred_std:.3f} "
                        f"tgt_v_std={tgt_std:.3f} "
                        f"corrN={corr_norm:.3f} "
                        f"fourier({fourier_loss_type})={loss_fourier.item():.4f} "
                        f"w_t[min,max]=[{w_t.min().item():.3f},{w_t.max().item():.3f}] "
                        f"x0_pred[mean,std,min,max]=[{x0_pred.mean().item():.3f},"
                        f"{x0_pred.std().item():.3f},"
                        f"{x0_pred.min().item():.3f},"
                        f"{x0_pred.max().item():.3f}]"
                    )

            # -------------------------------------------------
            # optimizer step / EMA
            # -------------------------------------------------
            if ((i + 1) % accum_steps == 0) or ((i + 1) == l):
                if use_scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                ema.step_ema(ema_model, model.module)

            # -------------------------------------------------
            # local logging
            # -------------------------------------------------
            true_loss = float(loss.item() * accum_steps)
            running_loss_local += true_loss

            if is_main:
                # step = epoch * l + i
                step = global_step

                if logger is not None and (step % 10 == 0):
                    logger.add_scalar("loss", true_loss, global_step=step)
                    logger.add_scalar("loss_v", loss_v.item(), global_step=step)
                    logger.add_scalar("loss_fourier", loss_fourier.item(), global_step=step)
                    logger.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step=step)

                    if use_wandb:
                        wandb.log(
                            {
                                "train/loss": true_loss,
                                "train/loss_v": loss_v.item(),
                                "train/loss_fourier": loss_fourier.item(),
                                "train/lr": optimizer.param_groups[0]["lr"],
                                "debug/pred_v_std": pred_std if "pred_std" in locals() else predicted_v.detach().float().std().item(),
                                "debug/target_v_std": tgt_std if "tgt_std" in locals() else target_v.detach().float().std().item(),
                                "debug/corr_norm": corr_norm if "corr_norm" in locals() else 0.0,
                                "debug/x0_pred_mean": x0_pred.detach().float().mean().item(),
                                "debug/x0_pred_std": x0_pred.detach().float().std().item(),
                                "debug/x0_pred_min": x0_pred.detach().float().min().item(),
                                "debug/x0_pred_max": x0_pred.detach().float().max().item(),
                                "debug/w_t_min": w_t.min().item(),
                                "debug/w_t_max": w_t.max().item(),
                                "epoch": epoch,
                            },
                            step=step,
                        )

                with open(loss_log_path, "a") as f:
                    f.write(f"{epoch},{step},{true_loss:.6f}\n")

                if i % 10 == 0:
                    print(f"[Epoch {epoch} | Iter {i}/{l}] loss={true_loss:.4f}", flush=True)
            
            global_step += 1

        # -----------------------------------------------------
        # Reduce avg loss across all ranks
        # -----------------------------------------------------
        loss_tensor = torch.tensor(running_loss_local, device=device, dtype=torch.float32)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        running_loss_global = loss_tensor.item() / args.world_size
        avg_loss = running_loss_global / max(1, l)

        # -----------------------------------------------------
        # End epoch rank 0
        # -----------------------------------------------------
        if is_main:
            print(f"[Epoch {epoch}] avg_loss={avg_loss:.6f}", flush=True)

            if use_wandb:
                wandb.log(
                    {
                        "epoch/avg_loss": avg_loss,
                        "epoch": epoch,
                    },
                    step=global_step, #(epoch + 1) * l,
                )

            if save_this_epoch and first_batch_cpu is not None:
                save_images(
                    first_batch_cpu,
                    log_paths["samples"],
                    epoch,
                    f"original_{model_config.img_size}",
                )

        # -----------------------------------------------------
        # Sampling rank 0 only
        # -----------------------------------------------------
        if save_this_epoch and is_main:
            class_idx = random.choice([0, 1, 2, 3])
            class_name = idx_to_name[class_idx]

            print("\nGenerating samples...")
            print(f"[E{epoch}] Sampling class={class_idx} ({class_name})", flush=True)

            n_local = 1
            y_cond = torch.full((n_local,), int(class_idx), device=device, dtype=torch.long)

            with torch.no_grad():
                model.module.eval()
                ema_model.eval()

                sampled_images = diffusion.sample(
                    model.module,
                    n=n_local,
                    labels=y_cond,
                    cfg_scale=0.0,
                )
                ema_images = diffusion.sample(
                    ema_model,
                    n=n_local,
                    labels=y_cond,
                    cfg_scale=0.0,
                )

                model.module.train()

            # ---- W&B P(k) diagnostic ----
            if use_wandb and first_batch_cpu is not None:
                log_pk_comparison_to_wandb(
                    gen=ema_images.detach().cpu(),
                    orig=first_batch_cpu,
                    epoch=epoch,
                    step=global_step,
                    channel=2,
                    n_bins=64,
                    tag=f"EMA_{epoch}_{class_idx}",
                )
                
            # ---- sample diagnostics ----
            for tag_diag, img_diag in [("model", sampled_images), ("EMA", ema_images)]:
                s = img_diag.detach()
                print(
                    f"  [{tag_diag}] mean={s.mean():.3f} std={s.std():.3f} "
                    f"min={s.min():.3f} max={s.max():.3f}"
                )

                if logger is not None:
                    logger.add_scalar(f"sample_{tag_diag}/mean", s.mean().item(), global_step=epoch)
                    logger.add_scalar(f"sample_{tag_diag}/std", s.std().item(), global_step=epoch)
                    logger.add_scalar(f"sample_{tag_diag}/min", s.min().item(), global_step=epoch)
                    logger.add_scalar(f"sample_{tag_diag}/max", s.max().item(), global_step=epoch)

                if use_wandb:
                    wandb.log(
                        {
                            f"sample/{tag_diag}_mean": s.mean().item(),
                            f"sample/{tag_diag}_std": s.std().item(),
                            f"sample/{tag_diag}_min": s.min().item(),
                            f"sample/{tag_diag}_max": s.max().item(),
                            "epoch": epoch,
                        },
                        step=global_step, #(epoch + 1) * l,
                    )

            if first_batch_cpu is not None:
                ref = first_batch_cpu
                print(
                    f"  [target] mean={ref.mean():.3f} std={ref.std():.3f} "
                    f"min={ref.min():.3f} max={ref.max():.3f}"
                )

                if use_wandb:
                    wandb.log(
                        {
                            "target/mean": ref.mean().item(),
                            "target/std": ref.std().item(),
                            "target/min": ref.min().item(),
                            "target/max": ref.max().item(),
                            "epoch": epoch,
                        },
                        step=global_step, #(epoch + 1) * l,
                    )

            tag = class_name
            save_images(
                sampled_images,
                log_paths["samples"],
                epoch,
                f"sample_{tag}_{model_config.img_size}",
            )
            save_images(
                ema_images,
                log_paths["samples"],
                epoch,
                f"sampleEMA_{tag}_{model_config.img_size}",
            )

        # -----------------------------------------------------
        # Checkpoints rank 0
        # -----------------------------------------------------
        if is_main:
            latest_path = model_dir / "latest.pt"

            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "prediction_type": "v",
                "model_state_dict": model.module.state_dict(),
                "ema_state_dict": ema_model.state_dict(),
                "ema_step": ema.step,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_loss,
                "wandb_run_id": (
                    wandb.run.id
                    if (is_main and use_wandb and wandb.run is not None)
                    else None
                ),
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
                "np_rng_state": np.random.get_state(),
                "py_rng_state": random.getstate(),
            }

            if use_scaler:
                ckpt["scaler_state_dict"] = scaler.state_dict()

            torch.save(ckpt, latest_path)
            torch.save(model.module.state_dict(), model_dir / "model_last.pt")
            torch.save(ema_model.state_dict(), model_dir / "ema_last.pt")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_ckpt_path = model_dir / "best.pt"

                best_ckpt = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "prediction_type": "v",
                    "model_state_dict": model.module.state_dict(),
                    "ema_state_dict": ema_model.state_dict(),
                    "ema_step": ema.step,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best_loss,
                }

                if use_scaler:
                    best_ckpt["scaler_state_dict"] = scaler.state_dict()

                torch.save(best_ckpt, best_ckpt_path)
                torch.save(model.module.state_dict(), model_dir / "model_best.pt")
                torch.save(ema_model.state_dict(), model_dir / "ema_best.pt")

                print(
                    f"🟢 New best model at epoch {epoch} "
                    f"(avg_loss={best_loss:.6f}) → {best_ckpt_path}"
                )

                if use_wandb:
                    wandb.run.summary["best_loss"] = best_loss
                    wandb.run.summary["best_epoch"] = epoch

            torch.cuda.empty_cache()

        dist.barrier()

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------
    if is_main and logger is not None:
        logger.flush()
        logger.close()

    if is_main and use_wandb:
        wandb.finish()

    dist.barrier()
    dist.destroy_process_group()