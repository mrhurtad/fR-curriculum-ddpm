# Conditional DDPM for f(R) matter-density maps

Class-conditional denoising diffusion model that generates 2D matter overdensity maps for four
f(R) modified-gravity models of the MG-Quijote simulations. The model is trained with a
progressive-resolution curriculum (64² → 128² → 256²) and a spectral (Fourier-amplitude) loss.

**Authors:** Saptarshi Pandey (IISER Pune), Elena Giusarma (Michigan Technological University),
Mauricio Reyes  (Michigan Technological University)

**Contact:** Elena Giusarma, egiusarm@mtu.edu

## Repository structure

```
ddpm.py                     entry point (train / sample)
ddpm/
  main.py                   argument parsing, run folders, dispatch
  train.py                  distributed training loop (DDP, AMP, EMA, spectral loss)
  test.py                   sampling with EMA weights
  models/unet_film_cond.py  U-Net with FiLM class + time conditioning
  schedulers/               cosine noise schedule, v-prediction, ancestral sampler
  dataset/fields.py         2.5D slab dataset (5 consecutive slabs as channels)
  utils/                    normalization, spectral losses, diagnostics
configs/                    one config per curriculum stage
data_prep/prepare_data.py   snapshots -> 2D slabs + normalization statistics
scripts/                    training and sampling launchers
```

## Installation

```bash
pip install -r requirements.txt
```

Building the dataset additionally needs [Pylians](https://pylians3.readthedocs.io)
(`readgadget`, `MAS_library`). Training and sampling require CUDA GPUs.

## 1. Data

```bash
python data_prep/prepare_data.py --config configs/ddpm_config_256.yml \
    --snapshots_root /path/to/MG-Quijote/Snapshots --n_realizations 22
```

Particles are assigned to a 256³ grid (CIC), converted to δ = ρ/ρ̄ − 1 and projected into
slabs one grid cell thick (3.9 h⁻¹ Mpc) along the three axes. Realizations are split 85/15 into
train/val. The model sees y = [asinh(δ/α) − μ]/σ with α = 0.8, where μ and σ are computed from the
training slabs.

## 2. Training

The three stages are run in order; each one is initialized from the weights of the previous one
(optimizer and scheduler are re-initialized). Before stages 128 and 256, set
`train_params.pretrained_path` to the `models/latest.pt` of the previous run.

```bash
bash scripts/train.sh configs/ddpm_config_64.yml
bash scripts/train.sh configs/ddpm_config_128.yml
bash scripts/train.sh configs/ddpm_config_256.yml
```

All visible GPUs are used (`CUDA_VISIBLE_DEVICES`). Set `train_params.use_wandb: true` and
`wandb_entity` to log to Weights & Biases.

| | 64² | 128² | 256² |
|---|---|---|---|
| Training input | random crop | random crop | full slab |
| Percentile clipping of y | 0.1–99.9 | 0.1–99.9 | none |
| Batch size per GPU × accumulation | 8 × 2 | 4 × 2 | 2 × 4 |
| Epochs | 80 | 60 | 40 |
| Peak learning rate | 2×10⁻⁴ | 1×10⁻⁴ | 5×10⁻⁵ |
| Spectral-loss exponent p | 0 | −0.25 | −0.5 |

The loss is L = L_v + λ L_spec, with L_v the v-prediction MSE and L_spec the L1 distance between
the Fourier amplitudes of the predicted and true clean fields, weighted by (k/k_max)^p
(`train_params.lam_fourier`, `train_params.l1.k_weight_power`). λ = 0.1 in all stages.

## 3. Sampling

```bash
bash scripts/sample.sh configs/ddpm_config_256.yml runs/stage_256_<timestamp>/models/ema_best.pt
```

Generates 500 samples per class with the EMA weights (1000-step ancestral sampler, x̂₀ clamped to
[−10, 10]). Each sample is a stack of 5 slabs saved as `.npy` in normalized space; apply the inverse
transform δ = α sinh(σ y + μ) to recover the overdensity.

## 4. Monitoring with Weights & Biases

We monitored training with [Weights & Biases](https://wandb.ai). When enabled, the run logs the
total, v-prediction and spectral losses, the learning rate, and periodic comparisons of the power
spectra of generated (model and EMA) and training maps. To use it, log in once with
`wandb login` and set in the config:

train_params:
  use_wandb: true
  wandb_project: "ddpm-cosmology"
  wandb_entity: "<your-wandb-team-or-user>"

Logging is disabled by default (`use_wandb: false`), so training runs without a W&B account.

## License

MIT, see [LICENSE](LICENSE).
