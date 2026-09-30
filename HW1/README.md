# CS6180 HW1: Experiment Code and Results

Code and outputs for Homework 1. Problem 4 is a set of architecture ablations on a character-level nanoGPT. The numerical checks and figures for Problems 1 and 2 are also included.

## Contents

| Path | Description |
|---|---|
| `nanoGPT/` | Snapshot of [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) (commit `3adf61e`, MIT license) with the HW1 changes |
| `nanoGPT/model.py`, `nanoGPT/train.py` | New options: `--norm_type` (`layernorm` / `rmsnorm`), `--mlp_type` (`gelu` / `swiglu`), `--pos_type` (`learned` / `none` / `rope`), `--n_kv_head` (grouped query attention), `--seed`, `--save_checkpoint`; every evaluation is logged to `loss_log.json` |
| `nanoGPT/model.diff`, `nanoGPT/train.diff` | Changes against upstream nanoGPT |
| `nanoGPT/sweep.py`, `pipeline.py`, `extra_dropout.py`, `final_extras.py` | Experiment drivers: learning-rate sweeps, extra seeds, follow-up runs |
| `nanoGPT/analyze.py` | Builds all figures and tables in `results/` from `runs/*/loss_log.json` |
| `nanoGPT/bench_variants.py`, `nanoGPT/bench_rmsnorm.py` | Parameter counts, training-step time, KV-cache size; fused vs. unfused RMSNorm timing |
| `nanoGPT/sample_char.py` | Samples text from a character-level checkpoint |
| `nanoGPT/runs/` | All 44 training runs: `loss_log.json` (every evaluation) and `train.log` |
| `nanoGPT/logs/` | Output of the experiment drivers and benchmarks |
| `theory/verify_conv.py` | Checks the convolution, im2col and input-gradient formulas against PyTorch autograd |
| `theory/make_theory_figs.py` | RoPE attention-score plot and the grammar-correction model diagram |
| `results/` | Figures (PNG), result tables (HTML), `summary.json`, `bench.json` |

## Results

Minimum validation loss on `shakespeare_char` (mean ± std over 3 seeds at each variant's best learning rate; default `config/train_shakespeare_char.py` otherwise).

| Variant | Best LR | Params | Min val loss | Δ vs. baseline | Welch t | ms / iter |
|---|---|---|---|---|---|---|
| Baseline (LayerNorm, GELU, learned PE, MHA) | 4e-3 | 10.75M | 1.4597 ± 0.0102 | | | 32.5 |
| RMSNorm | 3e-3 | 10.75M | 1.4606 ± 0.0035 | +0.0009 | +0.1 | 39.9* |
| SwiGLU | 3e-3 | 10.75M | 1.4872 ± 0.0076 | +0.0275 | +3.7 | 34.7 |
| NoPE | 5e-4 | 10.65M | 1.5229 ± 0.0034 | +0.0632 | +10.2 | 32.3 |
| RoPE | 1e-3 | 10.65M | 1.4717 ± 0.0020 | +0.0120 | +2.0 | 36.4 |
| GQA (6 query heads, 3 KV heads) | 3e-3 | 9.86M | 1.4523 ± 0.0069 | -0.0074 | -1.0 | 31.4 |

\* Unfused implementation. With PyTorch's fused `F.rms_norm` the same model runs at 31.8 ms/iter, versus 32.0 ms/iter for LayerNorm in the same benchmark (`logs/final_extras.log`).

The full learning-rate sweep is in `results/lr_table.html`; the per-variant comparison plots are `results/cmp_*.png`.

## Reproduce

```bash
cd nanoGPT
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py --compile=False                      # baseline
python train.py config/train_shakespeare_char.py --compile=False --norm_type=rmsnorm  # RMSNorm
python train.py config/train_shakespeare_char.py --compile=False --mlp_type=swiglu    # SwiGLU
python train.py config/train_shakespeare_char.py --compile=False --pos_type=none      # NoPE
python train.py config/train_shakespeare_char.py --compile=False --pos_type=rope      # RoPE
python train.py config/train_shakespeare_char.py --compile=False --n_kv_head=3        # GQA, group size 2
```

The full study as it was run (`sweep.py` skips runs that already exist in `runs/`):

```bash
python sweep.py lr_sweep > logs/sweep_lr.log
python sweep.py lrs 1 3e-3 > logs/sweep_lr3.log
python pipeline.py > logs/pipeline.log
python extra_dropout.py > logs/extra_dropout.log
python final_extras.py > logs/final_extras.log
python analyze.py
```

`--compile=False` is needed only because Triton is not available on Windows; it does not change the results.

Environment: Windows 11, NVIDIA RTX 5070 Ti (16 GB), PyTorch 2.9.1 + CUDA 12.8, bf16 autocast.

Not included: model checkpoints (`ckpt.pt`, about 130 MB each) and the generated dataset files (`data/shakespeare_char/input.txt`, `*.bin`, `meta.pkl`), which `prepare.py` recreates.

## Credits

nanoGPT is by Andrej Karpathy and is released under the MIT License (`nanoGPT/LICENSE`).
