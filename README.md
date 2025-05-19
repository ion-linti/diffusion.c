# diffusion.c — A Single-File DDIM Sampler in Pure CUDA/C

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()

> **Minimal. Hackable. Educational.**

## ✨ Features

| Area           | Details                                                                          |
| -------------- | -------------------------------------------------------------------------------- |
| Architecture   | U‑Net‑like encoder–decoder (4 down / 4 up blocks) with SiLU and skip connections |
| Time embedding | Sinusoidal; projected via linear layers                                          |
| Sampler        | Deterministic DDIM (β ∈ \[1 × 10⁻⁴, 2 × 10⁻²])                                   |
| Output         | 64 × 64 RGB PNG written by *stb\_image\_write*                                   |
| Build          | **Single `nvcc` command**, no CMake                                              |
| License        | MIT — copy, hack, embed anywhere                                                 |

---

## 📦 Requirements

| Software       | Version                    |
| -------------- | -------------------------- |
| CUDA Toolkit   | ≥ 11.8 (tested on 12.4)    |
| GPU Compute    | SM ≥ 60 (Pascal+)          |
| cuBLAS / cuDNN | Installed with the toolkit |

> **Windows** users: compile with *Developer Command Prompt for VS* or *MSYS2* and replace `-std=c++17` with `/std:c++17`.

---

## 🚀 Quick Start

Clone and build:

```bash
# SSH or HTTPS — pick one
git clone git@github.com:ion-linti/diffusion.c.git
cd diffusion.c

# Compile (adjust compute capability if needed)
nvcc -x cu -O3 -g -std=c++17 diffusion.c -lcublas -lcudnn -o diffusion
```

Run 50 DDIM steps with a fixed seed:

```bash
./diffusion weights.bin out.png 50 --seed 42
```

| Pos | CLI Arg       | Default    | Meaning                         |
| --- | ------------- | ---------- | ------------------------------- |
| 1   | `weights.bin` | *required* | Raw fp32 weight dump (NHWC)     |
| 2   | `out.png`     | *required* | Output filename                 |
| 3   | `steps`       | `50`       | DDIM steps ( ≤ 100 recommended) |
| 4   | `--seed`      | `1234`     | RNG seed                        |

The binary prints generation time in milliseconds.

---

## 🧠 Obtaining Weights

This repo ships **code only**. To generate images you must export a 64×64 U‑Net checkpoint 👉 **`weights.bin`**.

1. Train or fine‑tune any open‑source diffusion model at 64 × 64 (e.g. *Stable Diffusion 1.x* down‑scaled or *Denoising Diffusion GAN*).
2. Use `scripts/export_weights.py` (WIP) to dump **contiguous NHWC fp32** tensors in the exact layer order listed in `LOAD_WEIGHTS_ORDER.md`.

> **Tip:** You can begin with *random* weights to test the pipeline — images will be pure noise yet code path stays valid.

---

## 🛠️ Extending

| Idea                    | Hint                                                                                                                                          |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **fp16 / Tensor Cores** | Change master `typedef float` → `__half`, replace cuBLAS calls with `cublasGemmEx`, switch cuDNN conv algo to *IMPLICIT\_PRECOMP\_GEMM\_FP16* |
| **GroupNorm / Swish**   | Fuse via custom kernel; store `γ/β` alongside conv weights                                                                                    |
| **Text‑to‑Image**       | Swap time‑embedding linear with FiLM‑style text embedding, feed CLIP text latents                                                             |
| **Larger Resolution**   | Parameterize `H,W` and allocate dynamically; add pixel‑shuffle upsample                                                                       |
| **Better Samplers**     | Implement PNDM or DPM‑Solver; reuse the same UNet core                                                                                        |

---

## 📄 License

`diffusion.c` and all auxiliary files are released under the **MIT License**.

---

## 💬 Contact

\| Telegram | `@franzuzik` |
