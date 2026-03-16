# CLAUDE.md — Ground Truth for Matrix-Game-2.0 on Northeastern Explorer HPC

> This file is my authoritative context. I read it at the start of every session.
> Never contradict it without explicit user instruction. Update it when facts change.

---

## Project Identity

**Project:** Matrix-Game-2.0 — Real-time interactive video generation (25 FPS, keyboard + mouse input)
**Repo source:** https://github.com/SkyworkAI/Matrix-Game (cloned locally + on HPC)
**Goal:** Run Matrix-Game-2.0 end-to-end on Northeastern Explorer HPC
  1. Phase 4 — Batch validation via `inference.py` (non-interactive, sbatch)
  2. Phase 5 — Live interactive streaming via `inference_streaming.py` (interactive, srun --pty)
**Local repo path:** `/Volumes/Crucial_X9/Projects/Matrix_2d/Matrix-Game/Matrix-Game-2/`
**Project docs path:** `/Volumes/Crucial_X9/Projects/Matrix_2d/`

---

## My Role

I am the solutions architect and programmer guiding this project.
- I generate every file, script, and command needed
- The user runs commands **manually** on the HPC terminal — I never SSH in
- I update `logs.md`, `hpc_reference.md`, and `build_plan.md` as work progresses
- I tell the user exactly what to run, in what order, and what to look for
- I am honest about what is verified vs. assumed

---

## HPC Cluster Facts

| Item | Value |
|------|-------|
| Cluster | Northeastern Explorer |
| Login | `gupta.yashv@explorer.northeastern.edu` |
| Login node | `explorer-02` |
| Scratch base | `/scratch/gupta.yashv/` |
| Project scratch | `/scratch/gupta.yashv/matrix-game/` |
| Conda env path | `/scratch/gupta.yashv/matrix-game/conda-envs/matrix-game-2.0` |
| Repo on HPC | `/scratch/gupta.yashv/matrix-game/Matrix-Game/Matrix-Game-2/` |
| Weights dir | `/scratch/gupta.yashv/matrix-game/Matrix-Game-2.0/` |
| Outputs dir | `/scratch/gupta.yashv/matrix-game/outputs/` |
| Env loader | `~/load_matrix_env.sh` |

---

## Critical Constraints — Never Violate

1. **Home dir is 97% full.** Nothing large goes in `~`. Only small scripts (< 1 KB each).
2. **GPU requirement: A100 or H100 (≥40 GB VRAM).** V100 (16/32 GB) will OOM on this model.
3. **`inference_streaming.py` uses `input()`.** Must be run with `srun --pty`, never `sbatch`.
4. **`torch.compile` warmup takes 5–10 min on first run.** Never kill the process during this.
5. **All data, weights, envs, and outputs go to `/scratch/gupta.yashv/`.**
6. **Python version must be 3.10.x** in the conda env.
7. **PyTorch must be 2.4.0+cu121** — exact version for flash-attn compatibility.
8. **CUDA module must be `cuda/12.1.1`** — loaded before flash-attn install and at runtime.

---

## Environment Setup

```bash
# ~/load_matrix_env.sh — exact content required
# WARNING: Never use $(conda info --base) — gets Killed on this cluster
module load cuda/12.1.1
module load anaconda3/2024.06
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate /scratch/gupta.yashv/matrix-game/conda-envs/matrix-game-2.0
```

---

## Key Python Packages

| Package | Required Version | Notes |
|---------|-----------------|-------|
| torch | 2.4.0+cu121 | Install from pytorch whl index |
| torchvision | 0.19.0+cu121 | Same index as torch |
| flash-attn | latest | Requires cuda/12.1.1 loaded at build time |
| transformers | ≥4.49.0 | |
| diffusers | latest | |
| accelerate | ≥1.1.1 | |
| omegaconf | latest | |
| einops | latest | |
| safetensors | latest | |
| open_clip_torch | latest | |
| torchao | latest | |
| CLIP | git+openai/CLIP | |

---

## Model Weights

**HuggingFace repo:** `Skywork/Matrix-Game-2.0`
**Download location:** `/scratch/gupta.yashv/matrix-game/Matrix-Game-2.0/`

Required files:
- `Wan2.1_VAE.pth` — VAE decoder weights
- `*.safetensors` — diffusion model checkpoint (one or more files)

---

## Inference Scripts

### inference.py (batch, non-interactive)
```
python inference.py
  --config_path configs/inference_yaml/inference_universal.yaml
  --checkpoint_path <path>/<checkpoint>.safetensors
  --img_path <path>/test_image.png
  --output_folder /scratch/gupta.yashv/matrix-game/outputs/
  --num_output_frames 15
  --seed 42
  --pretrained_model_path /scratch/gupta.yashv/matrix-game/Matrix-Game-2.0
```

### inference_streaming.py (interactive, srun --pty only)
```
python inference_streaming.py
  --config_path configs/inference_yaml/inference_universal.yaml
  --checkpoint_path <path>/<checkpoint>.safetensors
  --output_folder /scratch/gupta.yashv/matrix-game/outputs/
  --seed 42
  --pretrained_model_path /scratch/gupta.yashv/matrix-game/Matrix-Game-2.0
```
Interactive prompts: image path → mouse action (`u`=neutral) → keyboard (`w`=forward) → `n` to stop

---

## SLURM Job Parameters

| Parameter | Batch job | Interactive session |
|-----------|-----------|-------------------|
| `--partition` | `gpu` | `gpu` |
| `--gres` | `gpu:a100:1` | `gpu:a100:1` |
| `--mem` | `64G` | `64G` |
| `--cpus-per-task` | `8` | `8` |
| `--time` | `01:00:00` | `08:00:00` |

Fallback GPU: `--gres=gpu:h100:1` if A100 unavailable.

---

## File Map (Local)

```
/Volumes/Crucial_X9/Projects/Matrix_2d/
├── CLAUDE.md                  ← this file (ground truth)
├── build_plan.md              ← full phased implementation plan
├── logs.md                    ← change log (every edit/command)
├── hpc_reference.md           ← HPC quick-reference commands
├── hpc_gpu_inventory.md       ← GPU/partition inventory
├── config.yaml                ← LiteLLM config (unrelated to Matrix-Game)
├── .sh                        ← NVIDIA NIM API keys (unrelated)
├── agent.md                   ← agent templates (unrelated)
├── venv/                      ← local venv (unrelated)
└── Matrix-Game/
    ├── Matrix-Game-1/         ← v1 (not used)
    └── Matrix-Game-2/         ← ACTIVE — this is what runs on HPC
        ├── inference.py
        ├── inference_streaming.py
        ├── configs/
        ├── pipeline/
        ├── wan/
        ├── utils/
        ├── demo_images/
        ├── setup.py
        └── requirements.txt
```

---

## Current Status

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1 — Audit | Not started | User needs to run audit commands |
| Phase 2 — Fix | Not started | Pending Phase 1 results |
| Phase 3 — load_matrix_env.sh | Not started | |
| Phase 4 — Batch job | Not started | |
| Phase 5 — Interactive streaming | Not started | |

**Last updated:** 2026-03-15

---

## How I Work

- I provide exact commands to copy-paste into the HPC terminal
- User reports back results (success/error messages)
- I update `logs.md` after every action
- I update `hpc_reference.md` when new commands are introduced
- I update this file when facts about the environment are confirmed
- I flag uncertainty explicitly with `[UNVERIFIED]` or `[ASSUMED]`

## Git / Commit Rules

- Only the user (`gupta.yashv`) is listed as author — never add Claude as co-author
- After every meaningful action, update `logs.md` and commit + push
- Commit messages are concise and factual — describe what changed and the outcome
