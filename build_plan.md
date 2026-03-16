# Build Plan — Matrix-Game-2.0 on Northeastern Explorer HPC

**Author:** Solutions Architect (Claude)
**User:** gupta.yashv
**Cluster:** Northeastern Explorer
**Created:** 2026-03-15
**Last reviewed:** 2026-03-15

---

## Overview

Matrix-Game-2.0 is a real-time interactive video generation model from SkyworkAI. Given a starting image and keyboard/mouse inputs, it generates a continuous video stream at 25 FPS using a causal diffusion transformer architecture.

**End goal:** Run `inference_streaming.py` interactively on an A100/H100 GPU on Explorer HPC, generating video from user-controlled inputs.

**Intermediate milestone:** Run `inference.py` as a batch job to validate the full stack (env, weights, imports, CUDA) before committing to interactive mode.

---

## Architecture Overview

```
User input (keyboard + mouse)
        ↓
inference_streaming.py
        ↓
CausalInferenceStreamingPipeline  (pipeline/)
        ↓
WanDiffusionWrapper               (wan/)  ← diffusion transformer
        ↓
VAEDecoderWrapper                 (demo_utils/)
        ↓
Wan2.1_VAE.pth + *.safetensors   (weights)
        ↓
MP4 output frames
```

**Key dependencies:**
- `torch.compile` used for speed — requires ~5–10 min warmup on first run
- `flash_attn` required for memory-efficient attention — must be compiled against CUDA 12.1
- Model loads entirely onto GPU — requires ≥40 GB VRAM (A100/H100 only)
- `inference_streaming.py` uses Python `input()` — requires a real TTY (`srun --pty`)

---

## Phase 1 — Audit Current HPC State

**Where:** Login node (`explorer-02`)
**Purpose:** Establish exactly what exists before doing any install work

### 1.1 Check home and disk usage
```bash
ls ~
df -h ~
df -h /scratch
```
**Pass criteria:**
- `load_matrix_env.sh` visible in `~`
- `/home` usage shown (expected ~97%)
- `/scratch` has ≥100 GB free

### 1.2 Check scratch directory structure
```bash
ls /scratch/gupta.yashv/
ls /scratch/gupta.yashv/matrix-game/ 2>/dev/null || echo "MISSING"
```
**Pass criteria:** `matrix-game/` exists

### 1.3 Check conda environment
```bash
module load anaconda3/2024.06
source $(conda info --base)/etc/profile.d/conda.sh
conda env list
```
**Pass criteria:** `/scratch/gupta.yashv/matrix-game/conda-envs/matrix-game-2.0` visible in list

### 1.4 Check Python and core packages
```bash
conda activate /scratch/gupta.yashv/matrix-game/conda-envs/matrix-game-2.0
python --version
python -c "import torch; print(torch.__version__)"
python -c "import flash_attn; print(flash_attn.__version__)"
python -c "import pipeline; import wan; import utils; print('All imports OK')"
```
**Pass criteria:**
- Python 3.10.x
- torch 2.4.0+cu121
- flash_attn imports cleanly
- pipeline/wan/utils all import

### 1.5 Check repo clone
```bash
ls /scratch/gupta.yashv/matrix-game/Matrix-Game/Matrix-Game-2/
```
**Pass criteria:** `inference.py`, `inference_streaming.py`, `configs/`, `pipeline/`, `wan/`, `utils/` present

### 1.6 Check model weights
```bash
ls -lh /scratch/gupta.yashv/matrix-game/Matrix-Game-2.0/
```
**Pass criteria:** `Wan2.1_VAE.pth` present + at least one `*.safetensors` file

### Phase 1 Decision Gate
After running all checks, report which items **PASS** and which **FAIL**.
Only run Phase 2 steps for failed items.

> **NOTE (confirmed 2026-03-15):** The import test `python -c "import pipeline; import wan; import utils"` **cannot be run on the login node**. `wan/modules/t5.py` calls `torch.cuda.current_device()` at class definition time, which requires a real GPU. This test must be skipped on login nodes — import correctness is validated by the batch job in Phase 4.

---

## Phase 2 — Fix Missing Pieces

**Where:** Login node (`explorer-02`)
**Run only the sub-steps corresponding to Phase 1 failures.**

### 2.1 Create scratch dir (if missing)
```bash
mkdir -p /scratch/gupta.yashv/matrix-game
mkdir -p /scratch/gupta.yashv/matrix-game/outputs
```

### 2.2 Create conda environment (if missing or broken)
```bash
module load cuda/12.1.1
module load anaconda3/2024.06
source $(conda info --base)/etc/profile.d/conda.sh
conda create --prefix /scratch/gupta.yashv/matrix-game/conda-envs/matrix-game-2.0 python=3.10 -y
conda activate /scratch/gupta.yashv/matrix-game/conda-envs/matrix-game-2.0
python --version   # must show 3.10.x
```

### 2.3 Install PyTorch (if wrong version or missing)
```bash
# Must have conda env active
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__)"   # must print 2.4.0+cu121
```

### 2.4 Install flash-attn (if missing — 15–30 min compile)
```bash
# MUST have cuda/12.1.1 loaded
module load cuda/12.1.1
# If nvcc not found:
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
pip install flash-attn --no-build-isolation
python -c "import flash_attn; print(flash_attn.__version__)"
```
> **Warning:** This compiles from source. Takes 15–30 min on login node. Run inside `tmux`.

### 2.5 Install remaining packages
```bash
pip install \
    opencv-python diffusers "transformers>=4.49.0" "tokenizers>=0.20.3" \
    "accelerate>=1.1.1" tqdm imageio easydict ftfy imageio-ffmpeg numpy \
    omegaconf einops av safetensors open_clip_torch \
    pycocotools lmdb matplotlib sentencepiece pydantic \
    scikit-image "huggingface_hub[cli]" dominate torchao flask flask-socketio wandb

pip install git+https://github.com/openai/CLIP.git
```

### 2.6 Install TensorRT packages (optional — skip if they fail)
```bash
pip install nvidia-pyindex nvidia-tensorrt pycuda onnx onnxruntime onnxscript onnxconverter_common \
    || echo "WARNING: TensorRT packages failed — safe to skip"
```

### 2.7 Clone repo (if missing)
```bash
cd /scratch/gupta.yashv/matrix-game
git clone https://github.com/SkyworkAI/Matrix-Game.git
```

### 2.8 Install repo in dev mode (if imports failed)
```bash
cd /scratch/gupta.yashv/matrix-game/Matrix-Game/Matrix-Game-2
python setup.py develop
python -c "import pipeline; import wan; import utils; print('All imports OK')"
```

### 2.9 Download model weights (if missing — run in tmux, takes 20–60 min)
```bash
tmux new -s download
# Inside tmux:
cd /scratch/gupta.yashv/matrix-game
source ~/load_matrix_env.sh
huggingface-cli download Skywork/Matrix-Game-2.0 --local-dir Matrix-Game-2.0
# Detach with Ctrl+B then D
# Reattach: tmux attach -t download
```
**Verify after completion:**
```bash
ls -lh /scratch/gupta.yashv/matrix-game/Matrix-Game-2.0/*.safetensors
ls -lh /scratch/gupta.yashv/matrix-game/Matrix-Game-2.0/Wan2.1_VAE.pth
```

---

## Phase 3 — Validate and Lock load_matrix_env.sh

**Where:** Login node
**Purpose:** Ensure the env loader script is correct so every subsequent session just runs `source ~/load_matrix_env.sh`

```bash
# Overwrite with exactly this content:
cat > ~/load_matrix_env.sh << 'EOF'
module load cuda/12.1.1
module load anaconda3/2024.06
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /scratch/gupta.yashv/matrix-game/conda-envs/matrix-game-2.0
EOF
chmod +x ~/load_matrix_env.sh
```

**Test it:**
```bash
source ~/load_matrix_env.sh
python -c "import torch; print(torch.__version__)"
```

---

## Phase 4 — Batch Validation Job

**Where:** Login node (submit) → compute node (runs automatically)
**Purpose:** Validate the entire pipeline (imports, weights loading, CUDA, inference) in a non-interactive batch job before attempting interactive streaming.

### 4.1 Confirm test image exists
```bash
ls /scratch/gupta.yashv/matrix-game/test_image.png 2>/dev/null || \
    cp /scratch/gupta.yashv/matrix-game/Matrix-Game/Matrix-Game-2/demo_images/universal/*.png \
       /scratch/gupta.yashv/matrix-game/test_image.png
```

### 4.2 Get checkpoint filename
```bash
ls /scratch/gupta.yashv/matrix-game/Matrix-Game-2.0/*.safetensors
```
Note the exact filename — use it in the SLURM script below.

### 4.3 Create SLURM batch script
File to create: `/scratch/gupta.yashv/matrix-game/test_run.sh`

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=matrix-game-test
#SBATCH --output=/scratch/gupta.yashv/matrix-game/test_%j.log
#SBATCH --error=/scratch/gupta.yashv/matrix-game/test_%j.err

source ~/load_matrix_env.sh
cd /scratch/gupta.yashv/matrix-game/Matrix-Game/Matrix-Game-2

python inference.py \
    --config_path configs/inference_yaml/inference_universal.yaml \
    --checkpoint_path /scratch/gupta.yashv/matrix-game/Matrix-Game-2.0/<CHECKPOINT>.safetensors \
    --img_path /scratch/gupta.yashv/matrix-game/test_image.png \
    --output_folder /scratch/gupta.yashv/matrix-game/outputs/ \
    --num_output_frames 15 \
    --seed 42 \
    --pretrained_model_path /scratch/gupta.yashv/matrix-game/Matrix-Game-2.0

echo "DONE — check outputs/"
```

> Replace `<CHECKPOINT>` with the actual filename from step 4.2.

### 4.4 Submit and monitor
```bash
sbatch /scratch/gupta.yashv/matrix-game/test_run.sh
squeue -u gupta.yashv
tail -f /scratch/gupta.yashv/matrix-game/test_<JOBID>.log
```

### Phase 4 Pass Criteria
- Job reaches `RUNNING` state (not stuck in `PENDING` forever)
- Log shows model loading messages (no ImportError, no FileNotFoundError)
- `torch.compile` warmup messages appear (normal — wait 5–10 min)
- MP4 file appears in `/scratch/gupta.yashv/matrix-game/outputs/`
- No Python traceback in `.log` or `.err` files

### Phase 4 Common Failures
| Error | Diagnosis | Fix |
|-------|-----------|-----|
| `ModuleNotFoundError: pipeline` | setup.py develop not run | Re-run Phase 2.8 |
| `FileNotFoundError: Wan2.1_VAE.pth` | Wrong `--pretrained_model_path` | Verify weights dir |
| `CUDA out of memory` | Got V100 instead of A100 | Re-submit specifying `gpu:a100:1` |
| `ImportError: flash_attn` | flash-attn not built with correct CUDA | Re-run Phase 2.4 with `cuda/12.1.1` loaded |
| Job stays PENDING | No A100 available | Try `--gres=gpu:h100:1` as fallback |

---

## Phase 5 — Interactive Streaming

**Where:** Compute node via `srun --pty`
**Purpose:** Run the live interactive inference session

> Do NOT attempt Phase 5 until Phase 4 fully passes.

### 5.1 Request interactive GPU node
```bash
srun --partition=gpu \
     --gres=gpu:a100:1 \
     --mem=64G \
     --cpus-per-task=8 \
     --time=08:00:00 \
     --pty /bin/bash
```
Wait for shell prompt to change (e.g. `[gupta.yashv@d3146 ~]$`).

```bash
nvidia-smi   # verify A100 or H100, NOT V100
```

### 5.2 Load environment and navigate to repo
```bash
source ~/load_matrix_env.sh
cd /scratch/gupta.yashv/matrix-game/Matrix-Game/Matrix-Game-2
```

### 5.3 Run interactive streaming
```bash
python inference_streaming.py \
    --config_path configs/inference_yaml/inference_universal.yaml \
    --checkpoint_path /scratch/gupta.yashv/matrix-game/Matrix-Game-2.0/<CHECKPOINT>.safetensors \
    --output_folder /scratch/gupta.yashv/matrix-game/outputs/ \
    --seed 42 \
    --pretrained_model_path /scratch/gupta.yashv/matrix-game/Matrix-Game-2.0
```

**First run sequence:**
1. `torch.compile` warmup — wait 5–10 min, do NOT interrupt
2. Prompt: enter image path → `/scratch/gupta.yashv/matrix-game/test_image.png`
3. Prompt: mouse action → `u` (neutral/stationary camera)
4. Prompt: keyboard action → `w` (move forward)
5. Model generates video frames — watch for output files in `outputs/`
6. Press `n` when prompted to stop generation

### 5.4 Download output to local machine
Run this in a **separate terminal on your local machine:**
```bash
scp gupta.yashv@explorer.northeastern.edu:/scratch/gupta.yashv/matrix-game/outputs/*.mp4 ~/Desktop/
```

### Phase 5 Pass Criteria
- Interactive prompts appear (image path, mouse action, keyboard action)
- Model generates at least one video chunk without error
- MP4 file in `outputs/` is playable and shows expected scene motion

---

## Verification Checklist

### Environment
- [ ] `/scratch/gupta.yashv/matrix-game/` exists
- [ ] `/scratch/gupta.yashv/matrix-game/outputs/` exists
- [ ] Conda env activates without error
- [ ] `python --version` → `3.10.x`
- [ ] `torch.__version__` → `2.4.0+cu121`
- [ ] `flash_attn` imports cleanly
- [ ] `import pipeline; import wan; import utils` all pass
- [ ] `~/load_matrix_env.sh` correct and tested

### Weights
- [ ] `Wan2.1_VAE.pth` present in `Matrix-Game-2.0/`
- [ ] `*.safetensors` present in `Matrix-Game-2.0/`

### Batch Job (Phase 4)
- [ ] SLURM job runs on A100 or H100
- [ ] No traceback in log/err files
- [ ] MP4 appears in `outputs/`

### Interactive Streaming (Phase 5)
- [ ] `srun --pty` allocates A100/H100
- [ ] `nvidia-smi` confirms correct GPU
- [ ] `torch.compile` warmup completes (≥5 min wait)
- [ ] Interactive prompts respond correctly
- [ ] Video output playable on local machine

---

## Rollback / Recovery

| Situation | Recovery Action |
|-----------|----------------|
| Env corrupted | `conda env remove --prefix .../matrix-game-2.0` then redo Phase 2.2–2.8 |
| flash-attn broken | `pip uninstall flash-attn`, reload `cuda/12.1.1`, reinstall |
| Weights corrupted | Delete `Matrix-Game-2.0/` and re-run Phase 2.9 |
| Job stuck pending | `scancel <JOBID>`, try H100 partition |
| srun session lost | Re-run step 5.1 — outputs already saved to scratch |
