# HPC Quick Reference — Northeastern Explorer

**User:** `gupta.yashv`
**Cluster:** `explorer.northeastern.edu`
**Updated:** 2026-03-15

---

## SSH & Connection

```bash
# Login
ssh gupta.yashv@explorer.northeastern.edu

# Login node you'll land on: explorer-02
```

---

## Environment Setup

```bash
# Load Matrix-Game environment (run this at the start of every session)
source ~/load_matrix_env.sh

# What that script does:
module load cuda/12.1.1
module load anaconda3/2024.06
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /scratch/gupta.yashv/matrix-game/conda-envs/matrix-game-2.0

# Manual conda activation (if load_matrix_env.sh fails)
# WARNING: Never use $(conda info --base) — gets Killed on this cluster
module load anaconda3/2024.06
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate /scratch/gupta.yashv/matrix-game/conda-envs/matrix-game-2.0
```

---

## Disk & Storage

```bash
# Check home directory usage
df -h ~

# Check scratch space
df -h /scratch

# Check your scratch directory contents
ls /scratch/gupta.yashv/

# Check matrix-game project dir
ls /scratch/gupta.yashv/matrix-game/

# Check weights
ls -lh /scratch/gupta.yashv/matrix-game/Matrix-Game-2.0/

# Check outputs
ls -lh /scratch/gupta.yashv/matrix-game/outputs/

# IMPORTANT: /home is ~97% full — never write large files there
```

---

## Conda Environment

```bash
# List all envs
conda env list

# Create env (if missing)
conda create --prefix /scratch/gupta.yashv/matrix-game/conda-envs/matrix-game-2.0 python=3.10 -y

# Activate
conda activate /scratch/gupta.yashv/matrix-game/conda-envs/matrix-game-2.0

# Deactivate
conda deactivate

# Remove env (nuclear option)
conda env remove --prefix /scratch/gupta.yashv/matrix-game/conda-envs/matrix-game-2.0
```

---

## Module Management

```bash
# List loaded modules
module list

# Load required modules
module load cuda/12.1.1
module load anaconda3/2024.06

# Check CUDA version
nvcc --version
echo $CUDA_HOME

# Set CUDA_HOME manually if needed
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
```

---

## SLURM — Job Submission

### Submit a batch job
```bash
sbatch /scratch/gupta.yashv/matrix-game/test_run.sh
```

### Check your jobs
```bash
squeue -u gupta.yashv
```

### Cancel a job
```bash
scancel <JOBID>
```

### Check job details
```bash
scontrol show job <JOBID>
```

### Watch job log in real time
```bash
tail -f /scratch/gupta.yashv/matrix-game/test_<JOBID>.log
```

### Check job output after completion
```bash
cat /scratch/gupta.yashv/matrix-game/test_<JOBID>.log
cat /scratch/gupta.yashv/matrix-game/test_<JOBID>.err
```

---

## SLURM — Interactive Sessions

### Request A100 interactive session (8 hours — for streaming inference)
```bash
srun --partition=gpu \
     --gres=gpu:a100:1 \
     --mem=64G \
     --cpus-per-task=8 \
     --time=08:00:00 \
     --pty /bin/bash
```

### Request A100 interactive session (2 hours — for quick testing)
```bash
srun --partition=gpu \
     --gres=gpu:a100:1 \
     --mem=64G \
     --cpus-per-task=8 \
     --time=02:00:00 \
     --pty /bin/bash
```

### Fallback: request H100 if A100 unavailable
```bash
srun --partition=gpu \
     --gres=gpu:h100:1 \
     --mem=64G \
     --cpus-per-task=8 \
     --time=08:00:00 \
     --pty /bin/bash
```

### Verify GPU type after srun allocates node
```bash
nvidia-smi
# Must show A100 or H100 — NOT V100 (V100 will OOM)
```

---

## SLURM — Cluster Info

```bash
# Show all partitions and their state
sinfo

# Show GPU availability
sinfo -o "%N %T %G" | grep -E "a100|h100|h200"

# Show partition details
scontrol show partitions

# Check your quotas
squota

# Check node details
scontrol show node d3146
```

### Key Partitions

| Partition | Max Time | Max GPUs | Use For |
|-----------|----------|----------|---------|
| `gpu` | 8 hours | unlimited | Standard GPU jobs |
| `gpu-short` | 2 hours | unlimited | Quick tests |
| `gpu-interactive` | 2 hours | unlimited | Interactive dev |
| `short` | 48 hours | 4 | CPU jobs |
| `sharing` | 1 hour | 277 | Low-priority fill |

---

## Python / Package Management

```bash
# Verify Python version (must be 3.10.x)
python --version

# Verify PyTorch (must be 2.4.0+cu121)
python -c "import torch; print(torch.__version__)"

# Verify CUDA available to PyTorch
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# Verify flash-attn
python -c "import flash_attn; print(flash_attn.__version__)"

# Verify all key imports
python -c "import pipeline; import wan; import utils; print('All imports OK')"

# Install PyTorch (exact version required)
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Install flash-attn (must have cuda/12.1.1 loaded first — takes 15-30 min)
module load cuda/12.1.1
pip install flash-attn --no-build-isolation

# Install repo in dev mode (run from Matrix-Game-2 directory)
cd /scratch/gupta.yashv/matrix-game/Matrix-Game/Matrix-Game-2
python setup.py develop
```

---

## Model Weights Download

```bash
# Start a tmux session (protects download if SSH disconnects)
tmux new -s download

# Inside tmux — download weights
cd /scratch/gupta.yashv/matrix-game
source ~/load_matrix_env.sh
huggingface-cli download Skywork/Matrix-Game-2.0 --local-dir Matrix-Game-2.0

# Detach from tmux (keep download running): Ctrl+B then D
# Reattach to tmux: tmux attach -t download
# List all tmux sessions: tmux ls
# Kill tmux session: tmux kill-session -t download
```

---

## Tmux Quick Reference

```bash
# New session
tmux new -s <name>

# Detach (keep running)
Ctrl+B then D

# List sessions
tmux ls

# Reattach
tmux attach -t <name>

# Kill session
tmux kill-session -t <name>
```

---

## Running Inference

### Run SLURM batch job (non-interactive)
```bash
sbatch /scratch/gupta.yashv/matrix-game/test_run.sh
```

### Run interactive streaming (MUST be on compute node via srun --pty)
```bash
# First: get compute node (see Interactive Sessions section above)
# Then on compute node:
source ~/load_matrix_env.sh
cd /scratch/gupta.yashv/matrix-game/Matrix-Game/Matrix-Game-2

python inference_streaming.py \
    --config_path configs/inference_yaml/inference_universal.yaml \
    --checkpoint_path /scratch/gupta.yashv/matrix-game/Matrix-Game-2.0/<CHECKPOINT>.safetensors \
    --output_folder /scratch/gupta.yashv/matrix-game/outputs/ \
    --seed 42 \
    --pretrained_model_path /scratch/gupta.yashv/matrix-game/Matrix-Game-2.0
```

**Interactive prompts during streaming:**
- Image path → `/scratch/gupta.yashv/matrix-game/test_image.png`
- Mouse action → `u` (neutral/stationary)
- Keyboard action → `w` (forward), `s` (back), `a` (left), `d` (right)
- Stop → `n`

**IMPORTANT: First run takes 5–10 min for `torch.compile` warmup. Do not kill.**

---

## File Transfer (Local ↔ HPC)

```bash
# Download outputs to local machine (run on LOCAL terminal, not HPC)
scp gupta.yashv@explorer.northeastern.edu:/scratch/gupta.yashv/matrix-game/outputs/*.mp4 ~/Desktop/

# Upload a file to HPC scratch
scp /local/path/to/file.png gupta.yashv@explorer.northeastern.edu:/scratch/gupta.yashv/matrix-game/

# Sync entire outputs folder
rsync -avz gupta.yashv@explorer.northeastern.edu:/scratch/gupta.yashv/matrix-game/outputs/ ~/Desktop/matrix-outputs/
```

---

## Key Paths

```
HOME:           ~  (or /home/gupta.yashv) — 97% FULL, use sparingly
SCRATCH:        /scratch/gupta.yashv/
PROJECT:        /scratch/gupta.yashv/matrix-game/
CONDA ENV:      /scratch/gupta.yashv/matrix-game/conda-envs/matrix-game-2.0
REPO:           /scratch/gupta.yashv/matrix-game/Matrix-Game/Matrix-Game-2/
WEIGHTS:        /scratch/gupta.yashv/matrix-game/Matrix-Game-2.0/
OUTPUTS:        /scratch/gupta.yashv/matrix-game/outputs/
ENV LOADER:     ~/load_matrix_env.sh
BATCH SCRIPT:   /scratch/gupta.yashv/matrix-game/test_run.sh
TEST IMAGE:     /scratch/gupta.yashv/matrix-game/test_image.png
```

---

## Troubleshooting

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `CondaError: Run 'conda init'` | conda.sh not sourced | Add `source $(conda info --base)/etc/profile.d/conda.sh` |
| `ImportError: flash_attn` | CUDA mismatch at build time | Reinstall with `cuda/12.1.1` loaded |
| `ModuleNotFoundError: pipeline` | setup.py not run | `cd .../Matrix-Game-2 && python setup.py develop` |
| `FileNotFoundError: Wan2.1_VAE.pth` | Wrong `--pretrained_model_path` | Verify path points to dir containing the file |
| `CUDA out of memory` | Got V100, not A100 | Re-request `--gres=gpu:a100:1` |
| `srun: Unable to allocate resources` | A100 nodes busy | Try `--gres=gpu:h100:1` |
| Job stuck in PENDING | No GPU free | Check `squeue`, try different GPU type |
| Program hangs ~10 min | `torch.compile` warmup | Normal — wait, do not kill |
| SSH timeout / disconnection | Idle connection | Use `tmux` for long-running jobs |
| `pip: command not found` | Env not activated | Run `source ~/load_matrix_env.sh` first |
