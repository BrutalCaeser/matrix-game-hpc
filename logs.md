# Change Log — Matrix-Game-2.0 HPC Deployment

All edits, commands run, outputs observed, and decisions made are recorded here in chronological order.
Format: `[DATE TIME] [WHO] [ACTION] — RESULT`

---

## Log Format Key

- `[ARCH]` — Action by Claude (architect/programmer)
- `[USER]` — Action by user on HPC terminal
- `[RESULT]` — Output observed
- `[DECISION]` — Decision made based on result
- `[BLOCKED]` — Something failed or is pending

---

## Session 1 — 2026-03-15

### 09:00 [ARCH] — Project initialization
Created foundational project files:
- `CLAUDE.md` — ground truth context file
- `build_plan.md` — full phased implementation plan (5 phases)
- `logs.md` — this file
- `hpc_reference.md` — HPC command quick reference

Reviewed local repo at `/Volumes/Crucial_X9/Projects/Matrix_2d/Matrix-Game/Matrix-Game-2/`.
Confirmed presence of: `inference.py`, `inference_streaming.py`, `configs/`, `pipeline/`, `wan/`, `utils/`, `setup.py`, `requirements.txt`.

Reviewed existing `hpc_gpu_inventory.md` — confirmed cluster has A100 (nodes d3146, d3203, d3149, d3204) and H200/H100 nodes available.

Reviewed existing `agent.md` — contains NVIDIA NIM API keys and model templates (unrelated to Matrix-Game deployment).

**Status:** Project structure established. Awaiting user to run Phase 1 audit commands.

---

## Pending — Phase 1 Audit

User needs to run the following on `explorer-02` and report back results:

```
Step 1.1: ls ~ && df -h ~ && df -h /scratch
Step 1.2: ls /scratch/gupta.yashv/matrix-game/
Step 1.3: module load anaconda3/2024.06 && source $(conda info --base)/etc/profile.d/conda.sh && conda env list
Step 1.4: conda activate .../matrix-game-2.0 && python --version && python -c "import torch; print(torch.__version__)" && python -c "import flash_attn; print(flash_attn.__version__)" && python -c "import pipeline; import wan; import utils; print('All imports OK')"
Step 1.5: ls /scratch/gupta.yashv/matrix-game/Matrix-Game/Matrix-Game-2/
Step 1.6: ls -lh /scratch/gupta.yashv/matrix-game/Matrix-Game-2.0/
```

---

---

## Session 2 — 2026-03-15 — Phase 1 Audit (Blocks 1–3)

### [USER] Block 1 — Disk check
```
ls ~  →  load_matrix_env.sh  microDLM  ondemand
df -h ~  →  155T total, 151T used, 5.0T avail, 97% (/home)
df -h /scratch  →  2.2P total, 1.2P used, 1003T avail, 54%
```
**[RESULT] PASS:**
- `load_matrix_env.sh` confirmed present in `~`
- Home is 97% full — confirmed, no large writes to `~`
- Scratch has **1003 TB free** — ample for weights, env, outputs

### [USER] Block 2 — Scratch dir check
```
ls /scratch/gupta.yashv/matrix-game/  →  conda-envs  conda-pkgs
```
**[RESULT] PARTIAL:**
- `/scratch/gupta.yashv/matrix-game/` exists ✓
- Only `conda-envs` and `conda-pkgs` subdirs present
- `Matrix-Game/` repo directory: **MISSING** → need to clone
- `Matrix-Game-2.0/` weights directory: **MISSING** → need to download

### [USER] Block 3 — Conda env list
```
conda env list shows:
  base         /shared/EL9/explorer/anaconda3/2024.06
  info6105_env /shared/EL9/explorer/anaconda3/2024.06/envs/info6105_env
  qiime2-amplicon-2025.7  ...
```
**[RESULT] FAIL:**
- `matrix-game-2.0` env NOT in conda env list
- `conda-envs/` directory exists on scratch but env may be empty/incomplete
- Need to check: `ls /scratch/gupta.yashv/matrix-game/conda-envs/`

### [DECISION] Phase 1 Audit — partial results
- Awaiting: Block 4 (package check), Block 5 (repo check), Block 6 (weights check)
- Also need: `ls /scratch/gupta.yashv/matrix-game/conda-envs/` to see env state
- Likely needed: create conda env, clone repo, download weights

<!-- NEW ENTRIES GO BELOW THIS LINE — most recent at bottom -->

---

## Session 3 — 2026-03-15 — Phase 2: Conda Env Rebuild

### [USER] Checked conda-envs dir
```
ls /scratch/gupta.yashv/matrix-game/conda-envs/  →  matrix-game-2.0
```
**[RESULT]** Directory existed but was NOT a valid conda environment.
```
conda env remove --prefix .../matrix-game-2.0
→ DirectoryNotACondaEnvironmentError: target directory exists but is not a conda environment
```
**[DECISION]** Manual removal required — ran `rm -rf` on the stale directory.

### [USER] Removed stale env dir
```bash
rm -rf /scratch/gupta.yashv/matrix-game/conda-envs/matrix-game-2.0
```
**[RESULT] SUCCESS** — directory removed.

### [USER] Created fresh conda env (Phase 2.2)
```bash
module load cuda/12.1.1
module load anaconda3/2024.06
source $(conda info --base)/etc/profile.d/conda.sh
conda create --prefix /scratch/gupta.yashv/matrix-game/conda-envs/matrix-game-2.0 python=3.10 -y
conda activate /scratch/gupta.yashv/matrix-game/conda-envs/matrix-game-2.0
python --version  →  Python 3.10.20
```
**[RESULT] PASS** — conda env created at correct path with Python 3.10.20.

### [ARCH] Git repo initialized locally
Initialized git repo in `/Volumes/Crucial_X9/Projects/Matrix_2d/` to track project docs (CLAUDE.md, build_plan.md, logs.md, hpc_reference.md, hpc_gpu_inventory.md).
Created GitHub repo: https://github.com/BrutalCaeser/matrix-game-hpc

### [USER] Installed PyTorch (Phase 2.3)
```bash
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__)"  →  2.4.0+cu121
```
**[RESULT] PASS** — PyTorch 2.4.0+cu121 confirmed.

### [USER] Cloned repo (Phase 2.7)
```bash
cd /scratch/gupta.yashv/matrix-game
git clone https://github.com/SkyworkAI/Matrix-Game.git
ls Matrix-Game/Matrix-Game-2/
→  README.md  assets  configs  demo_images  demo_utils  inference.py
   inference_streaming.py  pipeline  requirements.txt  setup.py  utils  wan
```
**[RESULT] PASS** — All required files and directories present.

### [USER] flash-attn install — Attempt 1 (Phase 2.4)
```bash
pip install flash-attn --no-build-isolation
```
**[RESULT] FAIL** — `ModuleNotFoundError: No module named 'psutil'` during metadata generation.

### [USER] Fixed missing psutil
```bash
pip install psutil
```
**[RESULT] PASS** — psutil installed.

### [USER] flash-attn install — Attempt 2
```bash
pip install flash-attn --no-build-isolation
```
**[RESULT] FAIL** — `[Errno 18] Invalid cross-device link`
- pip tried to move downloaded wheel from scratch temp dir → `/home/gupta.yashv/.cache/pip/wheels/`
- Home (`/home`) and scratch are different filesystems — cross-device rename not allowed
- Home is also 97% full — pip cache cannot live there
- **Note:** pip found a pre-built wheel (`flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`) — no compilation needed, just a download

### [ARCH] Diagnosis — conda source path issue
On explorer-01, `conda info --base` was `Killed` by the system (too heavy for login node process limits).
Fix: source conda.sh using the known hardcoded base path directly:
```bash
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
```
**[DECISION]** Always use the hardcoded path — never rely on `$(conda info --base)` on this cluster.
Updated `load_matrix_env.sh` logic accordingly.

### [USER] flash-attn install — Attempt 3 (in progress)
```bash
pip install flash-attn --no-build-isolation --cache-dir /scratch/gupta.yashv/pip-cache
```
Redirected pip cache to scratch to avoid cross-device link error and home disk pressure.
**[STATUS]** Awaiting result.
