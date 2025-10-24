#!/usr/bin/env python3
# All comments in English
import os, sys, shlex, subprocess

# --- Edit these to match your GPU node ---
CONDA_SH  = os.getenv("CONDA_SH",  "/mnt/shared-storage-user/chenshuizhou/miniconda3/etc/profile.d/conda.sh")
CONDA_ENV = os.getenv("CONDA_ENV", "e2former")
CUDA_LIB  = os.getenv("CUDA_LIB",  "/usr/local/cuda-12.1/lib64")  # change if needed

# Optional: preload (e.g., sklearn's libgomp). Leave empty if not needed.
EXTRA_PRELOAD = os.getenv("EXTRA_PRELOAD", "")

TARGET = os.getenv("TARGET_SCRIPT", "main.py")
DEFAULT_ARGS = (
    "--config-yml=configs/s2ef/MPTrj/e2former/e2former_67M_esen.yaml "
    "--mode=train --run-dir=runs --identifier=vscode_debug --timestamp-id=vscode_debug"
)
ARGS = os.getenv("TARGET_ARGS", DEFAULT_ARGS)

# For Remote-SSH, 127.0.0.1 is safer and requires no external exposure.
DEBUG_HOST = os.getenv("DEBUGPY_HOST", "127.0.0.1")
DEBUG_PORT = int(os.getenv("DEBUGPY_PORT", "5678"))
WAIT = os.getenv("DEBUGPY_WAIT", "1") in ("1", "true", "TRUE")
wait_flag = "--wait-for-client" if WAIT else ""

# Optional GPU debugging knobs
CUDA_VISIBLE = os.getenv("CUDA_VISIBLE_DEVICES", "")  # e.g. "0" or "0,1,2,3"
CUDA_SYNC = os.getenv("DEBUG_SYNC_CUDA", "0") in ("1","true","TRUE")

bash = f"""
set -eo pipefail
set -x

# 0) Optional: select GPUs and sync kernels for step debugging
{"export CUDA_VISIBLE_DEVICES="+shlex.quote(CUDA_VISIBLE) if CUDA_VISIBLE else ": # keep current GPUs"}
{"export CUDA_LAUNCH_BLOCKING=1" if CUDA_SYNC else ": # async CUDA"}

# 1) Proper conda activation in non-interactive shell
if [ -f {shlex.quote(CONDA_SH)} ]; then
  source {shlex.quote(CONDA_SH)}
  conda activate {shlex.quote(CONDA_ENV)}
else
  echo "[warn] conda.sh not found: {CONDA_SH}"
fi

# 2) Libraries and preload
export LD_LIBRARY_PATH="{CUDA_LIB}:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
{f'export LD_PRELOAD="$LD_PRELOAD:{shlex.quote(EXTRA_PRELOAD)}"' if EXTRA_PRELOAD else ': # no extra preload'}

# 3) Quick diagnostics
python -V
which python
conda info --envs || true
python - <<'PY'
import torch, os
print("torch.__version__      =", torch.__version__)
print("torch.version.cuda     =", torch.version.cuda)
print("cuda.is_available      =", torch.cuda.is_available())
print("LD_LIBRARY_PATH        =", os.environ.get("LD_LIBRARY_PATH",""))
print("CUDA_VISIBLE_DEVICES   =", os.environ.get("CUDA_VISIBLE_DEVICES","<unset>"))
PY

# 4) Run under debugpy (listen on 127.0.0.1:5678 for Remote-SSH)
python -m debugpy --listen {DEBUG_HOST}:{DEBUG_PORT} {wait_flag} {shlex.quote(TARGET)} {ARGS}
"""

print("======= debug_gpu_entry launch plan =======")
print(bash)
print("==========================================", flush=True)

proc = subprocess.Popen(["bash", "-lc", bash], stdout=sys.stdout, stderr=sys.stderr)
sys.exit(proc.wait())
