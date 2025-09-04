#!/usr/bin/env bash
set -euo pipefail

echo "========================================="
echo "DFS Fantasy Project Setup"
echo "========================================="

# --- 0) Sanity: python present ------------------------------------------------
if ! command -v python3 >/dev/null 2>&1; then
  echo "❌ Python 3 is not installed. Please install Python 3.9+ first."
  exit 1
fi
echo "✓ Python found: $(python3 --version)"

# --- 1) Install uv if missing, and ensure PATH immediately --------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "📦 Installing uv package manager..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Make uv available in *this* shell without re-login
  export PATH="$HOME/.local/bin:$PATH"
  if [ -f "$HOME/.local/bin/env" ]; then
    # shellcheck disable=SC1090
    source "$HOME/.local/bin/env" || true
  fi
  echo "✓ uv installed"
else
  echo "✓ uv already installed"
  # Also fix PATH if needed
  export PATH="$HOME/.local/bin:$PATH"
  [ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env" || true
fi

# Persist PATH so new shells see uv
if ! grep -q 'HOME/.local/bin' "$HOME/.bashrc" 2>/dev/null; then
  {
    echo 'export PATH="$HOME/.local/bin:$PATH"'
    echo 'test -f "$HOME/.local/bin/env" && . "$HOME/.local/bin/env" || true'
  } >> "$HOME/.bashrc"
fi

# --- 2) Prefer Python 3.11 for venv (stable cu128 wheels) ---------------------
PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "${PYVER}" != "3.11" ]; then
  echo "ℹ️  Current python is ${PYVER}; creating venv with Python 3.11 for PyTorch CUDA stability."
  export UV_PYTHON=3.11
fi

# --- 3) Use roomy cache/temp if /workspace exists -----------------------------
if [ -d /workspace ]; then
  export TMPDIR=/workspace/tmp
  export PIP_CACHE_DIR=/workspace/pip-cache
  mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
  echo "✓ Using /workspace for tmp/cache"
else
  echo "ℹ️  /workspace not found; using default tmp/cache locations."
fi

# --- 4) Create & activate venv ------------------------------------------------
echo "🔧 Creating virtual environment..."
uv venv .venv
echo "🔧 Activating virtual environment..."
# shellcheck disable=SC1091
source .venv/bin/activate
python -V

# Ensure pip exists/updated (some minimal venvs lack pip)
python -m ensurepip --upgrade
python -m pip install --upgrade pip

# --- 5) Install PyTorch first (GPU if available, else CPU) --------------------
GPU_AVAILABLE=0
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_AVAILABLE=1
fi

if [ "$GPU_AVAILABLE" -eq 1 ]; then
  echo "🎛️  GPU detected (nvidia-smi present). Installing PyTorch CUDA 12.8..."
  python -m pip install --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir \
    torch==2.8.0
else
  echo "🧠  No GPU detected. Installing CPU-only PyTorch..."
  python -m pip install --index-url https://download.pytorch.org/whl/cpu --no-cache-dir \
    torch==2.8.0
fi

# Quick CUDA sanity log (won't fail the script)
python - <<'PY' || true
import os, torch
print("[torch]", torch.__version__)
print("[cuda.is_available]", torch.cuda.is_available())
PY

# --- 6) Install project dependencies -----------------------------------------
echo "📦 Installing project dependencies..."
# If you have a requirements file:
if [ -f requirements.txt ]; then
  uv pip install -r requirements.txt
else
  # Or fall back to pyproject/uv lock if you use those
  uv pip install -e .
fi

# --- 7) Create directories & .env --------------------------------------------
echo "📁 Creating project directories..."
mkdir -p data models lineups logs

if [ ! -f .env ]; then
  echo "📝 Creating .env file..."
  cat > .env << 'EOF'
# DFS Environment Configuration
DATABASE_PATH=data/nfl_dfs.db
MODEL_DIR=models/
LINEUP_DIR=lineups/
LOG_LEVEL=INFO
EOF
  echo "✓ .env file created"
else
  echo "✓ .env file already exists"
fi

# --- 8) Add helpful runtime env defaults on venv activation -------------------
# Avoid CUDA-in-fork crashes; you can remove these after baking spawn/num_workers into code.
ACTIVATE_FILE=".venv/bin/activate"
if ! grep -q "DFS_DATALOADER_WORKERS" "$ACTIVATE_FILE"; then
  {
    echo ''
    echo '# DFS defaults to avoid CUDA-in-fork worker issues'
    echo 'export PYTORCH_DISTRIBUTED_DISABLE_FORK=1'
    echo 'export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}'
    echo 'export DFS_DATALOADER_WORKERS=${DFS_DATALOADER_WORKERS:-0}'
  } >> "$ACTIVATE_FILE"
  echo "✓ Added safe runtime env defaults to venv activation"
fi

# --- 9) Initialize database (best-effort) ------------------------------------
echo "🗄️  Initializing database (best-effort)..."
python - <<'PY' || true
import os, importlib
db_path = 'data/nfl_dfs.db'
if os.path.exists(db_path):
    print('✓ Database already exists')
else:
    tried=[]
    for mod, func in [('data','initialize_database'), ('db','init_db')]:
        try:
            m = importlib.import_module(mod)
            fn = getattr(m, func)
            try:
                if func == 'initialize_database':
                    fn(db_path)
                else:
                    fn()
                print(f'✓ Database initialized via {mod}.{func}')
                break
            except TypeError:
                # try both call styles
                try:
                    fn()
                    print(f'✓ Database initialized via {mod}.{func}()')
                    break
                except Exception as e:
                    tried.append(f"{mod}.{func} -> {e}")
        except Exception as e:
            tried.append(f"{mod}.{func} import/call failed -> {e}")
    else:
        print("⚠️  Could not auto-initialize DB; please run your DB init command manually.")
        for t in tried: print("   -", t)
PY

echo ""
echo "========================================="
echo "✅ Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  source .venv/bin/activate"
if [ "$GPU_AVAILABLE" -eq 1 ]; then
  echo "  # (GPU detected) Recommended training flags:"
  echo "  export CUDA_VISIBLE_DEVICES=0"
fi
echo "  uv run python run.py train --positions QB --tune-all --trials 5 --epochs 25"
echo ""
echo "Tips:"
echo "  - If you still see CUDA worker errors, keep DFS_DATALOADER_WORKERS=0 (already set on activate)."
echo "  - For long runs: nohup ./train_qb.sh &   (ask me to generate one)."
