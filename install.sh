#!/bin/bash

# --- Prevent crashing shell if sourced ---
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "⚠️  Please run this script with 'source install.sh' to activate the environment in your current shell."
  exit 1
fi

ENV_NAME=".venv"
REFRESH=false

# --- Parse command-line options ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    -r|--refresh)
      REFRESH=true
      shift
      ;;
    -*)
      echo "Unknown option: $1"
      echo "Usage: source install.sh [-r|--refresh]"
      exit 1
      ;;
    *)
      shift
      ;;
  esac
done

# --- Step 0: Detect architecture and write .python-version ---
ARCH=$(uname -m)
OS=$(uname -s)

if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
  echo "cpython-3.12.3-macos-aarch64-none" > .python-version
  echo "📄 Wrote ARM macOS Python version to .python-version"
else
  echo "3.12.3" > .python-version
  echo "📄 Wrote Linux/x86_64 Python version to .python-version"
fi

# --- Step 1: Handle refresh ---
if $REFRESH; then
  echo "🔄 Refreshing environment..."
  rm -rf "$ENV_NAME"
  rm -f uv.lock
fi

# --- Step 2: Check for uv ---
if ! command -v uv >/dev/null 2>&1; then
  echo "❌ Error: 'uv' is not installed."
  echo "Please install it using the official guide:"
  echo "🔗 https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

# --- Step 3: Create virtual environment ---
echo "📦 Creating virtual environment: $ENV_NAME"
uv venv "$ENV_NAME"

# --- Step 4: Activate the environment ---
# Use `.` (dot) for POSIX compatibility
if [ ! -d "$ENV_NAME/bin" ]; then
  . "$ENV_NAME/Scripts/activate"
else
  . "$ENV_NAME/bin/activate"
fi


# --- Step 5: Sync base dependencies from pyproject.toml ---
echo "🔧 Syncing dependencies from pyproject.toml..."
uv sync --active

# --- Step 6: Conditional CUDA vs CPU/MPS installation ---
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "🟢 CUDA (Nvidia GPUs)-enabled system detected. Installing CUDA compatible dependencies..."
  uv pip install -r requirements.cuda.txt
else
  echo "🟡 No CUDA (Nvidia GPUs)-enabled system detected. Installing CPU/MPS compatible dependencies..."
  uv pip install -r requirements.txt
fi

# --- Step 7: Re-activate (ensures correct env) ---
echo "✅ Setup complete. Environment '$ENV_NAME' is now active."
if [ ! -d "$ENV_NAME/bin" ]; then
  . "$ENV_NAME/Scripts/activate"
else
  . "$ENV_NAME/bin/activate"
fi