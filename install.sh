#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# SMT Artifact Installation Script
#
# This script:
#   1. Checks for Python 3.12
#   2. Creates a local virtual environment (.venv)
#   3. Installs Python dependencies
#   4. Creates utils/config.json from the example if necessary
#   5. Creates the output directory
#
# It does NOT run the evaluation and does NOT configure API keys.
# ============================================================

# Always work from the repository root, regardless of where the
# script is invoked from.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

echo "=========================================="
echo " Installing SMT artifact"
echo " Repository: ${ROOT_DIR}"
echo "=========================================="

# ------------------------------------------------------------
# 1. Locate Python 3.12
# ------------------------------------------------------------

if command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.12)"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"

    PY_VERSION="$("${PYTHON_BIN}" -c \
        'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

    if [[ "${PY_VERSION}" != "3.12" ]]; then
        echo "Error: SMT expects Python 3.12."
        echo "Found: Python ${PY_VERSION}"
        echo "Please install Python 3.12 and run this script again."
        exit 1
    fi
else
    echo "Error: Python 3.12 was not found."
    exit 1
fi

echo "[1/5] Using Python:"
"${PYTHON_BIN}" --version

# ------------------------------------------------------------
# 2. Create virtual environment
# ------------------------------------------------------------

if [[ ! -d ".venv" ]]; then
    echo "[2/5] Creating virtual environment .venv ..."
    "${PYTHON_BIN}" -m venv .venv
else
    echo "[2/5] Virtual environment .venv already exists."
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# ------------------------------------------------------------
# 3. Install dependencies
# ------------------------------------------------------------

echo "[3/5] Installing Python dependencies ..."

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# ------------------------------------------------------------
# 4. Prepare configuration
# ------------------------------------------------------------

echo "[4/5] Preparing configuration ..."

if [[ ! -f "utils/config.json" ]]; then
    if [[ -f "utils/config.example.json" ]]; then
        cp utils/config.example.json utils/config.json
        echo "Created utils/config.json from utils/config.example.json."
    else
        echo "Error: utils/config.example.json was not found."
        exit 1
    fi
else
    echo "utils/config.json already exists; leaving it unchanged."
fi

# ------------------------------------------------------------
# 5. Prepare output directory and verify dependencies
# ------------------------------------------------------------

echo "[5/5] Preparing output directory and checking installation ..."

mkdir -p output

python - <<'PY'
import openai
import requests

print("Dependency check succeeded.")
print("openai version :", openai.__version__)
print("requests version:", requests.__version__)
PY

echo
echo "=========================================="
echo " SMT installation completed successfully."
echo "=========================================="
echo
echo "Before running the artifact:"
echo
echo "  1. Edit utils/config.json and provide your API endpoint"
echo "     and API key."
echo
echo "  2. Activate the environment:"
echo
echo "       source .venv/bin/activate"
echo
echo "  3. Run the evaluation:"
echo
echo "       python main.py"
echo