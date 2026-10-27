#!/usr/bin/env bash
# One-step install for macOS and Linux.
#
#   ./install.sh            core + scientific stack (numpy, scipy, matplotlib, pydantic, structlog)
#   ./install.sh --dev      also pytest, ruff, mypy, pandas
#   ./install.sh --pymc     also PyMC for NUTS estimation (large download)
#
# Creates ./.venv so nothing is installed into (or breaks) your system Python or Homebrew
# packages. No sudo, no --break-system-packages.
set -euo pipefail

cd "$(dirname "$0")"

EXTRAS="analysis"
for arg in "$@"; do
  case "$arg" in
    --dev)  EXTRAS="all" ;;
    --pymc) EXTRAS="${EXTRAS},estimation" ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

# Pick the newest suitable interpreter.
PY=""
for cand in python3.14 python3.13 python3.12 python3.11 python3.10 python3 python; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
    PY="$cand"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.10 or newer is required." >&2
  if [ "$(uname)" = "Darwin" ]; then
    echo "On macOS:  brew install python   (or download it from https://www.python.org/downloads/)" >&2
  fi
  exit 1
fi
echo "Using $($PY --version) at $(command -v $PY)"

if [ ! -d .venv ]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --quiet --upgrade pip
python -m pip install --quiet -e ".[${EXTRAS}]"

echo "Smoke test..."
python - <<'PYEOF'
from pipeline import APGIPipeline
from config import CONFIG
r = APGIPipeline({**CONFIG, "seed": 42}).step(x_e=0.5, x_i=0.2, x_hat_e=0.3, x_hat_i=0.1)
assert {"S", "theta", "B"} <= set(r), r.keys()
print("  pipeline step OK  (S=%.4f, theta=%.4f)" % (r["S"], r["theta"]))
import models, estimation  # noqa: F401
print("  model + estimation packages import OK")
PYEOF

cat <<MSG

Installed. To use it in a new terminal:

    cd "$PWD"
    source .venv/bin/activate
    apgi --steps 100 --seed 42         # command-line run
    python examples/01_basic_usage.py  # an example
    python -m pytest tests -q          # needs ./install.sh --dev

MSG
