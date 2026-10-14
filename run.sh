#!/usr/bin/env bash
# Create the virtualenv, install dependencies, and start the API.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"

if ! "$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)'; then
  echo "Python 3.12 or newer is required. Found: $("$PY" --version 2>&1)" >&2
  exit 1
fi

if [[ ! -d "$ROOT/.venv" ]]; then
  echo "Creating virtual environment..."
  "$PY" -m venv "$ROOT/.venv"
fi

# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "$ROOT/requirements.txt"
python -m playwright install chromium

mkdir -p "$ROOT/database" "$ROOT/logs" "$ROOT/screenshots" "$ROOT/browser-profile"

read_env() {
  local key="$1"
  local default="$2"
  local line=""
  if [[ -f "$ROOT/.env" ]]; then
    line="$(grep -E "^${key}=" "$ROOT/.env" | tail -n 1 || true)"
  fi
  if [[ -z "$line" ]]; then
    printf '%s' "$default"
  else
    printf '%s' "${line#*=}"
  fi
}

HOST_VALUE="$(read_env HOST 0.0.0.0)"
PORT_VALUE="$(read_env PORT 8000)"

echo "Starting AI browser agent on http://${HOST_VALUE}:${PORT_VALUE}"
exec python -m uvicorn app.main:app --host "$HOST_VALUE" --port "$PORT_VALUE"
