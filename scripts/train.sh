#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="${1:-configs/train_dmd_qwen_t2i.yaml}"
python3 -m spc.train_dmd_qwen_t2i "$CONFIG"
