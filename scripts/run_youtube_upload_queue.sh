#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

{
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] youtube upload queue run"
  args=(scripts/youtube_upload_queue.py upload-next --limit "${UPLOAD_LIMIT:-1}")
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    args+=(--dry-run)
  else
    args+=(--approved --allow-public)
  fi
  .venv/bin/python "${args[@]}"
} >> logs/youtube-upload-queue.log 2>&1
