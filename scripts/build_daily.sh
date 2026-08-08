#!/bin/zsh
# Daily build of The Daily Prompt. Installed as a launchd agent — see README.
set -e
cd "$(dirname "$0")/.."

echo "=== build $(date '+%Y-%m-%d %H:%M') ==="
.venv/bin/python -m pipeline.main

osascript -e 'display notification "Today'"'"'s edition is ready — open app/index.html" with title "The Daily Prompt 📰"' || true
