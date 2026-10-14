#!/usr/bin/env bash
# Run the brain in the foreground (the systemd user service runs this same script).
# Holds a sleep/idle inhibitor so the workstation cannot suspend while it serves.
cd "$(dirname "$0")/../brain" || exit 1
export HF_HUB_OFFLINE=1 DO_NOT_TRACK=1 PYTHONUNBUFFERED=1
UV=$(command -v uv || echo "$HOME/.local/bin/uv")
if command -v systemd-inhibit >/dev/null 2>&1; then
  exec systemd-inhibit --what=sleep:idle --who=skeleton-brain --why="Halloween skeleton brain" \
    "$UV" run --frozen --offline skeleton-brain serve "$@"
fi
exec "$UV" run --frozen --offline skeleton-brain serve "$@"
