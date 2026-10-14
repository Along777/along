#!/usr/bin/env bash
# Run the brain in the foreground (the systemd user service runs this same script).
# Holds a sleep/idle inhibitor so the workstation cannot suspend while it serves.
cd "$(dirname "$0")/../brain" || exit 1
export HF_HUB_OFFLINE=1 DO_NOT_TRACK=1 PYTHONUNBUFFERED=1
UV=$(command -v uv || echo "$HOME/.local/bin/uv")
WHY=(--what=sleep:idle --who=skeleton-brain --why="Halloween skeleton brain")
# Probe first: if policy refuses the inhibitor, run without it rather than not at all.
if command -v systemd-inhibit >/dev/null 2>&1 && systemd-inhibit "${WHY[@]}" true 2>/dev/null; then
  exec systemd-inhibit "${WHY[@]}" "$UV" run --frozen --offline skeleton-brain serve "$@"
fi
echo "note: could not hold a sleep inhibitor - make sure Automatic Suspend is off" >&2
exec "$UV" run --frozen --offline skeleton-brain serve "$@"
