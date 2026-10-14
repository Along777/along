#!/usr/bin/env bash
# One-time setup for the brain (the Linux workstation). Idempotent: re-run until every step is ok.
#   bash scripts/setup_brain.sh            (from the ai-skeleton folder)
# Needs no sudo itself. Anything that does (Ollama install, firewall) is printed for you to run.
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT=$PWD
BRAIN="$ROOT/brain"
FAILED=()
MIN_OLLAMA="0.12.7"   # first version with qwen3-vl

say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m   ok:\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m warn:\033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m fail:\033[0m %s\n' "$*"; FAILED+=("$1"); }
version_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]; }

say "GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader || true
  VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -dc '0-9')
  if [ "${VRAM_MB:-0}" -ge 12000 ] 2>/dev/null; then ok "enough VRAM for the 8b vision model"
  else warn "under 12 GB VRAM - set vision_model/reply_model to qwen3-vl:4b-instruct in brain/config.local.toml"; fi
else
  warn "nvidia-smi not found - install the driver (sudo ubuntu-drivers autoinstall; reboot) or expect slow CPU replies"
fi

say "Disk"
AVAIL_GB=$(df -P -BG "$HOME" | tail -1 | awk '{print $4}' | tr -dc '0-9')
if [ "${AVAIL_GB:-0}" -lt 15 ]; then err "only ${AVAIL_GB} GB free in $HOME - need ~15 GB (models + Python)"
else ok "${AVAIL_GB} GB free"; fi

say "uv (Python manager, user-level)"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  else
    err "curl missing - run: sudo apt install -y curl   (then re-run this script)"
  fi
fi
if command -v uv >/dev/null 2>&1; then ok "$(uv --version)"; else err "uv not installed"; fi

say "Python environment (brain)"
if uv sync --project "$BRAIN" "$@"; then ok "brain environment ready"; else err "uv sync failed"; fi
read -r VISION REPLY < <(uv run --project "$BRAIN" python -c \
  "from skeleton_brain import config; c=config.load(); print(c.ollama.vision_model, c.ollama.reply_model)" 2>/dev/null)
VISION=${VISION:-qwen3-vl:8b-instruct}; REPLY=${REPLY:-$VISION}
echo "   vision model: $VISION   reply model: $REPLY"

say "Ollama"
if ! command -v ollama >/dev/null 2>&1; then
  err "ollama is not installed - run:  curl -fsSL https://ollama.com/install.sh | sh   (asks for your password)"
else
  OV=$(ollama --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -n1)
  if version_ge "${OV:-0}" "$MIN_OLLAMA"; then ok "ollama $OV"
  else err "ollama ${OV:-?} is too old for qwen3-vl - update: curl -fsSL https://ollama.com/install.sh | sh"; fi
  if curl -sf http://127.0.0.1:11434/api/version >/dev/null 2>&1; then ok "server answering"
  else err "ollama server not answering - start it: sudo systemctl start ollama"; fi
  ENVS=$(systemctl show ollama -p Environment 2>/dev/null || true)
  if echo "$ENVS" | grep -q 'OLLAMA_MAX_LOADED_MODELS=1' && [ "$VISION" != "$REPLY" ]; then
    err "OLLAMA_MAX_LOADED_MODELS=1 (set by the local_llm kit) but vision and reply models differ - they would swap every turn. Use one model for both."
  fi
  for m in $(printf '%s\n%s\n' "$VISION" "$REPLY" | sort -u); do
    echo "--- ollama pull $m"
    if ollama pull "$m"; then ok "$m"; else err "pull $m failed"; fi
  done
fi

say "Speech-to-text + voice models"
if uv run --project "$BRAIN" skeleton-brain fetch-models; then ok "models in brain/models/"; else err "fetch-models failed"; fi

say "Service (systemd --user, not started at boot until you enable it)"
UNIT_DIR="$HOME/.config/systemd/user"
mkdir -p "$UNIT_DIR"
sed "s|@ROOT@|$ROOT|g" "$ROOT/scripts/skeleton-brain.service" > "$UNIT_DIR/skeleton-brain.service"
if systemctl --user daemon-reload 2>/dev/null; then
  ok "installed $UNIT_DIR/skeleton-brain.service"
  echo "   start:  systemctl --user start skeleton-brain      logs: journalctl --user -u skeleton-brain -f"
  echo "   boot:   systemctl --user enable skeleton-brain     (do this the week of Halloween)"
else
  warn "no systemd user session here - run the brain with: bash scripts/run_brain.sh"
fi
if [ "$(loginctl show-user "$USER" -p Linger 2>/dev/null)" != "Linger=yes" ]; then
  warn "linger is off: the service stops when you log out. Fix once: sudo loginctl enable-linger $USER"
fi

say "Network"
if systemctl is-active --quiet ufw 2>/dev/null; then
  SUBNET=$(ip -o -f inet addr show scope global | awk '{print $4}' | head -n1)
  warn "ufw firewall is on - let the porch in:  sudo ufw allow from ${SUBNET:-192.168.1.0/24} to any port 8080 proto tcp"
fi
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo "   this machine: $(hostname)  ${IP:-?}"
echo "   porch setting:  [brain] url = \"http://${IP:-<this-ip>}:8080\"   (give this machine a DHCP reservation)"

say "Doctor"
if [ ${#FAILED[@]} -eq 0 ]; then
  uv run --project "$BRAIN" skeleton-brain doctor || FAILED+=("doctor")
fi

say "Summary"
if [ ${#FAILED[@]} -eq 0 ]; then ok "all steps passed"; exit 0; fi
err "issues: ${FAILED[*]}  (fix and re-run; the script is idempotent)"
exit 1
