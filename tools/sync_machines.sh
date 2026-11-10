#!/usr/bin/env bash
# sync_machines.sh — keep every compute machine's SynthStar + argus_image CODE current
# from the Mac (the source of truth). Run on the Mac.
#
# CODE ONLY. Never touches a machine's own config.toml, data/, stars/, outputs/, runs/,
# .venv/ — those are machine-specific and stay local (that is why config.toml + data/gaia
# are gitignored on `overhaul`).
#
#   git machines (newton): git fetch + reset --hard origin/<branch>  (respects .gitignore)
#   rsync machines (huepil, longleaf, aurora): rsync the pure-code dirs, --delete on packages
#
# Requires the Mac repo to be committed + pushed first (so git machines can pull the same code).
set -uo pipefail

MAC_SYN="$HOME/MESA_ARGUS"           # synthstar repo (source)
MAC_AS="$HOME/ArgusSim"              # argus_image lives in image_maker/
BRANCH="overhaul"

# name:user:type:synthstar_dir:argussim_dir   (empty argussim_dir = generation-only, no argus_image)
HOSTS=(
  "newton:argus:git:/home/argus/SynthStar:"
  "huepil:nillmill:rsync:/home/nillmill/argus_gpu/MESA_Argus:/home/nillmill/argus_gpu/ArgusSim"
  "longleaf:nillmill:rsync:/nas/longleaf/home/nillmill/argus_gpu/MESA_Argus:/nas/longleaf/home/nillmill/argus_gpu/ArgusSim"
  "aurora:nillmill:rsync:/home/nillmill/argus_gpu/MESA_Argus:/home/nillmill/argus_gpu/ArgusSim"
)

EXCL=(--exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='*.egg-info' --exclude='.venv')
SSH="ssh -o BatchMode=yes -o ConnectTimeout=12 -o LogLevel=ERROR"

# --- guard: Mac repo must be clean (committed) so rsync and git machines get the same code ---
if ! git -C "$MAC_SYN" diff --quiet || ! git -C "$MAC_SYN" diff --cached --quiet; then
  echo "ABORT: $MAC_SYN has uncommitted changes — commit + push first, then re-run." >&2
  exit 1
fi

for entry in "${HOSTS[@]}"; do
  IFS=: read -r name user type synp asp <<<"$entry"
  echo "=== $name ($type) ==="
  # reachable?
  if ! $SSH "$user@$name" true 2>/dev/null; then echo "  unreachable — skipped"; continue; fi
  # target present? (don't create new installs here — aurora bare until set up once)
  if ! $SSH "$user@$name" "test -d $synp" 2>/dev/null; then echo "  $synp absent — needs one-time setup, skipped"; continue; fi

  if [ "$type" = git ]; then
    $SSH "$user@$name" "cd '$synp' && git fetch origin $BRANCH -q && git reset --hard origin/$BRANCH -q && echo '  -> '\$(git rev-parse --short HEAD)" 2>&1 | tail -1
  else
    rsync -az --delete "${EXCL[@]}" "$MAC_SYN/synthstar/"     "$user@$name:$synp/synthstar/"       2>&1 | tail -1
    rsync -az          "${EXCL[@]}" "$MAC_SYN/make_stars.py"  "$user@$name:$synp/"                 2>&1 | tail -1
    rsync -az          "${EXCL[@]}" "$MAC_SYN/tools/"         "$user@$name:$synp/tools/"           2>&1 | tail -1
    if [ -n "$asp" ] && $SSH "$user@$name" "test -d $asp" 2>/dev/null; then
      rsync -az --delete "${EXCL[@]}" "$MAC_AS/image_maker/"  "$user@$name:$asp/image_maker/"      2>&1 | tail -1
    fi
    echo "  -> synced synthstar + make_stars + tools${asp:+ + image_maker}"
  fi
done
echo "=== sync complete ==="
