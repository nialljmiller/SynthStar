#!/usr/bin/env bash
# sync_machines.sh — keep every compute machine's SynthStar + argus_image CODE current
# from the Mac (the source of truth). Run on the Mac.
#
# CODE ONLY. Never touches a machine's own config.toml / data / stars / outputs / .venv
# (those are gitignored on the branches, so they survive a git reset / are excluded from rsync).
#
# Adaptive per directory: if it's a git clone -> git fetch + reset --hard <branch> (a clean pull).
# If it's not (no auth for the private ArgusSim yet) -> rsync the code from the Mac.
# Once a machine's ArgusSim is also a git clone, this auto-switches it to git pull — no edit needed.
set -uo pipefail

MAC_SYN="$HOME/MESA_ARGUS"            # synthstar repo (source)  -> SynthStar@overhaul
MAC_AS="$HOME/ArgusSim"               # argus_image in image_maker -> ArgusSim@image-maker
SSH="ssh -o BatchMode=yes -o ConnectTimeout=12 -o LogLevel=ERROR"

# host user   synthstar_dir                                    argussim_dir (- = none, generation-only)
HOSTS=(
  "newton   argus    /home/argus/SynthStar                             -"
  "huepil   nillmill /home/nillmill/argus_gpu/MESA_Argus               /home/nillmill/argus_gpu/ArgusSim"
  "longleaf nillmill /nas/longleaf/home/nillmill/argus_gpu/MESA_Argus  /nas/longleaf/home/nillmill/argus_gpu/ArgusSim"
  "aurora   nillmill /home/nillmill/argus_gpu/MESA_Argus               /home/nillmill/argus_gpu/ArgusSim"
  "vega     nill     /home/nill/MESA_Argus                             /home/nill/ArgusSim"
)

# sync_dir user host dir branch rsync_srcs...   (rsync_srcs: Mac paths to mirror into dir/ when not git)
sync_dir() {
  local user=$1 host=$2 dir=$3 branch=$4; shift 4
  if ! $SSH "$user@$host" "test -d '$dir'" 2>/dev/null; then echo "  $dir absent — skip"; return; fi
  if $SSH "$user@$host" "test -d '$dir/.git'" 2>/dev/null; then
    $SSH "$user@$host" "cd '$dir' && git fetch origin $branch -q && git reset --hard origin/$branch -q && echo '  git  $dir -> '\$(git rev-parse --short HEAD)" 2>&1 | tail -1
  else
    local s base
    for s in "$@"; do
      [ -e "$s" ] || continue
      base=$(basename "$s")
      if [ -d "$s" ]; then
        rsync -az --delete --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='*.egg-info' \
          "$s/" "$user@$host:$dir/$base/" >/dev/null 2>&1
      else
        rsync -az "$s" "$user@$host:$dir/" >/dev/null 2>&1
      fi
    done
    echo "  rsync $dir (no git/auth yet)"
  fi
}

if ! git -C "$MAC_SYN" diff --quiet || ! git -C "$MAC_SYN" diff --cached --quiet; then
  echo "ABORT: $MAC_SYN has uncommitted changes — commit + push first." >&2; exit 1
fi

for entry in "${HOSTS[@]}"; do
  read -r host user synp asp <<<"$entry"
  echo "=== $host ==="
  $SSH "$user@$host" true 2>/dev/null || { echo "  unreachable — skip"; continue; }
  sync_dir "$user" "$host" "$synp" overhaul "$MAC_SYN/synthstar" "$MAC_SYN/make_stars.py" "$MAC_SYN/tools"
  [ "$asp" != "-" ] && sync_dir "$user" "$host" "$asp" image-maker "$MAC_AS/image_maker"
done
echo "=== sync complete ==="
