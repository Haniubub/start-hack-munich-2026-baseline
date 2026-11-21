#!/bin/bash
# Vergleicht Git-Stände auf der Nebius-VM (Xeon Platinum 8468, dieselbe CPU wie in der Wertung).
#
# In einem lokalen Terminal auf dem Mac (nicht in der SSH-Session), irgendwo im Repository:
#   bash runs/philip/vm/vm_vergleich.sh IP                    # max=origin/max/nacht noel=origin/noel ich=HEAD, 12 Runden
#   bash runs/philip/vm/vm_vergleich.sh IP 20                 # dieselben Stände, 20 Runden
#   bash runs/philip/vm/vm_vergleich.sh IP 12 alt=HEAD~1 neu=HEAD noel=origin/noel
#
# Verglichen werden Commits, nicht der Arbeitsstand: Änderungen vorher committen (und `git fetch`, wenn die
# Stände der anderen aktuell sein sollen). Jeder Stand wird mit `git archive` nach ~/nacht-bench/<label> auf der VM
# kopiert; dort laufen alle abwechselnd in frischen Prozessen mit der offiziellen Kommandozeile
# (python benchmark.py --cpus 1 --repeats 3), eine Aufwärmrunde zählt nicht. Auf der VM wird nur ~/nacht-bench
# angefasst. Tabelle und Rohdaten landen in _nacht/vm/ im Repository (oder in $OUT).
#
# Umgebungsvariablen: KEY (SSH-Key, Standard ~/.ssh/id_ed25519), VM_USER (Standard philip), OUT (Zielordner).
set -euo pipefail
if [ $# -lt 1 ]; then
  echo "Aufruf: bash $0 IP [RUNDEN] [label=ref ...]"
  exit 1
fi
IP="$1"
shift
ROUNDS=12
if [ $# -gt 0 ]; then
  case "$1" in
    *=*) ;;
    *) ROUNDS="$1"; shift ;;
  esac
fi
if [ $# -eq 0 ]; then
  set -- max=origin/max/nacht noel=origin/noel ich=HEAD
fi

KEY="${KEY:-$HOME/.ssh/id_ed25519}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
OUT="${OUT:-$REPO/_nacht/vm}"
TARGET="${VM_USER:-philip}@$IP"
if [ ! -f "$KEY" ]; then
  echo "Key $KEY nicht gefunden. Das Skript muss auf dem Mac laufen (lokaler Tab), nicht auf der VM."
  exit 1
fi
OPTIONS=(-i "$KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new)

# Erst alle Stände prüfen, dann hochladen.
VERSIONS="$(mktemp)"
trap 'rm -f "$VERSIONS"' EXIT
LABELS=""
for spec in "$@"; do
  label="${spec%%=*}"
  ref="${spec#*=}"
  case "$label" in
    ""|*[!A-Za-z0-9_-]*) echo "Label '$label' darf nur Buchstaben, Ziffern, - und _ enthalten (Aufruf: label=ref)."; exit 1 ;;
  esac
  if ! git -C "$REPO" rev-parse --verify --quiet "$ref^{commit}" >/dev/null; then
    echo "Unbekannter Stand: $ref"
    exit 1
  fi
  echo "$label $(git -C "$REPO" log -1 --format='%h %ci %s' "$ref")" >> "$VERSIONS"
  LABELS="$LABELS $label"
done
cat "$VERSIONS"

echo "== Verbindung zu $TARGET =="
if ! ssh "${OPTIONS[@]}" "$TARGET" 'echo "ok: $(hostname), $(grep -m1 "model name" /proc/cpuinfo | cut -d: -f2), $(nproc) CPUs"; rm -rf ~/nacht-bench; mkdir -p ~/nacht-bench'; then
  echo "Keine Verbindung. Läuft die VM, stimmt die IP?"
  exit 1
fi

echo "== Stände hochladen =="
for spec in "$@"; do
  label="${spec%%=*}"
  ref="${spec#*=}"
  git -C "$REPO" archive --prefix="$label/" "$ref" -- . ':!media' | ssh "${OPTIONS[@]}" "$TARGET" 'tar -x -C ~/nacht-bench'
done
scp "${OPTIONS[@]}" "$HERE/run.sh" "$HERE/vm_ab.py" "$TARGET:nacht-bench/"
scp "${OPTIONS[@]}" "$VERSIONS" "$TARGET:nacht-bench/VERSIONEN.txt"

mkdir -p "$OUT"
STAMP="$(date +%m%d-%H%M)"
echo "== Benchmark ($ROUNDS Runden) =="
ssh "${OPTIONS[@]}" "$TARGET" "bash ~/nacht-bench/run.sh $ROUNDS$LABELS" 2>&1 | tee "$OUT/vm-ergebnis-$STAMP.txt"
scp "${OPTIONS[@]}" "$TARGET:nacht-bench/ergebnis.json" "$OUT/vm-ergebnis-$STAMP.json"
echo
echo "Fertig. Tabelle: $OUT/vm-ergebnis-$STAMP.txt"
