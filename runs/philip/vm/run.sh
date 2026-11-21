#!/bin/bash
# Läuft auf der VM (Linux) und wird von vm_vergleich.sh gestartet.
#   bash run.sh RUNDEN label [label ...]
# Jedes Label ist ein Verzeichnis neben diesem Skript mit einem vollständigen Checkout.
# Python-Umgebung: $PY, sonst ~/fayer-muc/.venv, sonst wird ~/nacht-bench-venv mit den festen Versionen angelegt.
set -euo pipefail
cd "$(dirname "$0")"
if [ $# -lt 2 ]; then
  echo "Aufruf: bash run.sh RUNDEN label [label ...]"
  exit 1
fi
ROUNDS="$1"
shift
FIRST="$1"
PY="${PY:-}"

if [ -z "$PY" ]; then
  for candidate in "$HOME/fayer-muc/.venv/bin/python" "$HOME/nacht-bench-venv/bin/python"; do
    if [ -x "$candidate" ] && "$candidate" -c "import jax, scipy, matplotlib, importlib.metadata as m; m.version('differometor')" >/dev/null 2>&1; then
      PY="$candidate"
      break
    fi
  done
fi
if [ -z "$PY" ]; then
  echo "Kein fertiges venv gefunden, lege ~/nacht-bench-venv mit den festen Versionen an (dauert ein, zwei Minuten) ..."
  if ! python3 -m venv "$HOME/nacht-bench-venv" >/dev/null 2>&1; then
    # Frische Ubuntu-Images haben python3-venv nicht dabei.
    echo "python3-venv fehlt, installiere es mit apt (sudo) ..."
    sudo -n apt-get update -qq
    sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip
    rm -rf "$HOME/nacht-bench-venv"
    python3 -m venv "$HOME/nacht-bench-venv"
  fi
  "$HOME/nacht-bench-venv/bin/python" -m pip install -q --upgrade pip
  "$HOME/nacht-bench-venv/bin/python" -m pip install -q -r "$FIRST/requirements.txt"
  # benchmark.py liest die Version der installierten Distribution; importiert wird trotzdem der jeweilige Stand.
  "$HOME/nacht-bench-venv/bin/python" -m pip install -q --no-deps -e "$FIRST"
  PY="$HOME/nacht-bench-venv/bin/python"
fi

echo "Python: $PY"
"$PY" - "$FIRST/requirements.txt" <<'PYTHON'
import importlib.metadata as metadata
import sys

pins = {}
for line in open(sys.argv[1]):
    line = line.strip()
    if line and not line.startswith("#") and "==" in line:
        name, version = line.split("==")
        pins[name] = version
wrong = [f"{name} {metadata.version(name)} statt {pins[name]}" for name in ("jax", "jaxlib", "numpy", "scipy", "optax") if name in pins and metadata.version(name) != pins[name]]
print("Pakete: " + ", ".join(f"{name} {metadata.version(name)}" for name in ("jax", "numpy", "scipy")))
if wrong:
    print("ACHTUNG, weicht von requirements.txt ab: " + "; ".join(wrong))
PYTHON

cat VERSIONEN.txt 2>/dev/null || true
# Jede Variante muss ihr eigenes differometor importieren (nicht ein editierbar installiertes aus ~/fayer-muc).
VARIANTS=()
for label in "$@"; do
  origin="$(cd "$label" && "$PY" -c "import os, sys; sys.path.insert(0, os.getcwd()); import differometor; print(os.path.dirname(os.path.abspath(differometor.__file__)))")"
  if [ "$origin" != "$(cd "$label" && pwd -P)/differometor" ]; then
    echo "FEHLER: $label importiert differometor aus $origin"
    exit 1
  fi
  VARIANTS+=("$label=$label")
done

"$PY" -u vm_ab.py "$ROUNDS" "${VARIANTS[@]}"
