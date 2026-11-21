"""Wechselnde frische `benchmark.py --cpus 1 --repeats 3`-Läufe mehrerer Checkouts, Vergleich der First-Run-Phasen.

Aufruf: python vm_ab.py N label1=verzeichnis1 label2=verzeichnis2 ...

Jede Runde startet jede Variante einmal in einem frischen Prozess (offizielle Kommandozeile, kein XLA_FLAGS);
die Reihenfolge der Varianten dreht sich jede Runde um. TOTAL ist `timing_seconds.end_to_end` des Reports,
also die First-Run-Summe, nach der gewertet wird. Rohdaten landen in ergebnis.json neben diesem Skript.
"""
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

here = Path(__file__).resolve().parent
rounds = int(sys.argv[1])
variants = [argument.split("=", 1) for argument in sys.argv[2:]]
out = here / "out"
out.mkdir(exist_ok=True)
env = {key: value for key, value in os.environ.items() if key != "XLA_FLAGS"}

groups = {
    "imports": ["imports"],
    "build": ["initialize", "load", "build"],
    "lower": ["forward_lower", "sensitivity_lower", "gradient_lower"],
    "compile": ["forward_compile", "sensitivity_compile", "gradient_compile"],
    "forward": ["forward"],
    "gradient": ["gradient"],
    "sens+xfer": ["sensitivity", "host_transfer"],
}


def benchmark(label, repo, report):
    repo = Path(repo).resolve()
    command = [sys.executable, str(repo / "benchmark.py"), "--cpus", "1", "--repeats", "3", "--output", str(report)]
    result = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True)
    if "PASS" not in result.stdout:
        print(label, "FAILED\n", result.stdout[-800:], result.stderr[-2000:], flush=True)
        sys.exit(1)
    return json.loads(Path(report).read_text())


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


print(f"CPU: {cpu_model()} | Python {platform.python_version()} | {time.strftime('%Y-%m-%d %H:%M:%S %Z')}", flush=True)

# Eine Aufwärmrunde, die nicht zählt: Seiten-Cache, Bytecode-Caches und der Font-Cache von matplotlib.
for label, repo in variants:
    started = time.time()
    report = benchmark(label, repo, out / f"{label}-warmup.json")
    print(f"aufwärmen {label:8s} first-run {report['timing_seconds']['end_to_end'] * 1e3:8.1f} ms   ({time.time() - started:.0f} s Prozess)", flush=True)

phases = {label: [] for label, _ in variants}
totals = {label: [] for label, _ in variants}
warm = {label: [] for label, _ in variants}
errors = {label: [] for label, _ in variants}
environment = {}
for index in range(rounds):
    order = variants if index % 2 == 0 else variants[::-1]
    line = []
    for label, repo in order:
        report = benchmark(label, repo, out / f"{label}-{index}.json")
        phases[label].append({phase["key"]: phase["seconds"] for phase in report["phases"]})
        totals[label].append(report["timing_seconds"]["end_to_end"])
        warm[label].append(report["timing_seconds"]["phase_runtime_medians"])
        errors[label].append(report["validation"]["max_absolute_log_sensitivity_error"])
        environment[label] = report["environment"]
        line.append(f"{label} {totals[label][-1] * 1e3:7.1f}")
    print(f"runde {index + 1}/{rounds}: " + "   ".join(line) + "  ms", flush=True)


def row(name, values_of):
    cells = []
    for label, _ in variants:
        values = [value * 1e3 for value in values_of(label)]
        cells.append(f"{statistics.median(values):10.1f} /{min(values):7.1f}".rjust(20))
    print(f"{name:14s}" + "".join(cells))


print()
print(f"n={rounds}  {'':8s}" + "".join(f"{label:>20s}" for label, _ in variants) + "   (median / min, ms)")
row("TOTAL", lambda label: totals[label])
for name, keys in groups.items():
    row(name, lambda label, keys=keys: [sum(run[key] for key in keys) for run in phases[label]])
row("warm total", lambda label: [sum(run.values()) for run in warm[label]])
row("warm forward", lambda label: [run["forward"] for run in warm[label]])
row("warm gradient", lambda label: [run["gradient"] for run in warm[label]])
print(f"{'log error':14s}" + "".join(f"{max(errors[label]):20.4e}" for label, _ in variants))
print("alle Läufe PASS")

summary = {
    "measured_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    "cpu_model": cpu_model(),
    "rounds": rounds,
    "command": "python benchmark.py --cpus 1 --repeats 3 --output <report>.json",
    "variants": {
        label: {
            "first_run_totals_seconds": totals[label],
            "first_run_median_seconds": statistics.median(totals[label]),
            "first_run_min_seconds": min(totals[label]),
            "phases_seconds": phases[label],
            "warm_phase_medians_seconds": warm[label],
            "max_absolute_log_sensitivity_error": max(errors[label]),
            "environment": environment[label],
        }
        for label, _ in variants
    },
}
(here / "ergebnis.json").write_text(json.dumps(summary, indent=1))
