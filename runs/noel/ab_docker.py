"""Alternate fresh `--cpus 1` benchmark runs of several Docker images and compare first-run phases.

Usage: python runs/noel/ab_docker.py N label1=image1 label2=image2 ...
"""
import json
import statistics
import subprocess
import sys
from pathlib import Path

n = int(sys.argv[1])
variants = [arg.split("=", 1) for arg in sys.argv[2:]]
out = Path(__file__).resolve().parent / "out" / "docker"
out.mkdir(parents=True, exist_ok=True)
groups = {
    "imports": ["imports"], "build": ["initialize", "load", "build"],
    "compile": ["forward_lower", "forward_compile", "sensitivity_lower", "sensitivity_compile", "gradient_lower", "gradient_compile"],
    "forward": ["forward"], "gradient": ["gradient", "sensitivity", "host_transfer"],
}
data = {label: [] for label, _ in variants}
warm = {label: [] for label, _ in variants}
errs = {label: [] for label, _ in variants}
info = {}
for i in range(n):
    for label, image in variants:
        report = out / "benchmark-docker.json"
        if report.exists():
            report.unlink()
        result = subprocess.run(
            ["docker", "run", "--rm", "-v", f"{out}:/out", image, "--repeats", "3"], capture_output=True, text=True
        )
        if "PASS" not in result.stdout:
            print(label, "FAILED\n", result.stdout[-800:], result.stderr[-1500:])
            sys.exit(1)
        d = json.loads(report.read_text())
        report.rename(out / f"{label}-{i}.json")
        data[label].append({p["key"]: p["seconds"] for p in d["phases"]})
        warm[label].append(d["timing_seconds"]["runtime_median"])
        errs[label].append(d["validation"]["max_absolute_log_sensitivity_error"])
        info[label] = (d["environment"]["cpu_affinity_count"], d["environment"]["platform"])
print("environment:", info)
print(f"{'':10s}" + "".join(f"{label:>22s}" for label, _ in variants) + "   (median / min, ms)")


def line(name, getter):
    cells = []
    for label, _ in variants:
        values = [getter(r) * 1e3 for r in data[label]]
        cells.append(f"{statistics.median(values):11.1f} /{min(values):7.1f}")
    print(f"{name:10s}" + "".join(f"{c:>22s}" for c in cells))


line("TOTAL", lambda r: sum(r.values()))
for name, keys in groups.items():
    line(name, lambda r, keys=keys: sum(r[k] for k in keys))
print(f"{'warm':10s}" + "".join(f"{statistics.median(warm[l])*1e3:11.1f} /{min(warm[l])*1e3:7.1f}".rjust(22) for l, _ in variants))
print(f"{'log error':10s}" + "".join(f"{max(errs[l]):22.4e}" for l, _ in variants))
