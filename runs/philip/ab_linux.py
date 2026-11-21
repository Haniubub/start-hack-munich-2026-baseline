"""Alternate fresh `benchmark.py --cpus 1` runs of several checkouts (Linux) and compare first-run phases.

Like runs/noel/ab.py, but with the official Linux command line (`--cpus 1`, no XLA_FLAGS) and with the
order of the variants flipped every round.

Usage: python runs/philip/ab_linux.py N label1=repo1 label2=repo2 ...
"""
import json, os, statistics, subprocess, sys
from pathlib import Path
n = int(sys.argv[1]); variants = [a.split("=", 1) for a in sys.argv[2:]]
out = Path(__file__).resolve().parent / "out"; out.mkdir(parents=True, exist_ok=True)
env = {k: v for k, v in os.environ.items() if k != "XLA_FLAGS"}
groups = {"imports": ["imports"], "build": ["initialize", "load", "build"],
          "lower": ["forward_lower", "sensitivity_lower", "gradient_lower"], "compile": ["forward_compile", "sensitivity_compile", "gradient_compile"],
          "forward": ["forward"], "gradient": ["gradient"], "sens+xfer": ["sensitivity", "host_transfer"]}
data = {l: [] for l, _ in variants}; warm = {l: [] for l, _ in variants}; errs = {l: [] for l, _ in variants}
for i in range(n):
    for label, repo in (variants if i % 2 == 0 else variants[::-1]):
        repo = Path(repo).resolve(); report = out / f"{label}-{i}.json"
        r = subprocess.run([sys.executable, str(repo / "benchmark.py"), "--cpus", "1", "--repeats", "3", "--output", str(report)], cwd=repo, env=env, capture_output=True, text=True)
        if "PASS" not in r.stdout:
            print(label, "FAILED\n", r.stdout[-600:], r.stderr[-1500:]); sys.exit(1)
        d = json.loads(report.read_text())
        data[label].append({p["key"]: p["seconds"] for p in d["phases"]}); warm[label].append(d["timing_seconds"]["phase_runtime_medians"]); errs[label].append(d["validation"]["max_absolute_log_sensitivity_error"])
print(f"n={n}  {'':8s}" + "".join(f"{l:>20s}" for l, _ in variants) + "   (median / min, ms)")
def line(name, getter, src=data):
    print(f"{name:14s}" + "".join(f"{statistics.median(v):10.1f} /{min(v):7.1f}".rjust(20) for v in ([getter(r) * 1e3 for r in src[l]] for l, _ in variants)))
line("TOTAL", lambda r: sum(r.values()))
for name, keys in groups.items(): line(name, lambda r, keys=keys: sum(r[k] for k in keys))
line("warm total", lambda r: sum(r.values()), warm); line("warm forward", lambda r: r["forward"], warm); line("warm gradient", lambda r: r["gradient"], warm)
print(f"{'log error':14s}" + "".join(f"{max(errs[l]):20.4e}" for l, _ in variants))
