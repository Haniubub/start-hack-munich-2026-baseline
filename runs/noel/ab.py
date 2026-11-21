"""Alternate fresh benchmark runs of several repos and compare first-run phases.

Usage: python runs/noel/ab.py N label1=repo1 label2=repo2 ...

The repos must be in the same bytecode state. A `differometor/__pycache__` in only one of them
makes its import phase several milliseconds faster (compiling `fast.py` and `build.py` takes about
4 ms each), so the script stops when the states differ. Remove the folders or create them in all.
"""
import json, os, statistics, subprocess, sys
from pathlib import Path

n = int(sys.argv[1])
variants = [arg.split("=", 1) for arg in sys.argv[2:]]
out = Path(__file__).resolve().parent / "out"
env = dict(os.environ)
env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
           XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
groups = {
    "imports": ["imports"], "build": ["initialize", "load", "build"],
    "compile": ["forward_lower", "forward_compile", "sensitivity_lower", "sensitivity_compile", "gradient_lower", "gradient_compile"],
    "forward": ["forward"], "gradient": ["gradient", "sensitivity", "host_transfer"],
}
cached = {label: sorted(p.name for p in (Path(repo).resolve() / "differometor" / "__pycache__").glob("*.pyc")) for label, repo in variants}
if len({tuple(names) for names in cached.values()}) > 1:
    sys.exit(f"different bytecode caches in differometor/__pycache__, the import phases would not be comparable: {cached}")
data = {label: [] for label, _ in variants}
warm = {label: [] for label, _ in variants}
errs = {label: [] for label, _ in variants}
for i in range(n):
    for label, repo in variants:
        repo = Path(repo).resolve()
        report = out / f"ab-{label}-{i}.json"
        result = subprocess.run([sys.executable, str(repo / "benchmark.py"), "--repeats", "3", "--output", str(report)], cwd=repo, env=env, capture_output=True, text=True)
        if "PASS" not in result.stdout:
            print(label, "FAILED\n", result.stdout[-800:], result.stderr[-1500:]); sys.exit(1)
        d = json.loads(report.read_text())
        data[label].append({p["key"]: p["seconds"] for p in d["phases"]})
        warm[label].append(d["timing_seconds"]["runtime_median"])
        errs[label].append(d["validation"]["max_absolute_log_sensitivity_error"])
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
