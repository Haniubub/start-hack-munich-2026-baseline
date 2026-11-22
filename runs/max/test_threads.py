"""First use of put-off modules from several threads at once (differometor/_startup.py).

Usage: python runs/max/test_threads.py        (from the repository root, so that the carrier is found)

A `_DeferredModule` (hashlib, importlib.metadata, statistics, subprocess, numpy.ma, opt_einsum, ...) or
a `_LaterModule` (the HLO dialects) runs its code on the first real use. A second thread that touches
the module while that code runs has to wait for it, not read a name the code has not defined yet.
Each check runs in a process of its own, several times, because the failure is a race.
"""
import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HEADER = ("import argparse, hashlib, importlib.metadata, json, statistics, subprocess, sys, threading\n"
          "import jax, jax.numpy as jnp\nimport numpy as np\nimport differometor\n"
          "N = 6\nbarrier = threading.Barrier(N)\nresults = [None] * N\nerrors = []\n"
          "def go(work):\n"
          "    def one(i):\n"
          "        try:\n            barrier.wait()\n            results[i] = work(i)\n"
          "        except BaseException as error:\n            errors.append(type(error).__name__ + ': ' + str(error)[:120])\n"
          "    threads = [threading.Thread(target=one, args=(i,)) for i in range(N)]\n"
          "    [t.start() for t in threads]\n    [t.join() for t in threads]\n"
          "    print(json.dumps({'errors': sorted(set(errors)), 'results': results}))\n")
failures = []


def run(body, repeats=5, **environment):
    env = dict(os.environ, **environment)
    env.pop("PYTHONPATH", None)
    seen = []
    for _ in range(repeats):
        out = subprocess.run([sys.executable, "-c", HEADER + body], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
        if out.returncode:
            seen.append({"errors": ["process failed: " + (out.stderr.strip().splitlines() or ["?"])[-1]], "results": []})
        else:
            seen.append(json.loads(out.stdout.strip().splitlines()[-1]))
    return seen


def check(name, runs, expected):
    bad = [r for r in runs if r["errors"] or r["results"] != expected]
    print(("ok   " if not bad else "FAIL ") + f"{name} ({len(runs)} processes)" + (f"  [{bad[0]}]" if bad else ""))
    if bad:
        failures.append(name)


# A program that XLA really compiles (too large for the interpreter): the first lowering runs the
# HLO dialect modules, and JAX hashes with hashlib on the way.
COMPILE = ("def work(i):\n    x = jnp.asarray(np.arange(96, dtype=np.float64).reshape(12, 8))\n"
           "    f = jax.jit(lambda a: a @ a.T + jnp.sin(a @ a.T))\n    return round(float(f.lower(x).compile()(x)[0, 0]), 6)\n"
           "go(work)\n")
value = 140.0 + float(__import__("math").sin(140.0))
check("six threads compile their first XLA program at once", run(COMPILE), [round(value, 6)] * 6)
check("the same with only the report modules put off", run(COMPILE, DIFFEROMETOR_STARTUP="report"), [round(value, 6)] * 6)
check("the same with only the HLO dialects put off", run(COMPILE, DIFFEROMETOR_STARTUP="hlo"), [round(value, 6)] * 6)

check("six threads use hashlib for the first time", run("go(lambda i: hashlib.sha256(b'differometor').hexdigest()[:12])\n"),
      [__import__("hashlib").sha256(b"differometor").hexdigest()[:12]] * 6)
check("six threads use statistics for the first time", run("go(lambda i: statistics.median([3, 1, 2]))\n"), [2] * 6)
check("six threads use subprocess for the first time", run("go(lambda i: subprocess.run([sys.executable, '-c', 'print(7)'], capture_output=True, text=True).stdout.strip())\n"), ["7"] * 6)
check("six threads use importlib.metadata for the first time", run("go(lambda i: importlib.metadata.version('numpy') == np.__version__)\n"), [True] * 6)
check("six threads use numpy.ma for the first time", run("go(lambda i: float(np.ma.masked_array([1.0, 2.0, 3.0], mask=[0, 1, 0]).sum()))\n"), [4.0] * 6)
check("six threads plan an einsum (opt_einsum) for the first time",
      run("go(lambda i: float(jnp.einsum('ab,bc,cd->ad', *[jnp.eye(3) * (k + 1) for k in range(3)], optimize='greedy')[1, 1]))\n"), [6.0] * 6)
check("six threads read NumPy docstrings for the first time",
      run("import differometor._startup as st\ngo(lambda i: (st.numpy_docs() or True) and len(np.ndarray.__doc__ or '') > 1000)\n"), [True] * 6)

# Two put-off modules whose codes touch each other from two threads would wait for each other
# forever. After `_FIRST_USE_WAIT` a waiting thread reads the other module as it is (half-built),
# the way it reads an ordinary module that another thread is still importing.
CYCLE = """
import time, types
import differometor._startup as st
st._FIRST_USE_WAIT = 0.5
started = {'cycle_a': threading.Event(), 'cycle_b': threading.Event()}
class Loader:
    def __init__(self, other):
        self.other = other
    def exec_module(self, module):
        started[module.__name__].set()
        started[self.other].wait(10)                     # both codes are running now
        try:
            module.seen = sys.modules[self.other].value  # the other thread has not defined it yet
        except AttributeError:
            module.seen = 'half-built'
        module.value = module.__name__
for name, other in (('cycle_a', 'cycle_b'), ('cycle_b', 'cycle_a')):
    module = types.ModuleType(name)
    module.__dict__[st._PLAN] = st._DeferredPlan(Loader(other), {}, None)
    module.__class__ = st._DeferredModule
    sys.modules[name] = module
N = 2
barrier = threading.Barrier(N)
results = [None] * N
begin = time.perf_counter()
names = ('cycle_a', 'cycle_b')
# `seen` is the other module's value, or 'half-built' for the thread that gave up waiting first
go(lambda i: [sys.modules[names[i]].value, sys.modules[names[i]].seen in ('half-built', names[1 - i]), type(sys.modules[names[i]]).__name__])
assert time.perf_counter() - begin < 8, 'waited too long'
"""
check("two put-off modules that touch each other from two threads do not block for good",
      run(CYCLE, repeats=2), [["cycle_a", True, "module"], ["cycle_b", True, "module"]])

print("all checks passed" if not failures else f"{len(failures)} failed")
sys.exit(1 if failures else 0)
