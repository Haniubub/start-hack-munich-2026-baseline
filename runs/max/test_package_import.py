"""`import differometor`: one copy of _startup.py per process, and the MHLO library on first use.

Usage: python runs/max/test_package_import.py        (from the repository root, so that the carrier is found)

differometor/__init__.py takes the module the carrier (argparse.py) has already executed as
`differometor._startup`; differometor/_tuning.py opens jaxlib's `_mlirHlo` when the first MLIR
context is made. Each check runs in a process of its own.
"""
import json, os, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MHLO = "jaxlib.mlir._mlir_libs._mlirHlo"
failures = []


def run(code, cwd=ROOT, pythonpath=None, **environment):
    env = dict(os.environ, **environment)
    env.pop("PYTHONPATH", None)
    if pythonpath:
        env["PYTHONPATH"] = pythonpath
    out = subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env, capture_output=True, text=True)
    if out.returncode:
        failures.append("process failed: " + (out.stderr.strip().splitlines() or ["?"])[-1])
        return {}
    return json.loads(out.stdout.strip().splitlines()[-1])


def check(name, condition, detail=""):
    print(("ok   " if condition else "FAIL ") + name + (f"  [{detail}]" if detail and not condition else ""))
    if not condition:
        failures.append(name)


REPORT = ("import sys, json, differometor\nimport differometor._startup as st\n"
          "hook = sys.modules.get('_differometor_startup')\n"
          "print(json.dumps({'hook': hook is not None, 'same': st is hook, 'attribute': differometor._startup is st, 'name': st.__name__,\n"
          "                  'spec': getattr(getattr(st, '__spec__', None), 'name', None), 'late': callable(st.late), 'gc': st.state().get('gc_run')}))")

r = run("import argparse\nimport jax\n" + REPORT)
check("with the carrier, differometor._startup is the module the carrier executed", r.get("hook") and r.get("same") and r.get("attribute") and r.get("late"), str(r))
check("it carries a spec under the package name", r.get("spec") == "differometor._startup", str(r))
with_carrier = r

r = run("import argparse\nimport jax\n" + REPORT, DIFFEROMETOR_STARTUP_SHADOW="0")
check("with the carrier switched off the file is imported as usual", r.get("hook") is False and r.get("name") == "differometor._startup" and r.get("late"), str(r))

r = run(REPORT)
check("without `import argparse` the file is imported as usual", r.get("hook") is False and r.get("name") == "differometor._startup" and r.get("attribute"), str(r))

with tempfile.TemporaryDirectory() as elsewhere:
    r = run(REPORT, cwd=elsewhere, pythonpath=ROOT)
    check("from another directory (no carrier on the path) too", r.get("hook") is False and r.get("name") == "differometor._startup", str(r))

r = run("import argparse, importlib, json, sys\nimport jax\nimport differometor\nimport differometor._startup as st\nstate = st.state()\n"
        "importlib.reload(st)\nimport differometor._startup as again\n"
        "print(json.dumps({'state': st.state() is state, 'again': again is st, 'registered': sys.modules['differometor._startup'] is st}))")
check("importlib.reload works and keeps the state", r.get("state") and r.get("again") and r.get("registered"), str(r))

r = run("import argparse, json, sys\nimport jax, jax.numpy as jnp\nimport differometor\nimport differometor._tuning as tuning\n"
        "before = %r in sys.modules\n"
        "value = float(jax.jit(lambda x: jnp.sort(x)[0])(jnp.asarray([3.0, 1.0, 2.0])))\n"
        "print(json.dumps({'status': tuning.STATUS.get('lazy_mhlo'), 'before': before, 'after': %r in sys.modules, 'value': value}))" % (MHLO, MHLO))
check("the MHLO library is not opened by the import", r.get("status") == "on" and r.get("before") is False, str(r))
check("a program that XLA compiles opens it and is right", r.get("after") is True and r.get("value") == 1.0, str(r))

r = run("import argparse, json, sys, threading\nimport jax, jax.numpy as jnp\nimport differometor\nvalues = []\n"
        "def work(k):\n    values.append(float(jax.jit(lambda x: jnp.sort(x)[k])(jnp.asarray([3.0, 1.0, 2.0]))))\n"
        "threads = [threading.Thread(target=work, args=(k % 3,)) for k in range(6)]\n[t.start() for t in threads]\n[t.join() for t in threads]\n"
        "print(json.dumps({'values': sorted(values)}))")
check("six threads compiling their first programs at once", r.get("values") == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0], str(r))

print("all checks passed" if not failures else f"{len(failures)} failed")
sys.exit(1 if failures else 0)
