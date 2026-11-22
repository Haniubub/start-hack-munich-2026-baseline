"""DEFER_NUMPY_DOCS and the modules added to LAZY_MODULES for the interpreter (differometor/_startup.py).

Usage: python runs/max/test_numpy_docs.py        (from the repository root, so that the carrier is found)

Each check runs in a process of its own that starts like benchmark.py: the carrier first, then jax.
"""
import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HEADER = "import argparse, sys, json\nimport jax, jax.numpy as jnp\nimport numpy as np\nimport differometor\n"
failures = []


def run(body, **environment):
    env = dict(os.environ, **environment)
    env.pop("PYTHONPATH", None)
    out = subprocess.run([sys.executable, "-c", HEADER + body], cwd=ROOT, env=env, capture_output=True, text=True)
    if out.returncode:
        failures.append("process failed: " + out.stderr.strip().splitlines()[-1])
        return {}
    return json.loads(out.stdout.strip().splitlines()[-1])


def check(name, condition, detail=""):
    print(("ok   " if condition else "FAIL ") + name + (f"  [{detail}]" if detail and not condition else ""))
    if not condition:
        failures.append(name)


STATE = "state = getattr(sys, '_differometor_startup', {})\n"
r = run(STATE + "print(json.dumps({'early': state.get('early'), 'deferred': state.get('deferred'), 'doc': np.ndarray.__doc__ is None, 'scalar': np.float64.__doc__ is None,"
        " 'python_doc': bool(np.sum.__doc__), 'value': float(np.add(np.float64(1.5), 2.0)), 'jax': float(jnp.sum(jnp.arange(4.0)))}))")
check("the docstring modules are deferred", "npdocs" in (r.get("early") or []) and "numpy._core._add_newdocs" in (r.get("deferred") or []), str(r))
check("docstrings of C types are empty until asked for, Python docstrings are there", r.get("doc") and r.get("scalar") and r.get("python_doc"), str(r))
check("NumPy and JAX compute as usual", r.get("value") == 3.5 and r.get("jax") == 6.0, str(r))
r = run("import differometor._startup as st\nfirst = st.numpy_docs()\nprint(json.dumps({'ran': first, 'again': st.numpy_docs(), 'doc': len(np.ndarray.__doc__ or ''), 'scalar': len(np.float64.__doc__ or '')}))")
check("numpy_docs() runs both modules once, in NumPy's order", r.get("ran") == ["numpy._core._add_newdocs", "numpy._core._add_newdocs_scalars"] and r.get("again") == [], str(r))
check("afterwards the docstrings are NumPy's", r.get("doc", 0) > 1000 and r.get("scalar", 0) > 100, str(r))
r = run("import io, contextlib\nbuffer = io.StringIO()\nwith contextlib.redirect_stdout(buffer):\n    help(np.ndarray)\nprint(json.dumps({'text': 'An array object represents' in buffer.getvalue()}))")
check("help() shows the docstring", r.get("text"), str(r))
r = run("import numpy._core._add_newdocs as m\nm.add_newdoc\nprint(json.dumps({'doc': len(np.ndarray.__doc__ or '')}))")
check("touching the module runs it", r.get("doc", 0) > 1000, str(r))
r = run(STATE + "print(json.dumps({'early': state.get('early'), 'doc': np.ndarray.__doc__ is not None}))", DIFFEROMETOR_STARTUP="gc,lazy,defer,dataclass,jits,report,tpuprobe")
check("without `npdocs` NumPy imports its docstrings itself", "npdocs" not in (r.get("early") or ["npdocs"]) and r.get("doc"), str(r))
NEW = ["jax._src.jaxpr_util", "jax._src.path", "jaxlib.mlir.dialects.func", "jaxlib.mlir.dialects.mpmd", "jaxlib.mlir.dialects.sdy", "jaxlib.mlir.passmanager"]
r = run("names = %r\nlazy = lambda: [type(sys.modules.get(n)).__name__ == '_LazyModule' for n in names]\nbefore = lazy()\n"
        "value = float(jax.jit(lambda x: jnp.sort(x)[0])(jnp.asarray([3.0, 1.0, 2.0])))\nprint(json.dumps({'before': before, 'after': lazy(), 'value': value}))" % NEW)
check("the six modules are not loaded by the import", all(r.get("before", [False])), str(r))
check("a program that XLA compiles loads what it needs and is right", r.get("value") == 1.0 and not all(r.get("after", [True])), str(r))
print("all checks passed" if not failures else f"{len(failures)} failed")
sys.exit(1 if failures else 0)
