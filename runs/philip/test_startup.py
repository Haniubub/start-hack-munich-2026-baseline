"""Checks for the start-up measures in differometor/_startup.py that put work off or reuse compiled code.

    python runs/philip/test_startup.py      (from anywhere; prints one line per check, exits non-zero on failure)

Runs the header of benchmark.py in this process (with the carrier statistics.py), then uses the three
deferred modules for real and compares with what an ordinary import gives.
"""
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
os.chdir(root)
sys.path.insert(0, str(root))
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
for name in ("DIFFEROMETOR_STARTUP", "DIFFEROMETOR_STARTUP_SHADOW"):
    os.environ.pop(name, None)

failures = []


def check(name, condition, detail=""):
    print(("ok   " if condition else "FAIL ") + name + (f"  [{detail}]" if detail and not condition else ""))
    if not condition:
        failures.append(name)


import statistics  # noqa: E402  the carrier, found next to benchmark.py as in a benchmark run

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import differometor  # noqa: E402, F401

state = sys._differometor_startup
kind = lambda name: type(sys.modules[name]).__name__ if name in sys.modules else "absent"
deferred = ("http.server", "numpy.ma", "opt_einsum")

check("the real statistics module is in place", statistics.median([1, 2, 3]) == 2 and "carrier" not in (statistics.__doc__ or ""))
check("the three modules are deferred after the imports", all(kind(name) == "_DeferredModule" for name in deferred), str({n: kind(n) for n in deferred}))
check("nothing they import has been loaded", all(kind(name) == "absent" for name in ("http.client", "ssl", "numpy.ma.core", "opt_einsum.contract")))
import http.server  # noqa: E402
import numpy.ma  # noqa: E402
import opt_einsum  # noqa: E402, F401

check("importing them again does not load them", all(kind(name) == "_DeferredModule" for name in deferred))
check("numpy.unique does not load numpy.ma", np.array_equal(np.unique(np.array([3, 1, 3])), [1, 3]) and kind("numpy.ma") == "_DeferredModule")

# ---- dataclass code ----------------------------------------------------------------------------
import dataclasses  # noqa: E402

check("dataclass code is installed", str(state.get("dataclass_code", "")).startswith("on:"), str(state.get("dataclass_code")))


@dataclasses.dataclass(frozen=True, order=True)
class Fresh:
    a: int
    b: str = "x"
    c: list = dataclasses.field(default_factory=list, compare=False)


check("a dataclass that is not in the code file works", repr(Fresh(1)) == "Fresh(a=1, b='x', c=[])" and Fresh(1) == Fresh(1, c=[2]) and Fresh(1) < Fresh(2)
      and hash(Fresh(3)) == hash(Fresh(3)))
probe = """
import dataclasses, json, os, sys
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, os.getcwd())
if sys.argv[1] == "with":
    import statistics
import jax
from jax._src import core, pjit
out = {}
for cls in (pjit.PjitInfo, core.JaxprDebugInfo if hasattr(core, "JaxprDebugInfo") else core.DebugInfo, jax.ShapeDtypeStruct):
    for name in ("__init__", "__repr__", "__eq__", "__hash__", "__replace__", "__setattr__"):
        method = cls.__dict__.get(name)
        code = getattr(method, "__code__", None)
        if code is not None:
            out[f"{cls.__name__}.{name}"] = [code.co_code.hex(), repr(code.co_names), repr(code.co_varnames), code.co_filename, code.co_firstlineno]
out["installed"] = str(getattr(sys, "_differometor_startup", {}).get("dataclass_code"))
print(json.dumps(out))
"""
import json  # noqa: E402

runs = {}
for mode in ("with", "without"):
    env = dict(os.environ, DIFFEROMETOR_STARTUP_SHADOW="1" if mode == "with" else "0")
    result = subprocess.run([sys.executable, "-c", probe, mode], cwd=root, env=env, capture_output=True, text=True)
    runs[mode] = json.loads(result.stdout.strip().splitlines()[-1]) if result.returncode == 0 else {"error": result.stderr[-500:]}
methods = [key for key in runs["with"] if key != "installed"]
check("methods of JAX dataclasses have the same code with and without the code file",
      runs["with"].get("installed", "").startswith("on:") and runs["without"].get("installed") == "None" and len(methods) >= 3
      and all(runs["with"][key] == runs["without"].get(key) for key in methods), str({k: v for k, v in runs["with"].items() if k == "installed"}) + str(runs["without"].get("error", "")))

# ---- opt_einsum ---------------------------------------------------------------------------------
from jax._src.numpy import einsum as jax_einsum  # noqa: E402

a, b = np.arange(6.0).reshape(2, 3), np.arange(12.0).reshape(3, 4)
check("jnp.einsum works and loads opt_einsum", np.allclose(jnp.einsum("ij,jk->ik", a, b), a @ b) and kind("opt_einsum") == "module")
check("JAX's path optimizer is a subclass of the real class", issubclass(jax_einsum.Unoptimized, opt_einsum.paths.PathOptimizer)
      and opt_einsum.paths is sys.modules["opt_einsum.paths"] and callable(opt_einsum.contract))

# ---- numpy.ma -----------------------------------------------------------------------------------
masked = np.ma.masked_array([1.0, 2.0, 3.0], mask=[0, 1, 0])
check("a masked array loads numpy.ma", kind("numpy.ma") == "module" and float(masked.sum()) == 4.0 and np.ma.is_masked(masked))
from jax._src import core, dtypes  # noqa: E402
from jax._src.interpreters import mlir, pxla  # noqa: E402

tables = (core.pytype_aval_mappings, dtypes.canonicalize_value_handlers, mlir._constant_handlers, pxla.shard_arg_handlers)
check("JAX's tables know the real MaskedArray and no stand-in", all(np.ma.MaskedArray in table for table in tables)
      and not any(type(key).__name__ == "_StandInMeta" for table in tables for key in table))
messages = []
for call in (lambda: jnp.asarray(masked), lambda: jnp.sin(masked), lambda: jax.jit(lambda x: x + 1)(masked)):
    try:
        call()
        messages.append("no error")
    except Exception as error:
        messages.append(f"{type(error).__name__}: {error}")
check("masked arrays are rejected with JAX's own error", all(message.startswith("ValueError: numpy masked arrays are not supported") for message in messages), str(messages))

# ---- http.server --------------------------------------------------------------------------------
from jax._src import profiler  # noqa: E402

check("the profiler's handler class still waits for http.server", kind("http.server") == "_DeferredModule")


class Request:
    def makefile(self, *args, **kwargs):
        import io

        return io.BytesIO()

    def sendall(self, data):
        pass


handler = profiler._PerfettoServer(Request(), ("127.0.0.1", 0), type("Server", (), {})())
check("creating the handler loads http.server and gives a real request handler", kind("http.server") == "module"
      and isinstance(handler, http.server.SimpleHTTPRequestHandler)
      and [c.__name__ for c in profiler._PerfettoServer.__mro__][:3] == ["_PerfettoServer", "SimpleHTTPRequestHandler", "BaseHTTPRequestHandler"])
check("all three were loaded exactly once", sorted(state.get("deferred_loaded", [])) == sorted(deferred), str(state.get("deferred_loaded")))

print(f"\n{len(failures)} failed" if failures else "\nall checks passed")
sys.exit(1 if failures else 0)
