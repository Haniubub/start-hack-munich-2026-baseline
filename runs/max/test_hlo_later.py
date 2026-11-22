"""DEFER_HLO_DIALECTS (differometor/_startup.py): the StableHLO and CHLO modules run at the first lowering.

Usage: python runs/max/test_hlo_later.py        (from the repository root, so that the carrier is found)

A battery of programs that XLA compiles (every group of operations whose lowering JAX registers
with a name from the two modules, plus effects, control flow and sorting) runs in two processes
that start like benchmark.py: once with the measure and once without it (`DIFFEROMETOR_STARTUP`
without `hlo`). The results have to be identical bit for bit: the same programs are compiled.
"""
import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ALL = "gc,lazy,defer,dataclass,jits,report,tpuprobe,npdocs,hlo"
HEADER = "import argparse, sys, json\nimport jax, jax.numpy as jnp\nimport numpy as np\nimport differometor\n"
MODULES = ["jaxlib.mlir.dialects.stablehlo", "jaxlib.mlir.dialects.chlo", "jaxlib.mlir.dialects._stablehlo_ops_gen",
           "jaxlib.mlir._mlir_libs._stablehlo", "jaxlib.mlir._mlir_libs._chlo"]
failures = []

BATTERY = r'''
import hashlib
from jax._src.interpreters import mlir
jax.config.update("jax_enable_x64", True)
state = getattr(sys, "_differometor_startup", {})
out = {"early": state.get("early"), "after_import": [type(sys.modules.get(m)).__name__ for m in MODULES],
       "token_type": type(mlir.token_type).__name__}
rng = np.random.default_rng(0)
x = jnp.asarray(rng.uniform(0.1, 0.9, (3, 5))); y = jnp.asarray(rng.uniform(1.1, 3.0, (3, 5)))
z = jnp.asarray(rng.normal(size=(3, 5)) + 1j * rng.normal(size=(3, 5)))
i = jnp.asarray(rng.integers(1, 200, (3, 5))); k = jnp.asarray(rng.integers(0, 5, 4)); b = x > 0.5
import jax.scipy.special as sp
cases = {
    "unary": lambda: [f(x) for f in (jnp.negative, jnp.floor, jnp.ceil, jnp.isfinite, jnp.exp, jnp.log, jnp.expm1, jnp.log1p, jnp.tanh, jnp.tan,
                                     jnp.arcsin, jnp.arccos, jnp.arctan, jnp.sinh, jnp.cosh, jnp.arcsinh, jnp.arctanh, jnp.abs, jnp.sqrt, jnp.cbrt, jnp.sign)]
                     + [jnp.arccosh(y), jax.lax.rsqrt(y), jnp.nextafter(x, y), jnp.arctan2(x, y)],
    "complex": lambda: [z.real, z.imag, jax.lax.complex(x, y), jnp.abs(z), jnp.conj(z), jnp.exp(z), jnp.sqrt(z), jnp.maximum(z, z.conj()), jnp.minimum(z, 1j * z)],
    "binary": lambda: [x + y, x - y, x * y, x / y, jnp.remainder(y, x), jnp.maximum(x, 0.5), jnp.minimum(x, y), jnp.clip(x, 0.2, 0.7), x**y],
    "integer and bool": lambda: [~i, i & 7, i | 8, i ^ 5, jax.lax.population_count(i), jax.lax.clz(i), i << 2, i >> 1, jax.lax.shift_right_logical(i, jnp.ones_like(i)),
                                 i // 3, i % 7, ~b, b & (y > 2), b | (y > 2), b ^ (y > 2)],
    "comparisons": lambda: [x < y, x <= 0.5, x == x, x != y, jnp.where(b, x, y), jax.random.key(1) == jax.random.key(1), jax.random.key(1) != jax.random.key(2)],
    "reductions": lambda: [x.sum(), x.prod(1), x.max(0), x.min(), b.any(), b.all(1), jnp.bitwise_xor.reduce(i, axis=0), x.argmax(1), jnp.cumsum(x, 1), jnp.cumprod(x, 0), jax.lax.cummax(x, axis=1)],
    "special": lambda: [sp.gammaln(y), sp.digamma(y), sp.polygamma(2, y), sp.zeta(y + 1, y), sp.i1e(x), sp.erf(x), sp.erfc(x), sp.erfinv(x)],
    "scatter and gather": lambda: [jnp.zeros(5).at[k].add(1.5), jnp.ones(5).at[k].subtract(0.5), jnp.ones(5).at[k].multiply(2.0), x[:, k], jnp.sort(x, 1), jnp.argsort(x, 0)],
    "control flow": lambda: [jax.lax.while_loop(lambda s: s[0] < 5, lambda s: (s[0] + 1, s[1] * 1.5), (0, x))[1], jax.lax.cond(True, jnp.sin, jnp.cos, x),
                             jax.lax.scan(lambda c, t: (c + t, c * t), jnp.zeros(5), x)[1], jax.lax.fori_loop(0, 4, lambda n, c: c + n, 1.0)],
    "linear algebra and random": lambda: [jnp.linalg.solve(x[:, :3] + 3 * jnp.eye(3), y[:, 0]), jnp.linalg.eigh(x[:, :3] + x[:, :3].T)[0], jax.random.normal(jax.random.key(0), (4,)), jnp.fft.fft(x)],
    "gradients": lambda: [jax.grad(lambda t: jnp.sum(jnp.tanh(t) * sp.erf(t) + jnp.arcsinh(t)))(x), jax.jacfwd(lambda t: jnp.cumsum(jnp.log(t)))(y[0]), jax.vmap(jax.grad(lambda t: jnp.max(t) ** 2))(x)],
}
seen = []
def host(v):
    seen.append(float(np.sum(v))); return np.asarray(v) * 2
def effects(t):
    jax.debug.callback(lambda v: seen.append(float(v)), t.sum(), ordered=True)
    u = jax.experimental.io_callback(host, jax.ShapeDtypeStruct(t.shape, t.dtype), t, ordered=True)
    return u + jax.pure_callback(lambda v: np.asarray(v) + 1, jax.ShapeDtypeStruct(t.shape, t.dtype), t, vmap_method="sequential")
import jax.experimental
cases["effects and tokens"] = lambda: [effects(x)]
digest = {}
for name, f in cases.items():
    values = jax.jit(f)()
    blob = b"".join(np.ascontiguousarray(np.asarray(v)).tobytes() + str(np.asarray(v).dtype).encode() + str(np.asarray(v).shape).encode() for v in jax.tree_util.tree_leaves(values))
    digest[name] = hashlib.sha256(blob).hexdigest()
jax.effects_barrier()
out.update(digest=digest, seen=seen, after_battery=[type(sys.modules.get(m)).__name__ for m in MODULES], loaded=state.get("later_loaded"),
           token_type_after=type(mlir.token_type).__name__)
from jax._src.lib.mlir.dialects import hlo
out["attribute"] = [callable(hlo.negate), isinstance(hlo.AddOp, type), hlo.AddOp.OPERATION_NAME]
print(json.dumps(out))
'''


def run(body, startup=None):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("DIFFEROMETOR_STARTUP", None)
    if startup is not None:
        env["DIFFEROMETOR_STARTUP"] = startup
    out = subprocess.run([sys.executable, "-c", HEADER + "MODULES = %r\n" % MODULES + body], cwd=ROOT, env=env, capture_output=True, text=True)
    if out.returncode:
        failures.append("process failed: " + (out.stderr.strip().splitlines() or ["?"])[-1])
        print(out.stderr[-1500:])
        return {}
    return json.loads(out.stdout.strip().splitlines()[-1])


def check(name, condition, detail=""):
    print(("ok   " if condition else "FAIL ") + name + (f"  [{detail}]" if detail and not condition else ""))
    if not condition:
        failures.append(name)


on = run(BATTERY)
off = run(BATTERY, ALL.replace(",hlo", ""))
check("with the measure the modules are not run by the import", "hlo" in (on.get("early") or []) and on.get("after_import", [None])[:2] == ["_LaterModule", "_LaterModule"] and on.get("after_import", [0] * 5)[2:] == ["NoneType"] * 3, str(on.get("after_import")))
check("without it they are ordinary modules after the import", "hlo" not in (off.get("early") or ["hlo"]) and off.get("after_import") == ["module"] * 5, str(off.get("after_import")))
check("the first lowering runs both, and JAX's own names are the real objects afterwards",
      on.get("after_battery") == ["module"] * 5 and sorted(on.get("loaded") or []) == sorted(MODULES[:2]) and on.get("token_type") == "_Later" and on.get("token_type_after") == off.get("token_type_after") == "method", str(on.get("loaded")))
names = sorted(on.get("digest", {}))
check(f"{len(names)} groups of compiled programs", len(names) == 12 and names == sorted(off.get("digest", {})))
for name in names:
    check(f"identical results: {name}", on["digest"][name] == off["digest"].get(name))
check("ordered effects ran the same way", on.get("seen") == off.get("seen") and len(on.get("seen") or []) == 2, str(on.get("seen")))
check("the modules answer like ordinary modules afterwards", on.get("attribute") == off.get("attribute") == [True, True, "stablehlo.add"], str(on.get("attribute")))
r = run('''
from jax._src.lib.mlir.dialects import hlo
from jax._src.interpreters import mlir
before = type(sys.modules["jaxlib.mlir.dialects.stablehlo"]).__name__
later = mlir.min_hlo.args[0]
text = repr(later); chained = type(later.OPERATION_NAME).__name__ if False else None
kind = type(hlo.AddOp).__name__                      # after `import jax`: a read runs the module
print(json.dumps({"before": before, "repr": text, "read": kind, "later_is_real_now": later.__name__, "isinstance": isinstance(3, mlir.create_token) if isinstance(mlir.create_token, type) else "function",
                  "after": type(sys.modules["jaxlib.mlir.dialects.stablehlo"]).__name__}))''')
check("after the import a plain read of the module runs it", r.get("before") == "_LaterModule" and r.get("after") == "module" and r.get("read") != "_Later", str(r))
check("a name put away earlier is the real function to its user", r.get("later_is_real_now") == "minimum" and "later jaxlib.mlir.dialects.stablehlo.minimum" in r.get("repr", ""), str(r))
r = run("print(json.dumps({'kind': type(sys.modules['jaxlib.mlir.dialects.stablehlo']).__name__}))", "none")
check("DIFFEROMETOR_STARTUP=none leaves the import alone", r.get("kind") == "module", str(r))
print("all checks passed" if not failures else f"{len(failures)} failed")
sys.exit(1 if failures else 0)
