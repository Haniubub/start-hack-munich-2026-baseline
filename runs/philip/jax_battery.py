"""A broad set of JAX calls, printed as JSON. `test_startup.py` runs it with and without the start-up measures
and compares the two outputs: numbers, names, signatures, error messages and traceback frames must be the same.

    python runs/philip/jax_battery.py        (DIFFEROMETOR_STARTUP_SHADOW=0 switches the measures off)
"""
import inspect
import json
import os
import sys
import traceback
from pathlib import Path

root = Path(__file__).resolve().parents[2]
os.chdir(root)
sys.path.insert(0, str(root))
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"

import statistics  # noqa: E402, F401  the carrier, as in benchmark.py

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402

out = {}


def note(name, value):
    if isinstance(value, (tuple, list)):
        value = [np.asarray(item).tolist() for item in value]
    elif not isinstance(value, (str, bool, int, float, dict, type(None))):
        value = np.asarray(value).tolist()
    out[name] = value


def complex_pairs(value):
    value = np.asarray(value)
    return [value.real.tolist(), value.imag.tolist()]


rng = np.random.default_rng(0)
a = jnp.asarray(rng.normal(size=(4, 4)) + 4 * np.eye(4))
b = jnp.asarray(rng.normal(size=(4, 3)))
v = jnp.asarray(rng.normal(size=7))
z = jnp.asarray(rng.normal(size=5) + 1j * rng.normal(size=5))

# jax.numpy functions
for name, call in {
    "sum": lambda: jnp.sum(a, axis=0), "mean": lambda: jnp.mean(a), "var": lambda: jnp.var(a, axis=1, ddof=1), "std": lambda: jnp.std(v),
    "prod": lambda: jnp.prod(v), "cumsum": lambda: jnp.cumsum(v), "argmax": lambda: jnp.argmax(a, axis=1), "sort": lambda: jnp.sort(v),
    "where": lambda: jnp.where(v > 0, v, -v), "clip": lambda: jnp.clip(v, -0.5, 0.5), "concatenate": lambda: jnp.concatenate([v, v[:2]]),
    "stack": lambda: jnp.stack([v, 2 * v]), "linspace": lambda: jnp.linspace(0.0, 1.0, 5), "einsum": lambda: jnp.einsum("ij,jk->ik", a, b),
    "dot": lambda: jnp.dot(a, b), "matmul": lambda: a @ b, "tensordot": lambda: jnp.tensordot(a, b, axes=1), "outer": lambda: jnp.outer(v[:3], v[:2]),
    "solve": lambda: jnp.linalg.solve(a, b), "inv": lambda: jnp.linalg.inv(a), "det": lambda: jnp.linalg.det(a), "norm": lambda: jnp.linalg.norm(a),
    "eigvalsh": lambda: jnp.linalg.eigvalsh(a + a.T), "svdvals": lambda: jnp.linalg.svd(b, compute_uv=False), "fft": lambda: jnp.abs(jnp.fft.fft(v)),
    "polyval": lambda: jnp.polyval(v[:3], 0.3), "interp": lambda: jnp.interp(0.4, jnp.arange(7.0), v), "take": lambda: jnp.take(v, jnp.array([0, 3])),
    "pad": lambda: jnp.pad(v[:3], 2), "tile": lambda: jnp.tile(v[:2], 3), "roll": lambda: jnp.roll(v, 2), "tri": lambda: jnp.tril(a),
    "abs_complex": lambda: jnp.abs(z), "angle": lambda: jnp.angle(z), "log10": lambda: jnp.log10(jnp.abs(v) + 1), "exp": lambda: jnp.exp(v),
    "squeeze": lambda: jnp.squeeze(a[None, :1]), "isfinite": lambda: jnp.isfinite(v).all(), "unique": lambda: jnp.unique(jnp.array([3, 1, 3]), size=2),
}.items():
    note("np." + name, call())
note("np.complex", complex_pairs(z * jnp.conj(z) + jnp.exp(1j * z.real)))

# array methods
for name, call in {
    "sum": lambda: a.sum(), "mean": lambda: a.mean(axis=0), "reshape": lambda: a.reshape(2, 8), "T": lambda: b.T, "slice": lambda: a[1:, ::2],
    "at": lambda: v.at[0].set(9.0), "astype": lambda: v.astype(jnp.float32), "clip": lambda: v.clip(-1, 1), "argsort": lambda: v.argsort(),
    "ravel": lambda: b.ravel(), "round": lambda: v.round(2), "max": lambda: a.max(axis=1), "dot": lambda: a.dot(b), "cumprod": lambda: v[:4].cumprod(),
}.items():
    note("method." + name, call())

# transformations
loss = lambda p: jnp.mean(jnp.log10(jnp.abs(jnp.sin(p) * p) + 1.0)) + jnp.sum(p ** 2) / (1 + jnp.abs(p).max())
note("grad", jax.grad(loss)(v))
note("value_and_grad", jax.jit(jax.value_and_grad(loss))(v))
note("jacfwd", jax.jacfwd(lambda p: jnp.sin(p) * p[0])(v[:3]))
note("jacrev", jax.jacrev(lambda p: jnp.sin(p) * p[0])(v[:3]))
note("hessian", jax.hessian(loss)(v[:3]))
note("vmap", jax.vmap(lambda row: jnp.sum(row ** 2) / jnp.mean(jnp.abs(row)))(a))
note("jvp", jax.jvp(loss, (v,), (jnp.ones_like(v),)))
value, pullback = jax.vjp(lambda p: jnp.abs(p * (1 + 2j)), v)
note("vjp", [value, pullback(jnp.ones_like(value))[0]])
note("scan", lax.scan(lambda carry, x: (carry + x, carry * x), 0.0, v)[1])
note("cond", lax.cond(v[0] > 0, lambda: jnp.sum(v), lambda: jnp.mean(v)))
note("while", lax.while_loop(lambda s: s < 10.0, lambda s: s * 1.7 + 0.1, 0.5))
note("fori", lax.fori_loop(0, 5, lambda i, s: s + i * v[i], 0.0))
note("map", lax.map(lambda x: x ** 2 + 1, v))
note("jit_of_np", jax.jit(jnp.linalg.inv)(a))
note("eval_shape", str(jax.eval_shape(jnp.linalg.inv, jax.ShapeDtypeStruct((3, 3), jnp.float64))))
note("random", [jax.random.normal(jax.random.key(1), (3,)), jax.random.uniform(jax.random.split(jax.random.key(2))[0], (2,)),
                jax.random.permutation(jax.random.key(3), 5)])
import jax.scipy.linalg  # noqa: E402
import jax.scipy.special  # noqa: E402

note("scipy", [jax.scipy.special.logsumexp(v), jax.scipy.special.erf(v[:2]), jax.scipy.special.gammaln(jnp.abs(v[:2]) + 1),
               jax.scipy.linalg.solve_triangular(jnp.tril(a), b[:, 0], lower=True), jax.nn.softmax(v[:3]), jax.nn.relu(v[:3])])

# what a jit object of JAX looks like from outside
from jax._src.numpy import reductions, ufuncs  # noqa: E402

for module, names in ((reductions, ("_mean", "_var", "_reduce_sum")), (ufuncs, ("log10", "true_divide", "abs"))):
    for name in names:
        obj = getattr(module, name, None)
        if obj is None:
            continue
        keys = sorted(key for key in vars(obj) if key != "_jit_info")
        note(f"object.{name}", {
            "name": obj.__name__, "module": obj.__module__, "qualname": obj.__qualname__, "doc": (obj.__doc__ or "")[:60],
            "signature": str(inspect.signature(obj)), "dict": keys, "repr": repr(obj).split(" at 0x")[0], "callable": callable(obj),
            "isroutine": inspect.isroutine(obj), "wrapped": obj.__wrapped__.__name__, "fun": obj._fun.__name__,
            "jit_info": type(obj._jit_info).__name__, "has": [hasattr(obj, n) for n in ("lower", "trace", "eval_shape", "clear_cache", "nope")],
            "static": [list(obj._jit_info.static_argnums), list(obj._jit_info.static_argnames)], "inline": obj._jit_info.inline,
        })


class Holder:
    scaled = jax.jit(lambda self, x: 2 * x, static_argnums=0)
    mean = reductions._mean if hasattr(reductions, "_mean") else None

    def __hash__(self):
        return 7

    def __eq__(self, other):
        return isinstance(other, Holder)


note("bound", [Holder().scaled(v[:2]), type(Holder().scaled).__name__, Holder.__dict__["scaled"] is Holder.scaled])

# errors: same exception, same message, and no frame of the start-up measures in the traceback
for name, call in {
    "bad_type": lambda: jnp.sum("text"), "bad_shape": lambda: jnp.linalg.solve(a, v), "bad_axis": lambda: jnp.mean(v, axis=3),
    "bad_static": lambda: jax.jit(lambda x, n: x * n, static_argnames="m")(1.0, 2), "traced_bool": lambda: jax.jit(lambda x: 1 if x > 0 else 0)(1.0),
}.items():
    try:
        call()
        note("error." + name, "no error")
    except Exception as error:
        frames = [Path(frame.filename).name for frame in traceback.extract_tb(error.__traceback__)]
        note("error." + name, {"type": type(error).__name__, "message": str(error).split("\n")[0][:160], "frames": frames})

print(json.dumps(out, sort_keys=True))
