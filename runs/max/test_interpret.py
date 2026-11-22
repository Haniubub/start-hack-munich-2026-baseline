"""The NumPy interpreter for small programs (differometor/_interpret.py) against XLA.

Usage: PYTHONPATH=. python runs/max/test_interpret.py        (double precision, then single precision)

Every case goes through `jax.jit(f).lower(*args).compile()(*args)` twice, once interpreted and once
compiled by XLA (`_tuning.INTERPRET_SMALL_PROGRAMS = False`), and the two results are compared:
output structure, array type, dtype, shape, where NaN and infinity sit, and the values.
"""
import functools, os, subprocess, sys
import numpy as np
import jax

X64 = "x32" not in sys.argv
jax.config.update("jax_enable_x64", X64)
import jax.numpy as jnp
import differometor  # noqa: F401  (installs the measures)
from differometor import _interpret, _tuning
from jax._src import stages

F = jnp.float64 if X64 else jnp.float32
C = jnp.complex128 if X64 else jnp.complex64
TOLERANCE = 1e-13 if X64 else 2e-5
rng = np.random.default_rng(0)
x = jnp.asarray(rng.normal(size=(4, 7)), F)
y = jnp.asarray(rng.normal(size=(4, 7)) + 2.5, F)
v = jnp.asarray(rng.normal(size=7), F)
z = jnp.asarray(rng.normal(size=(4, 7)) + 1j * rng.normal(size=(4, 7)), C)
w = jnp.asarray(rng.normal(size=7) + 1j * rng.normal(size=7), C)
zeros = jnp.asarray([0.0, -0.0, 1.0, -2.0, np.inf, np.nan, 3.0], F)
failures, checked = [], [0, 0]


def staged(f, args, kwargs, interpret):
    _tuning.INTERPRET_SMALL_PROGRAMS = interpret
    try:
        lowered = jax.jit(f).lower(*args, **kwargs)
        compiled = lowered.compile()
        return lowered, compiled, compiled(*args, **kwargs)
    finally:
        _tuning.INTERPRET_SMALL_PROGRAMS = True


def same(name, a, b):
    la, ta = jax.tree_util.tree_flatten(a)
    lb, tb = jax.tree_util.tree_flatten(b)
    if ta != tb:
        return failures.append(f"{name}: output structure {ta} != {tb}")
    for i, (p, q) in enumerate(zip(la, lb)):
        if not isinstance(p, jax.Array):
            failures.append(f"{name}: output {i} is {type(p).__name__}, not a JAX array")
        if p.dtype != q.dtype or p.shape != q.shape:
            failures.append(f"{name}: output {i} is {p.dtype}{p.shape}, XLA gives {q.dtype}{q.shape}")
            continue
        p, q = np.asarray(p), np.asarray(q)
        if p.dtype.kind in "fc":
            if not (np.array_equal(np.isnan(p), np.isnan(q)) and np.array_equal(np.isinf(p), np.isinf(q))):
                failures.append(f"{name}: output {i} has NaN / infinity in other places")
                continue
            finite = np.isfinite(q)
            if finite.any() and not np.allclose(p[finite], q[finite], rtol=TOLERANCE, atol=TOLERANCE * 1e-3):
                failures.append(f"{name}: output {i} differs by {np.abs(p[finite] - q[finite]).max():.2e}")
        elif not np.array_equal(p, q):
            failures.append(f"{name}: output {i} differs")


def case(name, f, *args, interpreted=True, **kwargs):
    lowered, compiled, out = staged(f, args, kwargs, True)
    is_interpreted = type(compiled).__module__ == _interpret.__name__
    if is_interpreted != interpreted:
        failures.append(f"{name}: {'interpreted' if is_interpreted else 'compiled by XLA'}, expected the other")
    _, _, reference = staged(f, args, kwargs, False)
    same(name, out, reference)
    same(name + " (second call)", compiled(*args, **kwargs), reference)
    checked[0] += 1
    checked[1] += is_interpreted
    return lowered, compiled


# arithmetic, real and complex
case("arithmetic", lambda a, b: (a + b, a - b, a * b, a / b, -a), x, y)
case("complex arithmetic", lambda a, b: (a + b, a * b, a / b, jnp.conj(a), a.real, a.imag, jnp.abs(a)), z, z + 1)
case("mixed real and complex", lambda a, b: a * b + b / (a + 3), x, z)
case("scalars and broadcasting", lambda a, b, c: (a * 2.0 + b, a / c, c - b), x, v, 3.0)
case("functions", lambda a: (jnp.log(a), jnp.exp(a), jnp.sqrt(a), jnp.sin(a), jnp.cos(a), jnp.tanh(a), jnp.log1p(a), jnp.expm1(a)), y)
case("log and division at 0, inf and NaN", lambda a: (jnp.log(a), 1 / a, a / a, jnp.sqrt(a), jnp.abs(a)), zeros)
case("integer powers", lambda a: (a**2, a**3, a**-2, a**-1, a**0, a**5), y)
case("integer powers, complex", lambda a: (a**2, a**-2, a**3), z)
case("maximum and minimum", lambda a, b: (jnp.maximum(a, b), jnp.minimum(a, 0.5)), x, y)
case("comparisons and where", lambda a, b: (jnp.where(a > b - 2, a, b), jnp.where(a == 0, 1.0, a), a <= b, (a < 0) | (b > 3), ~(a > 0)), x, y)
case("abs of complex with its derivative", lambda a: jax.grad(lambda t: jnp.sum(jnp.abs(t)), holomorphic=False)(a), w)
# shapes
case("sums and means", lambda a: (a.sum(), a.sum(0), a.mean(1), a.max(), a.min(0), jnp.prod(a, axis=1)), x)
case("reshape, transpose, squeeze, expand", lambda a: (a.reshape(7, 4), a.T, a[None].squeeze(0), a[:, None, :] * jnp.ones((4, 3, 7), F)), x)
case("concatenate and static slices", lambda a, b: (jnp.concatenate([a, b]), jax.lax.slice(a, (1, 0), (3, 7), (1, 2))), x, y)
case("indexing (XLA)", lambda a: (a[1:3, ::2], a[0]), x, interpreted=False)
case("complex from parts", lambda a, b: jax.lax.complex(a, b), x, y)
case("conversions", lambda a, b: (a.astype(C), b.real.astype(F), (a > 0).astype(F), a.astype(jnp.int32)), x, z)
# program structure
case("arguments are results", lambda a, b: (a, {"b": b, "sum": a + b}, a), x, y)
case("keyword arguments", lambda a, *, scale, shift=1.0: a * scale + shift, x, scale=y, shift=2.0)
case("constant from the closure", lambda a: a * np.arange(7.0, dtype=np.float64 if X64 else np.float32) + 1.5, x)
case("nested jit", lambda a: jax.jit(lambda t: t * 2 + 1)(a) / jax.jit(jnp.abs)(a - 9), x)
case("no arguments", lambda: jnp.ones(3, F) * 2)


def host(a, b):
    return np.asarray(a) * 2 + np.asarray(b), np.asarray(np.sum(np.asarray(a)), np.asarray(a).dtype)


shapes = (jax.ShapeDtypeStruct((4, 7), F), jax.ShapeDtypeStruct((), F))
case("host callback", lambda a, b: jax.pure_callback(host, shapes, a, b * 3, vmap_method="sequential"), x, y)
case("host callback on arguments and computed values", lambda a, b: jax.pure_callback(host, shapes, a + 1, b, vmap_method="sequential")[0] / b, x, y)


@jax.custom_vjp
def clipped(t):
    return jnp.where(t > 0, t, 0.1 * t)


clipped.defvjp(lambda t: (clipped(t), t), lambda t, bar: (jnp.where(t > 0, bar, 0.1 * bar),))
case("custom derivative rule", lambda a: clipped(a) * 2, x)
case("gradient through a custom rule", jax.grad(lambda a: jnp.sum(clipped(a) ** 2)), x)


def readout(values, reference):  # the shape of the benchmark's readout
    carrier, signal, noise = values
    sensitivity = noise / jnp.abs(signal * carrier)
    return jnp.mean(jnp.log10(sensitivity / reference)), (sensitivity, jnp.abs(carrier) ** 2)


values, reference = (w, w * 2 + 1, jnp.abs(v) + 1), jnp.abs(v) + 2
_, compiled_vjp = case("vjp with auxiliary output", lambda vals, ref: jax.vjp(lambda t: readout(t, ref), vals, has_aux=True), values, reference)
loss, pullback, aux = compiled_vjp(values, reference)
case("pullback as an argument", lambda pull, bar: pull(bar), pullback, jnp.asarray(1.0, F))
case("value and gradient", jax.value_and_grad(lambda a, b: jnp.sum(jnp.log(jnp.abs(a) + b**2))), z, x)
# left to XLA: operations the interpreter does not know, integers, large programs
case("sort (XLA)", lambda a: jnp.sort(a, axis=1), x, interpreted=False)
case("integer arithmetic (XLA)", lambda a: a // 2 + a % 3, jnp.arange(12).reshape(3, 4), interpreted=False)
case("gather (XLA)", lambda a, i: a[i], v, jnp.asarray([0, 3, 5]), interpreted=False)
case("linear solve (XLA)", lambda a, b: jnp.linalg.solve(a @ a.T + 5 * jnp.eye(4, dtype=F), b), x, x[:, 0], interpreted=False)
case("more than SMALL_PROGRAM_OPERATIONS equations (XLA)", lambda a: functools.reduce(lambda t, i: jnp.sin(t) + i, range(_tuning.SMALL_PROGRAM_OPERATIONS + 5), a), x, interpreted=False)

# the stage objects behave like JAX's
lowered, compiled = case("stage objects", lambda a, b: (a * b).sum(), x, y)
ok = isinstance(lowered, stages.Lowered) and isinstance(compiled, stages.Compiled)
ok &= jax.block_until_ready(lowered) is lowered and jax.block_until_ready(compiled) is compiled
ok &= "_lowering" not in vars(lowered)  # nothing has been lowered up to here
ok &= "stablehlo" in lowered.as_text() and lowered.compile().as_text() is not None
ok &= compiled.out_info.shape == () and lowered.out_info.dtype == x.dtype
ok &= bool(np.isclose(float(compiled(x, y)), float(staged(lambda a, b: (a * b).sum(), (x, y), {}, False)[2]), rtol=TOLERANCE))
if not ok:
    failures.append("stage objects: an attribute or method does not behave like JAX's")
for label, bad in (("another structure", (x,)), ("another shape", (x, v)), ("another dtype", (x, z))):
    try:
        compiled(*bad)
        failures.append(f"errors: {label} was accepted")
    except TypeError:
        pass
_tuning.INTERPRET_SMALL_PROGRAMS = False
if type(jax.jit(lambda a: a + 1).lower(x).compile()).__module__ == _interpret.__name__:
    failures.append("switch: INTERPRET_SMALL_PROGRAMS = False still interprets")
_tuning.INTERPRET_SMALL_PROGRAMS = True

precision = "double" if X64 else "single"
print(f"{precision} precision: {checked[0]} cases, {checked[1]} interpreted, {len(failures)} failures")
for failure in failures:
    print("  FAILED", failure)
if X64 and "x64" not in sys.argv:
    code = subprocess.call([sys.executable, __file__, "x32"], env=dict(os.environ))
    sys.exit(1 if failures or code else 0)
sys.exit(1 if failures else 0)
