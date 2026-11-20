"""Host detectors against the plain JAX detectors: values, VJPs, jit and vmap.

Usage: python runs/noel/test_detectors.py <repo>
"""
import os, sys
os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["JAX_ENABLE_X64"] = "true"
sys.path.insert(0, sys.argv[1])
import jax, jax.numpy as jnp, numpy as np
import differometor as df
from differometor import components as c
rng = np.random.default_rng(1)
cx = lambda *shape: jnp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape))
rel = lambda a, b: float(np.abs(np.asarray(a) - np.asarray(b)).max() / max(np.abs(np.asarray(b)).max(), 1e-300))
worst = 0.0
def check(name, value):
    global worst; worst = max(worst, value); print(f"  {name:52s} {value:.2e}")
for label, carrier, signal in [("benchmark shapes (286,1) x (751,101)", cx(286, 1), cx(751, 101)), ("1-D (5,) x (13,)", cx(5), cx(13)), ("(4,3) x (11,3)", cx(4, 3), cx(11, 3)), ("(4,1) x (8,1)", cx(4, 1), cx(8, 1))]:
    print(label)
    host = df.signal_detector(carrier, signal); ref = c.signal_detector_jax(carrier, signal)
    check("signal_detector value", rel(host, ref)); assert host.shape == ref.shape and host.dtype == ref.dtype
    ct = cx(*ref.shape)
    gh = jax.vjp(df.signal_detector, carrier, signal)[1](ct); gr = jax.vjp(c.signal_detector_jax, carrier, signal)[1](ct)
    check("signal_detector vjp carrier", rel(gh[0], gr[0])); check("signal_detector vjp signal", rel(gh[1], gr[1]))
    assert gh[0].shape == carrier.shape and gh[1].shape == signal.shape
    ph = df.power_detector(carrier); pr = c.power_detector_jax(carrier)
    check("power_detector value", rel(ph, pr)); assert ph.dtype == pr.dtype and ph.shape == pr.shape
    ctp = jnp.asarray(rng.normal(size=pr.shape))
    check("power_detector vjp", rel(jax.vjp(df.power_detector, carrier)[1](ctp)[0], jax.vjp(c.power_detector_jax, carrier)[1](ctp)[0]))
    # jit + grad of a scalar readout like the benchmark
    def loss(det, pw):
        return lambda car, sig: jnp.mean(jnp.log10(jnp.abs(det(car, sig)))) + jnp.sum(pw(car))
    gh = jax.jit(jax.grad(loss(df.signal_detector, df.power_detector), argnums=(0, 1)))(carrier, signal)
    gr = jax.jit(jax.grad(loss(c.signal_detector_jax, c.power_detector_jax), argnums=(0, 1)))(carrier, signal)
    check("jit(grad(readout)) carrier", rel(gh[0], gr[0])); check("jit(grad(readout)) signal", rel(gh[1], gr[1]))
print("vmap")
car, sig = cx(6, 4, 1), cx(6, 9, 7)
check("vmap signal_detector", rel(jax.vmap(df.signal_detector)(car, sig), jax.vmap(c.signal_detector_jax)(car, sig)))
check("vmap power_detector", rel(jax.vmap(df.power_detector)(car), jax.vmap(c.power_detector_jax)(car)))
print("real input falls back to JAX:", df.power_detector(jnp.arange(3.0)))
c.HOST_DETECTORS = False
print("switch off -> forward mode works:", jax.jvp(df.power_detector, (cx(3),), (cx(3),))[1].shape)
c.HOST_DETECTORS = True
print("WORST", worst); assert worst < 1e-12
