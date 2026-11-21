"""Port selection on the detector output: `signal_detector(carrier, signal)[ports]` on the host.

With `components.HOST_DETECTOR_PORTS` the rows of an integer index array are picked in NumPy
instead of by an XLA gather (and its scatter in reverse mode). This test compares values and
gradients with plain JAX indexing (the switch off) and with the plain JAX detector, for

- one port, two ports (balanced readout), negative, repeated and out-of-range ports,
  int32 and int64 indices, traced and constant index arrays,
- jit, grad, vjp with a hand-made cotangent, vmap over stacked solutions, cond,
- every other way of using the output (integers, slices, masks, arithmetic, `jnp` functions,
  returning it from a jitted function), which must behave exactly as before.

It also counts the operations XLA has to compile for the benchmark readout, with the switch on
and off.

Usage: python runs/philip/test_ports.py
"""

import os
import re
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import numpy as np

import differometor as df
from differometor import components

rng = np.random.default_rng(2)
n, F = 7, 5
complex_array = lambda *shape: jnp.asarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
carrier, signal, noise = complex_array(n, 1), complex_array(2 * n + 3, F), jnp.asarray(rng.random(F) + 1.0)
same = lambda a, b: np.array_equal(np.asarray(a), np.asarray(b))
close = lambda a, b: np.allclose(np.asarray(a), np.asarray(b), rtol=1e-13, atol=0)


def with_switch(value, function, *args):
    saved, components.HOST_DETECTOR_PORTS = components.HOST_DETECTOR_PORTS, value
    try:
        return function(*args)
    finally:
        components.HOST_DETECTOR_PORTS = saved


def readout(carrier, signal, noise, ports):
    signals = df.signal_detector(carrier, signal)[ports]
    detected = signals.squeeze() if ports.shape[0] == 1 else signals[0] - 0.5 * signals[1]  # no cancellation for repeated ports
    return jnp.mean(jnp.log10(noise / jnp.abs(detected)))


def reference_readout(carrier, signal, noise, ports):
    signals = components.signal_detector_jax(carrier, signal)[ports]
    detected = signals.squeeze() if ports.shape[0] == 1 else signals[0] - 0.5 * signals[1]  # no cancellation for repeated ports
    return jnp.mean(jnp.log10(noise / jnp.abs(detected)))


# 1. Values and gradients for different port arrays, traced and constant, against plain JAX.
cases = {
    "one port": [3],
    "two ports": [1, 5],
    "negative": [-1, -7],
    "repeated": [2, 2],
    "out of range": [n + 2, 4],
    "below range": [-n - 3, 0],
}
for name, ports in cases.items():
    for dtype in (np.int64, np.int32):
        index = jnp.asarray(np.array(ports, dtype))
        expected = jax.value_and_grad(reference_readout, argnums=(0, 1, 2))(carrier, signal, noise, index)
        for label, function in (
            ("traced index", jax.jit(jax.value_and_grad(readout, argnums=(0, 1, 2)))),
            ("constant index", jax.jit(lambda c, s, w: jax.value_and_grad(readout, argnums=(0, 1, 2))(c, s, w, np.array(ports, dtype)))),
        ):
            arguments = (carrier, signal, noise, index)[: 4 if label == "traced index" else 3]
            host = with_switch(True, function, *arguments)
            plain = with_switch(False, jax.jit(jax.value_and_grad(readout, argnums=(0, 1, 2))), carrier, signal, noise, index)
            assert all(close(a, b) for a, b in zip(jax.tree_util.tree_leaves(host), jax.tree_util.tree_leaves(expected))), (name, label)
            assert all(close(a, b) for a, b in zip(jax.tree_util.tree_leaves(host), jax.tree_util.tree_leaves(plain))), (name, label)
print(f"{len(cases)} kinds of port arrays, int32 and int64, traced and constant: values and gradients as with plain JAX indexing")

# 2. vjp with a hand-made cotangent, vmap, cond, forward values under plain jit.
ports = jnp.asarray([1, -2, 1])
cotangent = complex_array(3, F)
picked = lambda c, s: df.signal_detector(c, s)[ports]
picked_reference = lambda c, s: components.signal_detector_jax(c, s)[ports]
value, pullback = jax.vjp(picked, carrier, signal)
value_ref, pullback_ref = jax.vjp(picked_reference, carrier, signal)
assert close(value, value_ref) and all(close(a, b) for a, b in zip(pullback(cotangent), pullback_ref(cotangent)))
assert close(jax.jit(picked)(carrier, signal), value_ref)
carriers, signals = jnp.stack([carrier, 2 * carrier]), jnp.stack([signal, signal + 1])
assert close(jax.vmap(picked)(carriers, signals), jax.vmap(picked_reference)(carriers, signals))
power = lambda pick: jax.vmap(jax.grad(lambda c, s: jnp.sum(jnp.abs(pick(c, s)) ** 2)))
assert close(jax.jit(power(picked))(carriers, signals), power(picked_reference)(carriers, signals))
branch = jax.jit(lambda c, s, flag: jax.lax.cond(flag, lambda: df.signal_detector(c, s)[ports], lambda: jnp.zeros((3, F), complex)))
assert close(branch(carrier, signal, True), value_ref) and same(branch(carrier, signal, False), np.zeros((3, F)))
print("vjp with a given cotangent, jit, vmap, grad under vmap and cond agree with plain JAX")


# 3. Everything else behaves as before (compared with the switch off, bit for bit).
def other_uses(c, s):
    output = df.signal_detector(c, s)
    assert isinstance(output, jax.Array) and output.shape == (n, F) and output.dtype == np.complex128
    mask = np.arange(n) % 2 == 0
    return (
        output,
        output[2],
        output[-1, 1],
        output[1:4],
        output[:, 2],
        output[mask],
        output[np.array([1, 3]), np.array([0, 2])],
        output[np.array([[0, 1], [2, 3]])],
        output[jnp.asarray(3)],
        jnp.abs(output) + output.real,
        2 * output - output.T[:, :n].sum(),
        jnp.sum(output, axis=0),
        output.reshape(-1)[::3],
    )


for transform in (lambda f: f, jax.jit):
    on = with_switch(True, transform(other_uses), carrier, signal)
    off = with_switch(False, transform(other_uses), carrier, signal)
    assert len(on) == len(off) and all(same(a, b) and a.shape == b.shape for a, b in zip(on, off))
gradient = lambda value: jax.jit(jax.grad(lambda c, s: jnp.sum(jnp.abs(other_uses(c, s)[1]) ** 2) + jnp.sum(jnp.abs(other_uses(c, s)[3])), argnums=(0, 1)))
assert all(same(a, b) for a, b in zip(with_switch(True, gradient(True), carrier, signal), with_switch(False, gradient(False), carrier, signal)))
assert isinstance(df.signal_detector(carrier, signal), jax.Array) and close(df.signal_detector(carrier, signal)[np.array([1])], components.signal_detector_jax(carrier, signal)[np.array([1])])
print("integers, slices, masks, index tuples, nested index arrays, arithmetic, jnp functions and returned outputs are unchanged")


# 4. What XLA has to compile for the benchmark readout.
def stages(value):
    def count():
        ports = jnp.asarray([3])
        forward = jax.jit(lambda c, s, w, p: jax.vjp(lambda c, s, w: readout(c, s, w, p), c, s, w))
        text = forward.lower(carrier, signal, noise, ports).as_text()
        _, pullback = forward(carrier, signal, noise, ports)
        reverse = jax.jit(lambda pullback, cotangent: pullback(cotangent)).lower(pullback, jnp.asarray(1.0)).as_text()
        ops = lambda text: sorted(set(re.findall(r"stablehlo\.(gather|scatter)", text)))
        return len(re.findall(r"stablehlo\.\w+", text)), ops(text), len(re.findall(r"stablehlo\.\w+", reverse)), ops(reverse)

    return with_switch(value, count)


on, off = stages(True), stages(False)
print(f"readout with pullback: {off[0]} operations {off[1]} -> {on[0]} {on[1]};  reverse: {off[2]} operations {off[3]} -> {on[2]} {on[3]}")
assert on[1] == [] and on[3] == [] and off[1] == ["gather"] and off[3] == ["scatter"]
print("OK")
