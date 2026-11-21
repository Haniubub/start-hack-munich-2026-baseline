"""Reverse pass of the fast path for arbitrary output cotangents, against the original JAX code.

The benchmark readout only puts a cotangent on the two signal rows of the noise detector. This
test uses a loss that reads every carrier, signal and noise value with random weights, so the
general adjoint solve of the fast path (a second factorization in the reverse pass) is exercised,
and compares the full gradient with `simulate_jax`. It also repeats the benchmark-style readout.

Two more checks on the fast path itself:
- The frequency block size must not matter (blocks of 1 and 2 against the default).
- When the constant entries of the signal matrix do not fit into the residuals of the reverse
  pass, the general adjoint has to fail with a clear message, and the usual readout still works.

Usage: python runs/noel/test_adjoint.py <repo>
"""
import json
import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
sys.path.insert(0, sys.argv[1])
import jax
import jax.numpy as jnp
import numpy as np

import differometor as df
from differometor import fast
from differometor.setups import constrain_inter_grid_cell_spaces
from differometor.simulate import simulate_jax

PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
data = os.path.join(sys.argv[1], "examples", "data")
fixture = json.load(open(os.path.join(data, "uifo_800_3000.json")))
reference = json.load(open(os.path.join(data, "uifo_800_3000_test_data.json")))
setup = df.Setup.from_data(fixture)
pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
initial = []
for pair in pairs:
    members = pair if isinstance(pair[0], list) else [pair]
    name, prop = members[0]
    initial.append(fixture["nodes" if name in fixture["nodes"] else "edges"][name]["properties"][prop])
index = np.array([0, 37, 100])
frequencies = jnp.asarray(np.asarray(reference["frequencies"])[index])
arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], frequencies, pairs)
ports = jnp.asarray(ports)
parameters = jnp.asarray(initial, dtype=jnp.float64)
assert df.simulate is not simulate_jax

rng = np.random.default_rng(7)
carrier0, signal0, noise0 = (np.asarray(x) for x in simulate_jax(**dict(arrays, optimized_parameters=parameters)))
weights = lambda x: jnp.asarray((rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape)) / (np.abs(x) + np.abs(x).max() * 1e-6))
wc, ws, wn = weights(carrier0), weights(signal0), jnp.asarray(rng.normal(size=noise0.shape) / noise0)


def general_loss(sim):
    def loss(p):
        carrier, signal, noise = sim(**dict(arrays, optimized_parameters=p))
        return jnp.real(jnp.sum(wc * carrier)) + jnp.real(jnp.sum(ws * signal)) + jnp.sum(wn * noise)

    return jax.jit(jax.value_and_grad(loss))


def readout_loss(sim):
    def loss(p):
        carrier, signal, noise = sim(**dict(arrays, optimized_parameters=p))
        detected = df.signal_detector(carrier, signal)[ports].squeeze()
        return jnp.mean(jnp.log10(noise / jnp.abs(detected))) + 1e-3 * jnp.sum(df.power_detector(carrier))

    return jax.jit(jax.value_and_grad(loss))


rel = lambda a, b: float(np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(np.asarray(b)).max())
worst = 0.0
for label, make in (("all outputs, random weights (general adjoint)", general_loss), ("benchmark-style readout", readout_loss)):
    value, gradient = make(df.simulate)(parameters)
    value_jax, gradient_jax = make(simulate_jax)(parameters)
    deviation = rel(gradient, gradient_jax)
    worst = max(worst, deviation)
    print(f"{label:48s} loss {float(value):+.9e} vs {float(value_jax):+.9e}   gradient vs JAX {deviation:.2e}")
print("WORST", worst)
assert worst < 1e-5

# Block size: the compiled functions read it in the host callbacks, so they can be reused.
blocks = 0.0
if hasattr(fast, "BLOCK_FREQUENCIES"):
    default = fast.BLOCK_FREQUENCIES
    for make in (general_loss, readout_loss):
        function = make(df.simulate)
        value, gradient = function(parameters)
        for size in (1, 2):
            fast.BLOCK_FREQUENCIES = size
            value_blocked, gradient_blocked = function(parameters)
            blocks = max(blocks, rel(gradient_blocked, gradient), abs(float(value_blocked - value)) / abs(float(value)))
        fast.BLOCK_FREQUENCIES = default
    print(f"block sizes 1 and 2 against {default}: largest deviation {blocks:.2e}")
    assert blocks < 1e-11

# Residual capacity: with room for one constant entry per unknown the signal matrix does not fit.
if hasattr(fast, "BASE_ENTRIES_PER_UNKNOWN"):
    default = fast.BASE_ENTRIES_PER_UNKNOWN
    fast.BASE_ENTRIES_PER_UNKNOWN = 1
    try:
        value, gradient = readout_loss(df.simulate)(parameters)
        reference_value, reference_gradient = readout_loss(simulate_jax)(parameters)
        deviation = rel(gradient, reference_gradient)
        print(f"small residual capacity, benchmark-style readout: gradient vs JAX {deviation:.2e}")
        assert deviation < 1e-5
        try:
            general_loss(df.simulate)(parameters)
        except Exception as error:
            assert "BASE_ENTRIES_PER_UNKNOWN" in str(error), error
            print("small residual capacity, general adjoint: fails with a clear message, as intended")
        else:
            raise AssertionError("the general adjoint should have reported the missing constant entries")
    finally:
        fast.BASE_ENTRIES_PER_UNKNOWN = default
