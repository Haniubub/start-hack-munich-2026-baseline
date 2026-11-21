"""Reverse pass of the fast path for arbitrary output cotangents, against the original JAX code.

The benchmark readout only puts a cotangent on the two signal rows of the noise detector. This
test uses a loss that reads every carrier, signal and noise value with random weights, so the
general adjoint solve of the fast path (a second factorization in the reverse pass) is exercised,
and compares the full gradient with `simulate_jax`. It also repeats the benchmark-style readout.

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
