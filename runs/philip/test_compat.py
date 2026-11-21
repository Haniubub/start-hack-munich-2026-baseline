"""Compatibility paths of the fast path must give the same results as the default path.

  1. Dense base matrices (callers that still pass dense arrays instead of `BaseMatrix`).
  2. No compiled CSR kernels (`fast._sparsetools is None`): the pre-elimination is skipped.
Usage: python runs/philip/test_compat.py [repo]
"""
import json
import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import jax
import jax.numpy as jnp
import numpy as np

import differometor as df
from differometor import fast
from differometor.setups import constrain_inter_grid_cell_spaces
from differometor.sparse import BaseMatrix

PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
data = os.path.join(ROOT, "examples", "data")
fixture = json.load(open(os.path.join(data, "uifo_800_3000.json")))
reference = json.load(open(os.path.join(data, "uifo_800_3000_test_data.json")))
setup = df.Setup.from_data(fixture)
pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
initial = []
for pair in pairs:
    name, prop = (pair if isinstance(pair[0], list) else [pair])[0]
    initial.append(fixture["nodes" if name in fixture["nodes"] else "edges"][name]["properties"][prop])
parameters = jnp.asarray(np.asarray(initial, float))
arrays, *_ = df.run_build_step(setup, [("f", "frequency")], jnp.asarray(reference["frequencies"][::5]), pairs)
assert all(isinstance(arrays[key], BaseMatrix) for key in ("carrier_matrix", "signal_matrix", "noise_matrix"))


def run(arrays):
    def loss(p):
        carrier, signal, noise = df.simulate(**dict(arrays, optimized_parameters=p))
        return jnp.sum(jnp.log(noise)) + jnp.sum(jnp.abs(carrier) ** 2) * 1e-3, (carrier, signal, noise)

    (_, outputs), gradient = jax.value_and_grad(loss, has_aux=True)(parameters)
    return [np.asarray(x) for x in (*outputs, gradient)]


rel = lambda a, b: float(np.abs(a - b).max() / np.abs(b).max())
default = run(arrays)

dense = dict(arrays)
for key in ("carrier_matrix", "signal_matrix", "noise_matrix"):
    dense[key] = jnp.asarray(np.asarray(arrays[key]))
worst = max(rel(a, b) for a, b in zip(run(dense), default))
print(f"dense base matrices:      max relative difference {worst:.1e}")
assert worst == 0.0

kernels, fast._sparsetools = fast._sparsetools, None
try:
    plain = run(arrays)
finally:
    fast._sparsetools = kernels
names = ("carrier", "signal", "noise", "gradient")
differences = {name: rel(a, b) for name, a, b in zip(names, plain, default)}
print("without the CSR kernels: ", ", ".join(f"{name} {value:.1e}" for name, value in differences.items()))
assert differences["carrier"] == 0.0 and differences["noise"] < 1e-6 and differences["gradient"] < 1e-6 and differences["signal"] < 1e-5
