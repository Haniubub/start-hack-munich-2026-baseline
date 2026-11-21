"""Check `fast._CondensedLU` (exact pre-elimination + SuperLU) against a plain SuperLU solve.

For the benchmark setup and the two aLIGO setups that take the fast path:
  - solutions of S x = b and S^T z = b with random right-hand sides, with and without elimination
  - row-wise backward error max |S x - b| / (|S||x| + |b|) over all rows and frequencies

With random right-hand sides the plain SuperLU solve (partial pivoting) leaves some rows with a
backward error of several percent, because the rows differ in scale by more than 20 orders of
magnitude. The pre-elimination keeps it below 1e-8 here.
Usage: python runs/philip/test_condensed.py [repo]
"""
import json
import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import jax.numpy as jnp
import numpy as np

import differometor as df
from differometor import fast
from differometor.setups import aligo, constrain_inter_grid_cell_spaces

PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]


def benchmark_setup():
    data = os.path.join(ROOT, "examples", "data")
    fixture = json.load(open(os.path.join(data, "uifo_800_3000.json")))
    reference = json.load(open(os.path.join(data, "uifo_800_3000_test_data.json")))
    setup = df.Setup.from_data(fixture)
    pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
    initial = []
    for pair in pairs:
        name, prop = (pair if isinstance(pair[0], list) else [pair])[0]
        initial.append(fixture["nodes" if name in fixture["nodes"] else "edges"][name]["properties"][prop])
    return setup, pairs, jnp.asarray(reference["frequencies"]), np.asarray(initial, float)


def check(name, setup, pairs, frequencies, parameters):
    arrays, *_ = df.run_build_step(setup, [("f", "frequency")], frequencies, pairs)
    A = {key: arrays[key] for key in fast.HOST_KEYS}
    entries = fast.EntryPlan(A).values(parameters)
    plan = fast.SystemPlan(A)
    carrier_lu, carrier_rhs = plan.carrier_matrix(entries[0])
    V, _ = plan.signal_values(entries[1], carrier_lu.solve(carrier_rhs))
    m, F = plan.m, V.shape[0]
    rows, cols = plan.matrix_rows, plan.matrix_cols
    values = np.ascontiguousarray(V.T[plan.in_matrix])
    condensed = fast._CondensedLU(m, rows, cols, values)
    stages = fast._CondensedLU.MAX_STAGES
    fast._CondensedLU.MAX_STAGES = 0
    plain = fast._CondensedLU(m, rows, cols, values)
    fast._CondensedLU.MAX_STAGES = stages
    rng = np.random.default_rng(1)
    b = rng.standard_normal((F, m, 2)) + 1j * rng.standard_normal((F, m, 2))
    absolute = np.abs(values)
    worst = 0.0
    print(f"{name}: {m} unknowns, {F} frequencies, {len(condensed.stages)} elimination stages -> {condensed.rest.size} left for SuperLU")
    for trans in ("N", "T"):
        x, reference = condensed.solve(b, trans), plain.solve(b, trans)
        r, c = (rows, cols) if trans == "N" else (cols, rows)
        product = np.zeros((F, m, 2), complex)
        scale = np.zeros((F, m, 2))
        for f in range(F):
            np.add.at(product[f], r, values[:, f, None] * x[f, c])
            np.add.at(scale[f], r, absolute[:, f, None] * np.abs(x[f, c]))
        residual = (np.abs(product - b) / (scale + np.abs(b))).max()
        residual_plain_x = np.zeros((F, m, 2), complex)
        for f in range(F):
            np.add.at(residual_plain_x[f], r, values[:, f, None] * reference[f, c])
        residual_plain = (np.abs(residual_plain_x - b) / (scale + np.abs(b))).max()
        difference = np.abs(x - reference).max() / np.abs(reference).max()
        worst = max(worst, residual)
        print(f"  {trans}: row-wise backward error {residual:.2e} (plain SuperLU {residual_plain:.2e}), max |x - x_plain| / max |x| {difference:.2e}")
    return worst


worst = check("benchmark UIFO", *benchmark_setup())
for kind in ("space_modulation", "frequency_modulation"):
    setup, pairs = aligo(kind)
    frequencies = jnp.asarray(json.load(open(os.path.join(ROOT, "examples", "data", "aligo_test_data.json")))["frequencies"])
    arrays, *_ = df.run_build_step(setup, [("f", "frequency")], frequencies, pairs)
    parameters = np.asarray(arrays["parameters"])[0, np.asarray(arrays["optimized_parameter_indices"])].real
    order = np.asarray(arrays["optimized_value_indices"])
    values = np.zeros(int(order.max()) + 1)
    values[order] = parameters
    worst = max(worst, check(f"aLIGO {kind}", setup, pairs, frequencies, values))
print("WORST row-wise backward error with elimination", worst)
assert worst < 1e-7
