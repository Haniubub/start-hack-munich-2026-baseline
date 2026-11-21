"""Host constants in the build output (`differometor.sparse.Constant`).

The build step returns the arrays that describe the structure of a setup as constants: jitted
functions that take the build output as an argument are specialized on them instead of receiving
them at run time. This test checks, on the benchmark setup:

- which entries are constants and which stay runtime arrays,
- that outputs and gradient are bit-identical to a build with JAX arrays (`STATIC = False`),
  also when single entries are replaced by JAX arrays,
- that rebuilding the same setup or changing the sweep values does not retrace, and that a
  constant with different content does (and gives the result for the new content),
- that the original JAX path gives the same result for constants and for JAX arrays,
- that arrays derived from a constant, copies and pickled constants are ordinary arrays,
- that `vmap` over the design parameters still works.

Usage: python runs/philip/test_static.py
"""

import copy
import json
import os
import pickle
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo))

import jax
import jax.numpy as jnp
import numpy as np

import differometor as df
from differometor import sparse
from differometor.setups import constrain_inter_grid_cell_spaces
from differometor.simulate import simulate_jax

PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
RUNTIME = {"optimized_parameters", "parameters", "carrier_changing_values", "signal_changing_values"}
data = repo / "examples" / "data"
fixture = json.loads((data / "uifo_800_3000.json").read_text())
reference = json.loads((data / "uifo_800_3000_test_data.json").read_text())
all_frequencies = np.asarray(reference["frequencies"])


def build(frequencies, static=True):
    sparse.STATIC = static
    try:
        setup = df.Setup.from_data(fixture)
        pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
        initial = []
        for pair in pairs:
            name, prop = pair[0] if isinstance(pair[0], list) else pair
            initial.append(fixture["nodes" if name in fixture["nodes"] else "edges"][name]["properties"][prop])
        arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], jnp.asarray(frequencies), pairs)
    finally:
        sparse.STATIC = True
    parameters = jnp.asarray(initial, dtype=jnp.float64)
    arrays["optimized_parameters"] = parameters
    return arrays, jnp.asarray(ports), parameters, pairs.index(["mb11", "tuning"])


traces = []


def loss(parameters, arrays, ports, simulate=df.simulate):
    traces.append(1)
    carrier, signal, noise = simulate(**dict(arrays, optimized_parameters=parameters))
    detected = df.signal_detector(carrier, signal)[ports].squeeze()
    sensitivity = noise / jnp.abs(detected)
    return jnp.mean(jnp.log10(sensitivity)) + 1e-3 * jnp.sum(df.power_detector(carrier)), (carrier, signal, noise)


value_and_grad = jax.jit(jax.value_and_grad(loss, has_aux=True))
same = lambda a, b: all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))

frequencies = all_frequencies[::10]
arrays, ports, parameters, tuning = build(frequencies)

# 1. What is constant, what is a runtime array.
for name, value in arrays.items():
    leaves = jax.tree_util.tree_leaves(value)
    if name in RUNTIME:
        assert len(leaves) == 1 and isinstance(value, jax.Array), name
    else:
        assert len(leaves) == 0, (name, type(value))
        for part in (value.index, value.values) if isinstance(value, sparse.BaseMatrix) else (value,):
            assert isinstance(part, sparse.Constant) and not part.flags.writeable, name
print(f"{len(arrays)} build outputs, {len(jax.tree_util.tree_leaves(arrays))} runtime arrays: {sorted(RUNTIME)}")

# 2. Same numbers as with JAX arrays throughout.
result = value_and_grad(parameters, arrays, ports)
assert len(traces) == 1
dynamic, *_ = build(frequencies, static=False)
assert len(jax.tree_util.tree_leaves(dynamic)) == 34 and not any(isinstance(v, sparse.Constant) for v in dynamic.values())
result_dynamic = value_and_grad(parameters, dynamic, ports)
assert len(traces) == 2 and same(result, result_dynamic)
mixed = dict(arrays)
for name in ("signal_matrix_indices", "noise_selection_vectors", "signal_function_input_indices"):
    mixed[name] = jnp.asarray(np.asarray(arrays[name]))
mixed["signal_matrix"] = sparse.BaseMatrix(*(jnp.asarray(np.asarray(x)) for x in (arrays["signal_matrix"].index, arrays["signal_matrix"].values)), arrays["signal_matrix"].shape)
assert same(result, value_and_grad(parameters, mixed, ports)) and len(traces) == 3
print("outputs and gradient identical for constants, JAX arrays and a mix of both")

# 3. Retracing follows the content.
again, *_ = build(frequencies)
assert all(a is not b for a, b in zip(arrays.values(), again.values()))
assert same(result, value_and_grad(parameters, again, ports)) and len(traces) == 3, "rebuilding the setup retraced"
shifted, *_ = build(frequencies * 1.5)
result_shifted = value_and_grad(parameters, shifted, ports)
assert len(traces) == 3 and not same(result[0], result_shifted[0]), "new sweep values must not retrace"
selection = np.array(arrays["noise_selection_vectors"])
selection[..., np.flatnonzero(selection.ravel())[0]] *= 2.0
changed = dict(arrays, noise_selection_vectors=sparse.constant(selection))
result_changed = value_and_grad(parameters, changed, ports)
assert len(traces) == 4, "a constant with new content must retrace"
expected = value_and_grad(parameters, dict(dynamic, noise_selection_vectors=jnp.asarray(selection)), ports)
assert same(result_changed, expected) and not same(result_changed[0], result[0])
compiled = jax.jit(lambda p, a: df.simulate(**dict(a, optimized_parameters=p))).lower(parameters, arrays).compile()
assert same(compiled(parameters, arrays), compiled(parameters, again))
try:
    compiled(parameters, changed)
    raise AssertionError("a compiled program accepted a constant with different content")
except TypeError:
    pass
print("no retrace for a rebuilt setup or new sweep values; new content retraces and is used")

# 4. The original JAX path, on two frequencies.
few, *_ = build(all_frequencies[[0, 60]])
few_dynamic, *_ = build(all_frequencies[[0, 60]], static=False)
original = jax.jit(lambda p, a: simulate_jax(**dict(a, optimized_parameters=p)))
out_constants, out_arrays = original(parameters, few), original(parameters, few_dynamic)
assert same(out_constants, out_arrays)
fast = df.simulate(**few)
worst = max(float(np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(np.asarray(b)).max()) for a, b in zip(fast, out_arrays))
assert worst < 1e-6, worst
print(f"original JAX path identical for constants and JAX arrays; fast path vs JAX {worst:.1e}")

# 5. Only arrays made by `constant` are constants.
index = arrays["signal_matrix_indices"]
count = lambda value: len(jax.tree_util.tree_leaves(value))
assert count(index) == 0
for derived in (index[0], index + 1, index.copy(), index.reshape(-1), index.T, index.astype(float), copy.copy(index), copy.deepcopy(index), pickle.loads(pickle.dumps(index))):
    assert count(derived) == 1, type(derived)
runtime = jnp.asarray(np.asarray(index))  # the way to a runtime JAX array (`jnp.asarray(index)` leaves a constant as it is)
assert isinstance(runtime, jax.Array) and count(runtime) == 1 and isinstance(jnp.array(index), jax.Array)
assert isinstance(index.at[0, 0].set(5), jax.Array) and isinstance(jnp.take(jnp.arange(5000.0), index), jax.Array)
assert index.block_until_ready() is index
try:
    index[0, 0] = 1
    raise AssertionError("a constant was writable")
except ValueError:
    pass
restored = pickle.loads(pickle.dumps({k: (np.asarray(v) if not isinstance(v, sparse.BaseMatrix) else v) for k, v in arrays.items() if k not in RUNTIME}))
restored.update({k: arrays[k] for k in RUNTIME})
assert same(result, value_and_grad(parameters, restored, ports))
print("slices, arithmetic, copies and pickled constants are ordinary arrays; constants are read-only")

# 6. Batches of design parameters.
batch = jnp.stack([parameters, parameters.at[tuning].add(1e-3)])  # the parameter benchmark.py perturbs
batched = jax.jit(jax.vmap(lambda p: value_and_grad(p, arrays, ports)[1]))(batch)
single = [value_and_grad(p, arrays, ports)[1] for p in batch]
assert np.allclose(np.asarray(batched), np.asarray(single), rtol=1e-12, atol=0)
print("vmap over the design parameters matches single calls")
print("OK")
