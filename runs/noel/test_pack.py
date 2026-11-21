"""Packed residuals of the fast path: `_pack` and `_unpack` must restore every array exactly.

Checks the arrays of the benchmark setup and synthetic arrays of other dtypes and shapes
(boolean, 32-bit, empty, zero-dimensional, seven dimensions), and that the statically computed
sizes (`_packed_sizes`, used for the callback result shapes) match what `_pack` returns.

Usage: python runs/noel/test_pack.py <repo>
"""
import json, os, sys
os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["JAX_ENABLE_X64"] = "true"
sys.path.insert(0, sys.argv[1])
import jax.numpy as jnp, numpy as np
import differometor as df
from differometor import fast
from differometor.setups import constrain_inter_grid_cell_spaces

PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
data = os.path.join(sys.argv[1], "examples", "data")
fixture = json.load(open(os.path.join(data, "uifo_800_3000.json"))); reference = json.load(open(os.path.join(data, "uifo_800_3000_test_data.json")))
setup = df.Setup.from_data(fixture)
pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], jnp.asarray(reference["frequencies"]), pairs)


def check(name, A):
    packed = fast._pack(A)
    assert [x.dtype for x in packed] == [np.int64, np.float64] and all(x.ndim == 1 for x in packed), name
    assert tuple(x.shape[0] for x in packed) == fast._packed_sizes(A), (name, [x.shape for x in packed], fast._packed_sizes(A))
    restored = fast._unpack(*packed)
    assert list(restored) == list(fast.PACKED_KEYS), name
    for key in fast.PACKED_KEYS:
        a, b = np.asarray(A[key]), restored[key]
        assert a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b), (name, key, a.dtype, b.dtype, a.shape, b.shape)
    print(f"{name}: {len(restored)} arrays restored exactly from {packed[0].nbytes + packed[1].nbytes} bytes")


benchmark = {key: np.asarray(arrays[key]) for key in fast.HOST_KEYS}
check("benchmark setup", benchmark)
assert fast._packed_sizes(arrays) == fast._packed_sizes(benchmark)  # JAX arrays give the same static sizes

rng = np.random.default_rng(0)
other = dict(benchmark)
other["optimized_parameter_indices"] = np.array([True, False, True])
other["optimized_value_indices"] = rng.integers(-5, 5, (1, 2, 3, 1)).astype(np.int32)
other["signal_changing_values"] = rng.normal(size=(5, 1)).astype(np.float32)
other["parameters"] = (rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3))).astype(np.complex64)
other["carrier_changing_values"] = np.empty((0, 3))
other["noise_function_indices"] = np.array(7, dtype=np.uint8)
other["noise_selection_vectors"] = rng.normal(size=(1, 2, 1, 2, 2, 3, 1))
other["carrier_function_indices"] = np.array([np.iinfo(np.int64).min, np.iinfo(np.int64).max])
check("other dtypes and shapes", other)
print("OK")
