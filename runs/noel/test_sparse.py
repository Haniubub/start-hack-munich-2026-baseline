"""Sparse system matrices (`fast.SparseMatrix`) between the build step and `simulate`.

- Coordinate form and dense form agree (`np.asarray`, `dense()`, `from_dense`).
- The signal matrix is the carrier matrix in both sideband blocks plus a unit diagonal for the
  signal degrees of freedom; the noise matrix is empty.
- `simulate` with dense matrices takes the original JAX path and gives exactly what `simulate_jax`
  gives for the sparse matrices; the fast path agrees with both.
- Matrices changed by hand (other constant values in the carrier and the signal matrix, also in a
  slot that is multiplied by the carrier, and constant entries in the noise matrix): outputs and
  gradient of the fast path agree with the JAX path.
- A derivative with respect to matrix values takes the JAX path.

Usage: python runs/noel/test_sparse.py <repo>
"""
import json, os, sys
os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["JAX_ENABLE_X64"] = "true"
sys.path.insert(0, sys.argv[1])
import jax, jax.numpy as jnp, numpy as np
import differometor as df
from differometor import fast
from differometor.fast import SparseMatrix
from differometor.setups import constrain_inter_grid_cell_spaces, aligo
from differometor.simulate import simulate_jax

PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
data = os.path.join(sys.argv[1], "examples", "data")
fixture = json.load(open(os.path.join(data, "uifo_800_3000.json"))); reference = json.load(open(os.path.join(data, "uifo_800_3000_test_data.json")))
setup = df.Setup.from_data(fixture)
pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
frequencies = jnp.asarray(np.asarray(reference["frequencies"])[[0, 50, 100]])
MATRICES = ("carrier_matrix", "signal_matrix", "noise_matrix")


def structure(name, arrays):
    carrier, signal, noise = (arrays[key] for key in MATRICES)
    assert all(isinstance(matrix, SparseMatrix) for matrix in (carrier, signal, noise)), name
    n, m = carrier.shape[0], signal.shape[0]
    assert carrier.shape == (n, n + 1) and signal.shape == (m, m + 1) and noise.shape == (m, m), name
    for matrix in (carrier, signal, noise):
        dense = np.asarray(matrix)
        assert dense.shape == matrix.shape and dense.dtype == np.complex128, name
        assert np.array_equal(dense, np.asarray(matrix.dense())), name
        again = SparseMatrix.from_dense(dense)
        assert np.array_equal(np.asarray(again.indices), np.asarray(matrix.indices)[:, np.lexsort(np.asarray(matrix.indices)[::-1])]), name
        assert np.array_equal(np.asarray(again), dense) and np.count_nonzero(dense) == matrix.values.shape[0], name
    C, S = np.asarray(carrier), np.asarray(signal)
    expected = np.zeros((m, m + 1), complex)
    expected[:n, :n] = C[:, :n]
    expected[n : 2 * n, n : 2 * n] = C[:, :n]
    expected[np.arange(2 * n, m), np.arange(2 * n, m)] = 1
    assert np.array_equal(S, expected) and noise.values.shape[0] == 0, name
    print(f"{name}: carrier {carrier}, signal {signal}, noise {noise}: structure as expected")


arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], frequencies, pairs)
structure("uifo", arrays)
for mode in ("space_modulation", "amplitude_modulation"):
    aligo_arrays, *_ = df.run_build_step(aligo(mode)[0], [("f", "frequency")], frequencies, aligo(mode)[1])
    structure(f"aligo {mode}", aligo_arrays)

# Dispatcher: dense matrices are not supported by the fast path and take the JAX path.
initial = []
for pair in pairs:
    name, prop = (pair if isinstance(pair[0], list) else [pair])[0]
    initial.append(fixture["nodes" if name in fixture["nodes"] else "edges"][name]["properties"][prop])
parameters = jnp.asarray(initial, dtype=jnp.float64)
sparse_arguments = dict(arrays, optimized_parameters=parameters)
dense_arguments = dict(sparse_arguments, **{key: jnp.asarray(np.asarray(arrays[key])) for key in MATRICES})
mixed_arguments = dict(sparse_arguments, noise_matrix=dense_arguments["noise_matrix"])
assert fast.supports(sparse_arguments) and not fast.supports(dense_arguments) and not fast.supports(mixed_arguments)
rel = lambda a, b: float(np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(np.asarray(b)).max())
reference_outputs = jax.jit(simulate_jax)(**sparse_arguments)
for label, arguments in (("dense matrices", dense_arguments), ("one dense matrix", mixed_arguments)):
    outputs = jax.jit(df.simulate)(**arguments)
    deviations = [rel(a, b) for a, b in zip(outputs, reference_outputs)]
    print(f"simulate with {label}: JAX path, deviation from simulate_jax with sparse matrices {max(deviations):.1e}")
    assert max(deviations) == 0.0
fast_outputs = jax.jit(df.simulate)(**sparse_arguments)
deviations = [rel(a, b) for a, b in zip(fast_outputs, reference_outputs)]
print("fast path against JAX (carrier, signal, noise): " + ", ".join(f"{d:.1e}" for d in deviations))
assert max(deviations) < 1e-6

# Matrices changed by hand: the fast path has to honour any constant entries, not only the ones
# the build step produces.
n, m = arrays["carrier_matrix"].shape[0], arrays["signal_matrix"].shape[0]
host = {key: np.asarray(arrays[key]) for key in ("carrier_matrix_indices", "signal_matrix_indices", "signal_carrier_placing_indices", "noise_system_matrix_indices")}
flat = lambda indices, size, width: set((np.asarray(indices[0]) % size * width + np.asarray(indices[1]) % width).tolist())
carrier, signal, noise = (arrays[key] for key in MATRICES)


def scaled(matrix, taken, width, factor, wanted=lambda row, column: row != column):
    """`matrix` with one constant entry scaled: the first one off the diagonal that no function writes."""
    indices, values = np.asarray(matrix.indices), np.asarray(matrix.values).copy()
    for position, (row, column) in enumerate(indices.T.tolist()):
        if row * width + column not in taken and wanted(row, column):
            values[position] *= factor
            return SparseMatrix(matrix.indices, jnp.asarray(values), matrix.shape), (row, column)
    raise AssertionError("no such entry")


written_signal = flat(host["signal_matrix_indices"], m, m + 1)
placing = flat(host["signal_carrier_placing_indices"], m, m + 1)
carrier_changed, where_carrier = scaled(carrier, flat(host["carrier_matrix_indices"], n, n + 1), n + 1, 0.9 - 0.05j)
signal_changed, where_signal = scaled(signal, written_signal | placing, m + 1, 0.85 + 0.1j)
# A constant value in a slot that is multiplied by the carrier: the reverse pass works on a plan
# without the constant entries and looks these values up. Both plans have to give the same values.
where_placing = divmod(min(placing), m + 1)
signal_changed = SparseMatrix(
    jnp.concatenate([signal_changed.indices, jnp.asarray(where_placing)[:, None]], axis=1),
    jnp.concatenate([signal_changed.values, jnp.asarray([0.02 - 0.01j])]),
    signal.shape,
)
A = dict(zip(fast.HOST_KEYS, map(np.asarray, fast._host_leaves(dict(arrays, signal_matrix=signal_changed)))))
complete, reduced = fast.SystemPlan(A, n), fast.SystemPlan(A, n, signal_base=False, noise_base=False)
placing_values = complete.signal_base[complete.placing_positions]
assert np.array_equal(placing_values, reduced.signal_base[reduced.placing_positions]) and np.any(placing_values == 0.02 - 0.01j)
assert reduced.size < complete.size
taken = flat(host["noise_system_matrix_indices"], m, m)
free = [i for i in range(m) if all((i * m + j) not in taken and (j * m + i) not in taken for j in (i, i + 1))][:1]
i = free[0]
noise_changed = SparseMatrix(
    jnp.asarray([[i, i, i + 1], [i, i + 1, i]]), jnp.asarray([0.3, 0.2 + 0.1j, 0.2 - 0.1j], dtype=jnp.complex128) * 1e-3, noise.shape
)
changed = dict(sparse_arguments, carrier_matrix=carrier_changed, signal_matrix=signal_changed, noise_matrix=noise_changed)
assert fast.supports(changed)
print(f"changed entries: carrier {where_carrier}, signal {where_signal}, new in placing slot {tuple(map(int, where_placing))}, noise around ({i}, {i})")
rng = np.random.default_rng(5)
outputs_jax = jax.jit(simulate_jax)(**changed)
weights = [jnp.asarray((rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape)) / (np.abs(x) + np.abs(x).max() * 1e-6)) for x in outputs_jax[:2]]
weights.append(jnp.asarray(rng.normal(size=outputs_jax[2].shape) / np.asarray(outputs_jax[2])))


def loss(sim):
    def function(p):
        outputs = sim(**dict(changed, optimized_parameters=p))
        return sum(jnp.real(jnp.sum(w * x)) for w, x in zip(weights, outputs)), outputs

    return jax.jit(jax.value_and_grad(function, has_aux=True))


(value, outputs), gradient = loss(df.simulate)(parameters)
(value_jax, outputs_jax), gradient_jax = loss(simulate_jax)(parameters)
deviations = [rel(a, b) for a, b in zip((*outputs, gradient), (*outputs_jax, gradient_jax))]
unchanged = [rel(a, b) for a, b in zip(outputs_jax, reference_outputs)]
print("changed matrices, fast path against JAX (carrier, signal, noise, gradient): " + ", ".join(f"{d:.1e}" for d in deviations))
print("effect of the changes on the JAX outputs (carrier, signal, noise): " + ", ".join(f"{d:.1e}" for d in unchanged))
assert max(deviations) < 1e-5 and min(unchanged) > 1e-6

# A derivative with respect to matrix values is not what the fast path differentiates: JAX path.
def by_matrix(sim):
    def function(scale):
        matrix = SparseMatrix(signal.indices, signal.values * scale, signal.shape)
        carrier_out, signal_out, noise_out = sim(**dict(sparse_arguments, signal_matrix=matrix))
        return jnp.sum(jnp.log(noise_out))

    return jax.jit(jax.grad(function))(jnp.asarray(1.0))


derivative, derivative_jax = by_matrix(df.simulate), by_matrix(simulate_jax)
print(f"derivative with respect to the signal matrix values: {float(derivative):+.6e} (dispatcher) and {float(derivative_jax):+.6e} (JAX)")
assert derivative == derivative_jax and np.isfinite(float(derivative))
print("OK")
