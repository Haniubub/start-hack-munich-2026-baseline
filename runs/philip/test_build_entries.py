"""The build derives the signal and noise base matrices from the carrier entries, without dense arrays.

Before, the build filled a dense signal matrix (identity diagonal, carrier matrix without its right-hand
side in both sideband blocks) and an empty dense noise matrix and searched both for nonzero entries.
The entries must be exactly the same: indices, values, data types and shapes.
Usage: python runs/philip/test_build_entries.py [repo]
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
from differometor.setups import aligo, constrain_inter_grid_cell_spaces, voyager
from differometor.sparse import BaseMatrix

PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
data = os.path.join(ROOT, "examples", "data")
frequencies = jnp.asarray(json.load(open(os.path.join(data, "aligo_test_data.json")))["frequencies"][::10])


def fixture(name):
    setup = df.Setup.from_data(json.load(open(os.path.join(data, name))))
    return setup, constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)


def check(name, setup, pairs):
    arrays, *_ = df.run_build_step(setup, [("f", "frequency")], frequencies, pairs)
    carrier, signal, noise = (arrays[key] for key in ("carrier_matrix", "signal_matrix", "noise_matrix"))
    assert all(isinstance(matrix, BaseMatrix) for matrix in (carrier, signal, noise)), name

    dense_carrier = np.asarray(carrier)
    system_size = dense_carrier.shape[0]
    full_size = signal.shape[0]
    signal_size = full_size - 2 * system_size
    assert dense_carrier.shape == (system_size, system_size + 1) and signal_size >= 1, name

    # The dense construction the build used before.
    dense_signal = np.zeros((full_size, full_size + 1), dtype=complex)
    np.fill_diagonal(dense_signal, 1.0)
    dense_signal[:system_size, :system_size] = dense_carrier[:, :system_size]
    dense_signal[system_size:-signal_size, system_size:-signal_size - 1] = dense_carrier[:, :system_size]
    before = BaseMatrix.from_dense(dense_signal)
    empty = BaseMatrix.from_dense(np.zeros((full_size, full_size)))

    for label, new, old in (("signal", signal, before), ("noise", noise, empty)):
        new_index, old_index = np.asarray(new.index), np.asarray(old.index)
        new_values, old_values = np.asarray(new.values), np.asarray(old.values)
        assert new.shape == old.shape, (name, label, new.shape, old.shape)
        assert new_index.dtype == old_index.dtype and new_values.dtype == old_values.dtype, (name, label)
        assert np.array_equal(new_index, old_index), (name, label, "index")
        assert np.array_equal(new_values, old_values), (name, label, "values")
        assert np.all(np.diff(new_index) > 0), (name, label, "order")
    assert np.array_equal(np.asarray(signal), dense_signal), name
    print(f"{name:36s} carrier {np.asarray(carrier.index).size:5d} entries, signal {np.asarray(signal.index).size:5d} of {full_size * (full_size + 1)}, noise 0: identical")


check("benchmark (uifo_800_3000)", *fixture("uifo_800_3000.json"))
check("uifo_20_5000", *fixture("uifo_20_5000.json"))
for mode in ("space_modulation", "frequency_modulation", "amplitude_modulation"):
    check(f"aLIGO {mode}", *aligo(mode))
check("Voyager", *voyager())
print("OK")
