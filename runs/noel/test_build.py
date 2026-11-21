"""Build step: the fast sideband expansion against a plain reference, and build outputs across commits.

    python runs/noel/test_build.py <repo>                    # `_sideband_layout` against the reference
    PYTHONHASHSEED=0 python runs/noel/test_build.py <repo> --dump out.npz
    python runs/noel/test_build.py <repo> --compare a.npz b.npz

The reference below is the expansion as the upstream `set_instructions` wrote it, with boolean
masks and one NumPy call per step. `_sideband_layout` has to return the same arrays and change its
arguments in the same way, also when rows and columns are one and the same array.

`--dump` writes the arrays of `run_build_step` for 27 setups (the benchmark setup, aLIGO, Voyager
and random UIFOs of three sizes) as dense arrays. Two dumps of the same code are identical only
with a fixed `PYTHONHASHSEED`, because the build iterates over a set.
"""
import json, os, sys
os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["JAX_ENABLE_X64"] = "true"
repo = sys.argv[1]
sys.path.insert(0, repo)
import numpy as np


def compare(first, second):
    a, b = np.load(first), np.load(second)
    assert sorted(a.files) == sorted(b.files), set(a.files) ^ set(b.files)
    different = [key for key in a.files if not (a[key].shape == b[key].shape and a[key].dtype == b[key].dtype and np.array_equal(a[key], b[key]))]
    for key in different:
        print("DIFFERENT", key, a[key].shape, b[key].shape, a[key].dtype, b[key].dtype)
    print(f"{len(a.files)} arrays compared, {len(different)} different")
    return not different


def dump(path):
    import jax.numpy as jnp
    import differometor as df
    from differometor.setups import constrain_inter_grid_cell_spaces, aligo, voyager, uifo

    fixture = json.load(open(f"{repo}/examples/data/uifo_800_3000.json")); reference = json.load(open(f"{repo}/examples/data/uifo_800_3000_test_data.json"))
    frequencies = json.load(open(f"{repo}/examples/data/aligo_test_data.json"))["frequencies"]
    properties = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
    out = {}

    def add(name, setup, pairs, values):
        arrays, *metadata = df.run_build_step(setup, [("f", "frequency")], jnp.asarray(values), pairs)
        for position, value in enumerate(metadata[:4]):
            out[f"{name}/metadata{position}"] = np.asarray(value)
        out[f"{name}/port_to_index"] = np.array(sorted(metadata[4].items()), dtype=object).astype(str)
        out.update({f"{name}/{key}": np.asarray(value) for key, value in arrays.items() if value is not None})

    setup = df.Setup.from_data(fixture)
    add("uifo_fixture", setup, constrain_inter_grid_cell_spaces(setup.parameters, properties), reference["frequencies"])
    add("uifo_fixture_plain_pairs", setup, [tuple(p) for p in setup.parameters if p[1] in properties][:40], reference["frequencies"][:5])
    add("uifo_fixture_no_pairs", setup, None, reference["frequencies"][:5])
    for mode in ("space_modulation", "frequency_modulation", "amplitude_modulation"):
        add(f"aligo_{mode}", *aligo(mode), frequencies)
        add(f"voyager_{mode}", *voyager(mode), frequencies)
        for size in (1, 2, 3):
            for seed in (0, 1):
                add(f"uifo_{size}_{seed}_{mode}", *uifo(size, random=True, mode=mode, random_seed=seed), frequencies[:7])
    np.savez(path, **out)
    print(len(out), "arrays from", len({key.split("/")[0] for key in out}), "setups")


def reference_layout(rows, columns, size, companions):
    only_row = (rows < size) & (columns >= size)
    only_column = (columns < size) & (rows >= size)
    both = (rows < size) & (columns < size)
    signal_block = (rows >= size) & (columns >= size)
    columns[signal_block] += size
    rows[signal_block] += size
    columns[only_row] += size
    rows[only_column] += size
    new_rows, new_columns = rows.copy(), columns.copy()
    new_companions = [None if companion is None else companion.copy() for companion in companions]
    for mask, row_shift, column_shift in ((only_row, size, 0), (only_column, 0, size), (both, size, size)):
        if mask.any():
            new_rows = np.concatenate([new_rows, rows[mask] + row_shift])
            new_columns = np.concatenate([new_columns, columns[mask] + column_shift])
            new_companions = [None if c is None else np.concatenate([n, c[mask]]) for n, c in zip(new_companions, companions)]
    return new_rows, new_columns, new_companions


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)


def layout_test(trials=20000):
    from differometor.build import _sideband_layout

    rng = np.random.default_rng(0)
    for trial in range(trials):
        size, count = int(rng.integers(1, 6)), int(rng.integers(0, 10))
        dtype = np.int32 if trial % 11 == 0 else np.int64
        rows = rng.integers(-1, 2 * size + 3, count).astype(dtype)
        columns = rows if trial % 7 == 0 else rng.integers(-1, 2 * size + 3, count).astype(dtype)
        companions = (rng.integers(0, 4, count), rng.integers(0, 3, count).astype(dtype) if trial % 2 else None)
        arguments = []
        for _ in range(2):  # separate copies for both implementations, keeping rows and columns aliased
            r = rows.copy()
            arguments.append((r, r if columns is rows else columns.copy(), size, tuple(None if c is None else c.copy() for c in companions)))
        expected, result = reference_layout(*arguments[0]), _sideband_layout(*arguments[1])
        assert same(expected[0], result[0]) and same(expected[1], result[1]), (trial, expected, result)
        assert all(same(a, b) for a, b in zip(expected[2], result[2])), (trial, expected, result)
        assert same(arguments[0][0], arguments[1][0]) and same(arguments[0][1], arguments[1][1]), (trial, "arguments after the call")
    print(f"_sideband_layout: {trials} random cases identical to the reference (results and changed arguments)")


if "--compare" in sys.argv:
    sys.exit(0 if compare(*sys.argv[sys.argv.index("--compare") + 1 :][:2]) else 1)
elif "--dump" in sys.argv:
    dump(sys.argv[sys.argv.index("--dump") + 1])
else:
    layout_test()
    print("OK")
