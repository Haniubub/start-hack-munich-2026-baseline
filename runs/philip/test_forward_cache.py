"""The forward cache of the fast path must never change a result.

Compares gradients from pullbacks that find their forward pass in the cache with pullbacks whose
entry has been evicted and with `fast.FORWARD_CACHE = False` (the stateless path).
Usage: python runs/philip/test_forward_cache.py [repo]
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
arrays, *_ = df.run_build_step(setup, [("f", "frequency")], jnp.asarray(reference["frequencies"][::10]), pairs)

simulate = lambda p: df.simulate(**dict(arrays, optimized_parameters=p))
index = pairs.index(["mb11", "tuning"])  # the parameter benchmark.py perturbs for its own check
points = [parameters, parameters.at[index].add(1e-3), parameters.at[index].add(-5e-4)]
cotangents = None


def gradients(use_pullbacks_late):
    """Gradients at three points; with `use_pullbacks_late` all forwards run before any pullback."""
    global cotangents
    results, pullbacks = [], []
    for p in points:
        outputs, pullback = jax.vjp(simulate, p)
        if cotangents is None:
            rng = np.random.default_rng(0)
            cotangents = tuple(jnp.asarray(rng.standard_normal(o.shape) + (1j * rng.standard_normal(o.shape) if np.iscomplexobj(o) else 0)) for o in outputs)
        if use_pullbacks_late:
            pullbacks.append(pullback)
        else:
            results.append(np.asarray(pullback(cotangents)[0]))
    for pullback in pullbacks:
        results.append(np.asarray(pullback(cotangents)[0]))
        results.append(np.asarray(pullback(cotangents)[0]))  # a pullback may be used more than once
    return results


fast.FORWARD_CACHE = False
stateless = gradients(False)
assert not fast._forward_cache
fast.FORWARD_CACHE = True
hits = gradients(False)
assert len(fast._forward_cache) == fast._FORWARD_CACHE_SIZE
late = gradients(True)  # the first point's entry is evicted before its pullback runs
for i, expected in enumerate(stateless):
    assert np.array_equal(hits[i], expected), f"cache hit differs at point {i}"
    assert np.array_equal(late[2 * i], expected) and np.array_equal(late[2 * i + 1], expected), f"late pullback differs at point {i}"
jitted = jax.jit(lambda p: jax.vjp(simulate, p)[1](cotangents)[0])
assert np.array_equal(np.asarray(jitted(points[1])), stateless[1])
print("forward cache: gradients identical with cache hits, evicted entries, repeated pullbacks and under jit")
