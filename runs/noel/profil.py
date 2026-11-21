"""Profile the host callbacks of the fast path on the benchmark workload.

Usage: python runs/noel/profil.py <repo> [forward|backward|build]
"""
import cProfile
import json
import os
import pstats
import sys
from pathlib import Path
from time import perf_counter as pc

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
repo = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repo))
what = sys.argv[2] if len(sys.argv) > 2 else "forward"

import jax
import jax.numpy as jnp
import numpy as np
import differometor as df
from differometor import fast
from differometor.setups import constrain_inter_grid_cell_spaces

PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
data = repo / "examples" / "data"
fixture = json.loads((data / "uifo_800_3000.json").read_text())
reference = json.loads((data / "uifo_800_3000_test_data.json").read_text())


def build_inputs():
    setup = df.Setup.from_data(fixture)
    pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
    initial = []
    for pair in pairs:
        members = pair if isinstance(pair[0], list) else [pair]
        values = [fixture["nodes" if name in fixture["nodes"] else "edges"][name]["properties"][prop] for name, prop in members]
        initial.append(values[0])
    parameters = jnp.asarray(initial, dtype=jnp.float64)
    frequencies = jnp.asarray(reference["frequencies"], dtype=jnp.float64)
    arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], frequencies, pairs)
    arrays["optimized_parameters"] = parameters
    return arrays, ports, parameters


if what == "build":
    t = pc(); build_inputs(); print("first build", pc() - t)
    profile = cProfile.Profile(); profile.enable(); build_inputs(); profile.disable()
    pstats.Stats(profile).sort_stats("cumulative").print_stats(35)
    sys.exit()

arrays, ports, parameters = build_inputs()
host = [np.asarray(arrays[k]) if not isinstance(arrays[k], tuple) else np.asarray(arrays[k]) for k in fast.HOST_KEYS]
p = np.asarray(parameters)

for _ in range(2):
    t = pc(); out = fast._host_forward(p, *host); t_fwd = pc() - t
print("forward seconds", t_fwd)
carrier, signal, noise, *helpers = out
carrier_bar = np.zeros_like(carrier)
signal_bar = np.zeros_like(signal)
rng = np.random.default_rng(0)
d = np.nonzero(np.asarray(arrays["noise_selection_vectors"])[0, 0])[0]
signal_bar[d] = rng.normal(size=(d.size, signal.shape[1])) + 1j * rng.normal(size=(d.size, signal.shape[1]))
noise_bar = rng.normal(size=noise.shape)
backward_keys = getattr(fast, "BACKWARD_KEYS", fast.HOST_KEYS)  # older states pass every host array
backward_host = [h for k, h in zip(fast.HOST_KEYS, host) if k in backward_keys]
args = (p, *backward_host, carrier, signal, noise, *helpers, carrier_bar, signal_bar, noise_bar)
for _ in range(2):
    t = pc(); g = fast._host_backward(*args); t_bwd = pc() - t
print("backward seconds", t_bwd)

profile = cProfile.Profile()
profile.enable()
for _ in range(5):
    if what == "forward":
        fast._host_forward(p, *host)
    else:
        fast._host_backward(*args)
profile.disable()
stats = pstats.Stats(profile)
stats.sort_stats("cumulative").print_stats(38)
