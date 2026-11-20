"""Profile the benchmark's build_inputs phase in a fresh process (first run)."""
import os, sys, json, cProfile, pstats
from time import perf_counter as pc
os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["JAX_ENABLE_X64"] = "true"
t0 = pc()
import jax, jax.numpy as jnp, numpy as np
import differometor as df
from differometor.setups import constrain_inter_grid_cell_spaces
print(f"imports {pc()-t0:.3f}s")
jax.config.update("jax_enable_x64", True); jax.devices("cpu")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
fixture = json.load(open(f"{ROOT}/examples/data/uifo_800_3000.json")); reference = json.load(open(f"{ROOT}/examples/data/uifo_800_3000_test_data.json"))
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
    expected = jnp.asarray(reference["sensitivities"], dtype=jnp.float64)
    arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], frequencies, pairs)
    arrays["optimized_parameters"] = parameters
    return arrays, jnp.asarray(ports), parameters, frequencies, expected, pairs, jnp.asarray(1.0, dtype=jnp.float64)
pr = cProfile.Profile()
for i in range(3):
    t = pc()
    if i == 0: pr.enable()
    out = build_inputs(); jax.block_until_ready(out)
    if i == 0: pr.disable()
    print(f"build run {i}: {pc()-t:.3f}s")
st = pstats.Stats(pr); st.sort_stats("cumulative").print_stats(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
if "--callers" in sys.argv: st.print_callers("asarray")
