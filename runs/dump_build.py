"""Dump run_build_step arrays for the benchmark setup (to compare builds across commits)."""
import os, sys, json
os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["JAX_ENABLE_X64"] = "true"
import numpy as np, jax.numpy as jnp
import differometor as df
from differometor.setups import constrain_inter_grid_cell_spaces
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fx = json.load(open(f"{ROOT}/examples/data/uifo_800_3000.json")); ref = json.load(open(f"{ROOT}/examples/data/uifo_800_3000_test_data.json"))
setup = df.Setup.from_data(fx)
pairs = constrain_inter_grid_cell_spaces(setup.parameters, ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"])
arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], jnp.asarray(ref["frequencies"]), pairs)
np.savez(sys.argv[1], ports=np.asarray(ports), **{k: np.asarray(v) for k, v in arrays.items() if v is not None})
