"""Fast path against the original JAX code at perturbed design parameters (outputs and gradient).

The benchmark only evaluates the saved design. An optimizer visits other parameter values, where
the linear systems are scaled and conditioned differently. Trial 0 is the saved design, trials 1-5
move every design variable by a growing random amount (reflectivities stay below 1).

Usage: python runs/noel/test_perturbed.py <repo>
"""
import json, os, sys
os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["JAX_ENABLE_X64"] = "true"
sys.path.insert(0, sys.argv[1])
import jax, jax.numpy as jnp, numpy as np
import differometor as df
from differometor import fast
from differometor.setups import constrain_inter_grid_cell_spaces
from differometor.simulate import simulate_jax
PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
data = os.path.join(sys.argv[1], "examples", "data")
fixture = json.load(open(os.path.join(data, "uifo_800_3000.json"))); reference = json.load(open(os.path.join(data, "uifo_800_3000_test_data.json")))
setup = df.Setup.from_data(fixture)
pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
initial, kinds = [], []
for pair in pairs:
    members = pair if isinstance(pair[0], list) else [pair]
    name, prop = members[0]
    initial.append(fixture["nodes" if name in fixture["nodes"] else "edges"][name]["properties"][prop]); kinds.append(prop)
initial = np.asarray(initial, float); kinds = np.asarray(kinds)
index = np.array([0, 50, 100])
frequencies = jnp.asarray(np.asarray(reference["frequencies"])[index])
arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], frequencies, pairs)
ports = jnp.asarray(ports)
def make(sim):
    def loss(p):
        out = sim(**dict(arrays, optimized_parameters=p))
        carrier, signal, noise = out
        detected = df.signal_detector(carrier, signal)[ports].squeeze()
        sens = noise / jnp.abs(detected)
        return jnp.mean(jnp.log10(sens)), (out, sens)
    return jax.jit(jax.value_and_grad(loss, has_aux=True))
fast_fn, jax_fn = make(df.simulate), make(simulate_jax)
rel = lambda a, b: float(np.abs(np.asarray(a) - np.asarray(b)).max() / max(np.abs(np.asarray(b)).max(), 1e-300))
rng = np.random.default_rng(3)
print("kinds:", {k: int((kinds == k).sum()) for k in np.unique(kinds)})
worst = 0.0
for trial in range(6):
    p = initial.copy()
    if trial:
        strength = [0.0, 0.01, 0.05, 0.2, 0.5, 1.0][trial]
        u = rng.uniform(-1, 1, p.size)
        refl = kinds == "reflectivity"
        p[refl] = np.clip(p[refl] + strength * 0.5 * u[refl] * (1 - p[refl]) - strength * 0.3 * np.abs(u[refl]) * p[refl], 0.01, 0.999999)
        p[kinds == "tuning"] += strength * 30 * u[kinds == "tuning"]
        for k in ("length", "mass", "power"):
            p[kinds == k] *= 1 + 0.5 * strength * u[kinds == k]
        p[kinds == "db"] += strength * 3 * u[kinds == "db"]
        p[kinds == "angle"] += strength * 20 * u[kinds == "angle"]
    pj = jnp.asarray(p)
    (lf, (of, sf)), gf = fast_fn(pj); (lj, (oj, sj)), gj = jax_fn(pj)
    devs = [rel(of[0], oj[0]), rel(of[1], oj[1]), rel(of[2], oj[2]), rel(sf, sj), rel(gf, gj)]
    logdev = float(np.abs(np.log(np.asarray(sf)) - np.log(np.asarray(sj))).max())
    worst = max(worst, logdev)
    print(f"trial {trial}: loss {float(lf):+.6f}  carrier {devs[0]:.1e} signal {devs[1]:.1e} noise {devs[2]:.1e} sensitivity {devs[3]:.1e} gradient {devs[4]:.1e}  max |dlog sens| {logdev:.1e}  finite {bool(np.isfinite(np.asarray(gf)).all())}")
print("WORST log sensitivity deviation", worst)
assert worst < 1e-6
