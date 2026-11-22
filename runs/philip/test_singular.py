"""A mirror with reflectivity exactly 1 or 0: only its own reflectivity may have a non-finite gradient.

    PYTHONPATH=. python runs/philip/test_singular.py      (prints one line per case, exits non-zero on failure)

sqrt(1 - R) and sqrt(R) have an infinite slope there. The dense JAX path (reverse mode) gives a
non-finite gradient for that reflectivity and finite gradients for everything else. The dual numbers
of the fast path used to turn the tuning of the same mirror into nan as well (0 * inf); `fast._chain`
keeps directions with zero tangent at zero. Found by another team (SudeepJoshi21, commit 61a6aa4).
"""
import json
import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
import jax
import jax.numpy as jnp
import numpy as np

import differometor as df
from differometor import fast
from differometor.setups import Setup, aligo
from differometor.simulate import simulate_jax

PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
LIMIT = 1e-8


def detected(outputs, ports):
    carrier, signal, noise = outputs
    signals = df.signal_detector(carrier, signal)[ports]
    return signals.squeeze() if ports.shape[0] == 1 else signals[0] - signals[1]


def gradients(node, value):
    setup, _ = aligo(mode="space_modulation")
    data = json.loads(json.dumps(setup.to_data()))
    data["nodes"][node]["properties"]["reflectivity"] = value
    setup = Setup.from_data(data)
    pairs = [list(pair) for pair in setup.parameters if pair[1] in PROPERTIES]
    frequencies = jnp.logspace(jnp.log10(20.0), jnp.log10(5000.0), 21)
    arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], frequencies, pairs)
    ports = jnp.asarray(ports)
    values = np.zeros(int(arrays["optimized_value_indices"].max()) + 1)
    values[np.asarray(arrays["optimized_value_indices"])] = np.asarray(arrays["parameters"][0, arrays["optimized_parameter_indices"]]).real
    p0 = jnp.asarray(values)
    arrays = {key: item for key, item in arrays.items() if key != "optimized_parameters"}
    on_fast = bool(fast.supports(dict(arrays, optimized_parameters=p0)))

    def run(simulate):
        def loss(p):
            out = simulate(**arrays, optimized_parameters=p)
            return jnp.mean(jnp.log10((out[2] / jnp.abs(detected(out, ports))).squeeze()))

        return np.asarray(jax.jit(jax.grad(loss))(p0))

    with np.errstate(divide="ignore", invalid="ignore"):
        return [".".join(pair) for pair in pairs], on_fast, run(df.simulate), run(simulate_jax)


failures = 0
for node, value in (("etmx", 1.0), ("srm", 0.0)):
    names, on_fast, ours, dense = gradients(node, value)
    bad_ours = [names[i] for i in np.flatnonzero(~np.isfinite(ours))]
    bad_dense = [names[i] for i in np.flatnonzero(~np.isfinite(dense))]
    both = np.isfinite(ours) & np.isfinite(dense)
    deviation = float(np.abs(ours[both] - dense[both]).max() / np.abs(dense[both]).max())
    ok = on_fast and bad_ours == bad_dense == [f"{node}.reflectivity"] and deviation <= LIMIT
    failures += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {node}.reflectivity = {value}: fast path {on_fast}, non-finite gradients fast {bad_ours}, dense {bad_dense}, "
          f"all others agree to {deviation:.1e}", flush=True)
print("\n1 or more failed" if failures else "\nall checks passed")
sys.exit(1 if failures else 0)
