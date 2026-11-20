"""Compare the dispatched `df.simulate` against saved baseline arrays (all 101 frequencies, all 193 gradients).

Usage: python runs/check_fast.py runs/baseline.npz [--fallback]
  --fallback  additionally run the original JAX path (simulate_jax) on 3 frequencies with the
              new empty link arrays and compare it to the fast path there.
"""
import os
import sys
import json

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
from time import perf_counter as pc

import jax
import jax.numpy as jnp
import numpy as np

import differometor as df
from differometor.setups import constrain_inter_grid_cell_spaces
from differometor.simulate import simulate_jax

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]


def load(idx=None):
    data = os.path.join(ROOT, "examples", "data")
    fixture = json.load(open(os.path.join(data, "uifo_800_3000.json")))
    reference = json.load(open(os.path.join(data, "uifo_800_3000_test_data.json")))
    setup = df.Setup.from_data(fixture)
    pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
    initial = []
    for pair in pairs:
        members = pair if isinstance(pair[0], list) else [pair]
        name, prop = members[0]
        initial.append(fixture["nodes" if name in fixture["nodes"] else "edges"][name]["properties"][prop])
    freqs = np.asarray(reference["frequencies"])
    ref = np.asarray(reference["sensitivities"])
    if idx is not None:
        freqs, ref = freqs[idx], ref[idx]
    arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], jnp.asarray(freqs), pairs)
    params = jnp.asarray(initial, dtype=jnp.float64)
    arrays["optimized_parameters"] = params
    return arrays, jnp.asarray(ports), params, jnp.asarray(ref), pairs


def readout(outputs, ports, ref):
    carrier, signal, noise = outputs
    signals = df.signal_detector(carrier, signal)[ports]
    detected = signals.squeeze() if ports.shape[0] == 1 else signals[0] - signals[1]
    sens = (noise / jnp.abs(detected)).squeeze()
    return jnp.mean(jnp.log10(sens / ref)), (sens, df.power_detector(carrier))


def run(sim, arrays, ports, params, ref):
    def loss(p):
        out = sim(**dict(arrays, optimized_parameters=p))
        l, (sens, powers) = readout(out, ports, ref)
        return l, (out, sens, powers)

    (l, (out, sens, powers)), g = jax.jit(jax.value_and_grad(loss, has_aux=True))(params)
    return dict(carrier=out[0], signal=out[1], noise=out[2], sensitivity=sens, powers=powers, gradient=g, loss=l)


rel = lambda a, b: float(np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(np.asarray(b)).max())

if __name__ == "__main__":
    base = np.load(sys.argv[1])
    arrays, ports, params, ref, pairs = load()
    assert df.simulate is not simulate_jax
    t = pc()
    res = run(df.simulate, arrays, ports, params, ref)
    print(f"fast path value_and_grad (incl. trace/compile): {pc() - t:.3f} s")
    for key in ("carrier", "signal", "noise", "sensitivity", "powers", "gradient"):
        print(f"  {key:12s} max |diff| / max |baseline| = {rel(res[key], base[key]):.3e}")
    g, gb = np.asarray(res["gradient"]), base["gradient"]
    print(f"  gradient per-entry max rel = {np.max(np.abs(g - gb) / (np.abs(gb) + 1e-300)):.3e}; "
          f"worst index {int(np.argmax(np.abs(g - gb)))}")
    lerr = np.abs(np.log(np.asarray(res["sensitivity"])) - np.log(np.asarray(ref))).max()
    lerr_b = np.abs(np.log(base["sensitivity"]) - np.log(np.asarray(ref))).max()
    print(f"  max |log sens - log ref| fast {lerr:.3e}  baseline {lerr_b:.3e}  (limit 1e-5)")

    # Eager call (no jit) and plain jit (gradient-check path in benchmark.py) must agree.
    eager = df.simulate(**arrays)
    jitted = jax.jit(lambda p, a: df.simulate(**dict(a, optimized_parameters=p)))(params, arrays)
    print(f"  eager vs jit signal: {rel(eager[1], jitted[1]):.1e}")

    if "--fallback" in sys.argv:
        idx = np.array([0, 37, 100])
        a3, p3, par3, ref3, _ = load(idx)
        t = pc()
        truth = run(simulate_jax, a3, p3, par3, ref3)
        print(f"original JAX on 3 frequencies with empty link arrays: {pc() - t:.1f} s")
        fast3 = run(df.simulate, a3, p3, par3, ref3)
        for key in ("carrier", "signal", "noise", "sensitivity", "gradient"):
            print(f"  fast vs JAX (3 freqs) {key:12s} {rel(fast3[key], truth[key]):.3e}")
        print(f"  baseline sensitivity at those freqs vs JAX: {rel(base['sensitivity'][idx], truth['sensitivity']):.3e}")
        # Differentiating w.r.t. a non-optimized input must fall back to the JAX pullback.
        f = lambda v: readout(df.simulate(**dict(a3, signal_changing_values=v)), p3, ref3)[0]
        gf = jax.grad(f)(a3["signal_changing_values"])
        gj = jax.grad(lambda v: readout(simulate_jax(**dict(a3, signal_changing_values=v)), p3, ref3)[0])(a3["signal_changing_values"])
        print(f"  grad w.r.t. frequencies, dispatcher vs JAX: {rel(gf, gj):.1e}")
