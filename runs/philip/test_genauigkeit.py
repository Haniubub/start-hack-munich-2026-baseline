"""Elementweise Genauigkeit des schnellen Wegs gegen den dichten JAX-Weg auf anderen Setups.

    PYTHONPATH=. python runs/philip/test_genauigkeit.py [name ...]

benchmark.py verlangt |ln(sens) - ln(referenz)| <= 1e-5 je Frequenz. `runs/check_setups.py` misst
dagegen normweise (max-Abweichung / max-Betrag) und sieht deshalb Fehler in kleinen Einträgen nicht.
Hier wird je Frequenz in ln gemessen: Rauschen, Signal am Detektor, Empfindlichkeit; dazu der
Gradient von mean(log10(sens)) (relativ zum grössten Eintrag und elementweise).
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
from differometor.setups import Setup, aligo, constrain_inter_grid_cell_spaces, voyager
from differometor.simulate import simulate_jax

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "examples", "data")
PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
LIMIT = 1e-5


def detected(outputs, ports):
    carrier, signal, noise = outputs
    signals = df.signal_detector(carrier, signal)[ports]
    return signals.squeeze() if ports.shape[0] == 1 else signals[0] - signals[1]


def uifo_fixture(name, frequencies):
    fixture = json.load(open(os.path.join(DATA, name)))
    setup = Setup.from_data(fixture)
    return setup, constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES), frequencies


def cases():
    public = jnp.asarray(json.load(open(os.path.join(DATA, "uifo_800_3000_test_data.json")))["frequencies"])
    wide = jnp.logspace(jnp.log10(20.0), jnp.log10(5000.0), 100)
    return {
        "uifo_800_3000": lambda: uifo_fixture("uifo_800_3000.json", public),
        "uifo_20_5000": lambda: uifo_fixture("uifo_20_5000.json", wide),
        "voyager_space": lambda: (*voyager("space_modulation"), wide),
        "voyager_space_800": lambda: (*voyager("space_modulation"), public),
        "aligo_space": lambda: (*aligo("space_modulation"), wide),
    }


def check(name, setup, pairs, frequencies):
    arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], frequencies, pairs)
    ports = jnp.asarray(ports)
    values = np.zeros(int(arrays["optimized_value_indices"].max()) + 1)
    values[np.asarray(arrays["optimized_value_indices"])] = np.asarray(arrays["parameters"][0, arrays["optimized_parameter_indices"]]).real
    p0 = jnp.asarray(values)
    arrays = {k: v for k, v in arrays.items() if k != "optimized_parameters"}
    on_fast = bool(fast.supports(dict(arrays, optimized_parameters=p0)))

    def run(sim):
        def f(p):
            out = sim(**arrays, optimized_parameters=p)
            sens = (out[2] / jnp.abs(detected(out, ports))).squeeze()
            return jnp.mean(jnp.log10(sens)), (out[2].squeeze(), jnp.abs(detected(out, ports)).squeeze(), sens)
        return jax.jit(jax.value_and_grad(f, has_aux=True))(p0)

    (_, (noise, sig, sens)), grad = run(df.simulate)
    (_, (noise_j, sig_j, sens_j)), grad_j = run(simulate_jax)
    ln = lambda a, b: np.abs(np.log(np.asarray(a)) - np.log(np.asarray(b)))
    e_noise, e_sig, e_sens = ln(noise, noise_j), ln(sig, sig_j), ln(sens, sens_j)
    grad, grad_j = np.asarray(grad), np.asarray(grad_j)
    g_norm = np.abs(grad - grad_j).max() / np.abs(grad_j).max()
    worst = int(e_sens.argmax())
    verdict = "ok" if e_sens.max() <= LIMIT else "UEBER LIMIT"
    print(f"{name}: weg={'schnell' if on_fast else 'jax'}, {p0.shape[0]} Designwerte, {len(frequencies)} Frequenzen")
    print(f"  ln(rauschen)  max {e_noise.max():.2e}  median {np.median(e_noise):.2e}")
    print(f"  ln(signal)    max {e_sig.max():.2e}  median {np.median(e_sig):.2e}")
    print(f"  ln(sens)      max {e_sens.max():.2e}  median {np.median(e_sens):.2e}  bei f={float(frequencies[worst]):.1f} Hz  -> {verdict} (Limit {LIMIT:g})")
    print(f"  gradient      max-Abw./max-Betrag {g_norm:.2e}", flush=True)
    return {"name": name, "fast": on_fast, "noise": float(e_noise.max()), "signal": float(e_sig.max()), "sens": float(e_sens.max()), "gradient": float(g_norm)}


if __name__ == "__main__":
    table = cases()
    names = sys.argv[1:] or list(table)
    results = [check(name, *table[name]()) for name in names]
    if os.environ.get("GENAUIGKEIT_JSON"):
        json.dump(results, open(os.environ["GENAUIGKEIT_JSON"], "w"), indent=1)
    sys.exit(1 if any(r["fast"] and r["sens"] > LIMIT for r in results) else 0)
