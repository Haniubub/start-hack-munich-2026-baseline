"""Reverse pass of the fast path: the shortcuts against the plain evaluation.

`fast.backward` takes three shortcuts:

- it only visits the signal matrix entries whose cotangent somebody reads (`wanted`), and forms
  just a sum over the frequencies for entries that only feed the cotangent of the carrier,
- for a cotangent that is a multiple of the detector weights at every frequency (a sensitivity
  readout) it takes the adjoint from the forward pass instead of solving again,
- it reuses the factorizations and matrix values of the forward pass (`kept`).

The reference switches all three off: every written entry is visited with its full row, the
adjoint is solved for, and everything is rebuilt from the entries. The test runs on the host,
without JAX tracing, for the benchmark setup and two aLIGO setups, with cotangents of several
kinds: the multiple of the detector weights, one detector row only, random on every output, and
single outputs. (`runs/noel/test_adjoint.py` compares the complete gradient with the original JAX
code.)

Usage: python runs/philip/test_reverse.py
"""

import json
import os
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo))

import jax.numpy as jnp
import numpy as np

import differometor as df
from differometor import fast
from differometor.setups import aligo, constrain_inter_grid_cell_spaces

PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
data = repo / "examples" / "data"
rng = np.random.default_rng(5)
random = lambda *shape: rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def benchmark_setup():
    fixture = json.loads((data / "uifo_800_3000.json").read_text())
    reference = json.loads((data / "uifo_800_3000_test_data.json").read_text())
    setup = df.Setup.from_data(fixture)
    pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
    return setup, pairs, np.asarray(reference["frequencies"])[::4]


def gradient(entry_plan, plan, entries, tape, results, cotangents, kept, restrict, tolerance):
    """One reverse pass; `restrict` passes the entry plan's `wanted` and `varying` on."""
    saved, fast.PARALLEL_TOLERANCE = fast.PARALLEL_TOLERANCE, tolerance
    try:
        options = dict(wanted=entry_plan.signal.wanted, varying=entry_plan.signal.entry_varying) if restrict else {}
        entries_bar = fast.backward(plan, entries[0], entries[1], *results, *cotangents, kept=kept, **options)
    finally:
        fast.PARALLEL_TOLERANCE = saved
    return entry_plan.pullback(tape, *entries_bar, value_count=parameters.shape[0], compact=restrict)


worst = 0.0
aligo_frequencies = np.asarray(json.loads((data / "aligo_test_data.json").read_text())["frequencies"])[::10]
cases = [("benchmark", *benchmark_setup())] + [(f"aLIGO {kind}", *aligo(kind), aligo_frequencies) for kind in ("space_modulation", "frequency_modulation")]
for name, setup, pairs, frequencies in cases:
    arrays, *_ = df.run_build_step(setup, [("f", "frequency")], jnp.asarray(frequencies), pairs)
    parameters = np.zeros(int(np.asarray(arrays["optimized_value_indices"]).max()) + 1)
    parameters[np.asarray(arrays["optimized_value_indices"])] = np.asarray(arrays["parameters"])[0, np.asarray(arrays["optimized_parameter_indices"])].real
    assert fast.supports(dict(arrays, optimized_parameters=jnp.asarray(parameters))), name
    A = {key: arrays[key] for key in fast.HOST_KEYS}
    entry_plan, plan = fast.EntryPlan(A), fast.SystemPlan(A)
    entries, tape = entry_plan.entries(parameters)
    *results, kept = fast.forward(plan, *entries)
    carrier, signal, noise, Y, MU = results
    n, m, F = plan.n, plan.m, frequencies.size
    layout = plan.reverse_layout(entry_plan.signal.wanted, entry_plan.signal.entry_varying)
    print(
        f"{name}: {plan.update_positions.size} written signal entries ({int((~plan.update_live).sum())} overwritten), "
        f"{layout.count} read by the pullback, {layout.rows.size - layout.count} more with a full row, {layout.sum_rows.size} as sums"
    )

    weights = np.conj(plan.detector_weights(carrier))
    read = np.nonzero(weights)[0]
    scale = lambda x: x / np.abs(x).max()
    parallel = np.zeros((m, F), complex)
    parallel[read] = weights[read, None] * random(F)
    one_row = np.zeros((m, F), complex)
    one_row[read[0]] = random(F)
    zero_c, zero_s, zero_n = np.zeros(n, complex), np.zeros((m, F), complex), np.zeros(F)
    noise_bar = rng.standard_normal(F) / noise
    cotangents = {
        "multiple of the detector weights": (scale(random(n)), scale(parallel), noise_bar),
        "one detector row": (zero_c, scale(one_row), noise_bar),
        "random on every output": (scale(random(n)), scale(random(m, F) / (np.abs(signal) + np.abs(signal).max() * 1e-6)), noise_bar),
        "signal only, detector weights": (zero_c, scale(parallel), zero_n),
        "noise only": (zero_c, zero_s, noise_bar),
        "carrier only": (scale(random(n)), zero_s, zero_n),
    }
    for kind, bars in cotangents.items():
        reference = gradient(entry_plan, plan, entries, tape, results, bars, kept=None, restrict=False, tolerance=-1.0)
        deviations = []
        for kept_work in (kept, None):
            for restrict in (True, False):
                for tolerance in (fast.PARALLEL_TOLERANCE, -1.0):
                    result = gradient(entry_plan, plan, entries, tape, results, bars, kept_work, restrict, tolerance)
                    deviations.append(float(np.abs(result - reference).max() / np.abs(reference).max()))
        worst = max(worst, max(deviations))
        print(f"  {kind:34s} max deviation over 8 variants {max(deviations):.1e}   (|gradient| max {np.abs(reference).max():.2e})")

print("WORST", worst)
assert worst < 1e-9
