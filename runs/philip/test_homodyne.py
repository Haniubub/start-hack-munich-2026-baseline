"""Balanced homodyne detectors on the fast path, compared with the original JAX implementation.

    python runs/philip/test_homodyne.py     (prints one line per case and WORST, exits non-zero on failure)

Cases: the kit's saved homodyne setup `uifo_20_5000.json`, the same with other phases (then the
cotangent of the signal is not a multiple of the detector weights and the reverse pass takes its
general route), the phase as a design variable, other readouts, random UIFOs with a homodyne
readout, Voyager, the stateless callbacks and `vmap`.
"""
import copy
import json
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
os.chdir(root)
sys.path.insert(0, str(root))
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import differometor as df  # noqa: E402
from differometor import fast  # noqa: E402
from differometor._simulate_dicht import simulate_jax  # noqa: E402
from differometor.setups import Setup, constrain_inter_grid_cell_spaces, uifo, voyager  # noqa: E402

PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
FREQUENCIES = jnp.asarray(np.logspace(np.log10(20), np.log10(5000), 5))
LIMIT_LOG_SENSITIVITY, LIMIT_GRADIENT = 1e-6, 1e-5
failures, worst = [], [0.0, 0.0]


def built(setup, pairs, values=None):
    arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], FREQUENCIES, pairs)
    if values is None:  # the values the setup has now
        base = np.real(np.asarray(arrays["parameters"])[0])
        slots, places = np.asarray(arrays["optimized_parameter_indices"]), np.asarray(arrays["optimized_value_indices"])
        values = np.zeros(places.max() + 1)
        values[places] = base[slots]
    arrays["optimized_parameters"] = jnp.asarray(values, dtype=jnp.float64)
    return arrays, jnp.asarray(ports)


def readouts(ports):
    def sensitivity(outputs):  # the readout of benchmark.py
        carrier, signal, noise = outputs
        signals = df.signal_detector(carrier, signal)[ports]
        detected = signals.squeeze() if ports.shape[0] == 1 else signals[0] - signals[1]
        return (noise / jnp.abs(detected)).squeeze()

    return {
        "sensitivity": lambda outputs: jnp.mean(jnp.log10(sensitivity(outputs))),
        "one port": lambda outputs: jnp.mean(jnp.log10(outputs[2] / jnp.abs(df.signal_detector(outputs[0], outputs[1])[ports][0]))),
        "noise and powers": lambda outputs: jnp.sum(jnp.log(outputs[2])) + jnp.sum(df.power_detector(outputs[0])),
    }, sensitivity


def errors_against_exact(arrays):
    """
    For a badly conditioned setup: how far the signal at the detector rows is from the exact
    solution, for the fast path and for `simulate_jax` (first frequency, relative to the value
    there). The exact solution comes from iterative refinement with residuals summed in
    rational arithmetic.
    """
    from fractions import Fraction

    values = np.asarray(arrays["optimized_parameters"])
    host = {key: arrays[key] for key in fast.HOST_KEYS}
    entry_plan, plan = fast.EntryPlan(host), fast.SystemPlan(host)
    plan.selection = entry_plan.selection(values)
    carrier_entries, signal_entries, _ = entry_plan.values(values)
    carrier_lu, carrier_rhs = plan.carrier_matrix(carrier_entries)
    V, _ = plan.signal_values(signal_entries, carrier_lu.solve(carrier_rhs))
    m = plan.m
    S, b = np.zeros((m, m), complex), np.zeros(m, complex)
    S[plan.matrix_rows, plan.matrix_cols] = V[plan.in_matrix][:, 0]
    b[plan.rhs_rows] = V[~plan.in_matrix][:, 0]
    columns = [np.nonzero(S[i])[0] for i in range(m)]

    def residual(x):
        r = np.zeros(m, complex)
        for i in range(m):
            real, imag = Fraction(b[i].real), Fraction(b[i].imag)
            for j in columns[i]:
                ar, ai, xr, xi = Fraction(S[i, j].real), Fraction(S[i, j].imag), Fraction(x[j].real), Fraction(x[j].imag)
                real -= ar * xr - ai * xi
                imag -= ar * xi + ai * xr
            r[i] = complex(float(real), float(imag))
        return r

    exact = np.linalg.solve(S, b)
    for _ in range(8):
        correction = np.linalg.solve(S, residual(exact))
        exact = exact + correction
        if np.abs(correction).max() <= 1e-17 * np.abs(exact).max():
            break
    detector = np.nonzero(np.asarray(plan.selection))[0]
    error = lambda signal: float(max(abs(signal[i] - exact[i]) / abs(exact[i]) for i in detector))
    return error(np.asarray(df.simulate(**arrays)[1])[:, 0]), error(np.asarray(simulate_jax(**arrays)[1])[:, 0]), float(np.linalg.cond(S))


def compare(name, arrays, ports, simulate=df.simulate):
    if not fast.supports(arrays):
        print(f"FAIL {name}: not on the fast path")
        failures.append(name)
        return
    losses, sensitivity = readouts(ports)
    parameters = arrays["optimized_parameters"]
    run = lambda sim, p: sim(**dict(arrays, optimized_parameters=p))
    reference = np.asarray(sensitivity(run(simulate_jax, parameters)))
    ours = np.asarray(sensitivity(run(simulate, parameters)))
    usable = np.isfinite(reference).all() and (reference > 0).all()
    if not usable:
        # No light or no signal at the detector: there is no sensitivity, compare the outputs themselves.
        relative = lambda a, b: float(np.abs(np.asarray(a) - np.asarray(b)).max() / max(float(np.abs(np.asarray(b)).max()), 1e-300))
        differences = [relative(a, b) for a, b in zip(run(simulate, parameters), run(simulate_jax, parameters))]
        ok = max(differences) <= 1e-9
        print(f"{'ok  ' if ok else 'FAIL'} {name:58s} no signal at the detector in either path; carrier, signal, noise differ by {max(differences):.1e}")
        if not ok:
            failures.append(name)
        return
    log_error = float(np.abs(np.log(ours) - np.log(reference)).max())
    gradient_error = 0.0
    for loss in losses.values():
        expected = np.asarray(jax.grad(lambda p: loss(run(simulate_jax, p)))(parameters))
        got = np.asarray(jax.grad(lambda p: loss(run(simulate, p)))(parameters))
        scale = np.abs(expected).max()
        if np.isfinite(expected).all() and scale > 0:
            gradient_error = max(gradient_error, float(np.abs(got - expected).max() / scale))
    ok = log_error <= LIMIT_LOG_SENSITIVITY and gradient_error <= LIMIT_GRADIENT
    if not ok and simulate is df.simulate:
        # The two paths disagree. On a badly conditioned system the dense solve of the original is
        # the inaccurate one: accept the case if the fast path is the one close to the exact solution.
        ours_exact, jax_exact, condition = errors_against_exact(arrays)
        accepted = ours_exact <= 1e-7 and jax_exact > 10 * ours_exact
        print(f"{'ok  ' if accepted else 'FAIL'} {name:58s} |dln s| {log_error:8.1e}   gradient {gradient_error:8.1e}   badly conditioned ({condition:.0e}): "
              f"at the detector rows the fast path is {ours_exact:.1e} from the exact solution, simulate_jax {jax_exact:.1e}")
        if not accepted:
            failures.append(name)
        return
    print(f"{'ok  ' if ok else 'FAIL'} {name:58s} |dln s| {log_error:8.1e}   gradient {gradient_error:8.1e}   {parameters.shape[0]} variables, {ports.shape[0]} port(s)")
    if not ok:
        failures.append(name)
    else:
        worst[0], worst[1] = max(worst[0], log_error), max(worst[1], gradient_error)


def saved_setup():
    return json.loads((root / "examples" / "data" / "uifo_20_5000.json").read_text())


def design(data, extra=()):
    setup = Setup.from_data(data)
    pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES) + [list(pair) for pair in extra]
    section = lambda name: data["nodes" if name in data["nodes"] else "edges"]
    values = []
    for pair in pairs:
        name, prop = pair[0] if isinstance(pair[0], list) else pair
        values.append(float(section(name)[name]["properties"][prop]))
    return setup, pairs, values


data = saved_setup()
homodyne = next(name for name, node in data["nodes"].items() if node["component"] == "qhd")
compare("uifo_20_5000 (saved setup, phase 180)", *built(*design(data)))
for phase in (37.0, 90.0, -120.0):
    changed = copy.deepcopy(data)
    changed["nodes"][homodyne]["properties"]["phase"] = phase
    compare(f"uifo_20_5000 with phase {phase:g}", *built(*design(changed)))
changed = copy.deepcopy(data)
changed["nodes"][homodyne]["properties"]["phase"] = 153.0
compare("uifo_20_5000, phase 153 as a design variable", *built(*design(changed, extra=[(homodyne, "phase")])))

setup, pairs, values = design(data)
arrays, ports = built(setup, pairs, values)
fast.FORWARD_CACHE = False
try:
    compare("uifo_20_5000 with stateless callbacks", arrays, ports)
finally:
    fast.FORWARD_CACHE = True
losses, _ = readouts(ports)
tuning = next(index for index, pair in enumerate(pairs) if (pair[0] if isinstance(pair[0], list) else pair)[1] == "tuning")
batch = jnp.stack([arrays["optimized_parameters"], arrays["optimized_parameters"].at[tuning].add(0.5)])
batched = lambda sim: np.asarray(jax.vmap(jax.grad(lambda p: losses["sensitivity"](sim(**dict(arrays, optimized_parameters=p)))))(batch))
expected, got = batched(simulate_jax), batched(df.simulate)
error = float(np.abs(got - expected).max() / np.abs(expected).max())
print(f"{'ok  ' if error <= LIMIT_GRADIENT else 'FAIL'} {'uifo_20_5000 under vmap of grad':58s} gradient {error:8.1e}")
if error > LIMIT_GRADIENT:
    failures.append("vmap")
worst[1] = max(worst[1], error)

found = 0
for size, seed in ((2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1)):
    setup, pairs, centers, boundaries = uifo(size, random=True, random_seed=seed, verbose=True)
    if "balanced_homodyne" not in boundaries.values():
        continue
    found += 1
    compare(f"random UIFO {size}x{size}, seed {seed}, homodyne", *built(setup, constrain_inter_grid_cell_spaces(pairs, PROPERTIES)))
if not found:
    print("FAIL no random UIFO with a homodyne readout among the seeds")
    failures.append("random")

setup, pairs = voyager()
compare("Voyager, phases among the design variables", *built(setup, [pair for pair in pairs if pair[1] in PROPERTIES + ["phase"]]))

print(f"WORST |dln s| {worst[0]:.1e}, gradient {worst[1]:.1e}" if not failures else f"{len(failures)} failed: {failures}")
sys.exit(1 if failures else 0)
