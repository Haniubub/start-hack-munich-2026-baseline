#!/usr/bin/env python3
"""The benchmark workload without JAX. NOT the scored benchmark.

Run: python benchmark_native.py --cpus 1 --output benchmark-native.json
     python benchmark_native.py --cpus 1 --setup aligo --output benchmark-native-aligo.json

What this is
    ``benchmark.py`` is the scored benchmark and stays untouched. It imports JAX itself and lets
    JAX trace and lower its readout, so a large part of its total is JAX, whatever the simulator
    does. This script runs the same workload (same public inputs, all frequencies, the full
    gradient, the same two correctness checks) and calls the simulator's host path directly:
    the build step, the sparse forward pass, the readout and the reverse pass, all in NumPy and
    the native LU kernel. It shows what the simulator itself costs.

What is the same as in ``benchmark.py``
    The inputs and their hashes, the build call (``df.Setup.from_data`` / ``aligo``,
    ``df.run_build_step``), the forward pass with the work a reverse pass needs, sensitivity,
    powers and the scalar loss, the reverse-mode gradient with respect to every design variable,
    the timer scope (imports, input I/O, build, forward, sensitivity, gradient) and the
    validation (reference sensitivities with the same tolerance, the two-step finite-difference
    check of the same gradient entry).

What is different
    JAX is not imported. There is nothing to trace, lower or compile, so those phases do not
    exist. The readout of ``benchmark.py`` (``readout`` and its pullback) is written out in NumPy
    below. ``differometor`` imports ``jax`` at module level for its JAX entry points; here a
    placeholder stands in for the module while the package is imported and refuses every call
    afterwards, so a code path that needs JAX fails loudly instead of running something else.

Only setups that the sparse host path covers (``differometor.fast.supports``) can run here.
"""
from time import perf_counter

IMPORT_START = perf_counter()
import os
import sys

os.environ["DIFFEROMETOR_TUNING"] = "off"  # the tuning measures patch JAX, which is not here
os.environ["DIFFEROMETOR_STARTUP"] = "none"  # the start-up measures prepare `import jax`


def _options(argv):
    """Read the command line by hand: `import argparse` would find the start-up carrier next to this file."""
    options = {"cpus": None, "repeats": 10, "output": None, "setup": "grid", "save_arrays": None}
    names = {"--cpus": "cpus", "--repeats": "repeats", "--output": "output", "--setup": "setup", "--save-arrays": "save_arrays"}
    usage = "usage: benchmark_native.py [--cpus N] [--repeats N] [--setup grid|aligo] [--output FILE.json] [--save-arrays FILE.npz]"
    index = 0
    while index < len(argv):
        item = argv[index]
        name, separator, value = item.partition("=")
        if item in ("-h", "--help"):
            print(__doc__)
            print(usage)
            raise SystemExit(0)
        if name not in names:
            raise SystemExit(f"{usage}\nunknown argument: {item}")
        if not separator:
            index += 1
            if index >= len(argv):
                raise SystemExit(f"{usage}\n{name} needs a value")
            value = argv[index]
        options[names[name]] = value
        index += 1
    for name in ("cpus", "repeats"):
        if options[name] is not None:
            try:
                options[name] = int(options[name])
            except ValueError:
                raise SystemExit(f"{usage}\n--{name} needs a whole number")
    if options["setup"] not in ("grid", "aligo"):
        raise SystemExit(f"{usage}\n--setup is grid or aligo")
    if options["repeats"] < 3:
        raise SystemExit("Use at least three timed repetitions.")  # as benchmark.py
    if options["output"] is None:
        options["output"] = "benchmark-native-aligo.json" if options["setup"] == "aligo" else "benchmark-native.json"
    if not options["output"].endswith(".json"):
        raise SystemExit("--output must use a .json filename.")
    if options["save_arrays"] is not None and not options["save_arrays"].endswith(".npz"):
        raise SystemExit("--save-arrays must use a .npz filename.")
    return options


OPTIONS = _options(sys.argv[1:])

# Same as benchmark.py: set the CPU affinity before numerical libraries start their workers.
if OPTIONS["cpus"] is not None:
    if not hasattr(os, "sched_getaffinity"):
        raise SystemExit("--cpus requires Linux CPU affinity (Linux or WSL).")
    allowed_cpus = sorted(os.sched_getaffinity(0))
    if not 1 <= OPTIONS["cpus"] <= len(allowed_cpus):
        raise SystemExit(f"--cpus must be between 1 and {len(allowed_cpus)}.")
    os.sched_setaffinity(0, allowed_cpus[: OPTIONS["cpus"]])
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(OPTIONS["cpus"])

import json
import types

import numpy as np


class _Placeholder:
    """Stands in for a JAX object that differometor only names while it is imported.

    Module-level code of the package decorates functions (`jax.custom_vjp`), registers classes
    (`jax.tree_util.register_pytree_node`) and sets options (`jax.config.update`). During the
    import these calls are accepted and do nothing. After `seal()` every call raises.
    """

    sealed = False

    def __init__(self, name):
        self.__dict__["_name"] = name

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _Placeholder(f"{self._name}.{name}")

    def __call__(self, *args, **kwargs):
        if _Placeholder.sealed:
            raise RuntimeError(f"{self._name} was called, but this benchmark runs without JAX")
        return _Placeholder(f"{self._name}(...)")

    def __repr__(self):
        return f"<placeholder for {self._name}>"


class _JaxPlaceholder(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _Placeholder(f"jax.{name}")


class _NumpyAsJaxNumpy(types.ModuleType):
    """`jax.numpy` for the host: the build step and the annotations of the package get NumPy."""

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(np, name)


def _install_placeholders():
    if "jax" in sys.modules or "jaxlib" in sys.modules:
        raise SystemExit("JAX is already imported in this process; benchmark_native.py has to start without it.")
    jax = _JaxPlaceholder("jax")
    jax.__path__ = []  # a package without files: `import jax.anything_else` fails
    jax.numpy = _NumpyAsJaxNumpy("jax.numpy")
    sys.modules["jax"] = jax
    sys.modules["jax.numpy"] = jax.numpy


_install_placeholders()
import differometor as df
from differometor import components, fast
from differometor.setups import constrain_inter_grid_cell_spaces

_Placeholder.sealed = True
IMPORT_SECONDS = perf_counter() - IMPORT_START

# Same constants as benchmark.py.
FIXTURE_SHA256 = "ff4ba353a2a03cdce7f3e61bad729c2e00888d842c28ba9cf699f95afb87e138"
REFERENCE_SHA256 = "726f2cfe6986cb991fa48d5a28127b9330c3dc7e69d5bd193c8748acef03adc3"
ALIGO_REFERENCE_SHA256 = "e3419da07888d0260800be08ba873dc22518de0b103582740cc447e0014c5cb4"
ALIGO_SETUP_SHA256 = "f77fa86eeae1771a9aa3a9305c490a7ef406df6a542c3e93de303d3070536390"
PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
LN10 = float(np.log(10.0))


def cpu_model():
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    import platform

    if platform.system() == "Darwin":
        import subprocess

        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    return platform.processor() or platform.machine()


def simulate(parameters, host_arrays, taped):
    """`df.simulate` on the host. With `taped`, the forward pass keeps what the reverse pass needs."""
    (carrier, signal, noise, Y, MU), work = fast._run_forward(parameters, host_arrays, taped=taped)
    return (carrier[:, None], signal, noise.reshape(-1)), ((carrier, signal, noise, Y, MU), work)


def readout(outputs, detector_ports, reference):
    """`readout` of benchmark.py in NumPy: loss, (sensitivity, powers) and what the pullback needs.

    benchmark.py:
        signals = df.signal_detector(carrier, signal)[detector_ports]
        detected = signals.squeeze() if detector_ports.shape[0] == 1 else signals[0] - signals[1]
        sensitivity = (noise / jnp.abs(detected)).squeeze()
        loss = jnp.mean(jnp.log10(sensitivity / reference))
        return loss, (sensitivity, df.power_detector(carrier))
    """
    carrier, signal, noise = outputs
    n = carrier.shape[0]
    rows, inside = components._port_rows(detector_ports, n)
    if not inside.all():
        raise ValueError("A detector port lies outside the carrier solution")
    field, upper, lower = carrier[rows], signal[rows], signal[n + rows]
    signals = np.conj(field) * upper + field * lower  # signal_detector, for the detector ports only
    detected = signals[0] if rows.shape[0] == 1 else signals[0] - signals[1]
    magnitude = np.abs(detected)
    sensitivity = noise / magnitude
    loss = np.mean(np.log10(sensitivity / reference))
    powers = components._power_host(carrier)
    return loss, (sensitivity, powers), (rows, field, upper, lower, detected, magnitude, sensitivity)


def readout_pullback(outputs, kept, cotangent):
    """Cotangents of (carrier, signal, noise) for the cotangent of the loss, in JAX's convention.

    JAX pairs complex cotangents without conjugation (see `_signal_host_pullback` in
    differometor/components.py): the pullback of `abs(z)` is `g * conj(z) / abs(z)`, that of
    `conj(c) * u` with respect to `c` is `conj(g * u)`. `fast._run_backward` expects exactly these.
    """
    carrier, signal, noise = outputs
    rows, field, upper, lower, detected, magnitude, sensitivity = kept
    n, count = carrier.shape[0], sensitivity.shape[0]
    sensitivity_bar = cotangent / (count * LN10 * sensitivity)  # mean(log10(s / reference))
    noise_bar = sensitivity_bar / magnitude  # s = noise / magnitude
    magnitude_bar = -sensitivity_bar * sensitivity / magnitude
    detected_bar = magnitude_bar * np.conj(detected) / magnitude  # magnitude = abs(detected)
    signals_bar = detected_bar[None, :] if rows.shape[0] == 1 else np.stack([detected_bar, -detected_bar])
    carrier_bar = np.zeros(carrier.shape, carrier.dtype)
    np.add.at(carrier_bar, rows, (np.conj(signals_bar * upper) + signals_bar * lower).sum(axis=1, keepdims=True))
    signal_bar = np.zeros(signal.shape, signal.dtype)
    np.add.at(signal_bar, rows, signals_bar * np.conj(field))
    np.add.at(signal_bar, n + rows, signals_bar * field)
    return carrier_bar, signal_bar, noise_bar


def main():
    is_aligo = OPTIONS["setup"] == "aligo"
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.dirname(os.path.abspath(df.__file__)) != os.path.join(here, "differometor"):
        raise SystemExit("differometor was not imported from the checkout next to this script.")
    phases = [{"key": "imports", "label": "Python / library imports (no JAX)", "category": "setup", "seconds": IMPORT_SECONDS}]

    def timed(key, label, category, function):
        start = perf_counter()
        result = function()
        seconds = perf_counter() - start
        phases.append({"key": key, "label": label, "category": category, "seconds": seconds})
        print(f"{label}: {seconds:.6f} s", flush=True)
        return result

    data_dir = os.path.join(here, "examples", "data")
    fixture_path = None if is_aligo else os.path.join(data_dir, "uifo_800_3000.json")
    reference_path = os.path.join(data_dir, "aligo_test_data.json" if is_aligo else "uifo_800_3000_test_data.json")
    reference_sha256 = ALIGO_REFERENCE_SHA256 if is_aligo else REFERENCE_SHA256
    sensitivity_tolerance = 1e-6 if is_aligo else 1e-5
    check_pair = ["sq1", "db"] if is_aligo else ["mb11", "tuning"]

    def read(path):
        with open(path, "rb") as handle:
            return handle.read()

    def load_inputs():
        fixture_bytes = read(fixture_path) if fixture_path else None
        reference_bytes = read(reference_path)
        return fixture_bytes, reference_bytes, json.loads(fixture_bytes) if fixture_bytes else None, json.loads(reference_bytes)

    fixture_bytes, reference_bytes, fixture, reference = timed("load", "Read and parse public inputs", "setup", load_inputs)
    # Integrity checks are validation, so they are outside measured phases (as in benchmark.py).
    import hashlib

    if hashlib.sha256(reference_bytes).hexdigest() != reference_sha256 or (
        fixture_bytes is not None and hashlib.sha256(fixture_bytes).hexdigest() != FIXTURE_SHA256
    ):
        raise SystemExit("Restore the unchanged public fixtures from the baseline commit.")

    def build_inputs():
        nonlocal fixture
        if is_aligo:
            from copy import deepcopy
            from differometor.setups import aligo

            setup, _ = aligo(mode="space_modulation")
            fixture = deepcopy(setup.to_data())
            pairs = [list(pair) for pair in setup.parameters if pair[1] in PROPERTIES]
        else:
            setup = df.Setup.from_data(fixture)
            pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
        initial = []
        for pair in pairs:
            members = pair if isinstance(pair[0], list) else [pair]
            values = [fixture["nodes" if name in fixture["nodes"] else "edges"][name]["properties"][prop] for name, prop in members]
            if any(value != values[0] for value in values):
                raise ValueError("Grouped length parameters must agree in the public fixture")
            initial.append(values[0])
        parameters = np.asarray(initial, dtype=np.float64)
        frequencies = np.asarray(reference["frequencies"], dtype=np.float64)
        expected = np.asarray(reference["sensitivities"], dtype=np.float64)
        arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], frequencies, pairs)
        arrays["optimized_parameters"] = parameters
        if not fast.supports(arrays):
            raise SystemExit("This setup is not covered by the sparse host path; it needs the JAX implementation.")
        host_arrays = tuple(arrays[key] for key in fast.HOST_KEYS)
        return host_arrays, np.asarray(ports), parameters, frequencies, expected, pairs

    host_arrays, ports, parameters, frequencies, expected, pairs = timed("build", "Build optical setup and prepare inputs", "setup", build_inputs)
    setup_sha256 = FIXTURE_SHA256
    if is_aligo:
        setup_sha256 = hashlib.sha256(json.dumps(fixture, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if setup_sha256 != ALIGO_SETUP_SHA256:
            raise SystemExit("Restore the public aLIGO optical setup; its physical input has changed.")
    print(f"Workload: {len(frequencies)} frequencies; {len(parameters)} physical design variables; no optimizer", flush=True)

    def one_pass(first):
        outputs, tape = timed("forward", "Forward simulation + AD residuals" if first else "Forward", "execute", lambda: simulate(parameters, host_arrays, taped=True))
        loss, (sensitivity, powers), kept = timed("sensitivity", "Sensitivity, powers and scalar loss" if first else "Sensitivity", "execute", lambda: readout(outputs, ports, expected))

        def reverse():
            carrier_bar, signal_bar, noise_bar = readout_pullback(outputs, kept, 1.0)
            return fast._run_backward(parameters, tape[0], tape[1], carrier_bar, signal_bar, noise_bar)

        gradient = timed("gradient", "Reverse-mode gradient", "execute", reverse)
        return outputs, sensitivity, powers, loss, gradient

    host = one_pass(first=True)
    first_phases = list(phases)
    total = sum(phase["seconds"] for phase in first_phases)
    print(f"Complete first-run total without JAX (excluding validation): {total:.6f} s", flush=True)

    repeat_samples = []
    for index in range(OPTIONS["repeats"]):
        phases.clear()
        host = one_pass(first=False)
        sample = {phase["key"]: phase["seconds"] for phase in phases}
        repeat_samples.append(sample)
        print(f"Repeat {index + 1:02d}/{OPTIONS['repeats']}: {sum(sample.values()):.6f} s", flush=True)

    # Validation only: no timers below contribute to the total. Same checks as benchmark.py.
    print("Validating outputs, reference sensitivity and finite-difference gradient...", flush=True)
    host_outputs, host_sensitivity, host_powers, host_loss, host_gradient = host
    for value in (*host_outputs, host_sensitivity, host_powers, host_loss, host_gradient):
        if not np.isfinite(value).all():
            raise ValueError("Non-finite physical output, sensitivity, loss or gradient")
    if host_sensitivity.shape != expected.shape or np.any(host_sensitivity <= 0):
        raise ValueError("Expected one positive sensitivity per frequency")
    log_error = np.abs(np.log(host_sensitivity) - np.log(expected))
    np.testing.assert_allclose(np.log(host_sensitivity), np.log(expected), atol=sensitivity_tolerance, rtol=0)
    if host_gradient.shape != parameters.shape or not np.any(host_gradient != 0):
        raise ValueError("Expected a finite, nonzero full parameter gradient")

    check_index = pairs.index(check_pair)

    def loss_only(p):
        return float(readout(simulate(p, host_arrays, taped=False)[0], ports, expected)[0])

    autodiff = float(host_gradient[check_index])
    finite_differences = []
    for check_step in (1e-3, 5e-4):
        shift = np.zeros_like(parameters)
        shift[check_index] = check_step
        estimate = (loss_only(parameters + shift) - loss_only(parameters - shift)) / (2 * check_step)
        np.testing.assert_allclose(autodiff, estimate, rtol=0.01, atol=1e-6)
        finite_differences.append({"step": check_step, "central_difference": estimate, "absolute_error": abs(autodiff - estimate)})
    np.testing.assert_allclose(finite_differences[0]["central_difference"], finite_differences[1]["central_difference"], rtol=0.01, atol=1e-6)
    if "jaxlib" in sys.modules or not isinstance(sys.modules.get("jax"), _JaxPlaceholder):
        raise RuntimeError("JAX was loaded during this run")

    import platform
    import statistics
    from datetime import datetime, timezone

    totals = [sum(sample.values()) for sample in repeat_samples]
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    report = {
        "benchmark": "differometor-native-aligo-not-scored" if is_aligo else "differometor-native-uifo-not-scored",
        "scored": False,
        "note": "Same workload and checks as benchmark.py, simulator host path called directly, JAX not imported. benchmark.py remains the scored benchmark.",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "harness_sha256": hashlib.sha256(read(os.path.abspath(__file__))).hexdigest(),
        "inputs": {
            "setup": "differometor.setups.aligo(mode='space_modulation')" if is_aligo else os.path.basename(fixture_path),
            "setup_sha256": setup_sha256,
            "reference": os.path.basename(reference_path),
            "reference_sha256": reference_sha256,
            "frequencies": len(frequencies),
            "design_variables": len(parameters),
        },
        "environment": {
            "cpu_model": cpu_model(),
            "cpu_affinity_count": len(affinity) if affinity is not None else None,
            "cpu_affinity_ids": affinity,
            "requested_logical_cpus": OPTIONS["cpus"],
            "logical_cpus_on_host": os.cpu_count(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "jax_imported": False,
            "native_lu": bool(fast.NATIVE_LU and fast._pattern_lu() is not None),
            "thread_limits": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        },
        "phases": first_phases,
        "repeat_samples": repeat_samples,
        "timing_seconds": {
            "end_to_end": total,
            "trace_lower_and_compile": 0.0,
            "runtime_samples": totals,
            "runtime_median": statistics.median(totals),
            "runtime_min": min(totals),
            "runtime_max": max(totals),
            "phase_runtime_medians": {key: statistics.median(sample[key] for sample in repeat_samples) for key in repeat_samples[0]},
        },
        "validation": {
            "all_outputs_and_gradients_finite": True,
            "upstream_sensitivity_check_passed": True,
            "sensitivity_log_atol": sensitivity_tolerance,
            "max_absolute_log_sensitivity_error": float(log_error.max()),
            "gradient_check": {"parameter": ".".join(check_pair), "autodiff": autodiff, "finite_differences": finite_differences, "rtol": 0.01, "atol": 1e-6, "passed": True},
            "gradient_shape": list(host_gradient.shape),
            "gradient_norm": float(np.linalg.norm(host_gradient)),
            "loss": float(host_loss),
            "output_shapes": {name: list(value.shape) for name, value in zip(("carrier", "signal", "noise"), host_outputs)},
        },
    }
    if OPTIONS["save_arrays"]:
        os.makedirs(os.path.dirname(os.path.abspath(OPTIONS["save_arrays"])), exist_ok=True)
        np.savez_compressed(
            OPTIONS["save_arrays"], carrier=host_outputs[0], signal=host_outputs[1], noise=host_outputs[2], sensitivity=host_sensitivity,
            powers=host_powers, gradient=host_gradient, parameters=parameters, frequencies=frequencies,
        )
    os.makedirs(os.path.dirname(os.path.abspath(OPTIONS["output"])), exist_ok=True)
    with open(OPTIONS["output"], "w") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"PASS (not scored, no JAX): reference sensitivity and gradient spot check; first run {total:.6f} s, full runtime median {statistics.median(totals):.6f} s")
    print(f"JSON: {OPTIONS['output']}")


if __name__ == "__main__":
    main()
