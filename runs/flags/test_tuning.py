"""Checks for differometor/_tuning.py: every measure must leave results and JAX's behaviour alone.

Usage: python runs/flags/test_tuning.py     (prints one line per check, exits non-zero on failure)
"""
import os
import sys
import threading
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.pop("DIFFEROMETOR_TUNING", None)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np

import differometor  # noqa: F401  (applies the tuning)
from differometor import _tuning as tuning

failures = []


def check(name, condition, detail=""):
    print(("ok   " if condition else "FAIL ") + name + (f"  [{detail}]" if detail and not condition else ""))
    if not condition:
        failures.append(name)


class Spy:
    """Counts and records calls of one of the original JAX functions the tuning module wraps."""

    def __init__(self, attribute):
        self.attribute, self.original, self.calls = attribute, getattr(tuning, attribute), []

    def __enter__(self):
        def recorder(*args, **keywords):
            if self.attribute == "_jax_compile":
                options = args[3]
                self.calls.append((dict(options.env_option_overrides), options.executable_build_options.debug_options.xla_backend_optimization_level,
                                   options.executable_build_options.debug_options.xla_disable_hlo_passes))
            else:
                self.calls.append(None)
            return self.original(*args, **keywords)
        setattr(tuning, self.attribute, recorder)
        return self

    def __exit__(self, *exc):
        setattr(tuning, self.attribute, self.original)


def describe(a):
    return (type(a), str(a.dtype), a.shape, a.weak_type, a.committed, str(a.sharding), str(a.aval), np.asarray(a).tobytes())


check("all measures installed", all(tuning.STATUS.get(k) == "on" for k in ("compile", "mlir_context", "asarray", "lazy_mhlo", "inline_jits_under_ad", "lowering_in_place")), str(tuning.STATUS))

# ---- jnp.asarray --------------------------------------------------------------------------------
original_asarray = tuning._jax_asarray
fast_cases = [([0.1 * i for i in range(193)], jnp.float64), ([1, 2.5, 3], jnp.float64), ((1.5, 2.5), jnp.float64), ([], jnp.float64),
              (1.0, jnp.float64), ([1.5, 2.5], jnp.float32), (2.5, np.float32), ([1.0, 2.0], "float64"), ([1.0, 2.0], float),
              ([1e400, -1e400, float("nan")], jnp.float64), ([2**70, 1.0], jnp.float64)]
with Spy("_jax_compile") as compiles, Spy("_jax_asarray") as passed_on:
    ours = [describe(jnp.asarray(value, dtype=dtype)) for value, dtype in fast_cases]
check("asarray: Python numbers with float dtype compile nothing", len(compiles.calls) == 0 and len(passed_on.calls) == 0, f"{len(compiles.calls)} compiles, {len(passed_on.calls)} passed on")
theirs = [describe(original_asarray(value, dtype=dtype)) for value, dtype in fast_cases]
check("asarray: same type, dtype, shape, weak type, commitment, device and bytes as JAX", ours == theirs,
      str([i for i, (a, b) in enumerate(zip(ours, theirs)) if a != b]))
other_cases = [([[1.0, 2.0], [3.0, 4.0]], jnp.float64), (np.arange(3.0), jnp.float64), ([1, 2], jnp.int32), ([1.0, 2.0], None),
               ([1 + 2j], jnp.complex128), ([True, 1.0], jnp.float64), (3, jnp.float64), ([np.float64(1.0)], jnp.float64), ([1.0, 2.0], jnp.complex128)]
with Spy("_jax_asarray") as passed_on:
    ours = [describe(jnp.asarray(value, dtype=dtype)) for value, dtype in other_cases]
theirs = [describe(original_asarray(value, dtype=dtype)) for value, dtype in other_cases]
check("asarray: every other input goes to JAX unchanged", len(passed_on.calls) == len(other_cases) and ours == theirs)
with Spy("_jax_asarray") as passed_on:
    jnp.asarray([1.0], dtype=jnp.float64, copy=True); jnp.asarray([1.0], dtype=jnp.float64, order="K"); jnp.asarray([1.0], dtype=jnp.float64, device=jax.devices("cpu")[0])
check("asarray: copy, order and device arguments go to JAX unchanged", len(passed_on.calls) == 3)
for bad in ([1.0, None], [1.0, "x"], [[1.0], [1.0, 2.0]]):
    errors = []
    for function in (jnp.asarray, original_asarray):
        try:
            function(bad, dtype=jnp.float64); errors.append(None)
        except Exception as error:
            errors.append((type(error), str(error)))
    check(f"asarray: same error for {bad!r}", errors[0] == errors[1] and errors[0] is not None, str(errors))
traced = jax.jit(lambda x: x + jnp.asarray([1.0, 2.0], dtype=jnp.float64))(jnp.zeros(2))
inside = []
jax.make_jaxpr(lambda x: inside.append(type(jnp.asarray([1.0, 2.0], dtype=jnp.float64))) or x)(1.0)
check("asarray: inside a traced function it stays a traced operation", np.array_equal(traced, [1.0, 2.0]) and issubclass(inside[0], jax.core.Tracer), str(inside))
with Spy("_jax_asarray") as passed_on:
    with jax.transfer_guard("log"):
        jnp.asarray([1.0], dtype=jnp.float64)
    with jax.debug_nans(True):
        jnp.asarray([1.0], dtype=jnp.float64)
check("asarray: transfer guard and debug_nans go to JAX unchanged", len(passed_on.calls) == 2)
tuning.FAST_ASARRAY = False
with Spy("_jax_asarray") as passed_on:
    jnp.asarray([1.0, 2.0], dtype=jnp.float64)
tuning.FAST_ASARRAY = True
check("asarray: switch off -> JAX", len(passed_on.calls) == 1)

# ---- compile options ----------------------------------------------------------------------------
def host_identity(x):
    return jax.pure_callback(lambda v: np.asarray(v), jax.ShapeDtypeStruct(x.shape, x.dtype), x)


def readout(z, noise):
    return jnp.mean(jnp.log10(noise / jnp.abs(host_identity(z)))), jnp.abs(z) * noise


def plain(z, noise):
    return jnp.mean(jnp.log10(noise / jnp.abs(z))), jnp.abs(z) * noise


def inputs(n, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(size=n) + 1j * rng.normal(size=n)), jnp.asarray(rng.uniform(1.0, 2.0, n))


def compiled_with(function, n, **jit_options):
    with Spy("_jax_compile") as spy:
        result = jax.jit(function, **jit_options)(*inputs(n))
    jax.block_until_ready(result)
    return spy.calls[-1], result


(overrides, level, passes), tuned = compiled_with(readout, 101)
check("compile: small program with a host callback gets classic emitter, one module, level 0, plain sums",
      overrides == {"xla_cpu_use_fusion_emitters": False, "xla_cpu_parallel_codegen_split_count": 1} and level == 0
      and passes == ("tree_reduction_rewriter" if tuning.PLAIN_SMALL_REDUCTIONS else ""), f"{overrides} {level} {passes!r}")
(overrides, level, _), _ = compiled_with(plain, 101)
check("compile: small program without host callback keeps LLVM optimization", len(overrides) == 2 and level == 3, f"{overrides} {level}")
(overrides, level, _), _ = compiled_with(readout, 5000)
check("compile: host callback but larger arrays keeps LLVM optimization", len(overrides) == 2 and level == 3, f"{overrides} {level}")


def long_program(x):
    for i in range(tuning.SMALL_PROGRAM_OPERATIONS):
        x = jnp.sin(x) + i
    return x


with Spy("_jax_compile") as spy:
    jax.jit(long_program)(jnp.ones(3))
check("compile: a program with many operations is compiled exactly as JAX asks", spy.calls[-1][0] == {} and spy.calls[-1][1] == 3, str(spy.calls[-1]))
(overrides, level, _), _ = compiled_with(readout, 101, compiler_options={"xla_cpu_use_fusion_emitters": True, "xla_backend_optimization_level": 2})
check("compile: jit(compiler_options=...) of the caller wins", overrides.get("xla_cpu_use_fusion_emitters") is True and overrides.get("xla_backend_optimization_level") == 2 and level == 3, f"{overrides} {level}")
tuning.PLAIN_SMALL_REDUCTIONS = True
(_, _, passes), _ = compiled_with(lambda z, noise: readout(z, noise * 1.0), 101)
tuning.PLAIN_SMALL_REDUCTIONS = False
check("compile: PLAIN_SMALL_REDUCTIONS disables the tree reduction pass", passes == "tree_reduction_rewriter", repr(passes))
saved = (tuning.CLASSIC_KERNEL_EMITTER, tuning.ONE_CODEGEN_MODULE, tuning.UNOPTIMIZED_SMALL_PROGRAMS)
tuning.CLASSIC_KERNEL_EMITTER = tuning.ONE_CODEGEN_MODULE = tuning.UNOPTIMIZED_SMALL_PROGRAMS = False
(overrides, level, passes), stock = compiled_with(lambda z, noise: readout(z, noise), 101)
tuning.CLASSIC_KERNEL_EMITTER, tuning.ONE_CODEGEN_MODULE, tuning.UNOPTIMIZED_SMALL_PROGRAMS = saved
check("compile: switches off -> options untouched", overrides == {} and level == 3 and passes == "", f"{overrides} {level}")
difference = max(float(np.max(np.abs(np.asarray(a) - np.asarray(b)) / np.abs(np.asarray(b)))) for a, b in zip(tuned, stock))
check("compile: tuned and stock program agree to rounding", difference < 1e-14, f"{difference:.2e}")

# a jaxlib that does not know an option: the compile is repeated with JAX's own options
four = jnp.arange(4.0)
real_name = tuning._EMITTER_OPTION
tuning._EMITTER_OPTION = "xla_cpu_option_that_does_not_exist"
with Spy("_jax_compile") as spy:
    value = jax.jit(lambda x: jnp.cos(host_identity(x)) * 3)(four)
check("fallback: unknown option -> same program compiled once more with JAX's own options", np.allclose(value, np.cos(np.arange(4.0)) * 3) and len(spy.calls) == 2
      and "xla_cpu_option_that_does_not_exist" in spy.calls[0][0] and spy.calls[1][0] == {} and spy.calls[1][1] == 3, str(spy.calls))
with Spy("_jax_compile") as spy:
    jax.jit(lambda x: jnp.cos(host_identity(x)) * 4)(four)
check("fallback: the unknown option is not tried again", len(spy.calls) == 1 and spy.calls[0][0] == {"xla_cpu_parallel_codegen_split_count": 1} and spy.calls[0][1] == 0, str(spy.calls))
tuning._EMITTER_OPTION = real_name
tuning._rejected_options.clear()
try:
    jax.jit(lambda x: jax.pure_callback(lambda v: np.zeros(7), jax.ShapeDtypeStruct((3,), x.dtype), x))(jnp.arange(3.0)).block_until_ready()
    raised = False
except Exception:
    raised = True
check("a wrong program still raises its own error", raised)

# ---- MLIR context -------------------------------------------------------------------------------
tuning._retire_shared_context()
with Spy("_jax_make_ir_context") as contexts:
    lowered = [jax.jit(lambda x, k=k: jnp.tanh(x) * k).lower(jnp.arange(3.0)) for k in range(1, 6)]
check("mlir: five lowerings share one context", len(contexts.calls) == 1, str(len(contexts.calls)))
modules = [low.compiler_ir("stablehlo") for low in lowered]
check("mlir: the modules are separate objects in the same context", len({id(m) for m in modules}) == 5 and all(m.context == modules[0].context for m in modules))
values = [np.asarray(low.compile()(jnp.arange(3.0))) for low in reversed(lowered)]
check("mlir: modules of one context compile independently, in any order", all(np.allclose(v, np.tanh(np.arange(3.0)) * k) for v, k in zip(values, range(5, 0, -1))))
tuning._retire_shared_context()
with Spy("_jax_make_ir_context") as contexts:
    for k in range(tuning.MLIR_CONTEXT_REUSES + 1):
        jax.jit(lambda x, k=k: x * k).lower(jnp.arange(2.0))
check("mlir: the context is replaced after MLIR_CONTEXT_REUSES lowerings", len(contexts.calls) == 2, str(len(contexts.calls)))
constant, many, two = np.arange(100000.0), jnp.ones(100000), jnp.ones(2)
tuning._retire_shared_context()
with Spy("_jax_make_ir_context") as contexts:
    jax.jit(lambda x: x * 3)(two)
    jax.jit(lambda x: x + constant)(many)  # large constant: compiled in the shared context, which is then retired
    jax.jit(lambda x: x * 2)(two)
    jax.jit(lambda x: x * 4)(two)
check("mlir: a program with a large constant retires the context", len(contexts.calls) == 2, str(len(contexts.calls)))
try:
    jax.jit(lambda x: jnp.reshape(x, (7,))).lower(jnp.arange(3.0))
except Exception:
    pass
try:
    jax.jit(lambda x: 1 / 0).lower(jnp.arange(3.0))
except ZeroDivisionError:
    pass
check("mlir: a failed lowering leaves the shared context usable", np.allclose(jax.jit(lambda x: x * 5)(jnp.arange(3.0)), np.arange(3.0) * 5))


def worker(index, results):
    def function(x):
        return jnp.sum(jnp.sin(host_identity(x)) * index)
    values = [float(jax.jit(function)(jnp.arange(4.0) + i)) for i in range(3)]
    gradient = np.asarray(jax.grad(lambda x: jnp.sum(jnp.sin(x) * index))(jnp.arange(4.0)))
    results[index] = (values, gradient, tuning._contexts.context)


results = {}
threads = [threading.Thread(target=worker, args=(i, results)) for i in range(1, 5)]
[t.start() for t in threads]; [t.join() for t in threads]
right = all(np.allclose(results[i][0], [np.sum(np.sin(np.arange(4.0) + k)) * i for k in range(3)]) and np.allclose(results[i][1], np.cos(np.arange(4.0)) * i) for i in range(1, 5))
check("mlir: four threads lower and compile at the same time, each with its own context", right and len(results) == 4 and len({id(results[i][2]) for i in results}) == 4)

exported = jax.export.export(jax.jit(lambda x: jnp.sin(x) * 2))(jnp.arange(3.0))
check("mlir: jax.export and calling the exported function (parses into another context)",
      np.allclose(exported.call(jnp.arange(3.0)), np.sin(np.arange(3.0)) * 2) and np.allclose(jax.jit(lambda x: exported.call(x) + 1)(jnp.arange(3.0)), np.sin(np.arange(3.0)) * 2 + 1))

# ---- lazy MHLO classes --------------------------------------------------------------------------
from jax._src.interpreters import mlir
from jax._src.lib.mlir import dialects, ir

if "jaxlib.mlir.dialects.mhlo" not in sys.modules:
    check("mhlo: the Python classes were not imported by lowering and compiling", type(dialects.mhlo) is tuning._MhloStandIn)
    with mlir.make_ir_context():
        parsed = ir.Module.parse("func.func @f(%a: tensor<2xf32>) -> tensor<2xf32> {\n  %0 = mhlo.add %a, %a : tensor<2xf32>\n  return %0 : tensor<2xf32>\n}")
    check("mhlo: the dialect is registered in new contexts", "mhlo.add" in str(parsed))
    check("mhlo: the classes load on first use", dialects.mhlo.AddOp.__name__ == "AddOp" and "jaxlib.mlir.dialects.mhlo" in sys.modules and type(dialects.mhlo) is not tuning._MhloStandIn)
else:
    check("mhlo: classes already imported by something else, nothing to check", True)

# ---- transformations on a differometor-like function --------------------------------------------
def model(p):
    return jnp.sum(jnp.abs(host_identity(p * (1 + 2j))) ** 2)


batch = jax.vmap(lambda p: jnp.sum(jnp.abs(p * (1 + 2j)) ** 2))(jnp.arange(6.0).reshape(2, 3))
check("jit, grad and vmap give the expected numbers", np.allclose(jax.jit(model)(jnp.arange(3.0)), 5 * np.sum(np.arange(3.0) ** 2))
      and np.allclose(jax.grad(lambda p: jnp.sum(jnp.abs(p * (1 + 2j)) ** 2))(jnp.arange(3.0)), 10 * np.arange(3.0)) and np.allclose(batch, [25.0, 250.0]))

# ---- inline jits under differentiation, lowering in place ----------------------------------------
def readout_like(x, reference):
    detected = jnp.abs(x[0] - x[1]).squeeze()
    sensitivity = (reference / detected).squeeze()
    return jnp.mean(jnp.log10(sensitivity / reference)) + jnp.sum(jnp.sin(x.real) * jnp.exp(-x.imag ** 2))


def variants(function, *arguments):
    """Values and derivatives of `function` through several transformations, as NumPy arrays."""
    value, pullback = jax.vjp(function, *arguments)
    out = [value, *pullback(jnp.ones_like(value))]
    out += [jax.jvp(function, arguments, tuple(jnp.ones_like(a) for a in arguments))[1]]
    out += [jax.jit(lambda *a: jax.vjp(function, *a)[1](1.0))(*arguments)[0]]
    out += [jax.vmap(lambda a: jax.grad(lambda r: function(a, r))(arguments[1]))(jnp.stack([arguments[0], 2 * arguments[0]]))]
    out += [jax.hessian(lambda r: function(arguments[0], r))(arguments[1])]
    return [np.asarray(item) for item in out]


def with_switches(function, *arguments, **switches):
    saved = {name: getattr(tuning, name) for name in switches}
    for name, value in switches.items():
        setattr(tuning, name, value)
    try:
        jax.clear_caches()
        return variants(function, *arguments)
    finally:
        for name, value in saved.items():
            setattr(tuning, name, value)


x0 = jnp.asarray(np.random.default_rng(3).normal(size=(2, 1, 7)) + 1j * np.random.default_rng(4).normal(size=(2, 1, 7)))
r0 = jnp.asarray(np.random.default_rng(5).uniform(1.0, 2.0, 7))
jax_way = with_switches(readout_like, x0, r0, INLINE_JITS_UNDER_AD=False, DIRECT_LOWERING_SMALL_PROGRAMS=False)
for title, switches in (("inline jits under differentiation", dict(INLINE_JITS_UNDER_AD=True, DIRECT_LOWERING_SMALL_PROGRAMS=False)),
                        ("lowering in place", dict(INLINE_JITS_UNDER_AD=False, DIRECT_LOWERING_SMALL_PROGRAMS=True)),
                        ("both", dict(INLINE_JITS_UNDER_AD=True, DIRECT_LOWERING_SMALL_PROGRAMS=True))):
    ours = with_switches(readout_like, x0, r0, **switches)
    worst = max(float(np.max(np.abs(a - b))) for a, b in zip(jax_way, ours))
    check(f"{title}: vjp, jvp, jit, vmap of grad and hessian as without it", len(jax_way) == len(ours) and worst <= 1e-13, f"largest difference {worst:.2e}")

with Spy("_jax_lower_jaxpr_to_module") as spy:
    jax.clear_caches()
    jax.block_until_ready(jax.jit(long_program)(jnp.arange(4.0)))
    small_then = tuning._is_small_jaxpr(jax.make_jaxpr(lambda v: jnp.sin(v) + 1)(jnp.arange(4.0)))
check("lowering in place: a long program is lowered the way JAX lowers it", len(spy.calls) >= 1 and small_then
      and not tuning._is_small_jaxpr(jax.make_jaxpr(long_program)(jnp.arange(4.0))))
check("lowering in place: nothing stays switched on after a lowering", not getattr(tuning._lowering, "in_place", False))
ordered = []
jax.clear_caches()
jax.jit(lambda v: (jax.experimental.io_callback(lambda a: ordered.append(float(a)) or a, v, v, ordered=True),
                   jax.experimental.io_callback(lambda a: ordered.append(float(a) + 1) or a, v, v, ordered=True))[1])(jnp.float64(5.0))
jax.effects_barrier()
check("lowering in place: ordered callbacks keep their order", ordered == [5.0, 6.0], str(ordered))

print(f"\n{len(failures)} failed" if failures else "\nall checks passed")
sys.exit(1 if failures else 0)
