"""Process-wide JAX / XLA settings that make the first run of small programs cheaper.

A first run of differometor spends more time in XLA compilation and JAX lowering than in the
physics: the compiled programs are tiny (a few host callbacks plus the caller's readout on about
100 numbers), so almost all of that time is fixed cost per program and per kernel. The measures
below remove fixed cost. Each one has its own switch. `differometor/__init__.py` calls `apply()`
once; the switches are read again on every compile / lowering / call, so they can also be changed
at run time (`differometor._tuning.UNOPTIMIZED_SMALL_PROGRAMS = False`). The ones that end in
`_AT_IMPORT` only act when `apply()` runs.

For experiments without editing this file: the environment variable `DIFFEROMETOR_TUNING` overrides
the defaults, e.g. `DIFFEROMETOR_TUNING=UNOPTIMIZED_SMALL_PROGRAMS=0,FAST_ASARRAY=0`, and
`DIFFEROMETOR_TUNING=off` leaves JAX completely untouched.

Everything here changes JAX for the whole process, also for programs that have nothing to do with
differometor. That is why every measure is written so that it cannot make a working program fail:
a compile that fails with changed options is repeated with the options JAX asked for, and a JAX
version without the patched internals is left alone (see `STATUS`). Numbers and tests: `runs/flags/`.
"""

import functools
import importlib
import inspect
import os
import sys
import threading
import types

import numpy as np

# --------------------------------------------------------------------------------------------
# Switches
# --------------------------------------------------------------------------------------------

# Which programs count as small. Programs with more operations than SMALL_PROGRAM_OPERATIONS are
# compiled exactly as JAX would compile them, whatever the switches below say.
SMALL_PROGRAM_OPERATIONS = 200
# Largest array (in elements) that a compiled kernel of a program may read or write for the two
# measures that change generated code quality or summation order. Arrays that only pass through
# host callbacks do not count.
SMALL_PROGRAM_ELEMENTS = 1024

# XLA compiles every fused elementwise kernel through its new MLIR "fusion emitter" pipeline and
# gives each kernel an LLVM module of its own. The classic emitter writes LLVM IR directly.
# Benchmark: compile 216 -> 131 ms. Same LLVM optimization level, kernels run equally fast
# (measured up to 4 million elements), outputs bit-identical on the machine measured.
CLASSIC_KERNEL_EMITTER = True

# The classic emitter spreads the kernels of a program over up to 32 LLVM modules so that it can
# compile them in parallel. Compiling is not parallel on one CPU, and every module has fixed cost.
# With this switch all kernels of a small program share one module. Benchmark: compile
# 131 -> 105 ms. Outputs bit-identical. Cost: on a machine with free cores a program with many
# kernels compiles slower, which is why SMALL_PROGRAM_OPERATIONS limits it.
ONE_CODEGEN_MODULE = True

# Small programs that call back into Python (host callbacks, which every differometor program
# has) and whose kernels only touch arrays of at most SMALL_PROGRAM_ELEMENTS elements are compiled
# without LLVM optimization (`xla_backend_optimization_level=0`, what
# `jax_disable_most_optimizations` does for every program). Benchmark: compile 105 -> 45 ms.
# Cost 1: results can differ in the last bit, because unoptimized code does not contract a*b+c
# into one FMA instruction (benchmark: sensitivity 1 ulp, gradient 1.3e-14 of its largest entry,
# carrier, signal, noise and powers unchanged). Cost 2: such kernels run 3 to 4 times slower,
# 7 us per call for the benchmark's readout on 101 numbers, 0.1 ms at the limit of 1024.
UNOPTIMIZED_SMALL_PROGRAMS = True

# XLA rewrites a sum over more than 32 elements into sums over blocks of 32 (a reduce-window kernel
# in an LLVM module of its own). With this switch small programs keep the plain sum. It changes
# the order of summation, so a mean or sum can differ in the last digits (benchmark: only the
# scalar loss, 1e-15 relative; the saved arrays stay identical). Benchmark:
# compile 103 -> 85 ms without UNOPTIMIZED_SMALL_PROGRAMS, 46 -> 43 ms with it. On since the
# branch philip-fastest (Xeon 8468, job prof-1726, 20 alternating runs: compile 38.4 -> 35.5 ms).
PLAIN_SMALL_REDUCTIONS = True

# JAX creates a new MLIR context for every program it lowers (2-3 ms each). With this switch the
# lowerings of a thread share one context. Outputs identical (the context only owns types and
# attributes). Benchmark: lower 38.6 -> 31.5 ms. Cost: a context keeps what was lowered in it alive
# until it is replaced; it is replaced after MLIR_CONTEXT_REUSES lowerings and as soon as a
# program that is not small (operations or array size, constants included) has been compiled.
SHARED_MLIR_CONTEXT = True
MLIR_CONTEXT_REUSES = 32

# `jnp.asarray(list_of_python_numbers, dtype=float_type)` compiles a small XLA program per shape
# only to copy the values. With this switch that case builds the same array directly (same type,
# dtype, weak-type flag, device, values). Every other call goes to JAX unchanged. Benchmark:
# build + lower 88 -> 61 ms (three programs of benchmark.py are no longer compiled).
FAST_ASARRAY = True

# The first MLIR context makes JAX import the Python classes of the MHLO dialect (160 generated
# classes, 5-10 ms) only to call the C function that registers the dialect. With this switch that
# function is called directly; the classes are imported the first time anything else asks for them.
LAZY_MHLO_CLASSES_AT_IMPORT = True

# Run every XLA program on the calling thread instead of handing it to a worker thread
# (`jax_cpu_enable_async_dispatch`). Only acts if the CPU backend does not exist yet.
INLINE_DISPATCH_AT_IMPORT = False

# Leave Python tracebacks out of the debug locations of lowered programs
# (`jax_include_full_tracebacks_in_locations`). Error messages from XLA then name the operation
# but not the Python call stack.
SHORT_LOCATIONS_AT_IMPORT = False

# What apply() did, per measure: "on", "off", or the reason it was skipped. For diagnosis only.
STATUS = {}

_SWITCHES = (
    "SMALL_PROGRAM_OPERATIONS", "SMALL_PROGRAM_ELEMENTS", "CLASSIC_KERNEL_EMITTER", "ONE_CODEGEN_MODULE",
    "UNOPTIMIZED_SMALL_PROGRAMS", "PLAIN_SMALL_REDUCTIONS", "SHARED_MLIR_CONTEXT", "MLIR_CONTEXT_REUSES", "FAST_ASARRAY",
    "LAZY_MHLO_CLASSES_AT_IMPORT", "INLINE_DISPATCH_AT_IMPORT", "SHORT_LOCATIONS_AT_IMPORT",
)

# --------------------------------------------------------------------------------------------
# Compile options
# --------------------------------------------------------------------------------------------

_EMITTER_OPTION = "xla_cpu_use_fusion_emitters"
_SPLIT_OPTION = "xla_cpu_parallel_codegen_split_count"
_LEVEL_OPTION = "xla_backend_optimization_level"
_TREE_REDUCTION_PASS = "tree_reduction_rewriter"

# Operations that XLA does not turn into a kernel of its own; their arrays do not count.
_NOT_A_KERNEL = frozenset({"stablehlo.custom_call", "func.func", "func.return", "stablehlo.return", "sdy.mesh"})

_rejected_options = set()  # option names this XLA build does not know
_jax_compile = None
_ranked_tensor_type = None
_compile_error = Exception


def _program_size(module, max_operations):
    """Return `(few, largest)` for a lowered program.

    `few` says whether the program has at most `max_operations` operations (the walk stops there).
    `largest` is the number of elements of the largest array that a compiled operation reads or
    writes. Arrays that only pass through host callbacks or from the arguments to the results do
    not count, constants do, and a dynamic shape counts as infinitely large.
    """
    sizes = {}
    largest = 0
    seen = 0
    stack = list(module.body.operations)
    while stack:
        operation = stack.pop().operation
        seen += 1
        if seen > max_operations:
            return False, largest
        if operation.name not in _NOT_A_KERNEL:
            for values in (operation.results, operation.operands):
                for value in values:
                    kind = value.type
                    size = sizes.get(kind)
                    if size is None:
                        size = float("inf")
                        if isinstance(kind, _ranked_tensor_type) and kind.has_static_shape:
                            size = 1
                            for extent in kind.shape:
                                size *= extent
                        sizes[kind] = size
                    if size > largest:
                        largest = size
        for region in operation.regions:
            for block in region.blocks:
                stack.extend(block.operations)
    return True, largest


def _cheapen(compile_options, module, has_callbacks):
    """Change `compile_options` in place and return a function that undoes it, or None if nothing changed."""
    few, largest = _program_size(module, SMALL_PROGRAM_OPERATIONS)
    small = few and largest <= SMALL_PROGRAM_ELEMENTS
    if not small:
        _retire_shared_context()
    if not few:
        return None

    overrides = list(compile_options.env_option_overrides)
    chosen = {name for name, _ in overrides}  # options the caller set itself (jit(compiler_options=...)) win
    added = []
    if CLASSIC_KERNEL_EMITTER and _EMITTER_OPTION not in chosen and _EMITTER_OPTION not in _rejected_options:
        added.append((_EMITTER_OPTION, False))
    if ONE_CODEGEN_MODULE and _SPLIT_OPTION not in chosen and _SPLIT_OPTION not in _rejected_options:
        added.append((_SPLIT_OPTION, 1))

    debug = compile_options.executable_build_options.debug_options
    level = debug.xla_backend_optimization_level
    passes = debug.xla_disable_hlo_passes
    new_level, new_passes = level, passes
    if small:
        if UNOPTIMIZED_SMALL_PROGRAMS and has_callbacks and level > 0 and _LEVEL_OPTION not in chosen:
            new_level = 0
        if PLAIN_SMALL_REDUCTIONS and "xla_disable_hlo_passes" not in chosen and _TREE_REDUCTION_PASS not in passes.split(","):
            new_passes = _TREE_REDUCTION_PASS if not passes else passes + "," + _TREE_REDUCTION_PASS

    if not added and new_level == level and new_passes == passes:
        return None
    if added:
        compile_options.env_option_overrides = overrides + added
    if new_level != level:
        debug.xla_backend_optimization_level = new_level
    if new_passes != passes:
        debug.xla_disable_hlo_passes = new_passes

    def undo():
        if added:
            compile_options.env_option_overrides = overrides
        debug = compile_options.executable_build_options.debug_options
        if new_level != level:
            debug.xla_backend_optimization_level = level
        if new_passes != passes:
            debug.xla_disable_hlo_passes = passes

    return undo


def _compile(backend, computation, devices, compile_options, host_callbacks, *rest, **keywords):
    """`jax._src.compiler.compile_or_get_cached` with cheaper options; falls back to JAX's own options."""
    undo = None
    if CLASSIC_KERNEL_EMITTER or ONE_CODEGEN_MODULE or UNOPTIMIZED_SMALL_PROGRAMS or PLAIN_SMALL_REDUCTIONS or SHARED_MLIR_CONTEXT:
        try:
            undo = _cheapen(compile_options, computation, bool(host_callbacks))
        except Exception as error:  # an unexpected JAX version: compile as JAX would
            STATUS["compile"] = f"options left alone: {type(error).__name__}: {error}"
            undo = None
    if undo is None:
        return _jax_compile(backend, computation, devices, compile_options, host_callbacks, *rest, **keywords)
    try:
        return _jax_compile(backend, computation, devices, compile_options, host_callbacks, *rest, **keywords)
    except _compile_error as error:
        # Either this XLA build does not know one of the options, or the program itself is wrong.
        # In both cases compile once more exactly as JAX asked; a wrong program fails again there.
        text = str(error)
        for name in (_EMITTER_OPTION, _SPLIT_OPTION):
            if "No such compile option" in text and name in text:
                _rejected_options.add(name)
                STATUS["compile"] = f"{name} is unknown to this jaxlib and no longer used"
        undo()
        return _jax_compile(backend, computation, devices, compile_options, host_callbacks, *rest, **keywords)


# --------------------------------------------------------------------------------------------
# MLIR context
# --------------------------------------------------------------------------------------------

_jax_make_ir_context = None
_contexts = threading.local()  # one shared context per thread: .context and .uses


def _make_ir_context():
    """`jax._src.interpreters.mlir.make_ir_context`, handing out the thread's context again."""
    if not SHARED_MLIR_CONTEXT:
        return _jax_make_ir_context()
    context = getattr(_contexts, "context", None)
    if context is None or _contexts.uses >= MLIR_CONTEXT_REUSES:
        context = _contexts.context = _jax_make_ir_context()
        _contexts.uses = 0
    _contexts.uses += 1
    return context


def _retire_shared_context():
    """The next lowering of this thread starts a new context; the old one dies with its last module."""
    _contexts.context = None


class _MhloStandIn(types.ModuleType):
    """`jax._src.lib.mlir.dialects.mhlo` until something other than the dialect registration is needed."""

    def __init__(self, owner, register):
        super().__init__("jaxlib.mlir.dialects.mhlo")
        self.__dict__["_owner"] = owner
        self.register_mhlo_dialect = register

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        real = importlib.import_module("jaxlib.mlir.dialects.mhlo")
        if vars(self._owner).get("mhlo") is self:
            self._owner.mhlo = real
        return getattr(real, name)


# --------------------------------------------------------------------------------------------
# jnp.asarray
# --------------------------------------------------------------------------------------------

_jax_asarray = None
_asarray_parts = None  # (canonical dtype function, TypedNdArray, device_put, trace_state_clean, config)
# The direct construction skips an XLA program. These settings give that program a visible effect
# (an error or a check), so with any of them set the call goes to JAX unchanged.
_ASARRAY_NEEDS_DEFAULT = ("jax_transfer_guard", "jax_transfer_guard_host_to_device", "jax_debug_nans",
                          "jax_debug_infs", "jax_no_execution", "jax_disable_jit")


def _asarray(a, dtype=None, order=None, *, copy=None, device=None, out_sharding=None):
    """`jnp.asarray`; Python numbers with an explicit float dtype become an array without compiling."""
    if FAST_ASARRAY and dtype is not None and order is None and copy is None and device is None and out_sharding is None:
        kind = type(a)
        if kind is float or ((kind is list or kind is tuple) and all(type(item) is float or type(item) is int for item in a)):
            canonical_dtype, typed_array, device_put, trace_state_clean, config = _asarray_parts
            if trace_state_clean() and not any(getattr(config, name, None) for name in _ASARRAY_NEEDS_DEFAULT):
                dtype = canonical_dtype(dtype, "asarray")
                if isinstance(dtype, np.dtype) and dtype.kind == "f":
                    # What jnp.array does next is np.asarray(a, dtype=dtype) followed by a
                    # convert_element_type that changes nothing but is compiled once per shape.
                    return device_put(typed_array(np.asarray(a, dtype=dtype), weak_type=False))
    return _jax_asarray(a, dtype, order, copy=copy, device=device, out_sharding=out_sharding)


# --------------------------------------------------------------------------------------------
# Installation
# --------------------------------------------------------------------------------------------

_applied = False


def _read_environment():
    """Apply `DIFFEROMETOR_TUNING`; returns False if it asks for no tuning at all."""
    text = os.environ.get("DIFFEROMETOR_TUNING", "").strip()
    if not text:
        return True
    if text.lower() in ("off", "0", "false", "no"):
        return False
    for item in text.split(","):
        name, _, value = item.partition("=")
        name = name.strip().upper()
        if name not in _SWITCHES:
            raise ValueError(f"DIFFEROMETOR_TUNING: unknown switch {name!r}, known: {', '.join(_SWITCHES)}")
        current = globals()[name]
        globals()[name] = int(value) if not isinstance(current, bool) else value.strip().lower() in ("1", "true", "on", "yes")
    return True


def _install_compile():
    global _jax_compile, _ranked_tensor_type, _compile_error
    from jax._src import compiler
    from jax._src.lib import _jax
    from jax._src.lib.mlir import ir

    _ranked_tensor_type = ir.RankedTensorType
    _compile_error = _jax.JaxRuntimeError
    _jax_compile = compiler.compile_or_get_cached
    compiler.compile_or_get_cached = _compile


def _install_mlir_context():
    global _jax_make_ir_context
    from jax._src.interpreters import mlir

    _jax_make_ir_context = mlir.make_ir_context
    mlir.make_ir_context = _make_ir_context


def _install_lazy_mhlo():
    from jax._src.lib.mlir import dialects

    if "mhlo" in vars(dialects) or "jaxlib.mlir.dialects.mhlo" in sys.modules:
        return  # already imported, nothing to save
    from jaxlib.mlir._mlir_libs import _mlirHlo

    dialects.mhlo = _MhloStandIn(dialects, _mlirHlo.register_mhlo_dialect)


def _install_asarray():
    global _jax_asarray, _asarray_parts
    import jax
    import jax.numpy
    from jax._src import core, dtypes, literals

    original = jax.numpy.asarray
    if list(inspect.signature(original).parameters) != list(inspect.signature(_asarray).parameters):
        raise TypeError("jnp.asarray has another signature than expected")
    _asarray_parts = (dtypes.check_and_canonicalize_user_dtype, literals.TypedNdArray, jax.device_put,
                      core.trace_state_clean, jax.config)
    _jax_asarray = original
    functools.update_wrapper(_asarray, original)
    jax.numpy.asarray = _asarray


def _configure(name, value):
    import jax

    jax.config.update(name, value)


def apply():
    """Install the measures once. Safe to call again; never raises because of an unexpected JAX."""
    global _applied
    if _applied:
        return
    _applied = True
    if not _read_environment():
        STATUS["all"] = "off (DIFFEROMETOR_TUNING)"
        return
    for name, install in (("compile", _install_compile), ("mlir_context", _install_mlir_context), ("asarray", _install_asarray)):
        try:
            install()
            STATUS[name] = "on"
        except Exception as error:  # other JAX version: leave that part of JAX alone
            STATUS[name] = f"skipped: {type(error).__name__}: {error}"
    for name, wanted, install in (
        ("lazy_mhlo", LAZY_MHLO_CLASSES_AT_IMPORT, _install_lazy_mhlo),
        ("inline_dispatch", INLINE_DISPATCH_AT_IMPORT, lambda: _configure("jax_cpu_enable_async_dispatch", False)),
        ("short_locations", SHORT_LOCATIONS_AT_IMPORT, lambda: _configure("jax_include_full_tracebacks_in_locations", False)),
    ):
        if not wanted:
            STATUS[name] = "off"
            continue
        try:
            install()
            STATUS[name] = "on"
        except Exception as error:
            STATUS[name] = f"skipped: {type(error).__name__}: {error}"
