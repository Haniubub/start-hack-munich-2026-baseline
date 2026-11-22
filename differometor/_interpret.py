"""Small programs run in a NumPy interpreter instead of being lowered and compiled.

`jax.jit(f).lower(*args).compile()` turns a traced program into MLIR, hands it to XLA and gets
machine code back. For the programs differometor is used with that is almost pure fixed cost: the
physics runs in host callbacks, and what is left for XLA are a few dozen elementwise operations on
about a hundred numbers (the caller's readout and its derivative). This module evaluates such a
program directly: every equation of the traced program is carried out with the NumPy function
that computes the same thing, host callbacks are called the way JAX calls them, and the results
come back as JAX arrays in the caller's output structure. Nothing is skipped and nothing is stored:
the same operations run on the same inputs, only without a compiler in between.

What is interpreted: a program with at most `_tuning.SMALL_PROGRAM_OPERATIONS` equations whose
equations are all in `_BUILDERS` below with the operand types listed there (floating point and
complex arithmetic, comparisons, a few shape operations, host callbacks, calls of custom
derivative rules). Any other program, and any request for non-default lowering or compile
options, takes JAX's own path unchanged. There is no limit on array sizes: NumPy's loops are
compiled code already.

The objects handed out are subclasses of `jax.stages.Lowered` and `jax.stages.Compiled`. Calling
the compiled object interprets; every other attribute or method (`as_text`, `compiler_ir`,
`cost_analysis`, `out_info`, ...) lowers or compiles the program with JAX on first use and answers
from that.

Differences to compiled code: results can differ in the last bit (NumPy and XLA use different
implementations of `log`, complex `abs` and summation order), and an output that is one of the
arguments is returned as that very array instead of a copy.

`install()` is called by `_tuning.apply()`. Switch: `_tuning.INTERPRET_SMALL_PROGRAMS`.
"""

import numpy as np

_jax_lower = None  # jax.stages.Traced.lower as JAX defines it
_parts = None  # (core, stages, tree_util, device_put, Unspecified)
_tuning = None


class _Unsupported(Exception):
    """This program is left to XLA."""


# --------------------------------------------------------------------------------------------
# One builder per primitive: (equation) -> function of the equation's inputs
# --------------------------------------------------------------------------------------------


def _kinds(equation):
    return "".join(np.dtype(v.aval.dtype).kind for v in equation.invars)


def _need(condition):
    if not condition:
        raise _Unsupported


def _elementwise(function, kinds):
    def build(equation):
        _need(all(kind in kinds for kind in _kinds(equation)) and not any(equation.params.get(k) for k in ("accuracy",)))
        return function

    return build


def _integer_pow(equation):
    _need(set(_kinds(equation)) <= set("fc"))
    exponent = int(equation.params["y"])

    def run(x):
        # the multiplications JAX lowers this to, in the same order
        if exponent == 0:
            return np.ones_like(x)
        y, acc = abs(exponent), None
        while y > 0:
            if y & 1:
                acc = x if acc is None else acc * x
            y >>= 1
            if y > 0:
                x = x * x
        return 1 / acc if exponent < 0 else acc

    return run


def _select_n(equation):
    _need(len(equation.invars) == 3 and np.dtype(equation.invars[0].aval.dtype).kind == "b")
    return lambda which, case_false, case_true: np.where(which, case_true, case_false)


def _convert_element_type(equation):
    new = np.dtype(equation.params["new_dtype"])
    _need(new.kind in "fcbiu" and np.dtype(equation.invars[0].aval.dtype).kind in "fcbiu")
    if new.kind != "c" and np.dtype(equation.invars[0].aval.dtype).kind == "c":
        return lambda x: np.real(x).astype(new)
    return lambda x: x.astype(new)


def _broadcast_in_dim(equation):
    _need(len(equation.invars) == 1)
    shape = tuple(int(extent) for extent in equation.params["shape"])
    dimensions = tuple(equation.params["broadcast_dimensions"])
    operand_shape = equation.invars[0].aval.shape
    inner = [1] * len(shape)
    for axis, dimension in enumerate(dimensions):
        inner[dimension] = operand_shape[axis]
    inner = tuple(inner)
    return lambda x: np.broadcast_to(x.reshape(inner), shape)


def _reshape_to_output(equation):
    _need(len(equation.invars) == 1 and equation.params.get("dimensions") is None or equation.primitive.name == "squeeze")
    shape = tuple(equation.outvars[0].aval.shape)
    return lambda x: x.reshape(shape)


def _transpose(equation):
    permutation = tuple(equation.params["permutation"])
    return lambda x: np.transpose(x, permutation)


def _reduce(function):
    def build(equation):
        _need(set(_kinds(equation)) <= set("fc"))
        axes = tuple(equation.params["axes"])
        return lambda x: function(x, axis=axes)

    return build


def _complex(equation):
    _need(_kinds(equation) == "ff")
    dtype = np.dtype(equation.outvars[0].aval.dtype)
    shape = tuple(equation.outvars[0].aval.shape)

    def run(real, imaginary):
        out = np.empty(shape, dtype)
        out.real = real
        out.imag = imaginary
        return out

    return run


def _concatenate(equation):
    dimension = int(equation.params["dimension"])
    return lambda *operands: np.concatenate(operands, axis=dimension)


def _slice(equation):
    strides = equation.params["strides"] or (1,) * len(equation.params["start_indices"])
    index = tuple(slice(int(a), int(b), int(c)) for a, b, c in zip(equation.params["start_indices"], equation.params["limit_indices"], strides))
    return lambda x: x[index]


_REAL, _FLOATING, _BOOL = "f", "fc", "b"
_BUILDERS = {
    "neg": _elementwise(np.negative, _FLOATING),
    "abs": _elementwise(np.abs, _FLOATING),
    "conj": _elementwise(np.conjugate, _FLOATING),
    "real": _elementwise(np.real, "c"),
    "imag": _elementwise(np.imag, "c"),
    "log": _elementwise(np.log, _FLOATING),
    "log1p": _elementwise(np.log1p, _FLOATING),
    "exp": _elementwise(np.exp, _FLOATING),
    "expm1": _elementwise(np.expm1, _FLOATING),
    "sqrt": _elementwise(np.sqrt, _FLOATING),
    "sin": _elementwise(np.sin, _FLOATING),
    "cos": _elementwise(np.cos, _FLOATING),
    "tanh": _elementwise(np.tanh, _FLOATING),
    "square": _elementwise(np.square, _FLOATING),
    "is_finite": _elementwise(np.isfinite, _REAL),
    "not": _elementwise(np.logical_not, _BOOL),
    "add": _elementwise(np.add, _FLOATING),
    "sub": _elementwise(np.subtract, _FLOATING),
    "mul": _elementwise(np.multiply, _FLOATING),
    "div": _elementwise(np.true_divide, _FLOATING),
    "pow": _elementwise(np.power, _FLOATING),
    "max": _elementwise(np.maximum, _REAL),
    "min": _elementwise(np.minimum, _REAL),
    "atan2": _elementwise(np.arctan2, _REAL),
    "eq": _elementwise(np.equal, _FLOATING),
    "ne": _elementwise(np.not_equal, _FLOATING),
    "lt": _elementwise(np.less, _REAL),
    "le": _elementwise(np.less_equal, _REAL),
    "gt": _elementwise(np.greater, _REAL),
    "ge": _elementwise(np.greater_equal, _REAL),
    "and": _elementwise(np.logical_and, _BOOL),
    "or": _elementwise(np.logical_or, _BOOL),
    "integer_pow": _integer_pow,
    "select_n": _select_n,
    "convert_element_type": _convert_element_type,
    "broadcast_in_dim": _broadcast_in_dim,
    "squeeze": _reshape_to_output,
    "reshape": _reshape_to_output,
    "transpose": _transpose,
    "reduce_sum": _reduce(np.sum),
    "reduce_prod": _reduce(np.prod),
    "reduce_max": _reduce(np.max),
    "reduce_min": _reduce(np.min),
    "complex": _complex,
    "concatenate": _concatenate,
    "slice": _slice,
    "copy": lambda equation: np.array,
    "copy_p": lambda equation: np.array,
}
# Equations that carry a program of their own which is what they compute: (parameter with the program).
_CALLS = {"jit": "jaxpr", "pjit": "jaxpr", "custom_vjp_call": "call_jaxpr", "custom_jvp_call": "call_jaxpr"}


def _plain(params):
    """A jit without shardings, layouts or compiler options: only its program matters."""
    unspecified = _parts[4]
    _need(not params.get("compiler_options_kvs"))
    _need(all(isinstance(s, unspecified) for s in (*params["in_shardings"], *params["out_shardings"])))
    _need(all(layout is None for layout in (*params["in_layouts"], *params["out_layouts"])))


# --------------------------------------------------------------------------------------------
# A prepared program
# --------------------------------------------------------------------------------------------


class _Program:
    """A traced program as a list of steps over a list of values (NumPy arrays).

    `jax_of` maps id(NumPy array) to the JAX array it is a view of, so that an argument that goes
    straight into a host callback or straight to the results travels as the array the caller
    passed, without a round trip.
    """

    def __init__(self, closed, budget):
        core = _parts[0]
        jaxpr = closed.jaxpr
        _need(not jaxpr.effects and not getattr(jaxpr, "is_high", False))
        budget[0] -= len(jaxpr.eqns)
        _need(budget[0] >= 0)
        slots = {}

        def slot(variable):
            return slots.setdefault(variable, len(slots))

        self.inputs = [slot(v) for v in jaxpr.invars]
        self.constants = [(slot(v), np.asarray(value)) for v, value in zip(jaxpr.constvars, closed.consts)]
        self.steps = []
        for equation in jaxpr.eqns:
            name = equation.primitive.name
            operands = []
            for v in equation.invars:  # a literal is a value of its own, a variable a slot
                if isinstance(v, core.Literal):
                    operands.append((None, np.asarray(v.val, dtype=v.aval.dtype)))
                else:
                    _need(v in slots)
                    operands.append((slots[v], None))
            for v in equation.outvars:
                _need(np.dtype(v.aval.dtype).kind in "fcbiu" and all(isinstance(extent, int) for extent in v.aval.shape))
            results = [(slot(v), np.dtype(v.aval.dtype), tuple(v.aval.shape)) for v in equation.outvars]
            if name == "pure_callback":
                self.steps.append((self._callback, equation.params["callback"], operands, results))
            elif name in _CALLS:
                if _CALLS[name] == "jaxpr":  # a nested jit
                    _plain(equation.params)
                inner = _Program(equation.params[_CALLS[name]], budget)
                _need(len(inner.inputs) == len(operands) and len(inner.outputs) == len(results))
                self.steps.append((self._call, inner, operands, results))
            else:
                builder = _BUILDERS.get(name)
                _need(builder is not None and len(results) == 1)
                self.steps.append((self._apply, builder(equation), operands, results))
        self.outputs = []
        for v in jaxpr.outvars:
            if isinstance(v, core.Literal):
                self.outputs.append((None, np.asarray(v.val, dtype=v.aval.dtype)))
            else:
                _need(v in slots)
                self.outputs.append((slots[v], None))
        self.size = len(slots)

    @staticmethod
    def _apply(function, arguments, results, values, jax_of):
        slot, dtype, shape = results[0]
        out = np.asarray(function(*arguments))
        if out.dtype != dtype:
            out = out.astype(dtype)
        if out.shape != shape:
            out = np.broadcast_to(out, shape)
        values[slot] = out

    @staticmethod
    def _call(inner, arguments, results, values, jax_of):
        for (slot, _, _), out in zip(results, inner.run(arguments, jax_of)):
            values[slot] = out

    @staticmethod
    def _callback(callback, arguments, results, values, jax_of):
        # as `jax._src.callback.pure_callback_impl`: JAX arrays in, NumPy arrays out
        device_put = _parts[3]
        outs = callback(*[jax_of.get(id(a)) if id(a) in jax_of else device_put(a) for a in arguments])
        if len(outs) != len(results):
            raise TypeError(f"pure_callback returned {len(outs)} results, expected {len(results)}")
        for (slot, dtype, shape), out in zip(results, outs):
            out = np.asarray(out)
            if out.shape != shape or out.dtype != dtype:
                raise TypeError(f"pure_callback returned {out.dtype}{list(out.shape)}, expected {dtype}{list(shape)}")
            # The callback may keep or reuse what it returned; XLA copies a result as well.
            values[slot] = out.copy()

    def run(self, arguments, jax_of):
        values = [None] * self.size
        for slot, argument in zip(self.inputs, arguments):
            values[slot] = argument
        for slot, constant in self.constants:
            values[slot] = constant
        for step, what, operands, results in self.steps:
            step(what, [values[slot] if constant is None else constant for slot, constant in operands], results, values, jax_of)
        return [values[slot] if constant is None else constant for slot, constant in self.outputs]


def _prepare(traced):
    """The `_Program` for a traced function, or None when JAX has to lower it."""
    try:
        params = traced._params
        _need(not traced._consts)
        _plain(params)
        closed = params["jaxpr"]
        for variable in (*closed.jaxpr.invars, *closed.jaxpr.outvars):
            _need(np.dtype(variable.aval.dtype).kind in "fcbiu" and all(isinstance(extent, int) for extent in variable.aval.shape))
        return _Program(closed, [_tuning.SMALL_PROGRAM_OPERATIONS])
    except Exception:  # an unknown operation or an unexpected JAX: leave the program to JAX
        return None


# --------------------------------------------------------------------------------------------
# The stage objects
# --------------------------------------------------------------------------------------------

_classes = None  # (interpreted Lowered, interpreted Compiled), built by install()


def _make_classes(stages, tree_util, jax_array, device_put):
    lowered_slots = frozenset(stages.Lowered.__slots__)
    compiled_slots = frozenset(stages.Compiled.__slots__) | {"_call"}

    class Lowered(stages.Lowered):
        """A lowering that only exists once something other than `compile()` asks for it."""

        def __init__(self, traced, program):
            self._traced, self._program = traced, program
            self.out_tree = traced.out_tree

        def __getattr__(self, name):  # reached only for what is not there yet
            if name not in lowered_slots:
                raise AttributeError(name)
            real = _jax_lower(self._traced)
            for slot in lowered_slots:
                setattr(self, slot, getattr(real, slot))
            return getattr(self, name)

        def compile(self, compiler_options=None, *, device_assignment=None):
            if compiler_options is None and device_assignment is None and _tuning.INTERPRET_SMALL_PROGRAMS:
                return _classes[1](self)
            return stages.Lowered.compile(self, compiler_options, device_assignment=device_assignment)

    class Compiled(stages.Compiled):
        """Calls interpret the program; anything else compiles it with JAX on first use."""

        def __init__(self, lowered):
            traced = lowered._traced
            self._lowered, self._program = lowered, lowered._program
            self._tree = traced._in_tree
            self._arguments = [(np.dtype(v.aval.dtype), tuple(v.aval.shape)) for v in traced._params["jaxpr"].jaxpr.invars]
            self._results = [np.dtype(v.aval.dtype) for v in traced._params["jaxpr"].jaxpr.outvars]
            self.out_tree = traced.out_tree

        def __getattr__(self, name):
            if name not in compiled_slots:
                raise AttributeError(name)
            real = stages.Lowered.compile(self._lowered)
            for slot in compiled_slots:
                setattr(self, slot, getattr(real, slot))
            return getattr(self, name)

        def __call__(self, *args, **kwargs):
            leaves, tree = tree_util.tree_flatten((args, kwargs))
            if tree != self._tree:
                raise TypeError(f"Function compiled with input pytree does not match called pytree: {self._tree} vs {tree}")
            jax_of, arguments = {}, []
            for leaf, (dtype, shape) in zip(leaves, self._arguments):
                array = np.asarray(leaf)
                if array.shape != shape or (array.dtype != dtype and hasattr(leaf, "dtype")):
                    raise TypeError(f"Argument types differ from the types for which this computation was compiled: got {array.dtype}{list(array.shape)}, compiled for {dtype}{list(shape)}")
                if array.dtype != dtype:  # a Python scalar
                    array = array.astype(dtype)
                elif type(leaf) is not np.ndarray:
                    jax_of[id(array)] = leaf if isinstance(leaf, jax_array) else device_put(array)
                arguments.append(array)
            with np.errstate(all="ignore"):  # XLA does not warn about 1/0 or log(0) either
                outs = self._program.run(arguments, jax_of)
            results = []
            for out, dtype in zip(outs, self._results):
                kept = jax_of.get(id(out))
                if kept is None:
                    # An array this run made. `device_put` hands it to JAX without a program of
                    # its own (`jnp.array` would compile a conversion once per shape and dtype).
                    kept = device_put(out if out.dtype == dtype else out.astype(dtype))
                results.append(kept)
            return tree_util.tree_unflatten(self.out_tree, results)

    return Lowered, Compiled


def _lower(self, *, lowering_platforms=None, _private_parameters=None):
    """`jax.stages.Traced.lower`: small programs of known operations are not lowered at all."""
    if lowering_platforms is None and _private_parameters is None and _tuning.INTERPRET_SMALL_PROGRAMS:
        program = _prepare(self)
        if program is not None:
            return _classes[0](self, program)
    return _jax_lower(self, lowering_platforms=lowering_platforms, _private_parameters=_private_parameters)


def install(tuning):
    """Replace `Traced.lower`. Raises for a JAX whose stages look different (then nothing changes)."""
    global _jax_lower, _parts, _classes, _tuning
    import inspect

    import jax
    from jax._src import core, stages, tree_util
    from jax._src.sharding_impls import UnspecifiedValue

    lower = stages.Traced.lower
    if tuple(inspect.signature(lower).parameters) != ("self", "lowering_platforms", "_private_parameters"):
        raise TypeError("Traced.lower has other parameters")
    for cls, slots in ((stages.Lowered, {"_lowering", "out_tree"}), (stages.Compiled, {"_executable", "out_tree", "_params"}), (stages.Traced, {"_params", "_in_tree", "out_tree", "_consts"})):
        if not slots <= set(cls.__slots__):
            raise TypeError(f"{cls.__name__} has other fields")
    _tuning, _parts = tuning, (core, stages, tree_util, jax.device_put, UnspecifiedValue)
    _classes = _make_classes(stages, tree_util, jax.Array, jax.device_put)
    _jax_lower = lower
    stages.Traced.lower = _lower
