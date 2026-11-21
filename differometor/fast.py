"""
Fast host-side simulation path: sparse LU solves with SuperLU and a hand-written adjoint.

`differometor.simulate.simulate` dispatches here for setups without parameter linking,
without homodyne detectors (qhd), with exactly one quantum noise detector and a single
carrier operating point. Everything else uses the original JAX implementation.

How it works
------------
- Matrix entries are produced by the unchanged component functions of
  `differometor.components`, re-bound to a tiny NumPy shim. The forward pass evaluates plain
  values. The reverse pass evaluates them on dual numbers (value + derivative), but only for the
  function inputs that are design variables. No XLA compilation and no hand-written derivatives
  of the physics formulas are needed.
- The carrier system and the signal system of every frequency are assembled as sparse
  CSC matrices and factorized with SuperLU, the solver behind `scipy.sparse.linalg.splu`.
  The compiled SuperLU module is loaded directly (`_load_superlu`), because importing
  `scipy.sparse.linalg` costs more time than the whole simulation.
- All frequencies share one sparsity pattern. The column order is computed once (COLAMD) and
  reused; per frequency there is one factorization and two solve calls.
- The quantum noise of the (single) detector is `Re(y^H N y)` with `y = S^-H w`; the
  same solve is reused as the adjoint helper for the gradient.
- Reverse mode is written by hand. Cotangents follow JAX's convention for complex
  values (bilinear pairing), so signal adjoints solve with `S^-T`, not `S^-H`.

The JAX integration (`simulate_fast`) is a `jax.custom_vjp` around two
`jax.pure_callback`s. The callbacks keep no Python state: the forward callback returns
every residual the backward pass needs as arrays that flow through JAX, and the backward
callback rebuilds the sparsity plan from them.

JAX copies every array it hands to a callback. The reverse pass therefore gets no dense noise
matrix and builds its plan from the index arrays alone: only matrix entries that depend on
parameters receive a cotangent, so the constant entries are not needed there.
"""

import importlib.machinery
import importlib.util
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

import differometor.components as components
from differometor.components import UNIT_VACUUM, H_PLANCK, F0

NOISE_SCALE = UNIT_VACUUM * H_PLANCK * F0 * 0.25


# ---------------------------------------------------------------------
# SuperLU without the SciPy package import
# ---------------------------------------------------------------------

_SUPERLU_NAME = "scipy.sparse.linalg._dsolve._superlu"


def _load_superlu():
    """
    The SuperLU extension module that backs `scipy.sparse.linalg.splu`.

    Importing `scipy.sparse.linalg` executes a few hundred Python modules (about 90 ms) although
    only this one compiled module is used here. It is therefore loaded from its file. The module
    is registered under its regular name, so a later `import scipy.sparse.linalg` reuses it. If
    the file cannot be found, the regular import is used.
    """
    if _SUPERLU_NAME in sys.modules:
        return sys.modules[_SUPERLU_NAME]
    try:
        root = importlib.util.find_spec("scipy").submodule_search_locations[0]
        folder = os.path.join(root, "sparse", "linalg", "_dsolve")
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            path = os.path.join(folder, "_superlu" + suffix)
            if os.path.exists(path):
                loader = importlib.machinery.ExtensionFileLoader(_SUPERLU_NAME, path)
                spec = importlib.util.spec_from_loader(_SUPERLU_NAME, loader, origin=path)
                module = importlib.util.module_from_spec(spec)
                loader.exec_module(module)
                sys.modules[_SUPERLU_NAME] = module
                return module
    except Exception:
        pass
    from scipy.sparse.linalg._dsolve import _superlu

    return _superlu


_superlu = _load_superlu()


def _splu(size, values, row_indices, column_pointers, column_order="COLAMD", small_supernodes=False):
    """
    LU factorization of a square CSC matrix, with the options `scipy.sparse.linalg.splu` passes.

    `column_order="NATURAL"` keeps the given column order (like `splu(..., permc_spec="NATURAL")`).

    `small_supernodes` sets SuperLU's `Relax` and `PanelSize` to 1. The systems here have about
    3.5 entries per row. With the default relaxed supernodes SuperLU pads the factors with explicit
    zeros (13120 stored entries instead of 8330 for the benchmark setup), which makes both the
    factorization and every solve slower. Pivoting is unchanged (partial pivoting).
    """
    options = dict(DiagPivotThresh=None, ColPerm=column_order, PanelSize=None, Relax=None)
    if column_order == "NATURAL":
        options["SymmetricMode"] = True
    if small_supernodes:
        options["PanelSize"] = 1
        options["Relax"] = 1
    return _superlu.gstrf(
        size,
        row_indices.size,
        np.ascontiguousarray(values, dtype=np.complex128),
        row_indices,
        column_pointers,
        csc_construct_func=None,
        ilu=False,
        options=options,
    )


# Columns of the detector basis Z = S^-T [e_d] kept per frequency (upper + lower sideband of one port).
DETECTOR_WIDTH = 2


# ---------------------------------------------------------------------
# Component functions on dual numbers
# ---------------------------------------------------------------------


class Dual:
    """Value plus derivatives with respect to k real inputs; `tan` has shape (k, *val.shape) or is 0."""

    __slots__ = ("val", "tan")
    __array_priority__ = 1000

    def __init__(self, val, tan=0.0):
        self.val, self.tan = val, tan

    @staticmethod
    def lift(x):
        return x if isinstance(x, Dual) else Dual(x, 0.0)

    def __neg__(self):
        return Dual(-self.val, -self.tan)

    def __add__(self, other):
        other = Dual.lift(other)
        return Dual(self.val + other.val, self.tan + other.tan)

    __radd__ = __add__

    def __sub__(self, other):
        other = Dual.lift(other)
        return Dual(self.val - other.val, self.tan - other.tan)

    def __rsub__(self, other):
        other = Dual.lift(other)
        return Dual(other.val - self.val, other.tan - self.tan)

    def __mul__(self, other):
        other = Dual.lift(other)
        return Dual(self.val * other.val, self.tan * other.val + self.val * other.tan)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = Dual.lift(other)
        quotient = self.val / other.val
        return Dual(quotient, (self.tan - quotient * other.tan) / other.val)

    def __rtruediv__(self, other):
        return Dual.lift(other).__truediv__(self)

    def __pow__(self, exponent):
        if isinstance(exponent, Dual):
            raise TypeError("Dual exponents are not supported")
        return Dual(self.val**exponent, exponent * self.val ** (exponent - 1) * self.tan)


def _unary(function, derivative):
    def wrapped(x):
        if not isinstance(x, Dual):
            return function(x)
        y = function(x.val)
        return Dual(y, derivative(x.val, y) * x.tan)

    return wrapped


class _NumpyShim:
    """Stand-in for `jax.numpy` inside the component functions (all inputs are real)."""

    pi, e, complex128, ndarray = np.pi, np.e, np.complex128, np.ndarray
    log10, log = staticmethod(np.log10), staticmethod(np.log)
    exp = staticmethod(_unary(np.exp, lambda x, y: y))
    sqrt = staticmethod(_unary(np.sqrt, lambda x, y: 0.5 / y))
    sin = staticmethod(_unary(np.sin, lambda x, y: np.cos(x)))
    cos = staticmethod(_unary(np.cos, lambda x, y: -np.sin(x)))
    sinh = staticmethod(_unary(np.sinh, lambda x, y: np.cosh(x)))
    cosh = staticmethod(_unary(np.cosh, lambda x, y: np.sinh(x)))
    arcsin = staticmethod(_unary(np.arcsin, lambda x, y: 1.0 / np.sqrt(1.0 - x * x)))
    radians = staticmethod(_unary(np.radians, lambda x, y: np.pi / 180.0))

    @staticmethod
    def conj(x):  # valid because every primal input is real
        return Dual(np.conj(x.val), np.conj(x.tan)) if isinstance(x, Dual) else np.conj(x)

    conjugate = conj

    @staticmethod
    def abs(x):
        if not isinstance(x, Dual):
            return np.abs(x)
        magnitude = np.abs(x.val)
        return Dual(magnitude, (np.conj(x.val) * x.tan).real / magnitude)

    @staticmethod
    def array(items, dtype=None):
        return list(items)


class _Output(list):
    """Component output vector; supports the `-standardize_output(...)` sign flip."""

    def __neg__(self):
        return _Output(-x for x in self)


def _rebind_component_functions():
    """Copies of `components.FUNCTIONS` whose global `jnp` is the dual-number NumPy shim."""
    standardize = lambda output: _Output(list(output) + [0.0] * (4 - len(output)))
    namespace = components.rebind_functions(_NumpyShim, {"standardize_output": standardize})
    return [namespace[function.__name__] for function in components.FUNCTIONS]


FUNCTIONS_NUMPY = _rebind_component_functions()


def _evaluate(function_index, inputs, columns):
    """
    inputs: (..., rows, k) -> values (..., rows, 4) and Jacobians (..., rows, 4, len(columns)).

    Only the input columns listed in `columns` carry a derivative; the others are plain values.
    """
    lead, width = inputs.shape[:-1], len(columns)
    seed = np.eye(width)
    arguments = [inputs[..., j] for j in range(inputs.shape[-1])]
    for position, j in enumerate(columns):
        arguments[j] = Dual(arguments[j], seed[position].reshape((width,) + (1,) * len(lead)))
    outputs = FUNCTIONS_NUMPY[function_index](arguments)
    values = np.empty(lead + (4,), complex)
    jacobian = np.zeros(lead + (4, width), complex)
    for o, item in enumerate(outputs):
        if isinstance(item, Dual):
            values[..., o] = item.val
            if not np.isscalar(item.tan):
                jacobian[..., o, :] = np.moveaxis(np.broadcast_to(item.tan, (width,) + lead), 0, -1)
        else:
            values[..., o] = item
    return values, jacobian


def _evaluate_values(function_index, inputs):
    """Values only (no derivatives): inputs (..., rows, k) -> (..., rows, 4)."""
    outputs = FUNCTIONS_NUMPY[function_index]([inputs[..., j] for j in range(inputs.shape[-1])])
    values = np.empty(inputs.shape[:-1] + (4,), complex)
    for o, item in enumerate(outputs):
        values[..., o] = item
    return values


def _group_sum(values, order, starts):
    """Sum the columns of `values` (R, E) that share a target slot (`order`/`starts` from `_grouping`)."""
    return np.add.reduceat(values[:, order], starts, axis=1)


def _grouping(targets):
    """Sort order, group starts and the distinct target slots for a scatter-add to `targets`."""
    order = np.argsort(targets, kind="stable")
    slots, starts = np.unique(targets[order], return_index=True)
    return order, starts, slots


class _EntryBlock:
    """One group of matrix entries (carrier, signal or noise) produced by component functions.

    `changing` lists the parameter columns that differ between the rows (the frequency sweep).
    Calls that read none of them give the same entries in every row and are evaluated once.

    `differentiable` marks the parameter slots that receive a design variable. The reverse pass
    only needs derivatives with respect to those, so every other function input is evaluated as a
    plain value (a mirror surface has 7 inputs, but only reflectivity and tuning are design
    variables; force terms have none).
    """

    def __init__(self, function_input_indices, function_indices, output_indices, changing=(), differentiable=None):
        self.inputs = np.asarray(function_input_indices)
        function_indices = np.asarray(function_indices)
        self.outputs = np.asarray(output_indices)
        self.varying = np.isin(self.inputs, np.asarray(changing, int)).any(axis=1)
        if differentiable is None:
            needed = np.ones(self.inputs.shape, bool)
        else:
            needed = differentiable[self.inputs]
        self.groups = []  # (function index, members, varying, differentiated input columns)
        for k in np.unique(function_indices):
            for varying in (True, False):
                members = np.nonzero((function_indices == k) & (self.varying == varying))[0]
                if members.size:
                    columns = np.nonzero(needed[members].any(axis=0))[0]
                    self.groups.append((int(k), members, varying, columns))
        # Scatter from entries to (call, output) slots; several entries may share a slot.
        self.scatter = _grouping(self.outputs[0] * 4 + self.outputs[1])

    def _gather(self, values):
        return values[:, self.outputs[0], self.outputs[1]]

    def values(self, parameters):
        """parameters: (R, P) -> entries (R, E), no derivatives."""
        rows, calls = parameters.shape[0], self.inputs.shape[0]
        values = np.empty((rows, calls, 4), complex)
        for function_index, members, varying, _ in self.groups:
            source = parameters if varying else parameters[:1]
            values[:, members] = _evaluate_values(function_index, source[:, self.inputs[members]])
        return self._gather(values)

    def run(self, parameters):
        """parameters: (R, P). Returns entries (R, E) and the Jacobian tape for the reverse pass."""
        rows, calls = parameters.shape[0], self.inputs.shape[0]
        values = np.empty((rows, calls, 4), complex)
        tape = []
        for function_index, members, varying, columns in self.groups:
            source = parameters if varying else parameters[:1]
            inputs = source[:, self.inputs[members]]
            if columns.size:
                values[:, members], jacobian = _evaluate(function_index, inputs, columns)
            else:
                values[:, members], jacobian = _evaluate_values(function_index, inputs), None
            tape.append(jacobian)
        return self._gather(values), tape

    def back(self, tape, entries_bar, parameter_count):
        """entries_bar: (R, E) -> real parameter cotangent summed over the rows (P,)."""
        rows, calls = entries_bar.shape[0], self.inputs.shape[0]
        order, starts, slots = self.scatter
        outputs_bar = np.zeros((rows, calls * 4), complex)
        if order.size:
            outputs_bar[:, slots] = _group_sum(entries_bar, order, starts)
        outputs_bar = outputs_bar.reshape(rows, calls, 4)
        parameters_bar = np.zeros(parameter_count)
        for (function_index, members, varying, columns), jacobian in zip(self.groups, tape):
            if jacobian is None:
                continue
            if varying:
                inputs_bar = np.einsum("rco,rcoj->cj", outputs_bar[:, members], jacobian).real
            else:  # identical Jacobian in every row
                inputs_bar = np.einsum("co,coj->cj", outputs_bar[:, members].sum(0), jacobian[0]).real
            parameters_bar += np.bincount(self.inputs[members][:, columns].ravel(), inputs_bar.ravel(), parameter_count)
        return parameters_bar


class EntryPlan:
    """Parameter expansion and component evaluation (mirrors the start of `simulate`)."""

    def __init__(self, A):
        parameters = np.asarray(A["parameters"])
        if np.any(parameters.imag != 0):
            raise ValueError("fast path requires real base parameters")
        self.base = parameters[0].real.copy()
        self.optimized_parameter_indices = np.asarray(A["optimized_parameter_indices"])
        self.optimized_value_indices = np.asarray(A["optimized_value_indices"])
        self.carrier_changing_indices = np.asarray(A["carrier_changing_parameter_indices"])
        self.carrier_changing_values = np.asarray(A["carrier_changing_values"])
        self.signal_changing_indices = np.asarray(A["signal_changing_parameter_indices"])
        self.signal_changing_values = np.asarray(A["signal_changing_values"])
        for values in (self.carrier_changing_values, self.signal_changing_values):
            if np.any(np.imag(values) != 0):
                raise ValueError("fast path requires real changing values")
        changing = self.signal_changing_indices
        # Parameter slots whose derivative reaches a design variable (see `pullback`).
        differentiable = np.zeros(self.base.size, bool)
        differentiable[self.optimized_parameter_indices] = True
        differentiable[self.carrier_changing_indices] = False
        differentiable[self.signal_changing_indices] = False
        block = lambda name, changing=(): _EntryBlock(
            A[name + "_function_input_indices"], A[name + "_function_indices"], A[name + "_output_indices"], changing, differentiable
        )
        self.carrier, self.signal, self.noise = block("carrier"), block("signal", changing), block("noise", changing)

    def _rows(self, optimized_parameters):
        P = self.base.copy()
        P[self.optimized_parameter_indices] = np.asarray(optimized_parameters, float)[self.optimized_value_indices]
        carrier_rows = np.tile(P, (self.carrier_changing_values.shape[1], 1))
        carrier_rows[:, self.carrier_changing_indices] = np.real(self.carrier_changing_values).T
        signal_rows = np.tile(carrier_rows[0], (self.signal_changing_values.shape[1], 1))
        signal_rows[:, self.signal_changing_indices] = np.real(self.signal_changing_values).T
        return carrier_rows[:1], signal_rows

    def values(self, optimized_parameters):
        """Carrier, signal and noise entries without derivatives (forward pass)."""
        carrier_rows, signal_rows = self._rows(optimized_parameters)
        return self.carrier.values(carrier_rows)[0], self.signal.values(signal_rows), self.noise.values(signal_rows)

    def entries(self, optimized_parameters):
        carrier_rows, signal_rows = self._rows(optimized_parameters)
        carrier, carrier_tape = self.carrier.run(carrier_rows)
        signal, signal_tape = self.signal.run(signal_rows)
        noise, noise_tape = self.noise.run(signal_rows)
        return (carrier[0], signal, noise), (carrier_tape, signal_tape, noise_tape)

    def pullback(self, tape, carrier_bar, signal_bar, noise_bar, value_count):
        carrier_tape, signal_tape, noise_tape = tape
        parameter_count = self.base.size
        signal_rows_bar = self.signal.back(signal_tape, signal_bar, parameter_count) + self.noise.back(
            noise_tape, noise_bar, parameter_count
        )
        signal_rows_bar[self.signal_changing_indices] = 0.0  # overwritten by the sweep values
        P_bar = self.carrier.back(carrier_tape, carrier_bar[None], parameter_count) + signal_rows_bar
        P_bar[self.carrier_changing_indices] = 0.0
        return np.bincount(self.optimized_value_indices, P_bar[self.optimized_parameter_indices], value_count)


# ---------------------------------------------------------------------
# Sparse linear systems
# ---------------------------------------------------------------------


def _flat_nonzero(matrix):
    """Flat indices of the nonzero entries of a complex matrix (faster than `np.flatnonzero`)."""
    parts = np.ascontiguousarray(matrix).view(np.float64).reshape(-1, 2) != 0
    return np.flatnonzero(parts[:, 0] | parts[:, 1])


def _last_writer(rows, columns, width):
    """Mask of scatter entries that survive `.at[rows, columns].set(...)` (last write wins)."""
    flat = rows * width + columns
    reverse_first = np.unique(flat[::-1], return_index=True)[1]
    mask = np.zeros(flat.size, bool)
    mask[flat.size - 1 - reverse_first] = True
    return mask


class SystemPlan:
    """Static sparsity data derived from the index arrays (cheap, rebuilt in every callback).

    The forward pass needs the complete sparsity pattern, so it scans the dense signal and noise
    matrix for their nonzeros. The reverse pass only needs the entries that depend on parameters:
    the ones written by component functions or multiplied by the carrier. `signal_base=False`
    restricts the signal pattern to those entries and `noise_base=False` leaves out the constant
    noise entries. Then no dense matrix is scanned and the noise matrix is not read at all.
    """

    def __init__(self, A, signal_base=True, noise_base=True):
        get = lambda key: np.asarray(A[key])
        self.carrier_base = get("carrier_matrix")
        self.n = n = self.carrier_base.shape[0]
        signal_matrix = get("signal_matrix")
        self.m = m = signal_matrix.shape[0]

        # Carrier system: pattern of the n x n matrix, right-hand side in column n.
        self.cmi = get("carrier_matrix_indices") % np.array([[n], [n + 1]])
        self.carrier_live = _last_writer(self.cmi[0], self.cmi[1], n + 1)
        mask = self.carrier_base[:, :-1] != 0
        in_matrix = self.cmi[1] < n
        mask[self.cmi[0][in_matrix], self.cmi[1][in_matrix]] = True
        self.carrier_cols, self.carrier_rows = np.nonzero(mask.T)
        self.carrier_rows = self.carrier_rows.astype(np.intc)
        self.carrier_ptr = np.concatenate([[0], np.cumsum(mask.sum(0))]).astype(np.intc)

        # Signal system: pattern of the augmented m x (m + 1) matrix in CSC order. Entries are
        # identified by the key `column * m + row`, which sorts them column by column.
        smi = get("signal_matrix_indices") % np.array([[m], [m + 1]])
        placing = get("signal_carrier_placing_indices") % np.array([[m], [m + 1]])
        self.signal_carrier_indices = get("signal_carrier_indices").ravel()
        key = lambda rows, cols: cols * m + rows
        written = [key(smi[0], smi[1]), key(placing[0], placing[1])]
        if signal_base:
            base_rows, base_cols = np.divmod(_flat_nonzero(signal_matrix), m + 1)
            written.append(key(base_rows, base_cols))
        keys = np.unique(np.concatenate(written))
        cols, rows = np.divmod(keys, m)
        self.rows, self.cols, self.size = rows, cols, rows.size
        self.signal_base = signal_matrix[rows, cols]
        self.update_positions = np.searchsorted(keys, key(smi[0], smi[1]))
        self.update_live = _last_writer(smi[0], smi[1], m + 1)
        self.placing_positions = np.searchsorted(keys, key(placing[0], placing[1]))
        self.placing_live = _last_writer(placing[0], placing[1], m + 1)
        # Entries that are complex-conjugated for the lower sideband (see `simulate`).
        self.conjugated = (
            ((rows >= n) & (rows < 2 * n) & (cols >= n) & (cols < 2 * n))
            | ((rows >= 2 * n) & (cols < n))
            | ((rows >= n) & (rows < 2 * n) & (cols >= 2 * n))
        )
        self.in_matrix = cols < m
        self.matrix_rows = rows[self.in_matrix].astype(np.intc)
        self.matrix_cols = cols[self.in_matrix]
        self.matrix_ptr = np.searchsorted(self.matrix_cols, np.arange(m + 1)).astype(np.intc)
        self.rhs_rows = rows[~self.in_matrix]

        # Noise matrix: base entries plus entries written by noise component functions.
        nmi = get("noise_system_matrix_indices") % m
        self.noise_live = _last_writer(nmi[0], nmi[1], m)
        if noise_base:
            noise_matrix = get("noise_matrix")
            base = _flat_nonzero(noise_matrix)
            base = base[~np.isin(base, nmi[0] * m + nmi[1])]
            base_rows, base_cols = np.divmod(base, m)
            self.noise_base = noise_matrix[base_rows, base_cols]
        else:
            base_rows = base_cols = np.empty(0, np.intp)
            self.noise_base = np.empty(0, complex)
        self.noise_rows = np.concatenate([base_rows, nmi[0][self.noise_live]])
        self.noise_cols = np.concatenate([base_cols, nmi[1][self.noise_live]])
        if noise_base:  # scatter plans for N y and N^H y (forward pass only)
            self.noise_by_row = _grouping(self.noise_rows)
            self.noise_by_col = _grouping(self.noise_cols)
        self.selection = get("noise_selection_vectors")[0, 0]
        # Rows the quantum noise detector reads. When they fit into DETECTOR_WIDTH, the forward pass
        # solves S^T z = e_d for each of them; Y and the signal adjoint are then combinations of z.
        self.detector = np.nonzero(self.selection)[0]
        self.use_basis = 0 < self.detector.size <= DETECTOR_WIDTH
        self.order = None

    def carrier_matrix(self, carrier_entries):
        n = self.n
        M = self.carrier_base.astype(complex)
        M[self.cmi[0][self.carrier_live], self.cmi[1][self.carrier_live]] = carrier_entries[self.carrier_live]
        return _splu(n, M[self.carrier_rows, self.carrier_cols], self.carrier_rows, self.carrier_ptr), M[:, -1]

    def signal_values(self, signal_entries, carrier):
        """Nonzero values of the augmented signal matrices, shape (F, size), plus placing pre-values."""
        F = signal_entries.shape[0]
        V = np.broadcast_to(self.signal_base.astype(complex), (F, self.size)).copy()
        V[:, self.update_positions[self.update_live]] = signal_entries[:, self.update_live]
        pre = V[:, self.placing_positions].copy()
        live = self.placing_live
        V[:, self.placing_positions[live]] = -pre[:, live] * carrier[self.signal_carrier_indices[live]]
        V[:, self.conjugated] = np.conj(V[:, self.conjugated])
        return V, pre

    def placing_inputs(self, signal_entries):
        """The `pre` values of `signal_values` alone: what the carrier is multiplied with, shape (F, P)."""
        F = signal_entries.shape[0]
        writer = np.full(self.size, -1)
        writer[self.update_positions[self.update_live]] = np.nonzero(self.update_live)[0]
        writer = writer[self.placing_positions]
        written = writer >= 0
        pre = np.broadcast_to(self.signal_base[self.placing_positions].astype(complex), (F, writer.size)).copy()
        pre[:, written] = signal_entries[:, writer[written]]
        return pre

    def _set_order(self, values_row):
        """Column order from COLAMD on one matrix, reused for all frequencies (shared pattern)."""
        first = _splu(self.m, values_row[self.in_matrix], self.matrix_rows, self.matrix_ptr)
        order = np.argsort(first.perm_c)
        counts = np.diff(self.matrix_ptr)[order]
        self.ordered_ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.intc)
        gather = np.repeat(self.matrix_ptr[order] - self.ordered_ptr[:-1], counts) + np.arange(self.ordered_ptr[-1])
        self.ordered_rows = self.matrix_rows[gather]
        self.ordered_values = np.nonzero(self.in_matrix)[0][gather]
        self.order = order

    def signal_factors(self, V):
        """One SuperLU factorization of B_f = S_f[:, order] per frequency (rows of `V`)."""
        if self.order is None:
            self._set_order(V[0])
        values = np.ascontiguousarray(V[:, self.ordered_values])
        rows, pointers, m = self.ordered_rows, self.ordered_ptr, self.m
        return [_splu(m, row, rows, pointers, "NATURAL", small_supernodes=True) for row in values]

    def signal_lu(self, values_row):
        return _OrderedLU(self.signal_factors(values_row[None])[0], self.order)

    def noise_values(self, noise_entries):
        F = noise_entries.shape[0]
        base = np.broadcast_to(self.noise_base.astype(complex), (F, self.noise_base.size))
        return np.concatenate([base, noise_entries[:, self.noise_live]], axis=1)

    def detector_weights(self, carrier):
        n, s = self.n, self.selection
        w = s.astype(complex).copy()
        w[:n] = s[:n] * carrier
        w[n : 2 * n] = np.conj(s[n : 2 * n] * carrier)
        return w


class _OrderedLU:
    """LU of B = A[:, order]; solves with A, A^T and A^H."""

    def __init__(self, lu, order):
        self.lu, self.order = lu, order

    def solve(self, b, trans="N"):
        if trans == "N":
            y = self.lu.solve(b)
            x = np.empty_like(y)
            x[self.order] = y
            return x
        return self.lu.solve(b[self.order], trans=trans)


def _scatter_sum(values, grouping, size):
    """Scatter-add the columns of `values` (F, E) into (F, size) target slots."""
    order, starts, slots = grouping
    result = np.zeros((values.shape[0], size), complex)
    if order.size:
        result[:, slots] = _group_sum(values, order, starts)
    return result


def forward(plan, carrier_entries, signal_entries, noise_entries):
    """Carrier, signal and noise outputs plus the adjoint helpers Y = S^-H w, MU and Z.

    Z[f] = S_f^-T [e_d for d in plan.detector] (zero-padded to DETECTOR_WIDTH columns). It gives
    Y = conj(Z conj(w_d)) here and the signal adjoint S^-T signal_bar in `backward` without a
    second factorization, whenever the cotangent of the signal lives on the detector rows.

    Per frequency this takes one factorization and two SuperLU solve calls with two right-hand
    sides each: first S^T Z = [e_d], then S [X, MU] = [rhs, (N + N^H) Y / 2]. Everything between
    the two solves is evaluated for all frequencies at once.
    """
    m = plan.m
    carrier_lu, carrier_rhs = plan.carrier_matrix(carrier_entries)
    carrier = carrier_lu.solve(carrier_rhs)
    V, _ = plan.signal_values(signal_entries, carrier)
    w = plan.detector_weights(carrier)
    N = plan.noise_values(noise_entries)
    F = V.shape[0]
    factors = plan.signal_factors(V)
    order = plan.order
    d, k = plan.detector, plan.detector.size
    Z = np.zeros((F, m, DETECTOR_WIDTH), complex)
    if plan.use_basis:
        basis = np.zeros((m, k), complex)
        basis[d, np.arange(k)] = 1.0
        basis = basis[order]
        for f, lu in enumerate(factors):
            Z[f, :, :k] = lu.solve(basis, trans="T")
        Y = np.conj(Z[:, :, :k] @ np.conj(w[d]))
    else:
        w_ordered = w[order]
        Y = np.empty((F, m), complex)
        for f, lu in enumerate(factors):
            Y[f] = lu.solve(w_ordered, trans="H")
    q = _scatter_sum(N * Y[:, plan.noise_cols], plan.noise_by_row, m)  # N y
    qh = _scatter_sum(np.conj(N) * Y[:, plan.noise_rows], plan.noise_by_col, m)  # N^H y
    power = np.einsum("fm,fm->f", np.conj(Y), q).real
    right_hand_sides = np.zeros((F, m, 2), complex)
    right_hand_sides[:, plan.rhs_rows, 0] = V[:, ~plan.in_matrix]
    right_hand_sides[:, :, 1] = 0.5 * (q + qh)
    solutions = np.empty((F, m, 2), complex)
    for f, lu in enumerate(factors):
        solutions[f] = lu.solve(right_hand_sides[f])
    X = np.empty((F, m), complex)
    MU = np.empty((F, m), complex)
    X[:, order] = solutions[:, :, 0]
    MU[:, order] = solutions[:, :, 1]
    noise = np.sqrt(2 * power * NOISE_SCALE)
    return carrier, X, noise, Y, MU, Z


def _signal_adjoint(plan, Z, signal_bar, general_adjoint):
    """lam_f = S_f^-T signal_bar_f for every frequency, shape (F, m).

    The detector basis Z covers a cotangent that lives on the detector rows (the usual readout).
    Any other cotangent needs the complete signal systems again: `general_adjoint(signal_bar)`.
    """
    support = np.nonzero(np.any(signal_bar != 0, axis=0))[0]
    d, k = plan.detector, plan.detector.size
    if plan.use_basis and np.isin(support, d).all():
        return np.einsum("fmk,fk->fm", Z[:, :, :k], signal_bar[:, d])
    return general_adjoint(signal_bar)


def backward(
    plan, carrier_entries, signal_entries, carrier, X, noise, Y, MU, Z, carrier_bar, signal_bar, noise_bar, general_adjoint
):
    """Cotangents of the carrier, signal and noise entries (JAX's bilinear convention).

    `plan` only has to contain the parameter-dependent entries (`SystemPlan(..., signal_base=False,
    noise_base=False)`): constant matrix entries receive no cotangent.
    """
    n, m = plan.n, plan.m
    carrier_lu, _ = plan.carrier_matrix(carrier_entries)
    pre = plan.placing_inputs(signal_entries)
    F = signal_entries.shape[0]
    carrier_bar = carrier_bar.astype(complex).copy()
    V_bar = np.empty((F, plan.size), complex)
    g = noise_bar * NOISE_SCALE / noise  # d noise / d power
    rows, cols = plan.rows[plan.in_matrix], plan.cols[plan.in_matrix]
    lam = _signal_adjoint(plan, Z, signal_bar, general_adjoint)
    V_bar[:, plan.in_matrix] = -lam[:, rows] * X[:, cols] - 2 * g[:, None] * np.conj(Y[:, rows]) * MU[:, cols]
    V_bar[:, ~plan.in_matrix] = lam[:, plan.rhs_rows]
    w_bar = 2 * (g[:, None] * np.conj(MU)).sum(0)

    N_bar = g[:, None] * np.conj(Y[:, plan.noise_rows]) * Y[:, plan.noise_cols]
    noise_entries_bar = np.zeros((F, plan.noise_live.size), complex)
    noise_entries_bar[:, plan.noise_live] = N_bar[:, plan.noise_base.size :]

    V_bar[:, plan.conjugated] = np.conj(V_bar[:, plan.conjugated])
    live = plan.placing_live
    placed = plan.placing_positions[live]
    sci = plan.signal_carrier_indices[live]
    np.add.at(carrier_bar, sci, -(V_bar[:, placed] * pre[:, live]).sum(0))
    V_bar[:, placed] = -V_bar[:, placed] * carrier[sci]
    signal_entries_bar = np.zeros_like(signal_entries)
    signal_entries_bar[:, plan.update_live] = V_bar[:, plan.update_positions[plan.update_live]]

    s = plan.selection
    carrier_bar += w_bar[:n] * s[:n] + np.conj(w_bar[n : 2 * n]) * s[n : 2 * n]
    lam_c = carrier_lu.solve(carrier_bar, trans="T")
    r, c = plan.cmi
    carrier_entries_bar = np.where(c == n, lam_c[r], -lam_c[r] * carrier[np.minimum(c, n - 1)])
    carrier_entries_bar = np.where(plan.carrier_live, carrier_entries_bar, 0)
    return carrier_entries_bar, signal_entries_bar, noise_entries_bar


# ---------------------------------------------------------------------
# JAX integration
# ---------------------------------------------------------------------

# Arrays the host callbacks need (the remaining `simulate` arguments only select the path).
HOST_KEYS = (
    "optimized_parameter_indices",
    "optimized_value_indices",
    "carrier_changing_parameter_indices",
    "carrier_changing_values",
    "signal_changing_parameter_indices",
    "signal_changing_values",
    "parameters",
    "carrier_matrix",
    "signal_matrix",
    "noise_matrix",
    "noise_selection_vectors",
    "carrier_function_input_indices",
    "carrier_function_indices",
    "carrier_output_indices",
    "carrier_matrix_indices",
    "signal_function_input_indices",
    "signal_function_indices",
    "signal_output_indices",
    "signal_matrix_indices",
    "signal_carrier_indices",
    "signal_carrier_placing_indices",
    "noise_function_input_indices",
    "noise_function_indices",
    "noise_output_indices",
    "noise_system_matrix_indices",
)

# The reverse pass does not read the dense noise matrix (see `SystemPlan`). Every array handed to a
# callback is copied, so the 9 MB matrix is left out of the residuals.
BACKWARD_KEYS = tuple(key for key in HOST_KEYS if key != "noise_matrix")


def supports(arrays):
    """Static (shape/dtype only) check whether the fast path covers these `simulate` arguments."""
    optimized = arrays["optimized_parameters"]
    return (
        arrays["linked_indices"].shape[0] == 0
        and arrays["qhd_parameter_indices"].shape[0] == 0
        and arrays["noise_selection_vectors"].shape[0] == 1
        and arrays["carrier_changing_values"].ndim == 2
        and arrays["carrier_changing_values"].shape[1] == 1
        and arrays["signal_changing_values"].ndim == 2
        and optimized.ndim == 1
        and not jnp.issubdtype(optimized.dtype, jnp.complexfloating)
    )


def _output_shapes(arrays):
    n = arrays["carrier_matrix"].shape[0]
    m = arrays["signal_matrix"].shape[0]
    F = arrays["signal_changing_values"].shape[1]
    noise_shape = (F,) if F != 1 else ()  # `noise.squeeze()` of shape (1, F)
    return n, m, F, noise_shape


def _host_forward(optimized_parameters, *host_arrays):
    A = dict(zip(HOST_KEYS, host_arrays))
    entries = EntryPlan(A).values(optimized_parameters)  # derivatives are only taped in the backward pass
    carrier, X, noise, Y, MU, Z = forward(SystemPlan(A), *entries)
    return carrier[:, None], np.ascontiguousarray(X.T), noise.reshape(-1), Y, MU, Z


def _host_backward(optimized_parameters, *rest):
    k = len(BACKWARD_KEYS)
    A = dict(zip(BACKWARD_KEYS, rest[:k]))
    carrier, signal, noise, Y, MU, Z, carrier_bar, signal_bar, noise_bar = rest[k:]
    F = Y.shape[0]
    entry_plan = EntryPlan(A)
    (carrier_entries, signal_entries, _), tape = entry_plan.entries(optimized_parameters)
    carrier = np.asarray(carrier)[:, 0]

    def general_adjoint(signal_bar):
        # A cotangent outside the detector rows: factorize the complete signal systems once more.
        plan = SystemPlan(A, noise_base=False)
        V, _ = plan.signal_values(signal_entries, carrier)
        lam = np.empty(signal_bar.shape, complex)
        for f in range(F):
            lam[f] = plan.signal_lu(V[f]).solve(signal_bar[f].astype(complex), trans="T")
        return lam

    entries_bar = backward(
        SystemPlan(A, signal_base=False, noise_base=False),
        carrier_entries,
        signal_entries,
        carrier,
        np.asarray(signal).T,
        np.asarray(noise).reshape(F),
        np.asarray(Y),
        np.asarray(MU),
        np.asarray(Z),
        np.asarray(carrier_bar)[:, 0],
        np.asarray(signal_bar).T,
        np.asarray(noise_bar, float).reshape(F),
        general_adjoint,
    )
    gradient = entry_plan.pullback(tape, *entries_bar, value_count=optimized_parameters.shape[0])
    return gradient.astype(optimized_parameters.dtype)


def _forward_callback(optimized_parameters, host):
    n, m, F, noise_shape = _output_shapes(host)
    result_shapes = (
        jax.ShapeDtypeStruct((n, 1), jnp.complex128),
        jax.ShapeDtypeStruct((m, F), jnp.complex128),
        jax.ShapeDtypeStruct((F,), jnp.float64),
        jax.ShapeDtypeStruct((F, m), jnp.complex128),
        jax.ShapeDtypeStruct((F, m), jnp.complex128),
        jax.ShapeDtypeStruct((F, m, DETECTOR_WIDTH), jnp.complex128),
    )
    carrier, signal, noise, Y, MU, Z = jax.pure_callback(
        _host_forward, result_shapes, optimized_parameters, *(host[k] for k in HOST_KEYS), vmap_method="sequential"
    )
    return (carrier, signal, noise.reshape(noise_shape)), (Y, MU, Z)


def _is_primal(x):
    return isinstance(x, jax.custom_derivatives.CustomVJPPrimal)


def _zeros_if_symbolic(value):
    if isinstance(value, jax.custom_derivatives.SymbolicZero):
        return jnp.zeros(value.shape, value.dtype)
    return value


@jax.custom_vjp
def _fast(optimized_parameters, host, others):
    outputs, _ = _forward_callback(optimized_parameters, host)
    return outputs


def _fast_fwd(optimized_parameters, host, others):
    leaves = jax.tree_util.tree_leaves((host, others), is_leaf=_is_primal)
    unwrap = lambda tree: jax.tree_util.tree_map(lambda x: x.value, tree, is_leaf=_is_primal)
    p, host, others = optimized_parameters.value, unwrap(host), unwrap(others)
    if any(leaf.perturbed for leaf in leaves):
        # Differentiation with respect to something other than the optimized parameters:
        # defer to the original JAX implementation and its pullback.
        from differometor.simulate import simulate_jax

        full = lambda p, host, others: simulate_jax(optimized_parameters=p, **host, **others)
        outputs, pullback = jax.vjp(full, p, host, others)
        return outputs, (pullback,)
    outputs, (Y, MU, Z) = _forward_callback(p, host)
    return outputs, (p, {key: host[key] for key in BACKWARD_KEYS}, outputs, Y, MU, Z)


def _fast_bwd(residuals, cotangents):
    cotangents = tuple(_zeros_if_symbolic(c) for c in cotangents)
    if len(residuals) == 1:
        return residuals[0](cotangents)
    p, host, (carrier, signal, noise), Y, MU, Z = residuals
    gradient = jax.pure_callback(
        _host_backward,
        jax.ShapeDtypeStruct(p.shape, p.dtype),
        p,
        *(host[k] for k in BACKWARD_KEYS),
        carrier,
        signal,
        noise,
        Y,
        MU,
        Z,
        *cotangents,
        vmap_method="sequential",
    )
    return gradient, None, None


_fast.defvjp(_fast_fwd, _fast_bwd, symbolic_zeros=True)


def simulate_fast(arrays):
    """Fast `simulate` for argument dicts accepted by `supports`."""
    host = {key: arrays[key] for key in HOST_KEYS}
    others = {key: value for key, value in arrays.items() if key not in HOST_KEYS and key != "optimized_parameters"}
    return _fast(arrays["optimized_parameters"], host, others)
