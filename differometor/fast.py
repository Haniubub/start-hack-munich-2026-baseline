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
`jax.pure_callback`s. The forward callback returns every residual the backward pass needs
as arrays that flow through JAX, so the backward callback can rebuild everything from them.
It only does so when it has to: a forward pass that is followed by a pullback keeps its
work on the host in a small cache (`FORWARD_CACHE`), and the pullback finds it through a key
that travels with the residuals.

The index arrays and base matrices from the build step are host constants
(`differometor.sparse.Constant`). They have no leaves as pytrees, so they are neither operands
of the callbacks nor residuals: the callbacks receive the constants themselves, and only the
design variables, the base parameters and the sweep values travel through JAX. Entries that
were replaced by JAX arrays travel through JAX as before.
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
from differometor.sparse import base_entries

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


def _load_sparsetools():
    """
    SciPy's compiled CSR kernels (`csr_matvecs`, `csc_matvecs`), loaded like `_load_superlu`
    without importing `scipy.sparse`. Returns None when the module cannot be found; the fast path
    then skips the pre-elimination in `_CondensedLU` and factorizes the full systems.
    """
    name = "scipy.sparse._sparsetools"
    if name in sys.modules:
        return sys.modules[name]
    try:
        root = importlib.util.find_spec("scipy").submodule_search_locations[0]
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            path = os.path.join(root, "sparse", "_sparsetools" + suffix)
            if os.path.exists(path):
                loader = importlib.machinery.ExtensionFileLoader(name, path)
                spec = importlib.util.spec_from_loader(name, loader, origin=path)
                module = importlib.util.module_from_spec(spec)
                loader.exec_module(module)
                if hasattr(module, "csr_matvecs") and hasattr(module, "csc_matvecs"):
                    return module
    except Exception:
        pass
    return None


_sparsetools = _load_sparsetools()


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
        # Per group: the entries that read its calls, as (entry, call within the group, output),
        # and the scatter from those entries back to (call, output) slots for the reverse pass.
        # Several entries may share a slot.
        self.readers, self.scatter = [], []
        for _, members, _, _ in self.groups:
            local = np.full(self.inputs.shape[0], -1)
            local[members] = np.arange(members.size)
            entries = np.nonzero(local[self.outputs[0]] >= 0)[0]
            call, output = local[self.outputs[0][entries]], self.outputs[1][entries]
            self.readers.append((entries, call, output))
            order, starts, slots = _grouping(call * 4 + output)
            self.scatter.append((entries[order], starts, slots))

    # Entries are stored entry by entry, shape (E, R): one contiguous row of R values (the sweep)
    # per matrix entry. Selecting or writing a set of entries then moves whole rows.

    def _inputs(self, source, members):
        """Function inputs of the calls `members`, shape (members, R, k), for parameters (R, P)."""
        return np.moveaxis(source.T[self.inputs[members]], 1, 2)

    def values(self, parameters):
        """parameters: (R, P) -> entries (E, R), no derivatives."""
        entries = np.empty((self.outputs.shape[1], parameters.shape[0]), complex)
        for (function_index, members, varying, _), (readers, call, output) in zip(self.groups, self.readers):
            source = parameters if varying else parameters[:1]
            outputs = _evaluate_values(function_index, self._inputs(source, members))
            entries[readers] = outputs[call, :, output]
        return entries

    def run(self, parameters):
        """parameters: (R, P). Returns entries (E, R) and the Jacobian tape for the reverse pass."""
        entries = np.empty((self.outputs.shape[1], parameters.shape[0]), complex)
        tape = []
        for (function_index, members, varying, columns), (readers, call, output) in zip(self.groups, self.readers):
            source = parameters if varying else parameters[:1]
            inputs = self._inputs(source, members)
            if columns.size:
                outputs, jacobian = _evaluate(function_index, inputs, columns)
            else:
                outputs, jacobian = _evaluate_values(function_index, inputs), None
            entries[readers] = outputs[call, :, output]
            tape.append(jacobian)
        return entries, tape

    def back(self, tape, entries_bar, parameter_count):
        """entries_bar: (E, R) -> real parameter cotangent summed over the rows (P,)."""
        rows = entries_bar.shape[1]
        parameters_bar = np.zeros(parameter_count)
        for (function_index, members, varying, columns), jacobian, (entries, starts, slots) in zip(
            self.groups, tape, self.scatter
        ):
            if jacobian is None:  # no design variable among the inputs of this group
                continue
            outputs_bar = np.zeros((members.size * 4, rows), complex)
            if slots.size:
                outputs_bar[slots] = np.add.reduceat(entries_bar[entries], starts, axis=0)
            outputs_bar = outputs_bar.reshape(members.size, 4, rows)
            if varying:
                inputs_bar = np.einsum("cor,croj->cj", outputs_bar, jacobian).real
            else:  # identical Jacobian in every row
                inputs_bar = np.einsum("co,coj->cj", outputs_bar.sum(2), jacobian[:, 0]).real
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
        return self.carrier.values(carrier_rows)[:, 0], self.signal.values(signal_rows), self.noise.values(signal_rows)

    def entries(self, optimized_parameters):
        carrier_rows, signal_rows = self._rows(optimized_parameters)
        carrier, carrier_tape = self.carrier.run(carrier_rows)
        signal, signal_tape = self.signal.run(signal_rows)
        noise, noise_tape = self.noise.run(signal_rows)
        return (carrier[:, 0], signal, noise), (carrier_tape, signal_tape, noise_tape)

    def pullback(self, tape, carrier_bar, signal_bar, noise_bar, value_count):
        carrier_tape, signal_tape, noise_tape = tape
        parameter_count = self.base.size
        signal_rows_bar = self.signal.back(signal_tape, signal_bar, parameter_count) + self.noise.back(
            noise_tape, noise_bar, parameter_count
        )
        signal_rows_bar[self.signal_changing_indices] = 0.0  # overwritten by the sweep values
        P_bar = self.carrier.back(carrier_tape, carrier_bar[:, None], parameter_count) + signal_rows_bar
        P_bar[self.carrier_changing_indices] = 0.0
        return np.bincount(self.optimized_value_indices, P_bar[self.optimized_parameter_indices], value_count)


# ---------------------------------------------------------------------
# Sparse linear systems
# ---------------------------------------------------------------------


def _last_writer(rows, columns, width):
    """Mask of scatter entries that survive `.at[rows, columns].set(...)` (last write wins)."""
    flat = rows * width + columns
    reverse_first = np.unique(flat[::-1], return_index=True)[1]
    mask = np.zeros(flat.size, bool)
    mask[flat.size - 1 - reverse_first] = True
    return mask


class SystemPlan:
    """Static sparsity data derived from the index arrays (cheap, rebuilt in every callback)."""

    def __init__(self, A):
        get = lambda key: np.asarray(A[key])
        # Base matrices arrive as sparse entries (`BaseMatrix`) or, from older callers, dense.
        carrier_index, carrier_values, (n, _) = base_entries(A["carrier_matrix"])
        signal_index, signal_values, (m, _) = base_entries(A["signal_matrix"])
        self.n, self.m = n, m
        # The carrier system is small (n x (n + 1)), so its base stays dense on the host.
        self.carrier_base = np.zeros((n, n + 1), complex)
        self.carrier_base.reshape(-1)[carrier_index] = carrier_values

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
        base_keys = key(*np.divmod(signal_index, m + 1))
        keys = np.unique(np.concatenate([base_keys, key(smi[0], smi[1]), key(placing[0], placing[1])]))
        cols, rows = np.divmod(keys, m)
        self.rows, self.cols, self.size = rows, cols, rows.size
        self.signal_base = np.zeros(keys.size, complex)
        self.signal_base[np.searchsorted(keys, base_keys)] = signal_values
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
        noise_index, noise_values, _ = base_entries(A["noise_matrix"])
        nmi = get("noise_system_matrix_indices") % m
        self.noise_live = _last_writer(nmi[0], nmi[1], m)
        constant = ~np.isin(noise_index, nmi[0] * m + nmi[1])
        base_rows, base_cols = np.divmod(noise_index[constant], m)
        self.noise_rows = np.concatenate([base_rows, nmi[0][self.noise_live]])
        self.noise_cols = np.concatenate([base_cols, nmi[1][self.noise_live]])
        self.noise_base = noise_values[constant]
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
        """Nonzero values of the augmented signal matrices, shape (size, F), plus placing pre-values."""
        V = np.empty((self.size, signal_entries.shape[1]), complex)
        V[:] = self.signal_base[:, None]
        if self.update_live.all():
            V[self.update_positions] = signal_entries
        else:
            V[self.update_positions[self.update_live]] = signal_entries[self.update_live]
        pre = V[self.placing_positions]
        live = self.placing_live
        V[self.placing_positions[live]] = -pre[live] * carrier[self.signal_carrier_indices[live], None]
        np.negative(V.imag, out=V.imag, where=self.conjugated[:, None])  # conjugate those rows in place
        return V, pre

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
        """Solver for S_f x = b and S_f^T z = b at every frequency (rows of `V`), see `_CondensedLU`."""
        return _CondensedLU(self.m, self.matrix_rows, self.matrix_cols, V, self.in_matrix)

    def signal_lu(self, values_row):
        if self.order is None:
            self._set_order(values_row)
        values = np.ascontiguousarray(values_row[self.ordered_values])
        lu = _splu(self.m, values, self.ordered_rows, self.ordered_ptr, "NATURAL", small_supernodes=True)
        return _OrderedLU(lu, self.order)

    def noise_values(self, noise_entries):
        """Noise matrix values, shape (F, entries), from the noise entries (E, F)."""
        F = noise_entries.shape[1]
        base = np.broadcast_to(self.noise_base.astype(complex), (F, self.noise_base.size))
        return np.concatenate([base, noise_entries[self.noise_live].T], axis=1)

    def detector_weights(self, carrier):
        n, s = self.n, self.selection
        w = s.astype(complex).copy()
        w[:n] = s[:n] * carrier
        w[n : 2 * n] = np.conj(s[n : 2 * n] * carrier)
        return w


class _BlockLU:
    """
    LU factorizations of F square matrices that share one sparsity pattern.

    The matrices are factorized as one block-diagonal system diag(B_0, ..., B_{F-1}) per chunk of
    frequencies, with a single SuperLU call instead of F. Pivoting never leaves a block, so the
    factors and every solution are bit-identical to F separate factorizations. What is saved is
    the fixed cost of each SuperLU call (setup, work arrays, Python round trip).
    """

    MAX_ROWS = 1 << 18  # rows per SuperLU call; bounds the size of one block-diagonal system

    def __init__(self, m, values, rows, pointers):
        F, entries = values.shape
        self.m, self.F = m, F
        step = max(1, self.MAX_ROWS // m)
        self.chunks = []
        for start in range(0, F, step):
            count = min(step, F - start)
            blocks = np.arange(count)[:, None]
            block_rows = (rows[None, :] + m * blocks).astype(np.intc).ravel()
            block_ptr = np.empty(count * m + 1, np.intc)
            block_ptr[:-1] = (pointers[None, :-1] + entries * blocks).ravel()
            block_ptr[-1] = count * entries
            block_values = values[start : start + count].ravel()
            lu = _splu(count * m, block_values, block_rows, block_ptr, "NATURAL", small_supernodes=True)
            self.chunks.append((start, count, lu))

    def solve(self, b, trans="N"):
        """Solve B_f x_f = b_f (or the transposed systems) for b of shape (F, m, k)."""
        m, k = self.m, b.shape[2]
        x = np.empty((self.F, m, k), complex)
        for start, count, lu in self.chunks:
            block = np.ascontiguousarray(b[start : start + count], dtype=complex).reshape(count * m, k)
            x[start : start + count] = lu.solve(block, trans=trans).reshape(count, m, k)
        return x


def _largest_square(values):
    """max over the frequencies of |v|^2 for entry values of shape (entries, F)."""
    parts = np.ascontiguousarray(values).view(np.float64).reshape(values.shape[0], values.shape[1], 2)
    return np.einsum("efk,efk->ef", parts, parts).max(axis=1, initial=0.0)


def _eliminable(size, rows, cols, square, one):
    """
    Mask of unknowns that can be eliminated exactly before the LU, without pivoting.

    Unknown i qualifies when the diagonal entry of row i is exactly 1 at every frequency and
    either all other entries of row i have modulus <= 1, or no other row refers to unknown i.
    Subtracting multiples of such a row from another row never enlarges that row beyond its own
    scale, so the elimination is as stable as LU with pivoting. The selected unknowns are not
    coupled to each other: their block of the matrix is the identity.

    `square` is the largest |v|^2 of every entry over the frequencies, `one` marks entries that
    are exactly 1 everywhere. Also returns the mask of entries that are nonzero somewhere.
    """
    diagonal = rows == cols
    unit = np.zeros(size, bool)
    unit[rows[diagonal]] = one[diagonal]
    live = square > 0
    off = live & ~diagonal
    r, c = rows[off], cols[off]
    row_max = np.zeros(size)
    np.maximum.at(row_max, r, square[off])
    row_count, column_count = np.bincount(r, minlength=size), np.bincount(c, minlength=size)
    undecided = unit & ((row_max <= 1 + 1e-9) | (column_count == 0))
    # Independent set, built in rounds: an undecided unknown is picked when it has the lowest
    # priority (expected fill, then index) among its undecided neighbours.
    priority = (row_count * column_count).astype(np.int64) * size + np.arange(size)
    last = np.iinfo(np.int64).max
    picked = np.zeros(size, bool)
    while undecided.any():
        p = np.where(undecided, priority, last)
        lowest = p.copy()
        np.minimum.at(lowest, r, p[c])
        np.minimum.at(lowest, c, p[r])
        new = undecided & (lowest == p)
        picked |= new
        neighbours = np.zeros(size, bool)
        neighbours[r[new[c]]] = True
        neighbours[c[new[r]]] = True
        undecided &= ~(new | neighbours)
    return picked, live


class _BlockCSR:
    """
    The sparse matrices -A_f of all frequencies as one CSR matrix over the index (f, unknown).

    `subtract(x)` computes x[rows] -= A x[cols] in place for x of shape (F, size, k), and with
    `transposed=True` x[cols] -= A^T x[rows]. Rows and columns of A must not overlap.
    """

    def __init__(self, size, rows, cols, values):
        entries, F = values.shape
        order = np.argsort(rows, kind="stable")
        count = np.bincount(rows, minlength=size)
        blocks = np.arange(F)[:, None]
        self.pointers = np.empty(F * size + 1, np.intc)
        self.pointers[:-1] = ((np.cumsum(count) - count)[None, :] + entries * blocks).ravel()
        self.pointers[-1] = entries * F
        self.indices = (cols[order][None, :] + size * blocks).astype(np.intc).ravel()
        self.data = np.ascontiguousarray(-values[order].T).ravel()
        self.total = F * size

    def subtract(self, x, transposed=False):
        flat = x.reshape(-1)
        kernel = _sparsetools.csc_matvecs if transposed else _sparsetools.csr_matvecs
        kernel(self.total, self.total, x.shape[2], self.pointers, self.indices, self.data, flat, flat)


class _Elimination:
    """
    One exact elimination step. With the unknowns ordered as (I, J) the matrix is

        S = [[1, S_IJ], [S_JI, S_JJ]],

    so x_I = b_I - S_IJ x_J and the remaining system is (S_JJ - S_JI S_IJ) x_J = b_J - S_JI b_I.
    The transposed system reduces to the transpose of the same matrix.

    Unknowns keep their numbers. The entry values of all stages live in one array `store` of
    shape (capacity, F); `ids` are the rows of `store` for the pattern (`rows`, `cols`), which is
    in CSC order. Entries of S_JJ that the step changes are updated in place, fill-in is appended.
    """

    def __init__(self, size, rows, cols, ids, picked, live, store, used):
        row_in, col_in = picked[rows], picked[cols]
        ij = np.nonzero(live & row_in & ~col_in)[0]
        ji = np.nonzero(live & ~row_in & col_in)[0]
        jj = np.nonzero(live & ~row_in & ~col_in)[0]
        self.upper = _BlockCSR(size, rows[ij], cols[ij], store[ids[ij]])  # S_IJ
        self.lower = _BlockCSR(size, rows[ji], cols[ji], store[ids[ji]])  # S_JI
        # S_JI S_IJ: every entry (r, i) meets every entry (i, j) of the same eliminated unknown i.
        by_unknown = np.argsort(rows[ij], kind="stable")
        count = np.bincount(rows[ij], minlength=size)
        start = np.cumsum(count) - count
        unknown = cols[ji]
        partners = count[unknown]
        left = np.repeat(np.arange(ji.size), partners)
        offset = np.arange(left.size) - np.repeat(np.cumsum(partners) - partners, partners)
        right = by_unknown[start[unknown[left]] + offset]
        keys, self.ids, self.changed, self.used = cols[jj] * size + rows[jj], ids[jj], ids[:0], used
        if left.size:
            order, starts, targets = _grouping(cols[ij[right]] * size + rows[ji[left]])
            at = np.minimum(np.searchsorted(keys, targets), max(keys.size - 1, 0))
            exists = keys[at] == targets if keys.size else np.zeros(targets.size, bool)
            fresh = used + np.arange(targets.size - exists.sum())
            self.used = used + fresh.size
            self.fits = self.used <= store.shape[0]
            if not self.fits:
                return
            sums = np.add.reduceat(store[ids[ji[left[order]]]] * store[ids[ij[right[order]]]], starts, axis=0)
            touched = self.ids[at[exists]]
            store[touched] -= sums[exists]
            store[fresh] = 0.0
            store[fresh] -= sums[~exists]
            self.changed = np.concatenate([touched, fresh])
            keys, self.ids = np.concatenate([keys, targets[~exists]]), np.concatenate([self.ids, fresh])
            order = np.argsort(keys, kind="stable")
            keys, self.ids = keys[order], self.ids[order]
        self.fits = True
        self.cols, self.rows = np.divmod(keys, size)


class _CondensedLU:
    """
    Solves S_f x = b and S_f^T z = b for F matrices that share one sparsity pattern.

    SuperLU spends a fixed amount of work on every column, and for systems this sparse (about
    3.5 entries per row) that fixed work dominates. Most unknowns of an interferometer signal
    matrix are simple, though: sources, read-only ports and fields that are a combination of a
    few others with coefficients of modulus <= 1. `_eliminable` picks such unknowns, and
    `_Elimination` removes them exactly with array operations over all frequencies. This repeats
    while a stage still removes a good share of the system. SuperLU (with partial pivoting, as
    before) only factorizes what is left: 200 of 751 unknowns for the benchmark setup.

    `values` has shape (entries, F) for the CSC pattern (`rows`, `cols`). With `select`, a
    boolean mask, `values` has one row per mask element and only the selected rows are entries.
    Right-hand sides and solutions have shape (F, size, k).
    """

    MAX_STAGES = 3
    MIN_SHARE = 0.25  # stop when a stage would remove less than this share of the unknowns
    ORDERING = "MMD_ATA"  # 5-10 % less SuperLU work than COLAMD on the reduced systems

    def __init__(self, size, rows, cols, values, select=None):
        entries, F = rows.size, values.shape[1]
        store = np.empty((3 * entries + 256, F), complex)  # entry values of all stages, with room for fill-in
        if select is None:
            store[:entries] = values
        else:
            np.compress(select, values, axis=0, out=store[:entries])
        ids, used = np.arange(entries), entries
        self.stages = []
        remaining = np.ones(size, bool)
        if _sparsetools is not None and self.MAX_STAGES:
            square = np.zeros(store.shape[0])
            one = np.zeros(store.shape[0], bool)
            square[:entries] = _largest_square(store[:entries])
            diagonal = np.nonzero(rows == cols)[0]
            one[diagonal] = (store[diagonal] == 1).all(axis=1)
            while len(self.stages) < self.MAX_STAGES:
                picked, live = _eliminable(size, rows, cols, square[ids], one[ids])
                if picked.sum() < self.MIN_SHARE * remaining.sum():
                    break
                stage = _Elimination(size, rows, cols, ids, picked, live, store, used)
                if not stage.fits:
                    break
                self.stages.append(stage)
                remaining &= ~picked
                rows, cols, ids, used = stage.rows, stage.cols, stage.ids, stage.used
                changed = store[stage.changed]
                square[stage.changed] = _largest_square(changed)
                one[stage.changed] = (changed == 1).all(axis=1)
        # What is left goes to SuperLU, renumbered 0..n-1. The column order comes from one
        # matrix and is reused for all frequencies (shared pattern).
        self.rest = rest = np.nonzero(remaining)[0]
        n = rest.size
        number = np.cumsum(remaining) - 1
        rows, cols = number[rows].astype(np.intc), number[cols]
        pointers = np.searchsorted(cols, np.arange(n + 1)).astype(np.intc)
        first = _splu(n, np.ascontiguousarray(store[ids, 0]), rows, pointers, self.ORDERING)
        order = np.argsort(first.perm_c)
        counts = np.diff(pointers)[order]
        ordered_ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.intc)
        gather = np.repeat(pointers[order] - ordered_ptr[:-1], counts) + np.arange(ordered_ptr[-1])
        self.lu = _BlockLU(n, np.ascontiguousarray(store[ids[gather]].T), rows[gather], ordered_ptr)
        self.ordered = rest[order]

    def solve(self, b, trans="N", overwrite=False):
        """x with S_f x_f = b_f (`trans="N"`) or S_f^T x_f = b_f (`trans="T"`).

        With `overwrite`, a contiguous complex `b` is used as the work array and returned.
        """
        transposed = trans == "T"
        if overwrite and isinstance(b, np.ndarray) and b.dtype == np.complex128 and b.flags.c_contiguous and b.flags.writeable:
            x = b
        else:
            x = np.array(b, dtype=complex, order="C")
        for stage in self.stages:  # right-hand side of the remaining system
            (stage.upper if transposed else stage.lower).subtract(x, transposed)
        if transposed:
            x[:, self.rest] = self.lu.solve(x[:, self.ordered], trans="T")
        else:
            x[:, self.ordered] = self.lu.solve(x[:, self.rest])
        for stage in reversed(self.stages):  # eliminated unknowns from the ones solved for
            (stage.lower if transposed else stage.upper).subtract(x, transposed)
        return x


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
    """Carrier (n,), signal (m, F) and noise (F,) outputs plus the adjoint helpers Y = S^-H w, MU and Z.

    Z[f] = S_f^-T [e_d for d in plan.detector] (zero-padded to DETECTOR_WIDTH columns). It gives
    Y = conj(Z conj(w_d)) here and the signal adjoint S^-T signal_bar in `backward` without a
    second factorization, whenever the cotangent of the signal lives on the detector rows.

    The last return value (carrier factorization, signal matrix values, placing pre-values) is
    what `backward` would otherwise compute a second time.

    Per frequency this takes one factorization and two solves with two right-hand sides each:
    first S^T Z = [e_d], then S [X, MU] = [rhs, (N + N^H) Y / 2]. Simple unknowns are eliminated
    exactly beforehand and the rest goes through SuperLU as one block-diagonal system for all
    frequencies (`_CondensedLU`). Everything between the two solves is evaluated for all
    frequencies at once.
    """
    m = plan.m
    carrier_lu, carrier_rhs = plan.carrier_matrix(carrier_entries)
    carrier = carrier_lu.solve(carrier_rhs)
    V, pre = plan.signal_values(signal_entries, carrier)
    w = plan.detector_weights(carrier)
    N = plan.noise_values(noise_entries)
    F = V.shape[1]
    factors = plan.signal_factors(V)
    d, k = plan.detector, plan.detector.size
    if plan.use_basis:
        basis = np.zeros((m, k), complex)
        basis[d, np.arange(k)] = 1.0
        Z = factors.solve(np.broadcast_to(basis, (F, m, k)), trans="T")
        Y = np.conj(Z @ np.conj(w[d]))
        if k < DETECTOR_WIDTH:
            Z = np.concatenate([Z, np.zeros((F, m, DETECTOR_WIDTH - k), complex)], axis=2)
    else:
        Z = np.zeros((F, m, DETECTOR_WIDTH), complex)
        # S^-H w = conj(S^-T conj(w))
        Y = np.conj(factors.solve(np.broadcast_to(np.conj(w)[:, None], (F, m, 1)), trans="T"))[:, :, 0]
    q = _scatter_sum(N * Y[:, plan.noise_cols], plan.noise_by_row, m)  # N y
    qh = _scatter_sum(np.conj(N) * Y[:, plan.noise_rows], plan.noise_by_col, m)  # N^H y
    power = np.einsum("fm,fm->f", np.conj(Y), q).real
    right_hand_sides = np.zeros((F, m, 2), complex)
    right_hand_sides[:, plan.rhs_rows, 0] = V[~plan.in_matrix].T
    right_hand_sides[:, :, 1] = 0.5 * (q + qh)
    solutions = factors.solve(right_hand_sides, overwrite=True)
    signal = np.ascontiguousarray(solutions[:, :, 0].T)
    MU = np.ascontiguousarray(solutions[:, :, 1])
    noise = np.sqrt(2 * power * NOISE_SCALE)
    return carrier, signal, noise, Y, MU, Z, (carrier_lu, V, pre)


def _signal_adjoint(plan, V, Z, signal_bar):
    """lam_f = S_f^-T signal_bar_f for every frequency; `signal_bar` and the result have shape (m, F)."""
    support = np.nonzero(np.any(signal_bar != 0, axis=1))[0]
    d, k = plan.detector, plan.detector.size
    if plan.use_basis and np.isin(support, d).all():
        return np.einsum("fmk,kf->mf", Z[:, :, :k], signal_bar[d])
    lam = np.empty(signal_bar.shape, complex)
    for f in range(V.shape[1]):
        lam[:, f] = plan.signal_lu(V[:, f]).solve(signal_bar[:, f].astype(complex), trans="T")
    return lam


def backward(
    plan, carrier_entries, signal_entries, carrier, signal, noise, Y, MU, Z, carrier_bar, signal_bar, noise_bar, kept=None
):
    """Cotangents of the carrier, signal and noise entries (JAX's bilinear convention).

    `signal` and `signal_bar` have shape (m, F), the adjoint helpers Y and MU (F, m). `kept` is the
    last return value of `forward` for the same inputs, when it is still available. The work on
    the signal matrix entries runs entry by entry, on arrays of shape (entries, F).
    """
    n, m = plan.n, plan.m
    if kept is None:
        carrier_lu, _ = plan.carrier_matrix(carrier_entries)
        V, pre = plan.signal_values(signal_entries, carrier)
    else:
        carrier_lu, V, pre = kept
    F = V.shape[1]
    carrier_bar = carrier_bar.astype(complex).copy()
    V_bar = np.empty((plan.size, F), complex)
    g = noise_bar * NOISE_SCALE / noise  # d noise / d power
    rows, cols = plan.rows[plan.in_matrix], plan.cols[plan.in_matrix]
    lam = _signal_adjoint(plan, V, Z, signal_bar)
    Y_rows, MU_rows = np.ascontiguousarray(Y.T), np.ascontiguousarray(MU.T)
    # V_bar = -lam[rows] * signal[cols] - 2 g conj(Y[rows]) * MU[cols] on the matrix entries,
    # built in place: every temporary here has 2 500 x F elements.
    entries = rows.size
    matrix_bar = V_bar[:entries] if plan.in_matrix[:entries].all() else np.empty((entries, F), complex)
    np.take(lam, rows, axis=0, out=matrix_bar, mode="clip")  # "clip" writes straight into `out`
    np.negative(matrix_bar, out=matrix_bar)
    matrix_bar *= signal[cols]
    noise_part = np.conjugate(Y_rows[rows])
    noise_part *= 2 * g
    noise_part *= MU_rows[cols]
    matrix_bar -= noise_part
    if matrix_bar.base is not V_bar:
        V_bar[plan.in_matrix] = matrix_bar
    V_bar[~plan.in_matrix] = lam[plan.rhs_rows]
    w_bar = 2 * (g[:, None] * np.conj(MU)).sum(0)

    N_bar = g[:, None] * np.conj(Y[:, plan.noise_rows]) * Y[:, plan.noise_cols]
    noise_entries_bar = np.zeros((plan.noise_live.size, F), complex)
    noise_entries_bar[plan.noise_live] = N_bar[:, plan.noise_base.size :].T

    np.negative(V_bar.imag, out=V_bar.imag, where=plan.conjugated[:, None])  # conjugate those rows in place
    live = plan.placing_live
    placed = plan.placing_positions[live]
    sci = plan.signal_carrier_indices[live]
    placed_bar = V_bar[placed]
    np.add.at(carrier_bar, sci, -(placed_bar * (pre if live.all() else pre[live])).sum(1))
    np.negative(placed_bar, out=placed_bar)
    placed_bar *= carrier[sci, None]
    V_bar[placed] = placed_bar
    signal_entries_bar = V_bar[plan.update_positions]
    signal_entries_bar[~plan.update_live] = 0.0

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


# Work of the forward pass that the backward pass can reuse: both plans, the matrix entries
# with their Jacobian tape, the signal matrix values, the carrier factorization, and the results
# with the adjoint helpers Y, MU and Z. With the cache on, a forward pass that is followed by a
# pullback leaves all of this on the host and hands JAX only the outputs and a key; the pullback
# finds its forward pass through the key. Nothing depends on the entry still being there: when
# it is gone the backward callback runs the forward pass again from its own inputs.
#
# With `FORWARD_CACHE = False` (read when a function is traced) the callbacks keep no state:
# Y, MU and Z flow through JAX as residuals and the backward callback rebuilds the rest.
FORWARD_CACHE = True
_FORWARD_CACHE_SIZE = 2
_forward_cache = {}


def _run_forward(optimized_parameters, host_arrays, taped):
    """The forward pass on the host: results (carrier, signal, noise, Y, MU, Z) and the reusable work."""
    A = dict(zip(HOST_KEYS, host_arrays))
    entry_plan, plan = EntryPlan(A), SystemPlan(A)
    if taped:  # a pullback will follow: evaluate the entries with derivatives once, here
        entries, tape = entry_plan.entries(optimized_parameters)
    else:
        entries, tape = entry_plan.values(optimized_parameters), None
    *results, kept = forward(plan, *entries)
    return tuple(results), (entry_plan, plan, entries, tape, kept)


def _run_backward(optimized_parameters, results, work, carrier_bar, signal_bar, noise_bar):
    """The backward pass on the host, from the results and the work of the forward pass."""
    carrier, signal, noise, Y, MU, Z = results
    entry_plan, plan, entries, tape, kept = work
    F = Y.shape[0]
    entries_bar = backward(
        plan,
        entries[0],
        entries[1],
        carrier,
        signal,
        noise,
        Y,
        MU,
        Z,
        np.asarray(carrier_bar)[:, 0],
        np.asarray(signal_bar),
        np.asarray(noise_bar, float).reshape(F),
        kept=kept,
    )
    gradient = entry_plan.pullback(tape, *entries_bar, value_count=optimized_parameters.shape[0])
    return gradient.astype(optimized_parameters.dtype)


def _host_forward(optimized_parameters, *host_arrays):
    """Stateless forward callback: the outputs and the adjoint helpers Y, MU, Z."""
    (carrier, signal, noise, Y, MU, Z), _ = _run_forward(optimized_parameters, host_arrays, taped=False)
    return carrier[:, None], signal, noise.reshape(-1), Y, MU, Z


def _host_forward_plain(optimized_parameters, *host_arrays):
    """Forward callback when no pullback follows: only the outputs."""
    return _host_forward(optimized_parameters, *host_arrays)[:3]


def _host_forward_cached(optimized_parameters, *host_arrays):
    """Forward callback in front of a pullback: the outputs and the key of the cache entry."""
    results, work = _run_forward(optimized_parameters, host_arrays, taped=True)
    token = int.from_bytes(os.urandom(6), "little") + 1  # exact in float64, never reused in practice
    while len(_forward_cache) >= _FORWARD_CACHE_SIZE:
        del _forward_cache[next(iter(_forward_cache))]
    _forward_cache[token] = (np.array(optimized_parameters, float), results, work)
    carrier, signal, noise = results[:3]
    return carrier[:, None], signal, noise.reshape(-1), np.float64(token)


def _host_backward(optimized_parameters, *rest):
    """Stateless backward callback: everything comes from the residuals."""
    k = len(HOST_KEYS)
    carrier, signal, noise, Y, MU, Z, carrier_bar, signal_bar, noise_bar = rest[k:]
    A = dict(zip(HOST_KEYS, rest[:k]))
    entry_plan, plan = EntryPlan(A), SystemPlan(A)
    entries, tape = entry_plan.entries(optimized_parameters)
    results = (np.asarray(carrier)[:, 0], np.asarray(signal), np.asarray(noise).reshape(-1), np.asarray(Y), np.asarray(MU), np.asarray(Z))
    return _run_backward(optimized_parameters, results, (entry_plan, plan, entries, tape, None), carrier_bar, signal_bar, noise_bar)


def _host_backward_cached(optimized_parameters, *rest):
    """Backward callback for `_host_forward_cached`: reuses the forward pass, or repeats it if it is gone."""
    k = len(HOST_KEYS)
    token, carrier_bar, signal_bar, noise_bar = rest[k:]
    cached = _forward_cache.get(int(token))
    if cached is not None and np.array_equal(cached[0], optimized_parameters):
        _, results, work = cached
    else:
        results, work = _run_forward(optimized_parameters, rest[:k], taped=True)
    return _run_backward(optimized_parameters, results, work, carrier_bar, signal_bar, noise_bar)


def _result_shapes(host):
    """Shapes of the forward outputs and of the adjoint helpers, plus the shape of `noise`."""
    n, m, F, noise_shape = _output_shapes(host)
    outputs = (
        jax.ShapeDtypeStruct((n, 1), jnp.complex128),
        jax.ShapeDtypeStruct((m, F), jnp.complex128),
        jax.ShapeDtypeStruct((F,), jnp.float64),
    )
    helpers = (
        jax.ShapeDtypeStruct((F, m), jnp.complex128),
        jax.ShapeDtypeStruct((F, m), jnp.complex128),
        jax.ShapeDtypeStruct((F, m, DETECTOR_WIDTH), jnp.complex128),
    )
    return outputs, helpers, noise_shape


def _call_host(function, shapes, optimized_parameters, host, *more):
    return jax.pure_callback(
        function, shapes, optimized_parameters, *(host[k] for k in HOST_KEYS), *more, vmap_method="sequential"
    )


def _is_primal(x):
    return isinstance(x, jax.custom_derivatives.CustomVJPPrimal)


def _zeros_if_symbolic(value):
    if isinstance(value, jax.custom_derivatives.SymbolicZero):
        return jnp.zeros(value.shape, value.dtype)
    return value


@jax.custom_vjp
def _fast(optimized_parameters, host, others):
    outputs, _, noise_shape = _result_shapes(host)
    carrier, signal, noise = _call_host(_host_forward_plain, outputs, optimized_parameters, host)
    return carrier, signal, noise.reshape(noise_shape)


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
    outputs, helpers, noise_shape = _result_shapes(host)
    # Under `vmap` all forward passes of a batch run before the first pullback, so the small
    # cache would only serve the last ones. Batched calls therefore take the stateless route.
    # (The tracer class has no public name any more; if this check stops matching, results stay
    # correct and a batch only gets slower.)
    batched = any(type(leaf).__name__ == "BatchTracer" for leaf in jax.tree_util.tree_leaves((p, host)))
    if FORWARD_CACHE and not batched:
        key = jax.ShapeDtypeStruct((), jnp.float64)
        carrier, signal, noise, token = _call_host(_host_forward_cached, outputs + (key,), p, host)
        return (carrier, signal, noise.reshape(noise_shape)), (p, host, token)
    carrier, signal, noise, Y, MU, Z = _call_host(_host_forward, outputs + helpers, p, host)
    return (carrier, signal, noise.reshape(noise_shape)), (p, host, carrier, signal, noise, Y, MU, Z)


def _fast_bwd(residuals, cotangents):
    cotangents = tuple(_zeros_if_symbolic(c) for c in cotangents)
    if len(residuals) == 1:
        return residuals[0](cotangents)
    p, host, *rest = residuals
    function = _host_backward_cached if len(rest) == 1 else _host_backward
    gradient = _call_host(function, jax.ShapeDtypeStruct(p.shape, p.dtype), p, host, *rest, *cotangents)
    return gradient, None, None


_fast.defvjp(_fast_fwd, _fast_bwd, symbolic_zeros=True)


def simulate_fast(arrays):
    """Fast `simulate` for argument dicts accepted by `supports`."""
    host = {key: arrays[key] for key in HOST_KEYS}
    others = {key: value for key, value in arrays.items() if key not in HOST_KEYS and key != "optimized_parameters"}
    return _fast(arrays["optimized_parameters"], host, others)
