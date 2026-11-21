"""
Host-side storage for the build outputs that describe the structure of a setup.

`Constant`: index arrays and other fixed arrays
-----------------------------------------------
Most arrays the build step returns never change for a given setup: the index arrays that say
which component function writes which matrix entry, the selection vectors of the detectors, the
constant matrix entries. As JAX arrays they were runtime inputs of every jitted function that
takes the build output as an argument (`benchmark.py` and `start_here.py` do): about thirty
arrays that are passed into the compiled program, copied into each host callback of the fast
path, returned again as residuals of the pullback and passed into the gradient program. On x86
Linux this bookkeeping costs about 20 ms of a 0.9 s first run.

A `Constant` is a read-only NumPy array that JAX treats like a Python constant: it is registered
as a pytree node without children, so a jitted function that receives one is specialized on its
content instead of taking it as a runtime input. Two constants are interchangeable when shape,
dtype and bytes agree, so rebuilding the same setup does not retrace, and a different setup does.
This is what already happens when the arrays are closed over (as in the examples).

What stays a runtime JAX array: `optimized_parameters`, the base `parameters` and the sweep
values (`carrier_changing_values`, `signal_changing_values`). Those are data, and they can still be
replaced, batched or differentiated. A constant cannot: `jax.grad` and `jax.vmap` see no array in
it. To do that with one of them, pass a JAX array instead: `jnp.asarray(np.asarray(constant))`
(`jnp.asarray(constant)` alone returns the constant unchanged). Every entry of the build output
may be replaced by a JAX array of the same shape, and `STATIC = False` makes the build step
return JAX arrays throughout.

Arrays derived from a constant (slices, arithmetic, copies) are ordinary arrays again.

`BaseMatrix`: constant matrix entries
-------------------------------------
The carrier, signal and noise base matrices produced by the build step are more than 99 % zeros
(the benchmark signal matrix has about 2 700 entries in 751 x 752). Kept dense, the three of them
are 20 MB that get copied again and again: once to the device in the build step, once into every
host callback of the fast path, and once more out of the forward pass, because the pullback keeps
them as residuals. On x86 Linux these copies cost about 50 ms of a 1 s first run.

`BaseMatrix` stores only the nonzero entries. It is registered as a JAX pytree, so it passes
through `jax.jit`, `jax.vjp` and `jax.pure_callback` wherever the dense array did. The fast path
reads the entries directly, the original JAX path (`simulate_jax`) rebuilds the dense matrix with
`dense_matrix`. `np.asarray(base_matrix)` gives the dense matrix as well.
"""

import jax
import jax.numpy as jnp
import numpy as np

STATIC = True
"""Whether the build step returns structure arrays as `Constant` (True) or as JAX arrays (False)."""


class _Content:
    """Pytree metadata of a `Constant`: compares equal for the same shape, dtype and bytes."""

    __slots__ = ("array", "_hash")

    def __init__(self, array):
        self.array, self._hash = array, None

    def __hash__(self):
        if self._hash is None:
            array = self.array
            self._hash = hash((array.shape, array.dtype.str, array.tobytes()))
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, _Content):
            return NotImplemented
        mine, theirs = self.array, other.array
        return mine is theirs or (
            mine.shape == theirs.shape
            and mine.dtype == theirs.dtype
            and hash(self) == hash(other)
            and mine.tobytes() == theirs.tobytes()
        )


class Constant(np.ndarray):
    """Read-only host array that JAX treats as a constant (a pytree node without leaves).

    Create one with `constant(values)`. Only arrays created that way are constants: views,
    copies and results of NumPy operations carry no content key and are ordinary arrays.
    """

    def __array_finalize__(self, source):
        self._content = None

    def __jax_array__(self):
        return jnp.asarray(np.asarray(self))

    @property
    def at(self):
        """`constant.at[...]` as for the JAX array with the same content."""
        return self.__jax_array__().at

    def block_until_ready(self):
        return self


def constant(values, dtype=None):
    """A `Constant` holding a private, read-only copy of `values`."""
    array = np.array(values, dtype=dtype).view(Constant)
    array.flags.writeable = False
    array._content = _Content(array)
    return array


def _constant_flatten(array):
    content = getattr(array, "_content", None)
    if content is None:  # derived from a constant: an ordinary array leaf
        return (np.asarray(array),), None
    return (), content


jax.tree_util.register_pytree_node(
    Constant, _constant_flatten, lambda content, children: children[0] if content is None else content.array
)


def structure_array(values, dtype=None):
    """A build output that describes structure: a `Constant`, or a JAX array when `STATIC` is off.

    The conversion happens on the host (`jnp.array(..., dtype=...)` would compile a conversion).
    """
    if STATIC:
        return constant(values, dtype)
    return jnp.asarray(np.asarray(values, dtype=dtype))


def device_array(value):
    """The JAX array for a `Constant`; anything else passes through unchanged."""
    return jnp.asarray(np.asarray(value)) if isinstance(value, Constant) else value


def flat_nonzero(matrix):
    """Row-major flat indices of the nonzero entries (for complex input faster than `np.flatnonzero`)."""
    matrix = np.ascontiguousarray(matrix)
    if matrix.dtype == np.complex128:
        parts = matrix.view(np.float64).reshape(-1, 2) != 0
        return np.flatnonzero(parts[:, 0] | parts[:, 1])
    return np.flatnonzero(matrix)


class BaseMatrix:
    """Nonzero entries of a matrix: sorted row-major flat `index`, complex `values`, dense `shape`."""

    __slots__ = ("index", "values", "shape")

    def __init__(self, index, values, shape):
        self.index, self.values, self.shape = index, values, shape

    @classmethod
    def from_dense(cls, matrix):
        """Extract the nonzero entries of a host (NumPy) matrix."""
        matrix = np.asarray(matrix)
        index = flat_nonzero(matrix)
        values = matrix.reshape(-1)[index].astype(np.complex128)
        return cls(structure_array(index), structure_array(values), tuple(matrix.shape))

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def ndim(self):
        return len(self.shape)

    def todense(self):
        """Dense JAX matrix; works on traced entries."""
        rows, columns = self.shape
        index, values = device_array(self.index), device_array(self.values)
        return jnp.zeros(rows * columns, values.dtype).at[index].set(values).reshape(self.shape)

    def __array__(self, dtype=None, copy=None):
        values = np.asarray(self.values)
        dense = np.zeros(self.shape, values.dtype)
        dense.reshape(-1)[np.asarray(self.index)] = values
        return dense if dtype is None else dense.astype(dtype)

    def __repr__(self):
        return f"BaseMatrix(shape={self.shape}, entries={getattr(self.index, 'shape', '?')})"


jax.tree_util.register_pytree_node(
    BaseMatrix,
    lambda matrix: ((matrix.index, matrix.values), matrix.shape),
    lambda shape, children: BaseMatrix(children[0], children[1], shape),
)


def dense_matrix(matrix):
    """The dense JAX matrix for a `BaseMatrix`; dense arrays pass through unchanged."""
    return matrix.todense() if isinstance(matrix, BaseMatrix) else matrix


def base_entries(matrix):
    """Host view `(flat index, values, shape)` of a `BaseMatrix` or of a dense matrix."""
    if isinstance(matrix, BaseMatrix):
        return np.asarray(matrix.index), np.asarray(matrix.values), tuple(matrix.shape)
    matrix = np.asarray(matrix)
    index = flat_nonzero(matrix)
    return index, matrix.reshape(-1)[index], tuple(matrix.shape)
