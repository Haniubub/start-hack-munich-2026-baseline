"""
Coordinate (sparse) storage for the constant entries of the system matrices.

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
        """Extract the nonzero entries of a host (NumPy) matrix and put them on the device."""
        matrix = np.asarray(matrix)
        index = flat_nonzero(matrix)
        values = matrix.reshape(-1)[index].astype(np.complex128)
        return cls(jnp.asarray(index), jnp.asarray(values), tuple(matrix.shape))

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def ndim(self):
        return len(self.shape)

    def todense(self):
        """Dense JAX matrix; works on traced entries."""
        rows, columns = self.shape
        return jnp.zeros(rows * columns, self.values.dtype).at[self.index].set(self.values).reshape(self.shape)

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
