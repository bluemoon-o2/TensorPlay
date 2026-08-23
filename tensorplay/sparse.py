"""Sparse tensor namespace, mirroring ``torch.sparse``.

Provides the COO/CSR operations under a dedicated namespace so user code can
read ``tensorplay.sparse.mm(s, d)`` the same way it reads ``torch.sparse.mm``.
"""

import tensorplay
from tensorplay._C import sparse_coo_tensor as sparse_coo_tensor

__all__ = [
    "coalesce",
    "sparse_coo_tensor",
    "sparse_mask",
    "mm",
    "sum",
    "to_dense",
    "to_sparse",
    "to_sparse_csr",
]


def coalesce(input):
    """Returns a coalesced copy of the sparse COO tensor ``input``."""
    return input.coalesce()


def sparse_mask(input, mask):
    """Returns a new sparse tensor with values of ``input`` at ``mask``'s indices."""
    return input.sparse_mask(mask)


def mm(sparse, dense):
    """Performs a matrix multiplication of a 2-D sparse COO/CSR tensor with a
    dense matrix.  Equivalent to ``torch.sparse.mm``."""
    return tensorplay.sparse_mm(sparse, dense)


def sum(input):
    """Returns the sum of all values of ``input`` as a 0-dim dense tensor."""
    return tensorplay.sparse_sum(input)


def to_dense(input):
    """Materializes a sparse COO/CSR tensor into a dense tensor."""
    return tensorplay.to_dense(input)


def to_sparse(input):
    """Converts a dense tensor into a coalesced sparse COO tensor."""
    return tensorplay.to_sparse(input)


def to_sparse_csr(input):
    """Converts a 2-D dense tensor into a sparse CSR tensor."""
    return tensorplay.to_sparse_csr(input)
