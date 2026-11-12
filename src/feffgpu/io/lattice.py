"""Ideal lattice blocks for examples and tests."""
import numpy as np


def bcc_block(L, a):
    """Sites (2 L^3, 3) of an L x L x L conventional bcc block, atom 0 at the origin; returns (sites, box = L a)."""
    c = np.stack(np.meshgrid(*[np.arange(L)] * 3, indexing="ij"), -1).reshape(-1, 3)
    return np.concatenate([2 * c, 2 * c + 1]) * (a / 2), L * a


def fcc_block(L, a):
    """Sites (4 L^3, 3) of an L x L x L conventional fcc block; returns (sites, box = L a)."""
    c = np.stack(np.meshgrid(*[np.arange(L)] * 3, indexing="ij"), -1).reshape(-1, 3).astype(float)
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    return (c[:, None, :] + basis[None]).reshape(-1, 3) * a, L * a
