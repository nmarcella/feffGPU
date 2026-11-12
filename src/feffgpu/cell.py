"""Periodic cells for the pair, path and RMC code: the minimum-image convention for a cubic box (a number), a general
cell (3 x 3, rows the lattice vectors, any angles) or no periodicity (None). Every class that takes a `box` accepts
all three; the general cell's minimum image is the fractional-coordinate rounding, exact while the distances of
interest stay below half the cell's smallest perpendicular width (checked against the path cutoff where the cell is
set)."""
from __future__ import annotations
import numpy as np
import torch


def as_cell(box, device=None):
    """None -> None; a number -> float; (3,) -> the diagonal cell; (3, 3) -> the cell as a float32 tensor."""
    if box is None:
        return None
    if torch.is_tensor(box):
        if box.dim() == 0:
            return float(box)
        c = box.to(device=device, dtype=torch.float32)
    else:
        if np.isscalar(box):
            return float(box)
        c = torch.as_tensor(np.asarray(box, float), device=device, dtype=torch.float32)
    return torch.diag(c) if c.dim() == 1 else c


def wrap(rel, box):
    """Minimum image of displacement vectors rel (..., 3): rel itself for None, the cubic rule for a number,
    fractional rounding for a cell."""
    if box is None:
        return rel
    if isinstance(box, float):
        return rel - box * torch.round(rel / box)
    cell = box.to(device=rel.device, dtype=rel.dtype)
    f = rel @ torch.linalg.inv(cell)
    return (f - torch.round(f)) @ cell


def min_width(box):
    """The smallest perpendicular width of the cell (A), the length for a cubic box, inf for None."""
    if box is None:
        return float("inf")
    if isinstance(box, float):
        return box
    c = np.asarray(box.detach().cpu().numpy(), float)
    vol = abs(np.linalg.det(c))
    return float(min(vol / np.linalg.norm(np.cross(c[(i + 1) % 3], c[(i + 2) % 3])) for i in range(3)))


def volume(box, n_default=None):
    """Volume (A^3) of the cell, None for a cluster."""
    if box is None:
        return None
    if isinstance(box, float):
        return box ** 3
    return float(abs(np.linalg.det(box.detach().cpu().numpy())))
