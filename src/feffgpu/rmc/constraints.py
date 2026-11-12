"""The limits of the search: the displacement limit (EvAX's box) and the closest distances between elements.

``Box``
    Every atom stays within ``max_displacement`` of its site of the fixed start lattice (``feffgpu.rmc.structure``):
    EvAX's ``Potential_type box`` with ``Maximal_displacement``. The lattice is never refitted, so a shift of all atoms
    together counts as a displacement of every atom. After an exchange or a hop the atom is held to its new site. Two
    ways to apply it: a projection (a displacement longer than the limit is scaled back to the limit along its
    direction; needed with gradient steps) and EvAX's rule (a random move that takes an atom outside is halved towards
    its start until it does not, at most 100 times, then rejected; the search implements the halving).
``ClosestDistances``
    For element pairs, the shortest distance that exists in the chemistry. EvAX rejects a move that brings a pair
    closer (``Potential_type hardsphere``). During gradient steps the limits act as harmonic springs (stiffness
    ``spring``, eV/A^2) below the limit; bred children and trial states with a pair below a limit (by more than a
    tolerance) are refused by the search.

Without limits, a reverse Monte Carlo model develops pairs at distances that do not exist in any real material; the
analysis tools report them (``feffgpu.analysis.short_pairs``).
"""
from __future__ import annotations
import numpy as np
import torch
from ..cell import as_cell, wrap

__all__ = ["Box", "ClosestDistances", "parse_pair_table", "nearest_sites"]


def nearest_sites(sites, cell, pos, site_of):
    """(S, n, 3): the site of every atom, the periodic image nearest to the atom.

    Parameters
    ----------
    sites : torch.Tensor, shape (m, 3)
        Site positions (A).
    cell : None, float or torch.Tensor (3, 3)
        As returned by ``feffgpu.cell.as_cell``.
    pos : torch.Tensor, shape (S, n, 3)
    site_of : torch.Tensor of int, shape (S, n)
    """
    s = sites[site_of]
    if cell is None:
        return s
    d = pos - s
    if isinstance(cell, float):
        return s + torch.round(d / cell) * cell                                              # an exact multiple of the cell (0 for most atoms)
    c = cell.to(device=pos.device, dtype=pos.dtype)
    return s + torch.round(d @ torch.linalg.inv(c)) @ c                                      # an integer combination of the lattice vectors


class Box:
    """The displacement limit from the sites of the fixed start lattice.

    Parameters
    ----------
    site_positions : array, shape (m, 3)
        The site table's positions (A).
    max_displacement : float
        The limit (A), > 0.
    cell : None, float or (3, 3)
        The periodic cell; displacements use the minimum image.
    device : str
    """

    def __init__(self, site_positions, max_displacement, cell=None, device="cuda"):
        if not max_displacement > 0:
            raise ValueError("the displacement limit must be positive (0 means no limit: do not build a Box)")
        self.device = torch.device(device)
        self.sites = torch.as_tensor(np.asarray(site_positions, float), device=self.device, dtype=torch.float32)
        self.r = float(max_displacement)
        self.cell = as_cell(cell, self.device)

    def site_positions(self, pos, site_of):
        """(S, n, 3): the site of every atom of the configurations ``pos`` (S, n, 3) with sites ``site_of`` (S, n), the
        periodic image nearest to the atom."""
        return nearest_sites(self.sites, self.cell, pos, site_of)

    def displacement(self, pos, site_of):
        """(S, n, 3): every atom's displacement from its site (A)."""
        return pos - self.site_positions(pos, site_of)

    def project(self, pos, site_of):
        """``pos`` (S, n, 3) with every displacement longer than the limit scaled back to the limit."""
        site = self.site_positions(pos, site_of)
        d = pos - site
        r = d.norm(dim=-1, keepdim=True)
        return torch.where(r > self.r, site + d * (self.r / r.clamp(min=1e-12)), pos)          # atoms inside the limit are not touched

    def outside(self, pos, site_of, tol=1e-4):
        """(S,) bool: a configuration with an atom farther than the limit from its site (by more than ``tol`` A)."""
        return (self.displacement(pos, site_of).norm(dim=-1) > self.r + tol).any(-1)

    @torch.no_grad()
    def at_limit(self, pos, site_of, tol=1e-3):
        """The share of atoms within ``tol`` (A) of the limit and the largest displacement (A)."""
        d = self.displacement(pos, site_of).norm(dim=-1)
        return float((d >= self.r - tol).float().mean()), float(d.max())


def parse_pair_table(table, elements):
    """``{"A-B": distance}`` -> a symmetric (n_el, n_el) numpy matrix over ``elements`` (sorted names); pairs not given
    are 0 (no limit). Pairs of an element the structure does not have are skipped."""
    n = len(elements)
    out = np.zeros((n, n))
    for key, val in (table or {}).items():
        parts = str(key).split("-")
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            raise ValueError(f"closest distance {key!r}: write the pair as 'A-B' with two element symbols")
        a, b = parts[0].strip(), parts[1].strip()
        if float(val) < 0:
            raise ValueError(f"closest distance {key}: must not be negative (got {val})")
        if a not in elements or b not in elements:
            continue
        i, j = elements.index(a), elements.index(b)
        out[i, j] = out[j, i] = float(val)
    return out


class ClosestDistances:
    """Closest distances between elements: springs for the gradient steps and the test that refuses a configuration.

    Parameters
    ----------
    symbols : array of str, shape (n,)
    table : dict
        ``{"A-B": distance}`` (A).
    cell : None, float or (3, 3)
    spring : float
        Stiffness of the springs below the limits (eV/A^2): energy ``0.5 spring (d_min - r)^2`` per pair closer than
        its limit.
    device : str
    """

    def __init__(self, symbols, table, cell=None, spring=50.0, device="cuda"):
        self.device = torch.device(device)
        symbols = np.asarray(symbols).astype(str)
        self.elements = sorted(set(symbols.tolist()))
        self.matrix = parse_pair_table(table, self.elements)
        code = np.array([self.elements.index(s) for s in symbols])
        iu = np.triu_indices(len(symbols), 1)
        rm = self.matrix[code[iu[0]], code[iu[1]]]
        keep = rm > 0
        self.pi = torch.as_tensor(iu[0][keep], device=self.device, dtype=torch.long)
        self.pj = torch.as_tensor(iu[1][keep], device=self.device, dtype=torch.long)
        self.rm = torch.as_tensor(rm[keep], device=self.device, dtype=torch.float32)
        self.cell = as_cell(cell, self.device)
        self.spring = float(spring)

    @property
    def active(self):
        """True if at least one pair of atoms has a limit."""
        return int(self.rm.numel()) > 0

    def as_dict(self):
        """``{"A-B": distance}`` of the limits in force for this structure (A)."""
        return {f"{a}-{b}": float(self.matrix[i, j]) for i, a in enumerate(self.elements) for j, b in enumerate(self.elements) if j >= i and self.matrix[i, j] > 0}

    def _depth(self, p, chunk=16):
        """(S, pairs): how far every limited pair lies below its limit (0 above it), in chunks of configurations."""
        out = []
        for s in range(0, p.shape[0], chunk):
            q = p[s:s + chunk]
            r = wrap(q[:, self.pi] - q[:, self.pj], self.cell).norm(dim=-1)
            out.append((self.rm[None] - r).clamp(min=0))
        return torch.cat(out) if len(out) > 1 else out[0]

    def energy(self, p):
        """The spring energy (S,) (eV) of configurations ``p`` (S, n, 3) and the number of pairs below a limit."""
        if not self.active:
            return torch.zeros(p.shape[0], device=p.device), 0
        viol = self._depth(p)
        return 0.5 * self.spring * (viol ** 2).sum(1), int((viol > 0).sum())

    def worst(self, p):
        """(S,): how far (A) the closest pair of every configuration lies below its limit (0: none below)."""
        if not self.active:
            return torch.zeros(p.shape[0], device=p.device)
        return self._depth(p).amax(1)

    @torch.no_grad()
    def below(self, p):
        """The number of pairs below their limit in configurations ``p`` (S, n, 3), summed, and the largest depth (A)."""
        if not self.active:
            return 0, 0.0
        viol = self._depth(p)
        return int((viol > 0).sum()), float(viol.max()) if viol.numel() else 0.0
