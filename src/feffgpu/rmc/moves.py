"""The moves of the search, all batched over the states of the population.

- ``start_population``: the start states, the start structure plus a small random displacement of every atom.
- ``uniform_step``: EvAX's random move (``Move_type move_all``): every coordinate of every atom moves by a uniform
  random step within +/- ``step`` (EvAX's ``Maximal_step_length``).
- ``exchange_atoms``: an atom of one element and an atom of another element exchange their positions and their sites
  (an addition to EvAX: a rearrangement of two species over their sites, on any structure).
- ``hop_atoms``: an atom moves onto a vacant site of its hop sublattice, from a neighboring site (an addition to
  EvAX).
- ``fire_relax`` with ``GradientObjective``: gradient steps of the FIRE minimizer (Bitzek et al., Phys. Rev. Lett.
  97, 170201 (2006)) on the misfit of every state (an addition to EvAX: EvAX moves atoms only at random).

Positions are tensors (P, C, n, 3): P states of C cells of n atoms. ``site_of`` (P, C, n) is the site of every atom
(``feffgpu.rmc.structure``). The random numbers come from one torch generator per sample, in a fixed order, so that a
run is reproduced by its seed.
"""
from __future__ import annotations
import math
import numpy as np
import torch

__all__ = ["start_population", "uniform_step", "exchange_atoms", "hop_atoms", "fire_relax", "GradientObjective"]


def start_population(base, states, kick, movable, gen, distribution="gaussian"):
    """The start states: ``states`` copies of ``base`` (C, n, 3) with a random displacement of every movable atom.

    ``distribution`` ``"gaussian"``: a normal displacement of standard deviation ``kick`` per coordinate;
    ``"uniform"``: EvAX's start, every coordinate uniform in ``[-kick, kick] / sqrt(2)`` (``kick`` is EvAX's
    ``Initial_displacement``). Returns (states, C, n, 3)."""
    dev = base.device
    shape = (int(states), *base.shape)
    if distribution == "gaussian":
        noise = kick * torch.randn(shape, device=dev, generator=gen)
    elif distribution == "uniform":
        noise = (kick / math.sqrt(2.0)) * (2 * torch.rand(shape, device=dev, generator=gen) - 1)
    else:
        raise ValueError(f"unknown start distribution {distribution!r}: 'gaussian' or 'uniform'")
    return base[None].repeat(int(states), 1, 1, 1) + noise * movable[None, None, :, None]


def uniform_step(pos, step, movable, gen):
    """EvAX's random move: ``pos`` (P, C, n, 3) plus a uniform step within +/- ``step`` (A) on every coordinate of every
    movable atom."""
    return pos + step * (2 * torch.rand(pos.shape, device=pos.device, generator=gen) - 1) * movable[None, None, :, None]


def exchange_atoms(pos, site_of, which, idx_a, idx_b, count, gen):
    """Exchanges of two atoms of different elements, in place.

    For every state flagged in ``which`` (P,) and every cell of it, ``count`` times: a random atom of ``idx_a`` and a
    random atom of ``idx_b`` exchange positions and sites. Returns the number of exchanges."""
    dev = pos.device
    done = 0
    for p in torch.nonzero(which).flatten().tolist():
        for c in range(pos.shape[1]):
            i = idx_a[torch.randint(idx_a.numel(), (count,), device=dev, generator=gen)]
            j = idx_b[torch.randint(idx_b.numel(), (count,), device=dev, generator=gen)]
            for ii, jj in zip(i.tolist(), j.tolist()):
                pi = pos[p, c, ii].clone()
                pos[p, c, ii] = pos[p, c, jj]
                pos[p, c, jj] = pi
                si = site_of[p, c, ii].clone()
                site_of[p, c, ii] = site_of[p, c, jj]
                site_of[p, c, jj] = si
                done += 1
    return done


class HopTables:
    """The hop sublattice of a site table as tensors on a device (for ``hop_atoms``)."""

    def __init__(self, sites, device="cuda"):
        dev = torch.device(device)
        self.sites = torch.as_tensor(sites.hop_sites, device=dev, dtype=torch.long)                     # (m_h,) table indices
        self.positions = torch.as_tensor(sites.positions[sites.hop_sites], device=dev, dtype=torch.float32)
        self.neighbors = torch.as_tensor(sites.hop_neighbors, device=dev, dtype=torch.long)
        local = np.full(sites.m, -1, int)
        local[sites.hop_sites] = np.arange(len(sites.hop_sites))
        self.local_of = torch.as_tensor(local, device=dev, dtype=torch.long)
        self.m = int(len(sites.hop_sites))


def hop_atoms(pos, site_of, which, hops: HopTables, count, jitter, gen, any_site=False):
    """Hops onto vacant sites of the hop sublattice, in place.

    For every state flagged in ``which`` (P,) and every cell of it, ``count`` times: a random vacant site is chosen;
    the atom of a random occupied neighboring site (any occupied site of the sublattice with ``any_site``) moves onto
    it, at the site plus a normal displacement of ``jitter`` (A) per coordinate. Returns (hops done, hops skipped:
    no vacant site or no occupied neighbor)."""
    idx = torch.nonzero(which).flatten().tolist()
    if not idx:
        return 0, 0
    dev = pos.device
    C = pos.shape[1]
    done = skipped = 0
    for p in idx:
        for c in range(C):
            loc = hops.local_of[site_of[p, c]]                                                     # (n,) local hop site of every atom (-1: none)
            members = torch.nonzero(loc >= 0).flatten()
            occupied = torch.zeros(hops.m, dtype=torch.long, device=dev).scatter_add_(0, loc[members], torch.ones_like(members))
            atom_at = torch.full((hops.m,), -1, dtype=torch.long, device=dev)
            atom_at[loc[members]] = members
            if bool((occupied > 1).any()):                                                         # two atoms on one site: no hops in this cell
                skipped += count
                continue
            for _ in range(count):
                vac = torch.nonzero(occupied == 0).flatten()
                if vac.numel() == 0:
                    skipped += 1
                    continue
                s_to = int(vac[torch.randint(vac.numel(), (1,), device=dev, generator=gen)])
                if any_site:
                    nb = torch.arange(hops.m, device=dev)
                else:
                    nb = hops.neighbors[s_to]
                    nb = nb[nb >= 0]
                occ = nb[occupied[nb] == 1]
                if occ.numel() == 0:
                    skipped += 1
                    continue
                s_from = int(occ[torch.randint(occ.numel(), (1,), device=dev, generator=gen)])
                atom = int(atom_at[s_from])
                new = hops.positions[s_to] + jitter * torch.randn(3, device=dev, generator=gen)
                pos[p, c, atom] = new
                site_of[p, c, atom] = hops.sites[s_to]
                atom_at[s_to], atom_at[s_from] = atom, -1
                occupied[s_to], occupied[s_from] = 1, 0
                done += 1
    return done, skipped


def fire_relax(objective, pos, steps, mask=None, dt=0.1, dt_max=1.0, maxstep=0.15, alpha0=0.1, f_inc=1.1, f_dec=0.5, n_min=5,
               f_alpha=0.99, fmax=None, project=None):
    """Batched FIRE minimization (Bitzek et al. 2006) of configurations ``pos`` (S, n, 3).

    Every configuration has its own time step and mixing parameter; the masses are 1.

    Parameters
    ----------
    objective : object
        ``energy_forces(pos) -> (E (S,), F (S, n, 3))``, energies in eV and forces in eV/A.
    pos : torch.Tensor, shape (S, n, 3)
    steps : int
        Number of FIRE steps.
    mask : torch.Tensor of bool, shape (S, n), optional
        The atoms allowed to move (default all).
    dt, dt_max, maxstep, alpha0, f_inc, f_dec, n_min, f_alpha : float
        FIRE's parameters; ``maxstep`` (A) caps the move of an atom in one step.
    fmax : float, optional
        Stop when the largest force on a movable atom of every configuration is below this (eV/A).
    project : callable, optional
        ``pos -> pos``, a hard constraint applied before the first evaluation and after every step (the displacement
        limit).

    Returns
    -------
    pos, E, F, steps_taken
    """
    pos = pos.detach().clone()
    if project is not None:
        pos = project(pos)
    S, n, _ = pos.shape
    dev = pos.device
    m = torch.ones(S, n, 1, device=dev) if mask is None else mask.to(dev).float()[:, :, None]
    v = torch.zeros_like(pos)
    dt_c = torch.full((S,), float(dt), device=dev)
    alpha = torch.full((S,), float(alpha0), device=dev)
    n_pos = torch.zeros(S, device=dev)
    E, F = objective.energy_forces(pos)
    F = F * m
    taken = 0
    for step in range(steps):
        if fmax is not None and bool((F.norm(dim=-1).amax(1) < fmax).all()):
            break
        P = (F * v).sum((1, 2))                                                    # power per configuration
        up = P > 0
        n_pos = torch.where(up, n_pos + 1, torch.zeros_like(n_pos))
        grow = up & (n_pos > n_min)
        dt_c = torch.where(grow, torch.clamp(dt_c * f_inc, max=dt_max), dt_c)
        alpha = torch.where(grow, alpha * f_alpha, alpha)
        dt_c = torch.where(up, dt_c, dt_c * f_dec)
        alpha = torch.where(up, alpha, torch.full_like(alpha, alpha0))
        v = torch.where(up[:, None, None], v, torch.zeros_like(v))
        v = v + dt_c[:, None, None] * F
        vn = v.norm(dim=(1, 2), keepdim=True)
        fn = F.norm(dim=(1, 2), keepdim=True)
        v = (1 - alpha[:, None, None]) * v + alpha[:, None, None] * F * (vn / (fn + 1e-30))
        v = v * m
        dr = dt_c[:, None, None] * v
        big = dr.norm(dim=-1, keepdim=True)
        dr = torch.where(big > maxstep, dr * maxstep / (big + 1e-30), dr)
        pos = pos + dr
        if project is not None:
            pos = project(pos)
        E, F = objective.energy_forces(pos)
        F = F * m
        taken = step + 1
    return pos, E, F, taken


class GradientObjective:
    """The objective of the gradient steps of a population and its misfits.

    The configurations of P states of C cells are stacked as (P x C, n, 3), state major. The spectrum of a state is
    the single scattering of its cells (exact, from their own atoms) plus the multiple scattering ``ms`` held for the
    state; its misfit is ``M = sum_e w_e R_e`` over the edges (R the edge's objective, w the spectrum weights). For
    FIRE, every configuration has the energy ``E = scale x M_state / C`` plus the springs of the closest distances.

    Parameters
    ----------
    edges : list of feffgpu.exafs.edge.EdgeModel
    cells_per_state : int
    closest : ClosestDistances, optional
    movable : torch.Tensor, shape (n,)
        1 for the atoms that move, 0 for the others.
    scale : float
        Energy of a unit misfit (eV): sets the size of the gradient steps.

    Attributes
    ----------
    weights : numpy.ndarray, shape (n_edges,)
        The spectrum weights (set by the search).
    ms : list of torch.Tensor
        Per edge the held multiple scattering of every state (P, NK).
    """

    def __init__(self, edges, cells_per_state, closest=None, movable=None, scale=1.0):
        self.edges = list(edges)
        self.C = int(cells_per_state)
        self.closest = closest
        n = self.edges[0].n
        self.movable = torch.ones(n, device=self.edges[0].device) if movable is None else movable
        self.scale = float(scale)
        self.weights = np.full(len(self.edges), 1.0 / len(self.edges))
        self.ms = None
        self.n_below = 0

    def spectra(self, pos, ms=None):
        """Per edge the spectra (P, NK) of the states ``pos`` (P x C, n, 3), with the multiple scattering ``ms`` (default:
        the held one)."""
        ms = self.ms if ms is None else ms
        P = pos.shape[0] // self.C
        out = []
        for i, e in enumerate(self.edges):
            h = e.static_histograms(pos).view(P, self.C, len(e.types), e.nr).mean(1)
            out.append(e.spectrum(h, e.tables, ms[i]))
        return out

    def misfits(self, pos):
        """(P,): the misfit of every state, differentiable in ``pos``."""
        M = 0.0
        for i, (e, chi) in enumerate(zip(self.edges, self.spectra(pos))):
            M = M + float(self.weights[i]) * e.residual(chi)
        return M

    def edge_ratios(self, pos, ms=None):
        """(P, n_edges): the objective of every edge for every state (no gradient)."""
        with torch.no_grad():
            return torch.stack([e.residual(chi) for e, chi in zip(self.edges, self.spectra(pos, ms))], 1)

    def energy_forces(self, pos):
        """FIRE's interface: energies (P x C,) and forces (P x C, n, 3); immobile atoms get no force."""
        p = pos.detach().requires_grad_(True)
        with torch.enable_grad():
            M = self.misfits(p)
            E = self.scale * M.repeat_interleave(self.C) / self.C
            if self.closest is not None and self.closest.active:
                Ew, self.n_below = self.closest.energy(p)
                E = E + Ew
            (g,) = torch.autograd.grad(E.sum(), p)
        return E.detach(), (-g * self.movable[None, :, None]).detach()

    def state_misfit_and_gradient(self, pos, ms):
        """The misfit of one state ``pos`` (C, n, 3) with its multiple scattering ``ms`` (per edge (1, NK)) and its
        gradient (C, n, 3), the cells' histograms averaged."""
        p = pos.detach().requires_grad_(True)
        with torch.enable_grad():
            M = 0.0
            for i, e in enumerate(self.edges):
                h = e.static_histograms(p).mean(0, keepdim=True)
                chi = e.spectrum(h, e.tables, ms[i].mean(0, keepdim=True))
                M = M + float(self.weights[i]) * e.residual(chi)[0]
            (g,) = torch.autograd.grad(M, p)
        return float(M.detach()), g
