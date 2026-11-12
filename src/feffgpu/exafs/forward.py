"""The EXAFS of atomic configurations, exact and differentiable: the forward model shared by MD-EXAFS and RMC-EXAFS.

A configuration is a set of atoms in a periodic cell (or a cluster). Its spectrum is the average over the absorbing
atoms of FEFF's chi(k), in FEFF's convention: S0^2 = 1 and no Debye-Waller factor. All disorder comes from the atom
positions; averaging over configurations (frames of a trajectory, states of a reverse Monte Carlo model) gives the
thermal and static damping.

Three classes:

``PathSpectra``
    FEFF's path finder and path stage for chosen absorbers of copies of a system: path lists held as atom indices, so
    the signal follows the atoms while a list is held, and the spectrum is differentiable in the positions.
``PairTables``
    Single scattering without sampling. A two-leg path depends only on the pair distance and the scatterer's
    potential, so the exact two-leg signal of the path stage is tabulated once per potential on a fine distance grid
    (0.002 A), and the single-scattering spectrum of a whole configuration is the histogram of its pair distances
    times the table. Every pair within the cutoff is included (no importance filter).
``MultipleScattering``
    The three- and four-leg paths of every absorber of every copy, enumerated by the path finder and evaluated by
    the path stage; one spectrum per absorber, a full pass, and the exact gradient in passes of absorbers.
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from ..stage import GPUPathStage
from ..pathfinder import GPUPathFinder
from ..cell import as_cell, min_width, wrap
from .transform import ETOK, ft, hanning_window, r_grid


class PathSpectra:
    """Spectra of chosen absorbers through FEFF's path finder and path stage, with an R-space misfit to a target.

    Parameters
    ----------
    phase : str or feffgpu.PhaseData
        The phase file (``phase.pad`` path or the object from ``feffgpu phase``).
    k : array, shape (NK,)
        The uniform k grid of the spectra (1/A), from zero.
    target : array, shape (NK,)
        The spectrum the misfit refers to, on ``k`` (zeros if no misfit is needed).
    e0 : float
        Energy shift of the model (eV): the paths are evaluated at ``sqrt(k^2 - e0 * ETOK)``.
    window : (float, float)
        ``kmin``, ``kmax`` of the Hanning window.
    r_range : (float, float)
        R window of the misfit (A).
    box : float, array (3, 3) or None
        The periodic cell: a cubic length, lattice vectors as rows, or None for a cluster.
    rmax : float
        Path cutoff (A, half the total path length, FEFF's RPATH).
    nlegmax : int
        Largest number of legs (2, 3 or 4).
    critpw : float
        FEFF's plane-wave importance filter (percent).
    chunk : int
        Paths per batch of the path stage.
    device : str
    dtype : torch.dtype
        ``torch.complex64`` (default) or ``torch.complex128``.
    kweight : int
    dk : float
        Sill width of the Hanning window.
    """

    def __init__(self, phase, k, target, e0, window=(3.0, 23.0), r_range=(1.0, 8.0), box=None, rmax=9.4, nlegmax=4, critpw=0.5,
                 merge_degenerate=False, chunk=2048, device="cuda", dtype=torch.complex64, kweight=2, dk=1.0):
        self.device, self.chunk, self.box = torch.device(device), int(chunk), as_cell(box, device)
        self.kweight = int(kweight)
        if self.box is not None and min_width(self.box) < 2 * float(rmax):
            raise ValueError(f"the periodic cell (smallest width {min_width(self.box):.2f} A) must exceed twice the path cutoff ({rmax} A) "
                             "for minimum-image paths: use a larger supercell or a smaller cutoff")
        self.finder = GPUPathFinder(phase, device=device, rmax=rmax, nlegmax=nlegmax, critpw=critpw, merge_degenerate=merge_degenerate)
        self.stage = GPUPathStage(phase, device=device, dtype=dtype)
        self.rdtype = self.stage.rdtype
        k = np.asarray(k, float)
        en = k ** 2 - float(e0) * ETOK
        q = np.sign(en) * np.sqrt(np.abs(en))
        self.q = torch.tensor(q, device=self.device, dtype=self.rdtype)
        self.S = self.stage.linear_matrix(q)
        self.kt = torch.tensor(k, device=self.device, dtype=torch.float32)
        self.win = hanning_window(self.kt, torch.tensor([window[0]], device=self.device), torch.tensor([window[1]], device=self.device), dk=dk)
        self._target = np.asarray(target, float)
        self.set_r_range(*r_range)
        self.norm = float((self.target_r.abs() ** 2).sum())
        self.lists, self.C, self.n, self.A = None, 0, 0, 0

    def set_r_range(self, r1, r2):
        """Set the R window of the misfit (A)."""
        r = r_grid(self.device)
        self.rmask = (r >= r1) & (r <= r2)
        self.set_target(self._target)

    def set_target(self, target):
        """Set the target spectrum (NK,) on the k grid; its transform is kept in double precision."""
        self._target = np.asarray(target, float) if not torch.is_tensor(target) else target.detach().cpu().numpy().astype(float)
        self.target = torch.as_tensor(self._target, device=self.device, dtype=torch.float64)
        self.target_r = ft(self.target[None], self.kt.double(), self.win.double(), kweight=self.kweight)[0][self.rmask]
        self.norm = float((self.target_r.abs() ** 2).sum())

    # -- path lists ---------------------------------------------------------------------------------------------
    @staticmethod
    def sample_absorbers(C, n, A, rng):
        """(C, A) random distinct absorber indices per copy."""
        return np.stack([rng.choice(n, size=min(A, n), replace=False) for _ in range(C)])

    def refresh(self, positions, absorbers, ipot=None):
        """Enumerate the paths of the chosen absorbers (C, A) at the current positions (C, n, 3) and hold them.

        The lists are stored per number of legs with global atom indices (copy c, atom i -> c n + i). ``ipot`` (n,)
        is each atom's scatterer potential in the phase file (default 1 for all). Returns the number of paths per
        number of legs."""
        pos = torch.as_tensor(positions, device=self.device, dtype=torch.float32)
        C, n, _ = pos.shape
        absorbers = np.asarray(absorbers)
        self.C, self.n, self.A = C, n, absorbers.shape[1]
        ipot = torch.ones(n, dtype=torch.long, device=self.device) if ipot is None else torch.as_tensor(ipot, device=self.device)
        parts = {}
        for c in range(C):
            for a in absorbers[c]:
                kept, _ = self.finder.enumerate(pos[c], self.box, int(a), ipot)
                for nleg, (coords, ipots, deg, xp, atoms) in kept.items():
                    if coords.shape[0] == 0:
                        continue
                    g = parts.setdefault(nleg, ([], [], [], [], []))
                    g[0].append(atoms + c * n)
                    g[1].append(torch.full((atoms.shape[0],), int(a) + c * n, device=self.device, dtype=torch.long))
                    g[2].append(ipots)
                    g[3].append(deg)
                    g[4].append(torch.full((atoms.shape[0],), c, device=self.device, dtype=torch.long))
        self.lists = {nleg: dict(atoms=torch.cat(g[0]), absorber=torch.cat(g[1]), ipots=torch.cat(g[2]), deg=torch.cat(g[3]),
                                 copy=torch.cat(g[4])) for nleg, g in parts.items()}
        self.counts = {nleg: int(v["atoms"].shape[0]) for nleg, v in self.lists.items()}
        return self.counts

    # -- spectra ----------------------------------------------------------------------------------------------
    def _chunk_chi(self, coords, ipots):
        cchi, reff = self.stage.cchi(coords, ipots)
        return self.stage.chi_fine(cchi, reff, self.q, self.S, "feff")

    def spectra(self, pos_flat, use_checkpoint=True, nlegs=None):
        """chi(k) per copy, (C, NK), averaged over that copy's absorbers.

        ``pos_flat`` (C n, 3) may require a gradient. ``nlegs`` restricts the sum to these numbers of legs (default
        all)."""
        out = torch.zeros(self.C, self.kt.shape[0], device=self.device, dtype=self.rdtype)
        for nleg, L in self.lists.items():
            if nlegs is not None and nleg not in nlegs:
                continue
            for s in range(0, L["atoms"].shape[0], self.chunk):
                atoms, absr = L["atoms"][s:s + self.chunk], L["absorber"][s:s + self.chunk]
                rel = pos_flat[atoms] - pos_flat[absr][:, None, :]
                rel = wrap(rel, self.box)
                coords = torch.cat([torch.zeros(rel.shape[0], 1, 3, device=self.device, dtype=rel.dtype), rel], 1).to(self.rdtype)
                ipots = L["ipots"][s:s + self.chunk]
                if use_checkpoint and torch.is_grad_enabled() and pos_flat.requires_grad:
                    chi = checkpoint(self._chunk_chi, coords, ipots, use_reentrant=False)
                else:
                    chi = self._chunk_chi(coords, ipots)
                out = out.index_add(0, L["copy"][s:s + self.chunk], L["deg"][s:s + self.chunk, None].to(self.rdtype) * chi)
        return out / self.A

    def misfit(self, chi_bar):
        """R-space misfit of one spectrum (NK,) to the target: ``sum |chi_R - target_R|^2 / sum |target_R|^2``."""
        cr = ft(chi_bar[None].to(torch.float64), self.kt.double(), self.win.double(), kweight=self.kweight)[0][self.rmask]
        return ((cr - self.target_r).abs() ** 2).sum() / self.norm

    @torch.no_grad()
    def spectrum_only(self, positions):
        """The mean spectrum (NK,) of the held lists at ``positions`` (C, n, 3), its misfit, and the spectra per copy."""
        pos = torch.as_tensor(positions, device=self.device, dtype=self.rdtype).reshape(-1, 3)
        chi = self.spectra(pos, use_checkpoint=False)
        return chi.mean(0), self.misfit(chi.mean(0)), chi


class PairTables:
    """Exact single scattering of whole configurations from tabulated two-leg signals.

    For every scatterer potential of the phase file, the two-leg chi(k) of the path stage is tabulated on a distance
    grid from ``r_min`` to ``rmax`` in steps of ``dr``. The single-scattering spectrum of a configuration is the
    histogram of its absorber-neighbor distances (linear interpolation between grid points, so it is differentiable
    in the positions) times the table.

    Parameters are those of ``PathSpectra``, plus:

    r_min : float
        Start of the distance grid (A). Pairs closer than this are placed on the first grid point.
    dr : float
        Step of the distance grid (A).
    ms : bool
        Also build a path finder for multiple scattering (``self.sub``, a ``PathSpectra``). Not needed when
        ``MultipleScattering`` is used for that.

    Attributes
    ----------
    tables : dict
        ``{scatterer potential: tensor (nr, NK)}``.
    base : PathSpectra
        Holds the k grid, the window, the target and the misfit.
    """

    def __init__(self, phase, k, target, e0, window=(3.0, 23.0), r_range=(1.0, 8.0), box=None, rmax=9.4, r_min=1.5, dr=0.002, ms=True,
                 nlegmax=4, critpw=0.5, chunk=2048, device="cuda", dtype=torch.complex64, kweight=2, dk=1.0):
        self.device, self.box, self.rmax = torch.device(device), as_cell(box, device), float(rmax)
        self.sub = PathSpectra(phase, k, target, e0, window=window, r_range=r_range, box=box, rmax=rmax, nlegmax=nlegmax,
                               critpw=critpw, chunk=chunk, device=device, dtype=dtype, kweight=kweight, dk=dk) if ms else None
        stage = self.sub.stage if ms else GPUPathStage(phase, device=device, dtype=dtype)
        base = self.sub if ms else PathSpectra(phase, k, target, e0, window=window, r_range=r_range, box=box, rmax=rmax,
                                               nlegmax=2, critpw=0.0, chunk=chunk, device=device, dtype=dtype, kweight=kweight, dk=dk)
        self.base = base
        self.kt, self.q, self.S = base.kt, base.q, base.S
        # the 2-leg table per scatterer potential: cchi(E, r) -> chi(k) on the scan grid
        self.r0, self.dr = float(r_min), float(dr)
        self.nr = int(round((self.rmax - self.r0) / self.dr)) + 1
        rgrid = self.r0 + self.dr * torch.arange(self.nr, device=self.device, dtype=torch.float32)
        self.tables = {}
        nph = stage.P.nph
        with torch.no_grad():
            for ipot in range(1, nph + 1):
                rows = []
                for s in range(0, self.nr, chunk):
                    r = rgrid[s:s + chunk]
                    coords = torch.zeros(len(r), 2, 3, device=self.device, dtype=stage.rdtype)
                    coords[:, 1, 2] = r.to(stage.rdtype)
                    ipots = torch.tensor([[0, ipot]], device=self.device).expand(len(r), -1)
                    cchi, reff = stage.cchi(coords, ipots)
                    rows.append(stage.chi_fine(cchi, reff, self.q, self.S, "feff").float())
                self.tables[ipot] = torch.cat(rows)                                        # (nr, NK)
        self.C, self.n = 0, 0

    # -- single scattering over all pairs ---------------------------------------------------------------------
    def pair_distances(self, pos):
        """All ordered pairs (absorber a, neighbor j) within the cutoff of ``pos`` (C, n, 3), minimum image.

        Returns the distances (P,), differentiable in ``pos``, and the neighbors' scatterer potentials (P,): those
        of ``self.ipot`` (n,) if that attribute is set, else 1 for every atom (one element)."""
        C, n, _ = pos.shape
        rel = pos[:, :, None, :] - pos[:, None, :, :]                                      # (C, n, n, 3): j - a
        rel = wrap(rel, self.box)
        r2 = (rel ** 2).sum(-1)
        mask = (r2 <= self.rmax ** 2) & (r2 > 1e-6)
        c_idx, a_idx, j_idx = torch.nonzero(mask, as_tuple=True)
        r = torch.sqrt(r2[c_idx, a_idx, j_idx])
        return r, self.ipot[j_idx] if hasattr(self, "ipot") else torch.ones_like(j_idx)

    def single_scattering(self, pos):
        """The single-scattering chi(k) (NK,) of ``pos`` (C, n, 3), averaged over all copies with every atom an
        absorber. Differentiable in ``pos``."""
        C, n, _ = pos.shape
        r, ip = self.pair_distances(pos)
        chi = torch.zeros(self.kt.shape[0], device=self.device)
        u = (r - self.r0) / self.dr
        i0 = torch.floor(u).detach().clamp(0, self.nr - 2).long()
        f = (u - i0).clamp(0, 1)
        for ipot, table in self.tables.items():
            sel = ip == ipot
            if not bool(sel.any()):
                continue
            h = torch.zeros(self.nr, device=self.device, dtype=f.dtype)
            h = h.index_add(0, i0[sel], 1 - f[sel]).index_add(0, i0[sel] + 1, f[sel])
            chi = chi + h @ table
        return chi / (C * n)


class MultipleScattering:
    """The multiple-scattering paths (three and four legs, or three only) of every absorber of every copy.

    The paths are enumerated by the GPU path finder and evaluated by the path stage. ``rows`` holds one spectrum per
    absorber; ``full_pass`` fills it, ``chi_absorbers`` evaluates a subset with a gradient, ``full_gradient`` gives the
    exact gradient of a loss of the mean spectrum in passes of absorbers.

    Any mix of elements: ``ipot`` (n,) gives each atom's scatterer potential (default 1 for all) and ``absorbers``
    (A,) the atoms that absorb (default every atom). ``box`` is a cubic length, a 3 x 3 cell or None.
    """

    def __init__(self, phase, k, e0, box, rmax=7.0, critpw=2.0, chunk=1024, device="cuda", nlegmax=4):
        self.full = PathSpectra(phase, k, np.zeros(len(k)), e0, box=box, rmax=rmax, nlegmax=nlegmax, critpw=critpw, chunk=chunk, device=device)
        self.device, self.chunk, self.box = torch.device(device), int(chunk), as_cell(box, device)
        self.NK = len(k)
        self.lists, self.rows, self.n_abs = None, None, 0

    def enumerate(self, pos, ipot=None, absorbers=None):
        """Find the paths of every absorber of every copy of ``pos`` (C, n, 3).

        Atom indices are global (copy c, atom i -> c n + i). Row ``c A + j`` holds absorber j of copy c; ``row_of``
        maps a global atom index to its row (-1: not an absorber) and ``abs_atoms`` a row to its global atom index.
        Returns the number of paths per number of legs."""
        pos = torch.as_tensor(pos, device=self.device, dtype=torch.float32)
        C, n, _ = pos.shape
        ipot = torch.ones(n, dtype=torch.long, device=self.device) if ipot is None else torch.as_tensor(ipot, device=self.device, dtype=torch.long)
        absorbers = torch.arange(n, device=self.device) if absorbers is None else torch.as_tensor(absorbers, device=self.device, dtype=torch.long)
        A = int(absorbers.shape[0])
        parts = {}
        for c in range(C):
            for a in absorbers.tolist():
                kept, _ = self.full.finder.enumerate(pos[c], self.box, a, ipot)
                for nleg in (3, 4):
                    if nleg not in kept or kept[nleg][0].shape[0] == 0:
                        continue
                    coords, ipots, deg, xp, atoms = kept[nleg]
                    g = parts.setdefault(nleg, ([], [], [], []))
                    g[0].append(atoms + c * n); g[1].append(torch.full((atoms.shape[0],), a + c * n, device=self.device, dtype=torch.long))
                    g[2].append(ipots); g[3].append(deg)
        self.lists = {nleg: dict(atoms=torch.cat(g[0]), absorber=torch.cat(g[1]), ipots=torch.cat(g[2]), deg=torch.cat(g[3])) for nleg, g in parts.items()}
        self.C, self.n, self.A, self.n_abs = C, n, A, C * A
        self.abs_atoms = (absorbers[None, :] + n * torch.arange(C, device=self.device)[:, None]).reshape(-1)
        self.row_of = torch.full((C * n,), -1, device=self.device, dtype=torch.long)
        self.row_of[self.abs_atoms] = torch.arange(C * A, device=self.device)
        self.counts = {nleg: int(v["atoms"].shape[0]) for nleg, v in self.lists.items()}
        return self.counts

    def _chi_paths(self, pos_flat, L, idx, local, A, grad):
        out = torch.zeros(A, self.NK, device=self.device, dtype=self.full.rdtype)
        for s in range(0, len(idx), self.chunk):
            sl = idx[s:s + self.chunk]
            atoms, absr = L["atoms"][sl], L["absorber"][sl]
            rel = pos_flat[atoms] - pos_flat[absr][:, None, :]
            rel = wrap(rel, self.box)
            coords = torch.cat([torch.zeros(rel.shape[0], 1, 3, device=self.device, dtype=rel.dtype), rel], 1).to(self.full.rdtype)
            ipots = L["ipots"][sl]
            chi = checkpoint(self.full._chunk_chi, coords, ipots, use_reentrant=False) if grad else self.full._chunk_chi(coords, ipots)
            out = out.index_add(0, local[absr], L["deg"][sl, None].to(chi.dtype) * chi)
        return out

    def chi_absorbers(self, pos_flat, absorbers):
        """(A, NK): the multiple-scattering chi of the given absorbers (global atom indices), with a gradient if
        ``pos_flat`` (C n, 3) requires one."""
        A = int(absorbers.shape[0])
        local = torch.full((self.C * self.n,), -1, device=self.device, dtype=torch.long)
        local[absorbers] = torch.arange(A, device=self.device)
        out = torch.zeros(A, self.NK, device=self.device, dtype=self.full.rdtype)
        grad = torch.is_grad_enabled() and pos_flat.requires_grad
        for L in self.lists.values():
            idx = torch.nonzero(local[L["absorber"]] >= 0).squeeze(1)
            if len(idx):
                out = out + self._chi_paths(pos_flat, L, idx, local, A, grad)
        return out

    @torch.no_grad()
    def full_pass(self, pos_flat):
        """Every absorber's multiple-scattering chi at ``pos_flat`` (C n, 3) -> ``self.rows`` (n_abs, NK)."""
        local = self.row_of
        rows = torch.zeros(self.n_abs, self.NK, device=self.device, dtype=self.full.rdtype)
        for L in self.lists.values():
            rows = rows + self._chi_paths(pos_flat, L, torch.arange(L["atoms"].shape[0], device=self.device), local, self.n_abs, False)
        self.rows = rows
        return rows

    def full_gradient(self, pos_flat, loss_fn, absorbers_per_pass=512):
        """The exact gradient with respect to ``pos_flat`` of ``loss_fn(mean spectrum)``, the mean taken over every
        absorber, computed in passes of absorbers (the absorbers of a pass are live, the others come from ``rows``,
        so the loss is the true one)."""
        g = torch.zeros_like(pos_flat)
        base = self.rows.mean(0)
        for s in range(0, self.n_abs, absorbers_per_pass):
            sub = torch.arange(s, min(s + absorbers_per_pass, self.n_abs), device=self.device)
            p = pos_flat.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                live = self.chi_absorbers(p, self.abs_atoms[sub]).sum(0) / self.n_abs
                chi = base - self.rows[sub].sum(0) / self.n_abs + live
                (gs,) = torch.autograd.grad(loss_fn(chi), p)
            g += gs
        return g

    @property
    def mean(self):
        """The mean of the held rows over all absorbers, (NK,)."""
        return self.rows.mean(0)
