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
    The paths of three and more legs (up to nine) of every absorber of every copy, enumerated by the path finder and evaluated by
    the path stage; one spectrum per absorber, a full pass, and the exact gradient in passes of absorbers.
"""
from __future__ import annotations
import math
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
        Largest number of legs (2 to 9).
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

    # -- single scattering of chosen absorbers (one configuration, any cell) --------------------------------------
    @torch.no_grad()
    def histogram(self, pos, absorbers, ipot, box=None, batch=1024):
        """The distance histogram of chosen absorbers of one configuration, per scatterer potential.

        Parameters
        ----------
        pos : torch.Tensor, shape (n, 3)
            Positions (A) on this object's device.
        absorbers : torch.Tensor of int, shape (A,)
            Indices of the absorbing atoms.
        ipot : torch.Tensor of int, shape (n,)
            Scatterer potential of every atom (1 .. nph; 0 for an atom that is never a neighbor).
        box : None, float or tensor (3, 3)
            The cell of this configuration (as returned by ``feffgpu.cell.as_cell``); None for a cluster. It may
            differ from the cell given at construction, so that a trajectory with a changing cell can be processed.
        batch : int
            Absorbers per batch (bounds the memory).

        Returns
        -------
        hist : torch.Tensor, shape (nph + 1, nr), float64
            Row p: the histogram of absorber-neighbor distances for neighbors of potential p on the table's distance
            grid (the linear-interpolation weights of ``single_scattering``), summed over the absorbers. Every pair
            within ``rmax`` is counted; there is no importance filter.
        n_short : int
            Pairs closer than the start of the grid (they are placed on its first point).
        """
        nph = max(self.tables)
        hist = torch.zeros(nph + 1, self.nr, device=self.device, dtype=torch.float64)
        flat = hist.view(-1)
        short = 0
        rmax2 = self.rmax ** 2
        A = int(absorbers.shape[0])
        batch = max(1, int(batch))
        for s in range(0, A, batch):
            a = absorbers[s:s + batch]
            rel = wrap(pos[None, :, :] - pos[a][:, None, :], box)                  # (B, n, 3): neighbor - absorber
            r2 = (rel ** 2).sum(-1)
            mask = (r2 <= rmax2) & (r2 > 1e-6)
            _b, j = torch.nonzero(mask, as_tuple=True)
            r = torch.sqrt(r2[mask])
            short += int((r < self.r0).sum())
            u = (r - self.r0) / self.dr
            i0 = torch.floor(u).clamp(0, self.nr - 2).long()
            f = (u - i0).clamp(0, 1).double()
            idx = ipot[j] * self.nr + i0
            flat.index_add_(0, idx, 1.0 - f)
            flat.index_add_(0, idx + 1, f)
        return hist, short

    def table_stack(self, potentials=None, dtype=torch.float64):
        """The two-leg tables of the given scatterer potentials (default all), stacked: (P, nr, NK)."""
        potentials = sorted(self.tables) if potentials is None else list(potentials)
        return torch.stack([self.tables[p].to(dtype) for p in potentials])

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
    """The multiple-scattering paths (three legs up to ``nlegmax``, at most nine) of every absorber of every copy.

    The paths are enumerated by the GPU path finder and evaluated by the path stage. ``rows`` holds one spectrum per
    absorber; ``full_pass`` fills it, ``chi_absorbers`` evaluates a subset with a gradient, ``full_gradient`` gives the
    exact gradient of a loss of the mean spectrum in passes of absorbers.

    Any mix of elements: ``ipot`` (n,) gives each atom's scatterer potential (default 1 for all) and ``absorbers``
    (A,) the atoms that absorb (default every atom). ``box`` is a cubic length, a 3 x 3 cell or None.

    Which paths are computed is decided in one of two ways (``criteria_rule``):

    ``"feff"`` (default)
        FEFF's path finder on the structure as it is: every path is judged alone by FEFF's plane-wave importance,
        relative to the shortest single-scattering path, and kept from ``critpw`` percent on. The lists are found again
        whenever ``enumerate`` is called.
    ``"evax"``
        EvAX's rule: the paths are chosen once on the lattice sites (``set_lattice``). Paths are gathered into groups
        of like geometry the way EvAX's path finder does it (each direction of travel on its own, atoms within a
        tolerance of 0.5 A); a group is kept when the size of its signal on the sites is more than ``critpw`` percent
        of the size of the whole spectrum of the sites. A group is kept or dropped as a whole, for every atom, and
        ``enumerate`` then returns the same paths whatever the positions.
    """

    EVAX_KMAX = 20.0          # the importance of EvAX's rule is summed over k up to this (1/A): the k range of EvAX's FEFF runs

    def __init__(self, phase, k, e0, box, rmax=7.0, critpw=2.0, chunk=1024, device="cuda", nlegmax=4, criteria_rule="feff"):
        if criteria_rule not in ("feff", "evax"):
            raise ValueError(f"criteria_rule must be 'feff' or 'evax' (got {criteria_rule!r})")
        self.full = PathSpectra(phase, k, np.zeros(len(k)), e0, box=box, rmax=rmax, nlegmax=nlegmax, critpw=critpw, chunk=chunk, device=device)
        self.device, self.chunk, self.box = torch.device(device), int(chunk), as_cell(box, device)
        self.NK = len(k)
        self.k = np.asarray(k, float)
        self.critpw, self.criteria_rule = float(critpw), criteria_rule
        self.lists, self.rows, self.n_abs = None, None, 0
        self.lattice, self.types, self.lattice_n = None, [], 0
        self._batches = {}

    def _batch(self, nleg, grad=False):
        """Paths of ``nleg`` legs per batch of the path stage: ``chunk``, or fewer when the free memory of the GPU does
        not hold that many. The memory of a path grows with its number of legs and is several times larger with a
        gradient; it is measured once per number of legs on a batch of sixteen paths (about a tenth of a second)."""
        key = (int(nleg), bool(grad))
        if key in self._batches:
            return self._batches[key]
        size = self.chunk
        if self.device.type == "cuda":
            probe = 16
            coords = torch.zeros(probe, nleg, 3, device=self.device, dtype=self.full.rdtype)
            for j in range(1, nleg):                                    # a zigzag chain with legs of about 2.5 A
                coords[:, j, 0] = 2.0 * j
                coords[:, j, 1] = 1.5 * (j % 2)
            ipots = torch.ones(probe, nleg, dtype=torch.long, device=self.device)
            ipots[:, 0] = 0
            torch.cuda.synchronize(self.device)
            base = torch.cuda.memory_allocated(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            if grad:
                with torch.enable_grad():
                    c = coords.clone().requires_grad_(True)
                    checkpoint(self.full._chunk_chi, c, ipots, use_reentrant=False).sum().backward()
            else:
                with torch.no_grad():
                    self.full._chunk_chi(coords, ipots)
            per_path = max((torch.cuda.max_memory_allocated(self.device) - base) / probe, 1.0)
            free, _ = torch.cuda.mem_get_info(self.device)
            free += torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device)       # the allocator's own cache can be used again
            size = int(max(16, min(self.chunk, 0.5 * free / per_path)))
        self._batches[key] = size
        return size

    # -- EvAX's rule: the paths chosen on the lattice -----------------------------------------------------------------
    @staticmethod
    def _chain_coordinates(scat, tiny=1e-4):
        """EvAX's description of paths as chains of atoms (its path finder).

        ``scat`` (B, m, 3) holds the scatterers of B paths in one direction of travel, relative to the absorber
        (double precision). EvAX gives every atom of a chain that was not met before coordinates in a frame set by
        the atoms before it: the first scatterer its distance; the second its (x, y) in the plane of the absorber and
        the first two scatterers, x along the first; the third its (x, y, z) in that frame. The frame is reached by
        EvAX's three rotations (about z, y and x of the structure's axes); when the first two scatterers lie on one
        line with the absorber the last rotation is not defined and EvAX leaves it out, so the coordinates of the
        third atom then depend on how the path lies in the structure.

        Returns ``code`` (B, m): -1 for a new atom, else the position in the chain (1: the absorber) of the atom met
        again; and ``coord`` (B, m, 3), zero where an atom has fewer than three coordinates."""
        B, m, _ = scat.shape
        dev = scat.device
        code = torch.full((B, m), -1, dtype=torch.long, device=dev)
        coord = torch.zeros(B, m, 3, dtype=scat.dtype, device=dev)
        chain = torch.cat([torch.zeros(B, 1, 3, dtype=scat.dtype, device=dev), scat], 1)
        unique = torch.ones(B, dtype=torch.long, device=dev)                  # atoms met so far, the absorber included
        phi1 = torch.zeros(B, dtype=scat.dtype, device=dev)
        theta1, phi2 = torch.zeros_like(phi1), torch.zeros_like(phi1)

        def about_z(v, a):
            c, s = torch.cos(a), torch.sin(a)
            return torch.stack([v[:, 0] * c - v[:, 1] * s, v[:, 1] * c + v[:, 0] * s, v[:, 2]], 1)

        def about_y(v, a):
            c, s = torch.cos(a), torch.sin(a)
            return torch.stack([v[:, 0] * c + v[:, 2] * s, v[:, 1], v[:, 2] * c - v[:, 0] * s], 1)

        def about_x(v, a):
            c, s = torch.cos(a), torch.sin(a)
            return torch.stack([v[:, 0], v[:, 1] * c - v[:, 2] * s, v[:, 2] * c + v[:, 1] * s], 1)

        for i in range(m):
            s = scat[:, i]
            for pos in range(1, i + 1):                                       # the most recent equal atom wins
                same = (chain[:, pos - 1] - s).norm(dim=1) < 1e-3
                code[:, i] = torch.where(same, torch.full_like(code[:, i], pos), code[:, i])
            new = code[:, i] < 0
            unique = unique + new.long()
            first, second, third = new & (unique == 2), new & (unique == 3), new & (unique >= 4)
            if bool(first.any()):
                rho = s.norm(dim=1)
                flat = (s[:, 0].abs() < tiny) & (s[:, 1].abs() < tiny)
                p1 = torch.where(flat, torch.zeros_like(rho), torch.atan2(s[:, 1], s[:, 0]))
                t1 = torch.acos((s[:, 2] / rho.clamp_min(1e-12)).clamp(-1, 1))
                phi1, theta1 = torch.where(first, p1, phi1), torch.where(first, t1, theta1)
                coord[:, i, 0] = torch.where(first, rho, coord[:, i, 0])
            v = about_y(about_z(s, -phi1), math.pi / 2 - theta1)
            if bool(second.any()):
                flat = (v[:, 2].abs() < tiny) & (v[:, 1].abs() < tiny)
                p2 = torch.where(flat, torch.zeros_like(phi2), torch.atan2(v[:, 2], v[:, 1]))
                phi2 = torch.where(second, p2, phi2)
                w = about_x(v, -phi2)
                coord[:, i, 0] = torch.where(second, w[:, 0], coord[:, i, 0])
                coord[:, i, 1] = torch.where(second, w[:, 1], coord[:, i, 1])
            if bool(third.any()):
                coord[:, i] = torch.where(third[:, None], about_x(v, -phi2), coord[:, i])
        return code, coord

    @torch.no_grad()
    def set_lattice(self, sites, ipot=None, absorbers=None, kweight=2, tolerance=0.5):
        """Choose the multiple-scattering paths on the lattice ``sites`` (n, 3) the way EvAX does (``criteria_rule = "evax"``).

        Every path of every absorber within the cutoff is enumerated on the sites and described as EvAX describes it:
        a chain of atoms in one direction of travel (``_chain_coordinates``); a path that is not its own reverse gives
        two chains. Chains form groups: two chains belong together when they meet atoms again in the same pattern,
        have the same elements in the same order, and the coordinates of every atom differ by at most ``tolerance``
        (A; EvAX's ``Pathfinder_robustness``, default 0.5). As in EvAX, the first chain met founds a group and every
        later chain joins the first group whose founder it matches; the chains are met absorber by absorber, each
        next atom in the order of its distance from the atom before.

        The importance of a group is the size of (its number of chains per absorber) x (the signal of its founder on
        the sites), divided by the size of the sum of that over all groups (single scattering included), in percent;
        the size is the root of the sum of ``(k^kweight chi)^2`` over k from 0 to 20 1/A (the grid of EvAX's FEFF runs,
        whatever the grid of this object). Groups above ``critpw`` percent are
        kept with every chain they hold. A path then counts once for each of its directions that was kept: FEFF's
        degeneracy of two when both were, one when only one was. That happens to paths whose first two scatterers lie
        on a line with the absorber: EvAX's coordinates split their chains into small groups that stay below the
        threshold, while the opposite direction of the same paths is kept.

        ``self.types`` then lists every group from a quarter of the threshold on, in the form of EvAX's own list
        (``partialEXAFS.dat``): ``legs``, ``half_length`` of the founder (A), ``degeneracy`` (chains per absorber),
        ``importance`` (percent), ``potentials`` (of the founder's scatterers, in the order of travel) and ``kept``.
        Groups of two legs (single scattering) are listed for that comparison only: single scattering is always
        computed for every pair within the cutoff, from the pair tables. The sum that the importance is divided by
        holds the paths of up to ``nlegmax`` legs of this object (EvAX: ``N_legs``, default 8), so a smaller
        ``nlegmax`` shifts the importance of groups near the threshold. Chains of equal distance are met in an order
        that EvAX does not fix, which also matters only for groups within a few percent of the threshold.

        Returns the number of paths per number of legs."""
        dev = self.device
        sites = torch.as_tensor(sites if torch.is_tensor(sites) else np.asarray(sites, float), device=dev, dtype=torch.float32)
        n = sites.shape[0]
        ipot = torch.ones(n, dtype=torch.long, device=dev) if ipot is None else torch.as_tensor(ipot, device=dev, dtype=torch.long)
        absorbers = torch.arange(n, device=dev) if absorbers is None else torch.as_tensor(absorbers, device=dev, dtype=torch.long)
        A = int(absorbers.shape[0])
        tolerance = float(tolerance)
        finder = self.full.finder
        held, finder.critpw = finder.critpw, 0.0                     # every candidate path, no filter
        per = {}
        try:
            for a in absorbers.tolist():
                kept, _ = finder.enumerate(sites, self.box, a, ipot)
                for nleg, (coords, ipots, deg, xp, atoms) in kept.items():
                    if deg.shape[0]:
                        g = per.setdefault(nleg, ([], [], [], [], []))
                        g[0].append(coords); g[1].append(ipots); g[2].append(deg); g[3].append(atoms)
                        g[4].append(torch.full((deg.shape[0],), a, device=dev, dtype=torch.long))
        finally:
            finder.critpw = held
        kk = torch.as_tensor(self.k, device=dev, dtype=torch.float64)
        kw = kk ** int(kweight) * (kk <= self.EVAX_KMAX + 1e-9)       # EvAX sums the sizes over FEFF's grid, 0 to 20 1/A
        total = torch.zeros(self.NK, device=dev, dtype=torch.float64)
        work = {}
        for nleg in sorted(per):
            coords, ipots, deg, atoms, absr = (torch.cat(v) for v in per[nleg])
            P = coords.shape[0]
            directions = torch.round(deg).long()
            if not bool(((directions == 1) | (directions == 2)).all()):
                raise RuntimeError(f"paths of {nleg} legs with a degeneracy other than one or two: {torch.unique(directions).tolist()}")
            two = directions == 2
            every = torch.arange(P, device=dev)
            path = torch.cat([every, every[two]])                    # the path of every chain
            backward = torch.cat([torch.zeros(P, dtype=torch.bool, device=dev), torch.ones(int(two.sum()), dtype=torch.bool, device=dev)])
            scat = coords[:, 1:].double()
            chains = torch.cat([scat, scat[two].flip(1)])
            pots = torch.cat([ipots[:, 1:], ipots[two][:, 1:].flip(1)]).long()
            code, coord = self._chain_coordinates(chains)
            C = chains.shape[0]
            closed = torch.cat([torch.zeros(C, 1, 3, dtype=chains.dtype, device=dev), chains, torch.zeros(C, 1, 3, dtype=chains.dtype, device=dev)], 1)
            legs = (closed[:, 1:] - closed[:, :-1]).norm(dim=2)
            order = torch.arange(C, device=dev)                      # EvAX's order: by absorber, then leg by leg by length
            for column in [legs[:, j] for j in range(nleg - 2, -1, -1)] + [absr[path].double()]:
                order = order[torch.argsort(torch.round(column[order] * 1e4), stable=True)]
            code, coord, pots, path, backward, legs = code[order], coord[order], pots[order], path[order], backward[order], legs[order]
            _, bucket = torch.unique(torch.cat([code, pots], 1), dim=0, return_inverse=True)
            group = torch.full((C,), -1, dtype=torch.long, device=dev)
            founders = []
            for b in range(int(bucket.max()) + 1):
                idx = torch.nonzero(bucket == b).squeeze(1)          # in the order of meeting
                c = coord[idx]
                free = torch.ones(idx.shape[0], dtype=torch.bool, device=dev)
                while bool(free.any()):
                    f = int(torch.nonzero(free)[0])
                    near = free & ((c - c[f]).norm(dim=2) <= tolerance).all(1)
                    group[idx[near]] = len(founders)
                    founders.append(int(idx[f]))
                    free &= ~near
            fo = torch.tensor(founders, device=dev, dtype=torch.long)
            G = fo.shape[0]
            count = torch.zeros(G, device=dev, dtype=torch.float64).index_add_(0, group, torch.ones(C, device=dev, dtype=torch.float64)) / A
            fp = path[fo]
            step = self._batch(nleg)
            chi = torch.cat([self.signals(coords[fp[s:s + step]], ipots[fp[s:s + step]])[0].double() for s in range(0, G, step)])
            total += (count[:, None] * chi).sum(0)
            work[nleg] = dict(ipots=ipots, atoms=atoms, absr=absr, deg=deg, path=path, backward=backward, group=group, count=count, chi=chi,
                              half=legs[fo].sum(1) / 2, pots=pots[fo], P=P)
        norm = float(((total * kw) ** 2).sum()) ** 0.5
        if norm <= 0:
            raise ValueError("the lattice has no scattering path within the cutoff")
        self.lattice, self.types, self.lattice_n = {}, [], n
        for nleg in sorted(work):
            w = work[nleg]
            importance = 100.0 * w["count"] * ((w["chi"] * kw) ** 2).sum(1).sqrt() / norm
            keep = importance > self.critpw
            index = torch.full((keep.shape[0],), -1, device=dev, dtype=torch.long)
            for g in torch.argsort(w["half"]).tolist():
                if float(importance[g]) >= 0.25 * self.critpw or bool(keep[g]):
                    index[g] = len(self.types)                        # the row of this group in self.types
                    self.types.append(dict(legs=nleg, half_length=float(w["half"][g]), degeneracy=float(w["count"][g]), importance=float(importance[g]),
                                           potentials=[int(v) for v in w["pots"][g].tolist()], kept=bool(keep[g])))
            if nleg == 2:
                continue                                              # single scattering is exact for every pair (the pair tables)
            kept_chain = keep[w["group"]]
            P = w["P"]
            weight = torch.zeros(P, device=dev, dtype=w["deg"].dtype).index_add_(0, w["path"][kept_chain], torch.ones(int(kept_chain.sum()), device=dev, dtype=w["deg"].dtype))
            forward, back = kept_chain & ~w["backward"], kept_chain & w["backward"]
            ty = torch.full((P,), -1, device=dev, dtype=torch.long)
            ty[w["path"][forward]] = index[w["group"][forward]]
            ty_back = torch.full((P,), -1, device=dev, dtype=torch.long)
            ty_back[w["path"][back]] = index[w["group"][back]]
            rows = weight > 0
            if bool(rows.any()):
                self.lattice[nleg] = dict(atoms=w["atoms"][rows], absorber=w["absr"][rows], ipots=w["ipots"][rows], deg=weight[rows],
                                          type=ty[rows], type_back=ty_back[rows])
        self._lattice_absorbers = absorbers
        return {nleg: int(v["atoms"].shape[0]) for nleg, v in self.lattice.items()}

    def _lists_from_lattice(self, C, n):
        if self.lattice is None:
            raise RuntimeError("criteria_rule = 'evax' chooses the paths on the lattice: call set_lattice(sites, ...) first")
        if n != self.lattice_n:
            raise ValueError(f"the configuration has {n} atoms but the lattice of the paths has {self.lattice_n}")
        shift = n * torch.arange(C, device=self.device)
        return {nleg: dict(atoms=(v["atoms"][None] + shift[:, None, None]).reshape(-1, v["atoms"].shape[1]),
                           absorber=(v["absorber"][None] + shift[:, None]).reshape(-1), ipots=v["ipots"].repeat(C, 1), deg=v["deg"].repeat(C),
                           type=v["type"].repeat(C), type_back=v["type_back"].repeat(C)) for nleg, v in self.lattice.items()}

    @torch.no_grad()
    def type_signals(self, pos_flat):
        """The mean signal per absorber of every kept group of multiple-scattering paths (``criteria_rule = "evax"``),
        at the positions ``pos_flat`` (C n, 3): (number of entries of ``self.types``, NK). Rows of groups that were
        not kept, and of the groups of two legs, are zero. The rows add up to the multiple-scattering spectrum."""
        out = torch.zeros(len(self.types), self.NK, device=self.device, dtype=torch.float64)
        for L in self.lists.values():
            step = self._batch(L["atoms"].shape[1] + 1)
            for s in range(0, L["atoms"].shape[0], step):
                chi = self.path_signals(pos_flat, L, s, s + step)[0].double()
                for key in ("type", "type_back"):                    # a path counts once in the group of each kept direction
                    t = L[key][s:s + step]
                    out.index_add_(0, t[t >= 0], chi[t >= 0])
        return out / max(self.n_abs, 1)

    def enumerate(self, pos, ipot=None, absorbers=None):
        """Find the paths of every absorber of every copy of ``pos`` (C, n, 3).

        Atom indices are global (copy c, atom i -> c n + i). Row ``c A + j`` holds absorber j of copy c; ``row_of``
        maps a global atom index to its row (-1: not an absorber) and ``abs_atoms`` a row to its global atom index.
        Returns the number of paths per number of legs. With ``criteria_rule = "evax"`` the paths are those chosen on
        the lattice (``set_lattice``), whatever the positions."""
        pos = torch.as_tensor(pos, device=self.device, dtype=torch.float32)
        C, n, _ = pos.shape
        ipot = torch.ones(n, dtype=torch.long, device=self.device) if ipot is None else torch.as_tensor(ipot, device=self.device, dtype=torch.long)
        absorbers = torch.arange(n, device=self.device) if absorbers is None else torch.as_tensor(absorbers, device=self.device, dtype=torch.long)
        A = int(absorbers.shape[0])
        if self.criteria_rule == "evax":
            self.lists = self._lists_from_lattice(C, n)
        else:
            parts = {}
            for c in range(C):
                for a in absorbers.tolist():
                    kept, _ = self.full.finder.enumerate(pos[c], self.box, a, ipot)
                    for nleg in sorted(kept):
                        if nleg < 3 or kept[nleg][0].shape[0] == 0:
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

    def set_cell(self, box):
        """Use another periodic cell from now on (a cubic length, a 3 x 3 cell, or None): for configurations whose
        cell changes, such as the frames of a constant-pressure trajectory. The cell must be at least twice as wide
        as the path cutoff."""
        cell = as_cell(box, self.device)
        rmax = float(self.full.finder.rmax)
        if cell is not None and min_width(cell) < 2 * rmax - 1e-6:
            raise ValueError(f"the periodic cell (smallest width {min_width(cell):.2f} A) must exceed twice the path cutoff ({rmax} A) "
                             "for minimum-image paths: use a larger supercell or a smaller cutoff")
        self.box = cell
        self.full.box = cell

    def signals(self, coords, ipots):
        """The signals of explicit paths: ``coords`` (B, nleg, 3) in A with the absorber first at the origin, ``ipots``
        (B, nleg) with 0 for the absorber. Returns chi (B, NK) on this object's k grid (without the degeneracy) and
        the effective half path lengths (B,) in A."""
        cchi, reff = self.full.stage.cchi(coords.to(self.full.rdtype), ipots)
        return self.full.stage.chi_fine(cchi, reff, self.full.q, self.full.S, "feff"), reff

    def path_signals(self, pos_flat, L, start=0, stop=None):
        """The signals of the paths ``start:stop`` of one list ``L`` (an entry of ``self.lists``) at the positions
        ``pos_flat`` (C n, 3): chi (B, NK) without the degeneracy, and the effective half path lengths (B,). Use it to
        group paths, for example by the elements on them or by their length."""
        atoms, absr = L["atoms"][start:stop], L["absorber"][start:stop]
        rel = wrap(pos_flat[atoms] - pos_flat[absr][:, None, :], self.box)
        coords = torch.cat([torch.zeros(rel.shape[0], 1, 3, device=self.device, dtype=rel.dtype), rel], 1)
        return self.signals(coords, L["ipots"][start:stop])

    def _chi_paths(self, pos_flat, L, idx, local, A, grad):
        out = torch.zeros(A, self.NK, device=self.device, dtype=self.full.rdtype)
        step = self._batch(L["atoms"].shape[1] + 1, grad)
        for s in range(0, len(idx), step):
            sl = idx[s:s + step]
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
