"""The forward model of one absorption edge of a structure, compared with a measured spectrum.

``EdgeModel`` is a ``SpectrumComparison`` (the measured chi(k), the window, S0^2 and the E0 shift on the model, the
R-factor) that also knows the structure: the element of every atom, the periodic cell, the absorbing element and the
scattering potential of every element in the phase file. It calculates the spectrum of configurations of that
structure (copies of the cell, frames of a trajectory, states of a reverse Monte Carlo model):

- single scattering, exact for every pair: the absorber-neighbor distances of every configuration are histogrammed
  on the 0.002 A grid of the pair tables (``feffgpu.exafs.PairTables``: the two-leg signal of the path stage of every
  scattering potential), and the histogram times the tables is the spectrum, averaged over the absorbers. The
  histogram is differentiable in the positions, so the misfit has an exact gradient;
- multiple scattering (three- and four-leg paths), exact for the paths it holds: FEFF's path finder enumerates the
  paths of every absorber and the path stage evaluates them (``feffgpu.exafs.MultipleScattering``).

The calculated spectrum is FEFF's (S0^2 = 1, no Debye-Waller factor): the disorder is in the positions.

Pair types. The pairs of every neighbor element are counted between a lowest and a highest distance (by default from
``pair_min`` to ``rpath``). ``pair_windows`` can split that range into windows (pair types); the windows do not
change the spectrum, they only order the sum.
"""
from __future__ import annotations
import numpy as np
import torch
from ..cell import as_cell, min_width, wrap
from ..pad import load_phase
from .comparison import SpectrumComparison
from .forward import PairTables, MultipleScattering
from .transform import FitWindow
from .wavelet import make_wavelet

__all__ = ["EdgeModel", "PairType"]

OBJECTIVES = {"r_factor": "r", "wavelet": "w", "cauchy_wavelet": "cw"}


class PairType:
    """One pair type of an edge: the pairs of the absorber with the atoms of one element at distances ``lo <= r < hi``.

    Attributes
    ----------
    name : str
        ``"absorber-element"``.
    element : str
    lo, hi : float
        The distance window (A).
    ipot : int
        The scattering potential of the element in the phase file.
    """
    __slots__ = ("name", "element", "lo", "hi", "ipot")

    def __init__(self, name, element, lo, hi, ipot):
        self.name, self.element, self.lo, self.hi, self.ipot = str(name), str(element), float(lo), float(hi), int(ipot)

    def __repr__(self):
        return f"{self.name} {self.lo:.2f}-{self.hi:.2f} A"


class EdgeModel(SpectrumComparison):
    """The spectrum of one absorption edge of a structure and its comparison with a measured spectrum.

    Parameters
    ----------
    name : str
        Label of the spectrum. Spectra with the same name (the same edge measured on several samples) share S0^2 and
        the E0 shift in a joint fit (``feffgpu.exafs.fit_shared``).
    k : array, shape (NK,)
        The uniform k grid from zero, step 0.05 1/A (``feffgpu.exafs.k_grid``).
    data : array, shape (NK,)
        The measured chi(k) on that grid (``feffgpu.exafs.to_grid``), as measured.
    window : FitWindow
        The k window, the k weight and the R window of the comparison.
    phase : PhaseData, str or Path
        The phase file of this edge (anything ``feffgpu.load_phase`` accepts). Its potential 0 is the absorber.
    symbols : array of str, shape (n,)
        The element of every atom of the structure.
    cell : None, float, (3,) or (3, 3)
        The periodic cell (A): a cubic length, three lengths, lattice vectors as rows, or None for a cluster.
    absorber : str
        The absorbing element.
    potentials : dict, optional
        ``{element: scattering potential index}`` (1 or more) in the phase file. Required for every element the model
        sees, unless the phase file has a single scattering potential (then every element uses it).
    e0 : float
        Energy shift (eV) at which the pair tables and the paths are evaluated (the start of the E0 shift).
    s02 : float
        Amplitude factor (start value if ``fit_s02``).
    fit_s02, fit_e0 : bool
        Whether a fit may change S0^2 and the E0 shift.
    rpath : float
        Pair and path cutoff (A, FEFF's RPATH: half the path length). The cell must be at least twice as wide.
    nleg : int
        Largest number of legs of the paths (2: single scattering only, 3 or 4).
    criteria : float
        The importance filter of the multiple-scattering paths (percent).
    criteria_rule : str
        What ``criteria`` measures. ``"feff"`` (default): FEFF's plane-wave importance of each path of the structure as
        it is, relative to the shortest single-scattering path. ``"evax"``: EvAX's rule: the size of the signal of each
        group of paths on the lattice, relative to the whole spectrum of the lattice; the paths are then chosen once
        (``ms.set_lattice``) and kept for every structure.
    group_tolerance : float
        For ``criteria_rule = "evax"``: paths whose atoms lie within this distance (A) of each other, atom by atom,
        belong to one group (EvAX's ``Pathfinder_robustness``).
    multiple_scattering : bool
        False: single scattering only.
    pair_windows : dict, optional
        ``{element: [b0, b1, ...]}``: the pairs with that element are counted between ``b0`` and the last boundary,
        split at the inner boundaries (A). Default for every element: ``[pair_min, rpath]``.
    pair_min : float
        Start of the pair tables (A): the shortest pair distance that can be counted.
    table_pad : float
        The pair tables reach ``rpath + table_pad`` (A).
    chunk : int
        Paths per batch of the path stage.
    state_chunk : int
        Configurations per batch of the pair histograms.
    ignore : sequence of str
        Elements the forward model does not see (they neither scatter nor absorb).
    objective : str
        ``"r_factor"`` (xraylarch's R-factor on chi(R), the default), ``"wavelet"`` (EvAX's wavelet residual) or
        ``"cauchy_wavelet"`` (xraylarch's Cauchy wavelet).
    wavelet_r0 : float, optional
        EvAX's ``R0_for_WT`` for the ``"wavelet"`` objective (A; default the middle of the R window).
    device : str

    Attributes
    ----------
    types : list of PairType
    tables : torch.Tensor, shape (T, nr, NK)
        The pair table of every pair type.
    pairs : PairTables
    ms : MultipleScattering or None
    abs_idx : torch.Tensor
        Indices of the absorbing atoms; ``nA`` their number.
    keep : torch.Tensor
        Indices of the atoms the model sees; ``abs_keep`` the absorbers' positions among them.
    ipot : torch.Tensor, shape (n,)
        Scattering potential of every atom (0 for an ignored atom).
    """

    def __init__(self, name, k, data, window: FitWindow, phase, symbols, cell, absorber, potentials=None, e0=0.0, s02=1.0,
                 fit_s02=False, fit_e0=False, rpath=6.0, nleg=4, criteria=2.5, multiple_scattering=True, pair_windows=None,
                 pair_min=1.5, table_pad=1.0, chunk=2048, state_chunk=4, ignore=(), objective="r_factor", wavelet_r0=None,
                 device="cuda", criteria_rule="feff", group_tolerance=0.5):
        if objective not in OBJECTIVES:
            raise ValueError(f"{name}: unknown objective {objective!r}; use one of {sorted(OBJECTIVES)}")
        if int(nleg) not in (2, 3, 4):
            raise ValueError(f"{name}: nleg must be 2, 3 or 4 (got {nleg})")
        super().__init__(name, k, data, window, s02=s02, e0=e0, fit_s02=fit_s02, fit_e0=fit_e0, device=device)
        phase = load_phase(phase)
        self.phase = phase
        self.symbols = np.asarray(symbols).astype(str)
        self.n = len(self.symbols)
        self.absorber = str(absorber)
        self.cell = as_cell(cell, self.device)
        self.rmax, self.critpw, self.nlegmax = float(rpath), float(criteria), int(nleg)
        if self.cell is not None and min_width(self.cell) < 2 * self.rmax - 1e-9:
            raise ValueError(f"{name}: the periodic cell (smallest width {min_width(self.cell):.2f} A) must be at least twice the pair and "
                             f"path cutoff rpath ({self.rmax} A): use a larger supercell or a smaller rpath")
        self.abs_idx = torch.tensor(np.nonzero(self.symbols == self.absorber)[0], device=self.device, dtype=torch.long)
        self.nA = int(self.abs_idx.shape[0])
        if self.nA == 0:
            raise ValueError(f"{name}: the structure has no {self.absorber} atom to absorb")
        self.ignore = set(str(v) for v in ignore)
        if self.absorber in self.ignore:
            raise ValueError(f"{name}: the absorber {self.absorber} cannot be ignored")
        self.keep = torch.tensor([i for i, sy in enumerate(self.symbols) if sy not in self.ignore], device=self.device, dtype=torch.long)
        kept_pos = {int(i): j for j, i in enumerate(self.keep.tolist())}
        self.abs_keep = torch.tensor([kept_pos[int(i)] for i in self.abs_idx.tolist()], device=self.device, dtype=torch.long)
        self.elements = sorted(set(self.symbols.tolist()) - self.ignore)
        self.ipot_of_symbol = self._potentials(potentials, phase)
        self.ipot = torch.tensor([self.ipot_of_symbol[s] if s not in self.ignore else 0 for s in self.symbols], device=self.device, dtype=torch.long)
        self.elem_idx = {e: i for i, e in enumerate(self.elements)}
        self.elem = torch.tensor([self.elem_idx.get(s, -1) for s in self.symbols], device=self.device)
        self.state_chunk, self.table_pad = int(state_chunk), float(table_pad)
        # the tables do not depend on the cell; the cell enters the pair distances only
        self.pairs = PairTables(phase, self.k, self.data, self.e0, window=self.window, r_range=self.r_range, box=None, rmax=self.rmax + self.table_pad,
                                r_min=pair_min, ms=False, chunk=chunk, device=self.device, kweight=self.kweight, dk=self.dk)
        self.r0, self.dr, self.nr = self.pairs.r0, self.pairs.dr, self.pairs.nr
        use_ms = bool(multiple_scattering) and self.nlegmax >= 3
        self.criteria_rule, self.group_tolerance = criteria_rule, float(group_tolerance)
        self.ms = MultipleScattering(phase, self.k, self.e0, cell, rmax=self.rmax, critpw=self.critpw, chunk=chunk, device=self.device,
                                     nlegmax=self.nlegmax, criteria_rule=criteria_rule) if use_ms else None
        self.objective, self.space = objective, OBJECTIVES[objective]
        if objective != "r_factor":
            self.wavelet = make_wavelet(objective, self.k, window, r0=wavelet_r0, device=self.device)
            self.wavelet.set_target(self.data)
        self.types, self._tables = [], None
        self.set_types(pair_windows or {})

    def _potentials(self, potentials, phase):
        nph = int(phase.nph)
        given = {str(k): int(v) for k, v in (potentials or {}).items()}
        out = {}
        for el in self.elements:
            if el in given:
                out[el] = given[el]
            elif nph == 1:
                out[el] = 1
            else:
                raise ValueError(f"{self.name}: no scattering potential for {el}; the phase file has potentials 1 to {nph}: give potentials = {{ {el} = ... }}")
        for el, ip in out.items():
            if not 1 <= ip <= nph:
                raise ValueError(f"{self.name}: potential {ip} of {el} is not a scattering potential of the phase file (1 to {nph})")
        return out

    # -- pair types --------------------------------------------------------------------------------------------
    def set_types(self, boundaries):
        """Define the pair types from ``{element: [b0, b1, ...]}`` (A); elements not listed get one window from the start
        of the pair tables to ``rpath``. Returns the list of ``PairType``."""
        unknown = sorted(set(boundaries) - set(self.elements))
        if unknown:
            raise ValueError(f"{self.name}: pair windows for elements the structure does not have (or that are ignored): {unknown}")
        types = []
        top = self.rmax + self.table_pad - 2 * self.dr
        for el in self.elements:
            bs = [float(v) for v in boundaries.get(el, [self.r0, self.rmax])]
            if len(bs) < 2 or any(b1 <= b0 for b0, b1 in zip(bs, bs[1:])):
                raise ValueError(f"{self.name}, neighbor {el}: the window boundaries must be at least two increasing distances (got {bs})")
            if bs[0] < self.r0 - 1e-9:
                raise ValueError(f"{self.name}, neighbor {el}: the first boundary {bs[0]} A is below the start of the pair tables ({self.r0} A); "
                                 "lower pair_min")
            if bs[-1] < self.rmax - 1e-9 or bs[-1] > top + 1e-9:
                raise ValueError(f"{self.name}, neighbor {el}: the last boundary {bs[-1]} A must reach the cutoff rpath ({self.rmax} A) and stay "
                                 f"below {top:.3f} A")
            for lo, hi in zip(bs[:-1], bs[1:]):
                types.append(PairType(f"{self.absorber}-{el}", el, lo, hi, self.ipot_of_symbol[el]))
        self.types, self._tables = types, None
        return self.types

    @property
    def tables(self):
        """(T, nr, NK): the pair table of the scattering potential of every pair type."""
        if self._tables is None:
            self._tables = torch.stack([self.pairs.tables[t.ipot] for t in self.types])
        return self._tables

    # -- single scattering --------------------------------------------------------------------------------------
    def static_histograms(self, pos):
        """The absorber-neighbor distances of every configuration on the grid of the pair tables.

        Parameters
        ----------
        pos : array or torch.Tensor, shape (S, n, 3)
            Positions (A) of S configurations. With a gradient, the histogram is differentiable through the linear
            interpolation weights.

        Returns
        -------
        torch.Tensor, shape (S, T, nr)
            Per configuration and pair type, the pair count on every grid point (summed over the absorbers).
        """
        pos = torch.as_tensor(pos, device=self.device, dtype=torch.float32)
        S, n, _ = pos.shape
        T = len(self.types)
        out = torch.zeros(S, T, self.nr, device=self.device)
        eye = torch.eye(n, device=self.device, dtype=torch.bool)[self.abs_idx]                 # (nA, n) self pairs
        for s0 in range(0, S, self.state_chunk):
            ps = pos[s0:s0 + self.state_chunk]
            Sc = ps.shape[0]
            rel = wrap(ps[:, self.abs_idx][:, :, None, :] - ps[:, None, :, :], self.cell)      # (Sc, nA, n, 3)
            r2 = (rel ** 2).sum(-1)
            r2 = torch.where(eye[None], torch.full_like(r2, float("inf")), r2)
            h = torch.zeros(Sc * T * self.nr, device=self.device)
            for t, ty in enumerate(self.types):
                sel = (r2 >= ty.lo ** 2) & (r2 < ty.hi ** 2) & (self.elem[None, None, :] == self.elem_idx[ty.element])
                s_idx, a_idx, j_idx = torch.nonzero(sel, as_tuple=True)
                if s_idx.numel() == 0:
                    continue
                r = torch.sqrt(r2[s_idx, a_idx, j_idx])
                u = (r - self.r0) / self.dr
                i0 = torch.floor(u).clamp(0, self.nr - 2).long()
                f = (u - i0).clamp(0, 1)
                key = (s_idx * T + t) * self.nr
                h = h.index_add(0, key + i0, 1 - f).index_add(0, key + i0 + 1, f)
            out[s0:s0 + Sc] = h.view(Sc, T, self.nr)
        return out

    def spectrum(self, hist, tables=None, ms_mean=None):
        """The calculated chi(k) of histograms ``hist`` (S, T, nr), averaged over the absorbers, plus a multiple-scattering
        spectrum ``ms_mean`` (S, NK) if given. Returns (S, NK)."""
        chi = torch.einsum("str,trk->sk", hist, self.tables if tables is None else tables) / self.nA
        if ms_mean is not None:
            chi = chi + ms_mean
        return chi

    # -- multiple scattering -------------------------------------------------------------------------------------
    @torch.no_grad()
    def ms_pass(self, pos, enumerate_paths=True):
        """The multiple-scattering chi(k) (S, NK) of configurations ``pos`` (S, n, 3), averaged over the absorbers.

        The paths of every absorber of every configuration are found again (``enumerate_paths``) or the paths of the
        last search are evaluated at the new positions. Zero without multiple scattering."""
        pos = torch.as_tensor(pos, device=self.device, dtype=torch.float32)
        S = pos.shape[0]
        if self.ms is None:
            return torch.zeros(S, self.NK, device=self.device)
        if self.ignore:
            pos = pos[:, self.keep]
        if enumerate_paths or self.ms.lists is None:
            self.ms.enumerate(pos, self.ipot[self.keep], self.abs_keep)
        rows = self.ms.full_pass(pos.reshape(-1, 3))
        rows = rows.real if rows.is_complex() else rows
        return rows.view(S, self.nA, -1).mean(1).float()

    @torch.no_grad()
    def spectra(self, pos, multiple_scattering=True):
        """The calculated chi(k) (S, NK) of configurations ``pos`` (S, n, 3): exact single scattering plus, if asked and
        available, the multiple scattering of freshly found paths. Unscaled (S0^2 = 1, at the shift ``e0``)."""
        pos = torch.as_tensor(pos, device=self.device, dtype=torch.float32)
        chi = self.spectrum(self.static_histograms(pos))
        if multiple_scattering and self.ms is not None:
            chi = chi + self.ms_pass(pos)
        return chi

    def describe(self):
        obj = {"r_factor": "R-factor", "wavelet": "EvAX's wavelet residual", "cauchy_wavelet": "xraylarch's Cauchy wavelet residual"}[self.objective]
        rule = "EvAX's rule on the lattice" if self.criteria_rule == "evax" else "FEFF's rule on each structure"
        paths = f"paths to {self.rmax} A, {self.nlegmax} legs, {self.critpw} % filter ({rule})" if self.ms is not None else f"single scattering to {self.rmax} A"
        return (super().describe() + f"; {self.nA} {self.absorber} absorbers; {obj}; {paths}; potentials "
                + ", ".join(f"{el} {ip}" for el, ip in self.ipot_of_symbol.items()) + "; pair types " + ", ".join(repr(t) for t in self.types))
