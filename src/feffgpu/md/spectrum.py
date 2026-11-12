"""The EXAFS spectrum of one frame for one absorbing element, decomposed by path type.

The spectrum of a frame is the mean over the chosen absorbing atoms of FEFF's chi(k): S0^2 = 1 and no Debye-Waller
factor, because the disorder is in the positions. It has two parts.

Single scattering (two legs)
    ``feffgpu.exafs.PairTables`` tabulates the exact two-leg signal of the path stage once per scatterer potential on
    a fine distance grid. Here the absorber-neighbor distances of the frame (minimum image, every pair within the
    cutoff, no importance filter) are histogrammed on that grid with the same linear-interpolation weights, one
    histogram per scatterer potential, and multiplied by the tables. This gives the single-scattering part by
    neighbor element at no extra cost, and the histogram itself is kept so that any distance range can be selected
    later without a new calculation.
Multiple scattering (three and four legs)
    ``feffgpu.exafs.MultipleScattering`` enumerates the paths of every chosen absorber with FEFF's path finder (and
    its importance filter, ``criteria``); the path stage evaluates them. Each path is added to the part of its path
    type (``feffgpu.md.paths``) and of its effective length (half the path length, in bins of ``ms_bin_width``), so
    the multiple-scattering part can be selected by type and by length later.

The sum of the parts by type equals the total; the sum of the single- and multiple-scattering parts equals the total.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
import torch
from ..cell import as_cell, min_width
from ..exafs import MultipleScattering, PairTables
from ..pad import BOHR, PhaseData
from .paths import PathTypes

__all__ = ["FrameSpectrum", "EdgeSpectra", "choose_absorbers", "phase_k_range"]


def phase_k_range(phase):
    """Largest k (1/A) of a phase file's real-axis grid: spectra must not reach beyond it."""
    return float(phase.xk[: phase.ne1].max() / BOHR)


def choose_absorbers(candidates, count, seed, frame_index, element):
    """The absorbing atoms of one frame.

    Parameters
    ----------
    candidates : array of int
        Indices of the atoms of the absorbing element.
    count : int or "all"
        How many to use; a number at least as large as the candidates means all of them.
    seed : int
    frame_index : int
        Position of the frame in the trajectory.
    element : str

    Returns
    -------
    numpy.ndarray of int
        Sorted indices. A subset is drawn with ``feffgpu.parallel.keyed_rng(seed, "frame", frame_index, element)``, so
        it depends only on the seed, the frame and the element, never on the order of computation or the shards.
    """
    from ..parallel import keyed_rng
    cand = np.asarray(candidates, dtype=np.int64)
    if count == "all" or int(count) >= len(cand):
        return np.sort(cand)
    rng = keyed_rng(int(seed), "frame", int(frame_index), str(element))
    return np.sort(rng.choice(cand, size=int(count), replace=False))


@dataclass
class FrameSpectrum:
    """The spectrum of one frame for one edge.

    Attributes
    ----------
    chi : numpy.ndarray, shape (NK,)
        Total chi(k), the mean over the frame's absorbers.
    chi_ss, chi_ms : numpy.ndarray, shape (NK,)
        Single- and multiple-scattering parts.
    chi_types : numpy.ndarray, shape (T, NK)
        Parts by path type, in the order of ``PathTypes.names``.
    ss_histogram : numpy.ndarray, shape (P, NR)
        Absorber-neighbor distance histogram per scatterer potential on the pair-table grid, per absorber (the
        linear-interpolation weights of ``PairTables``).
    ms_binned : numpy.ndarray, shape (M, NB, NK)
        Multiple-scattering parts by type and by effective-length bin.
    n_absorbers : int
    n_paths : dict
        Number of multiple-scattering paths by number of legs.
    n_short_pairs : int
        Pairs closer than the start of the pair table (placed on its first point; should be zero).
    """
    chi: np.ndarray
    chi_ss: np.ndarray
    chi_ms: np.ndarray
    chi_types: np.ndarray
    ss_histogram: np.ndarray
    ms_binned: np.ndarray
    n_absorbers: int
    n_paths: dict = field(default_factory=dict)
    n_short_pairs: int = 0


class EdgeSpectra:
    """Spectra of frames for one absorbing element: single scattering from pair tables, multiple scattering from the
    path finder and the path stage, decomposed by path type.

    Parameters
    ----------
    phase : str, Path or feffgpu.PhaseData
        The phase file of this edge (``phase.pad`` or the object from ``feffgpu phase``). Potential 0 is the
        absorber; ``ipot_of`` gives the scatterer potential of every element.
    absorber : str
        Element of the absorbing atoms.
    ipot_of : dict
        ``{element: scatterer potential index}`` for every element whose atoms can be neighbors.
    elements : list of str
        Every element of the system.
    k : array, shape (NK,)
        The uniform k grid from zero (``feffgpu.exafs.k_grid``), 1/A.
    rpath : float
        Path cutoff (A): half the total path length, FEFF's RPATH. Single scattering includes every pair up to it.
    nleg : int
        Largest number of legs: 2 (single scattering only), 3 or 4.
    criteria : float
        FEFF's plane-wave importance filter for multiple-scattering paths (percent of the shortest single-scattering
        path's importance).
    pair_rmin, pair_dr : float
        Start and step of the pair-table distance grid (A).
    ms_bin_width : float
        Width of the effective-length bins of the multiple-scattering parts (A); 0 for one bin.
    device : str
        Torch device.
    absorber_batch : int
        Absorbers per batch of the distance histogram; 0 chooses it from the free GPU memory.
    path_chunk : int
        Paths per batch of the path stage; 0 chooses it from the free GPU memory.
    ms_group : int
        Absorbers whose paths are enumerated together (bounds the memory of the path lists).
    """

    def __init__(self, phase, absorber, ipot_of, elements, k, rpath=6.0, nleg=4, criteria=2.0, pair_rmin=1.0, pair_dr=0.002,
                 ms_bin_width=0.5, device="cuda", absorber_batch=0, path_chunk=0, ms_group=64):
        from .potentials import load_phase
        self.phase = phase if (isinstance(phase, PhaseData) or hasattr(phase, "iz")) else load_phase(phase)
        self.device = torch.device(device)
        self.absorber = str(absorber)
        self.k = np.asarray(k, float)
        self.NK = len(self.k)
        self.rpath, self.nleg, self.criteria = float(rpath), int(nleg), float(criteria)
        if self.nleg not in (2, 3, 4):
            raise ValueError(f"nleg must be 2, 3 or 4 (got {nleg}); the path finder of this package stops at four legs")
        kmax_phase = phase_k_range(self.phase)
        if self.k[-1] > kmax_phase + 1e-6:
            raise ValueError(f"kmax {self.k[-1]:.2f} 1/A is beyond the phase file's k range ({kmax_phase:.2f} 1/A): lower kmax or build "
                             "the phase file with a larger k range")
        self.ipot_of = {str(e): int(p) for e, p in ipot_of.items()}
        nph = int(self.phase.nph)
        bad = {e: p for e, p in self.ipot_of.items() if not (1 <= p <= nph)}
        if bad:
            raise ValueError(f"scatterer potentials {bad} are not in the phase file (potentials 1 to {nph})")
        self.ss_potentials = sorted(set(self.ipot_of.values()))
        elem_of_pot = {}
        for e, p in sorted(self.ipot_of.items(), key=lambda t: t[1]):
            elem_of_pot.setdefault(p, e)
        self.ss_elements = [elem_of_pot[p] for p in self.ss_potentials]
        self.types = PathTypes(self.absorber, self.ss_elements, list(elements), self.nleg, elem_of_pot)
        self.ms_bin_width = float(ms_bin_width)
        self.n_bins = 1 if self.ms_bin_width <= 0 else int(math.ceil(self.rpath / self.ms_bin_width - 1e-9))
        self.bin_edges = np.array([0.0, self.rpath]) if self.n_bins == 1 else self.ms_bin_width * np.arange(self.n_bins + 1)
        self.absorber_batch, self.path_chunk, self.ms_group = int(absorber_batch), int(path_chunk), max(1, int(ms_group))
        table_chunk = self.path_chunk if self.path_chunk > 0 else 1024
        zeros = np.zeros(self.NK)
        self.pairs = PairTables(self.phase, self.k, zeros, 0.0, window=(self.k[1], self.k[-1]), r_range=(1.0, self.rpath), box=None,
                                rmax=self.rpath, r_min=pair_rmin, dr=pair_dr, ms=False, chunk=table_chunk, device=str(self.device))
        self.r0, self.dr, self.nr = self.pairs.r0, self.pairs.dr, self.pairs.nr
        self.ms = None
        if self.nleg >= 3:
            self.ms = MultipleScattering(self.phase, self.k, 0.0, None, rmax=self.rpath, critpw=self.criteria, chunk=table_chunk,
                                         device=str(self.device), nlegmax=self.nleg)
        self._chunks = {}
        self._ipot_cache = (None, None)

    # -- bookkeeping -----------------------------------------------------------------------------------------------
    @property
    def tables(self):
        """The two-leg tables of the scatterer potentials in use, (P, NR, NK) float32 numpy, in the order of
        ``ss_elements``."""
        return np.stack([self.pairs.tables[p].cpu().numpy() for p in self.ss_potentials])

    @property
    def r_table(self):
        """The distance grid of the pair tables (A), (NR,)."""
        return self.r0 + self.dr * np.arange(self.nr)

    def atom_potentials(self, symbols):
        """The scatterer potential of every atom (n,) as a tensor; raises if an element has no potential."""
        key = tuple(symbols)
        if self._ipot_cache[0] == key:
            return self._ipot_cache[1]
        sym = np.asarray(symbols)
        missing = sorted({str(s) for s in sym} - set(self.ipot_of))
        lone = missing == [self.absorber] and int((sym == self.absorber).sum()) == 1     # a single absorbing atom is never a neighbor
        if missing and not lone:
            raise ValueError(f"elements {missing} have no scatterer potential in the {self.absorber} edge's phase file (potentials: "
                             f"{self.ipot_of}); give them in the edge's ipots table or use a phase file that includes them")
        t = torch.tensor([self.ipot_of.get(str(s), 0) for s in sym], device=self.device, dtype=torch.long)
        self._ipot_cache = (key, t)
        return t

    def check_cell(self, cell):
        """Raise if the periodic cell is too narrow for minimum-image paths up to ``rpath``."""
        if cell is None:
            return
        w = min_width(as_cell(cell, self.device))
        if w < 2 * self.rpath - 1e-6:
            raise ValueError(f"the cell's smallest width ({w:.3f} A) is less than twice rpath ({self.rpath} A): repeat the cell for the "
                             "calculation ([lattice] replicate = [2, 2, 2] or as needed) or lower rpath")

    def _free_bytes(self):
        if self.device.type != "cuda":
            return 2 ** 30
        free, _total = torch.cuda.mem_get_info(self.device)
        return int(free)

    def _absorber_batch_for(self, n_atoms, n_absorbers):
        if self.absorber_batch > 0:
            return min(self.absorber_batch, n_absorbers)
        per_absorber = 64 * max(n_atoms, 1)                                 # bytes: displacements, squared distances, masks
        return int(max(1, min(n_absorbers, 0.25 * self._free_bytes() // per_absorber, 4096)))

    def _chunk_for(self, nleg):
        """Paths per batch of the path stage: the manual value, or from the free GPU memory and the memory a small
        test batch of this number of legs needs."""
        if self.path_chunk > 0:
            return self.path_chunk
        if nleg in self._chunks:
            return self._chunks[nleg]
        if self.device.type != "cuda":
            self._chunks[nleg] = 256
            return 256
        probe = 32
        ip = self.ss_potentials[0]
        coords = torch.zeros(probe, nleg, 3, device=self.device, dtype=torch.float32)
        for j in range(1, nleg):                                            # a zigzag chain with 2.5 A legs
            coords[:, j, 0] = 2.0 * j
            coords[:, j, 1] = 1.5 * (j % 2)
        ipots = torch.full((probe, nleg), ip, device=self.device, dtype=torch.long)
        ipots[:, 0] = 0
        torch.cuda.synchronize(self.device)
        base = torch.cuda.memory_allocated(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        with torch.no_grad():
            self.ms.signals(coords, ipots)
        peak = torch.cuda.max_memory_allocated(self.device) - base
        per_path = max(peak / probe, 1.0)
        chunk = int(max(16, min(8192, 0.4 * self._free_bytes() / per_path)))
        self._chunks[nleg] = chunk
        return chunk

    # -- single scattering -----------------------------------------------------------------------------------------
    def _single_scattering(self, pos, cell, absorbers, ipot):
        """Histogram (nph + 1, NR) float64 of absorber-neighbor distances, summed over the absorbers, and the number
        of pairs closer than the table start (``PairTables.histogram`` with this frame's cell)."""
        batch = self._absorber_batch_for(pos.shape[0], int(absorbers.shape[0]))
        return self.pairs.histogram(pos, absorbers, ipot, box=as_cell(cell, self.device), batch=batch)

    # -- multiple scattering ---------------------------------------------------------------------------------------
    def _multiple_scattering(self, pos, cell, absorbers, ipot):
        """Multiple-scattering parts (M, NB, NK) float64 summed over the absorbers, and the path counts."""
        M, NB = len(self.types.ms), self.n_bins
        out = torch.zeros(M * NB, self.NK, device=self.device, dtype=torch.float64)
        counts = {}
        if self.ms is None or M == 0:
            return out.view(M, NB, self.NK), counts
        self.ms.set_cell(cell)                                                     # this frame's cell for the path finder
        npot = int(self.phase.nph) + 1
        width = self.ms_bin_width
        for g in range(0, int(absorbers.shape[0]), self.ms_group):
            group = absorbers[g:g + self.ms_group]
            self.ms.enumerate(pos[None], ipot, group)
            for nleg, L in self.ms.lists.items():
                P = int(L["atoms"].shape[0])
                counts[nleg] = counts.get(nleg, 0) + P
                if P == 0:
                    continue
                lut = self.types.lookup(nleg, npot, self.device)
                seq = L["ipots"]
                tidx = lut[seq[:, 1], seq[:, 2]] if nleg == 3 else lut[seq[:, 1], seq[:, 2], seq[:, 3]]
                if bool((tidx < 0).any()):
                    raise RuntimeError("a multiple-scattering path has no path type (inconsistent potentials)")
                chunk = self._chunk_for(nleg)
                for s in range(0, P, chunk):
                    chi, reff = self.ms.path_signals(pos, L, s, s + chunk)         # (B, NK), (B,)
                    if NB == 1:
                        b = torch.zeros_like(tidx[s:s + chunk])
                    else:
                        b = torch.floor(reff.double() / width).long().clamp(0, NB - 1)
                    out.index_add_(0, tidx[s:s + chunk] * NB + b, L["deg"][s:s + chunk, None].double() * chi.double())
        return out.view(M, NB, self.NK), counts

    # -- one frame -------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def spectrum(self, positions, cell, absorbers, symbols):
        """The spectrum of one frame.

        Parameters
        ----------
        positions : array, shape (n, 3)
            Cartesian positions (A).
        cell : array (3, 3) or None
            Periodic cell (lattice vectors as rows), or None for a cluster.
        absorbers : array of int, shape (A,)
            Indices of the absorbing atoms (atoms of this edge's element).
        symbols : array of str, shape (n,)
            Element of every atom.

        Returns
        -------
        FrameSpectrum
        """
        self.check_cell(cell)
        A = int(len(absorbers))
        if A == 0:
            raise ValueError(f"no absorbing {self.absorber} atom in the frame")
        pos = torch.as_tensor(np.asarray(positions, float), device=self.device, dtype=torch.float32)
        ab = torch.as_tensor(np.asarray(absorbers, dtype=np.int64), device=self.device)
        ipot = self.atom_potentials(symbols)
        hist, short = self._single_scattering(pos, cell, ab, ipot)
        hist = hist[self.ss_potentials] / A                                        # (P, NR) per absorber
        tables = self.pairs.table_stack(self.ss_potentials)
        chi_ss_types = torch.einsum("pr,prk->pk", hist, tables)                    # (P, NK)
        ms, counts = self._multiple_scattering(pos, cell, ab, ipot)
        ms = ms / A
        chi_ms_types = ms.sum(1)                                                   # (M, NK)
        chi_types = torch.cat([chi_ss_types, chi_ms_types], 0)
        chi_ss, chi_ms = chi_ss_types.sum(0), chi_ms_types.sum(0)
        return FrameSpectrum(chi=(chi_ss + chi_ms).cpu().numpy(), chi_ss=chi_ss.cpu().numpy(), chi_ms=chi_ms.cpu().numpy(),
                             chi_types=chi_types.cpu().numpy(), ss_histogram=hist.cpu().numpy(), ms_binned=ms.cpu().numpy(),
                             n_absorbers=A, n_paths={int(k): int(v) for k, v in counts.items()}, n_short_pairs=int(short))
