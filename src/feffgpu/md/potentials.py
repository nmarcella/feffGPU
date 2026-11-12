"""Potentials and phase shifts of an edge: an explicit choice between three modes.

``"file"``
    The user's own phase file: a ``phase.pad`` written by FEFF, or a pickle written by ``feffgpu phase``. A pickle is
    a Python object file: load only phase files that you made yourself.
``"reference"`` (default)
    One phase file per edge, computed once from a reference cluster cut from a chosen frame around a chosen absorbing
    atom (by default the first frame of the run and the first atom of the absorbing element): the potentials (free
    atoms, optionally the self-consistency loop) and the phase shifts are calculated on the GPU
    (``feffgpu.cli.build_phase``). The cluster's FEFF input, the phase file and a summary are kept in the output
    folder (``potentials/<edge name>/``) and reused by later runs and by every shard.
``"per_frame"``
    The same calculation repeated: new potentials every ``potentials_every`` selected frames, each from the frame
    that starts its block of frames.

The potential index of every element is fixed by the input of the calculation: ``0`` is the absorbing atom and
``1, 2, ...`` the elements in alphabetical order (the rule of ``md_exafs`` 2.0). For a phase file given by the user,
an element's potential is the first scatterer potential with its atomic number, unless the edge's ``ipots`` table
says otherwise. The same indices are used for every frame.
"""
from __future__ import annotations
import json
import math
import os
import pickle
import time
from pathlib import Path
import numpy as np
from ..io.structure import SYMBOLS, Z_OF
from ..pad import load_phase

__all__ = ["potentials_of_phase", "needed_elements", "cut_cluster", "reference_input", "PhaseProvider", "EDGE_NAMES"]

EDGE_NAMES = ("K", "L1", "L2", "L3", "M1", "M2", "M3", "M4", "M5")


def needed_elements(symbols, absorber):
    """The elements that need a scatterer potential: every element of the system, the absorbing element too unless it
    has a single atom (then it is only ever the absorber)."""
    sym = np.asarray(symbols)
    out = sorted(set(str(s) for s in sym))
    if int((sym == absorber).sum()) <= 1 and absorber in out:
        out.remove(absorber)
    return out


def potentials_of_phase(phase, absorber, elements, override=None):
    """``{element: scatterer potential}`` for a phase file.

    Parameters
    ----------
    phase : feffgpu.PhaseData
    absorber : str
        The absorbing element; it must be the phase file's potential 0.
    elements : list of str
        Elements that need a potential (``needed_elements``).
    override : dict, optional
        ``{element: potential}`` given by the user; checked against the phase file's atomic numbers.

    Returns
    -------
    dict
    """
    iz = [int(z) for z in phase.iz]
    if iz[0] != Z_OF[absorber]:
        raise ValueError(f"the phase file's absorbing potential is Z = {iz[0]} ({SYMBOLS[iz[0]]}), not {absorber}")
    out = {}
    override = {str(k): int(v) for k, v in (override or {}).items()}
    for el in elements:
        if el in override:
            p = override[el]
            if not (1 <= p < len(iz)):
                raise ValueError(f"ipots: {el} = {p} is not a scatterer potential of the phase file (1 to {len(iz) - 1})")
            if iz[p] != Z_OF[el]:
                raise ValueError(f"ipots: potential {p} of the phase file is Z = {iz[p]} ({SYMBOLS[iz[p]]}), not {el}")
            out[el] = p
            continue
        match = [p for p in range(1, len(iz)) if iz[p] == Z_OF[el]]
        if not match:
            have = ", ".join(f"{p}: {SYMBOLS[z]}" for p, z in enumerate(iz))
            raise ValueError(f"the phase file has no scatterer potential for {el} (its potentials are {have})")
        out[el] = match[0]
    return out


def cut_cluster(positions, symbols, cell, center, radius):
    """The atoms within ``radius`` of atom ``center``, positions relative to it, with every periodic image needed.

    Parameters
    ----------
    positions : array, shape (n, 3)
    symbols : array of str, shape (n,)
    cell : array (3, 3) or None
    center : int
    radius : float
        A.

    Returns
    -------
    rel : numpy.ndarray, shape (m, 3)
        The center first (at the origin), then the others by increasing distance.
    sym : numpy.ndarray of str, shape (m,)
    dist : numpy.ndarray, shape (m,)
    """
    pos = np.asarray(positions, float)
    sym = np.asarray(symbols)
    rel = pos - pos[center]
    if cell is None:
        shifts = np.zeros((1, 3))
    else:
        c = np.asarray(cell, float)
        frac = rel @ np.linalg.inv(c)
        rel = (frac - np.round(frac)) @ c
        vol = abs(np.linalg.det(c))
        widths = [vol / np.linalg.norm(np.cross(c[(i + 1) % 3], c[(i + 2) % 3])) for i in range(3)]
        reach = [int(math.ceil(radius / w + 0.5)) for w in widths]
        grid = np.stack(np.meshgrid(*[np.arange(-m, m + 1) for m in reach], indexing="ij"), -1).reshape(-1, 3)
        shifts = grid @ c
    allrel = (rel[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
    allsym = np.tile(sym, len(shifts))
    d = np.linalg.norm(allrel, axis=1)
    keep = d <= radius
    allrel, allsym, d = allrel[keep], allsym[keep], d[keep]
    order = np.argsort(d, kind="stable")
    allrel, allsym, d = allrel[order], allsym[order], d[order]
    if d[0] > 1e-6:
        raise RuntimeError("the cluster center is missing")
    return allrel, allsym, d


def reference_input(rel, sym, absorber, edge="K", scf_radius=0.0, phase_kmax=22.0, rpath=6.0, cards=(), title=""):
    """The FEFF input text of a reference cluster and the potential index of every element.

    Parameters
    ----------
    rel : array, shape (m, 3)
        Positions relative to the absorbing atom (first row, at the origin), A.
    sym : array of str, shape (m,)
    absorber : str
    edge : str
        Absorption edge (``"K"``, ``"L3"``, ...).
    scf_radius : float
        Radius of the self-consistency loop (A); 0 for no self-consistency (FEFF's default).
    phase_kmax : float
        Largest k of the phase grid (1/A), written as the EXAFS card.
    rpath : float
        Written as the RPATH card (it does not change the phase file).
    cards : sequence of str
        Extra FEFF cards, written as given.
    title : str

    Returns
    -------
    text : str
    ipot_of : dict
        ``{element: scatterer potential}``.
    """
    sym = np.asarray(sym)
    others = sorted(set(str(s) for s in sym[1:]))
    ipot_of = {el: i + 1 for i, el in enumerate(others)}
    lines = [f"TITLE {title}" if title else "TITLE md-exafs reference cluster", "CONTROL 1 1 1 1 1 1", "PRINT 0 0 0 0 0 0",
             f"EDGE {edge}", "S02 1.0"]
    if scf_radius and scf_radius > 0:
        lines.append(f"SCF {float(scf_radius):.3f} 0 30 0.2 1")
    lines += [f"EXAFS {float(phase_kmax):.2f}", f"RPATH {float(rpath):.3f}"]
    lines += [str(c) for c in cards]
    lines += ["", "POTENTIALS", f"    0  {Z_OF[absorber]:3d}  {absorber}"]
    lines += [f"    {ipot_of[el]}  {Z_OF[el]:3d}  {el}" for el in others]
    lines += ["", "ATOMS"]
    for j, (r, s) in enumerate(zip(rel, sym)):
        ip = 0 if j == 0 else ipot_of[str(s)]
        lines.append(f"  {r[0]:12.6f} {r[1]:12.6f} {r[2]:12.6f}  {ip}  {s:<3s} {np.linalg.norm(r):9.5f}")
    lines += ["END", ""]
    return "\n".join(lines), ipot_of


def _atomic_write(path, data, binary=False):
    tmp = Path(str(path) + f".tmp{os.getpid()}")
    with open(tmp, "wb" if binary else "w") as f:
        f.write(data)
    os.replace(tmp, path)


class PhaseProvider:
    """The phase file (and the element potentials) of one edge for every frame of a run.

    Parameters
    ----------
    edge : feffgpu.md.config.EdgeConfig
        The edge's settings (mode, phase file, reference frame and atom, cluster radius, ...).
    output : Path
        The run's output folder; phase files computed here go to ``output / "potentials" / edge.name``.
    selection : list of int
        Positions of the run's frames in the trajectory (for the reference frame and for the blocks of
        ``per_frame``).
    symbols : array of str
        Element of every atom.
    reader : feffgpu.io.trajectory.TrajectoryReader
        To read the reference frames.
    device : str
    log : callable
        Receives progress lines.
    """

    def __init__(self, edge, output, selection, symbols, reader, device="cuda", log=print):
        self.edge, self.output, self.selection = edge, Path(output), list(selection)
        self.symbols = np.asarray(symbols)
        self.reader, self.device, self.log = reader, device, log
        self.folder = self.output / "potentials" / edge.name
        self.elements = needed_elements(self.symbols, edge.element)
        self._loaded = {}

    # which phase file a frame uses ------------------------------------------------------------------------------------
    def source_frame(self, frame_index):
        """The frame whose cluster defines the potentials used for ``frame_index`` (None in ``file`` mode)."""
        mode = self.edge.potentials
        if mode == "file":
            return None
        if mode == "reference":
            return int(self.edge.reference_frame) if self.edge.reference_frame is not None else self.selection[0]
        j = self.selection.index(int(frame_index))
        every = max(1, int(self.edge.potentials_every))
        return self.selection[(j // every) * every]

    def key(self, frame_index):
        """A label of the phase file used for ``frame_index``: ``"file"``, ``"reference"`` or ``"frame_<i>"``."""
        mode = self.edge.potentials
        if mode == "file":
            return "file"
        if mode == "reference":
            return "reference"
        return f"frame_{self.source_frame(frame_index)}"

    def phase_for(self, frame_index):
        """``(phase, ipot_of, key)`` for a frame, computing and caching the phase file if needed."""
        key = self.key(frame_index)
        if key not in self._loaded:
            if key == "file":
                phase = load_phase(self.edge.phase)
                ipot_of = potentials_of_phase(phase, self.edge.element, self.elements, self.edge.ipots)
                self._loaded[key] = (phase, ipot_of, {"mode": "file", "phase": str(self.edge.phase)})
            else:
                self._loaded[key] = self._computed(key, self.source_frame(frame_index))
        phase, ipot_of, _info = self._loaded[key]
        return phase, ipot_of, key

    def info(self, key):
        """The summary of a phase file already loaded (mode, source frame and atom, potentials, timings)."""
        return self._loaded[key][2]

    def prepare(self):
        """Compute (or load) every phase file the run needs; returns their keys."""
        keys = []
        for f in self.selection:
            k = self.key(f)
            if k not in keys:
                self.phase_for(f)
                keys.append(k)
        return keys

    # computing -------------------------------------------------------------------------------------------------------
    def _computed(self, key, frame_index):
        self.folder.mkdir(parents=True, exist_ok=True)
        pkl, meta, inp = self.folder / f"{key}.pkl", self.folder / f"{key}.json", self.folder / f"{key}.feff.inp"
        if pkl.is_file() and meta.is_file():
            info = json.loads(meta.read_text())
            if info.get("settings") == self._settings(frame_index):
                with open(pkl, "rb") as f:
                    phase = pickle.load(f)
                self.log(f"[{self.edge.name}] potentials: using {pkl.relative_to(self.output)} (frame {info['frame']}, "
                         f"atom {info['atom']})")
                return phase, {str(k): int(v) for k, v in info["ipot_of"].items()}, info
            self.log(f"[{self.edge.name}] potentials: settings changed since {pkl.name} was made; computing it again")
        frame = self.reader.read_frame(frame_index)
        cand = np.nonzero(frame.symbols == self.edge.element)[0]
        if len(cand) == 0:
            raise ValueError(f"frame {frame_index} has no {self.edge.element} atom")
        atom = int(self.edge.reference_atom) if self.edge.reference_atom is not None else int(cand[0])
        if frame.symbols[atom] != self.edge.element:
            raise ValueError(f"reference_atom {atom} is a {frame.symbols[atom]} atom, not {self.edge.element}")
        rel, sym, _d = cut_cluster(frame.positions, frame.symbols, frame.cell, atom, self.edge.cluster_radius)
        present = set(str(s) for s in sym[1:])
        missing = [el for el in self.elements if el not in present]
        if missing:
            raise ValueError(f"[{self.edge.name}] the reference cluster (radius {self.edge.cluster_radius} A around atom {atom} of "
                             f"frame {frame_index}) contains no {', '.join(missing)} atom besides the absorber: every element needs "
                             "one to define its potential; use a larger cluster_radius or another reference_atom")
        title = f"md-exafs reference cluster: edge {self.edge.name}, frame {frame_index}, atom {atom}, {len(sym)} atoms"
        text, ipot_of = reference_input(rel, sym, self.edge.element, self.edge.edge, self.edge.scf_radius, self.edge.phase_kmax_resolved,
                                        self.edge.rpath, self.edge.cards, title)
        _atomic_write(inp, text)
        from ..cli import build_phase
        self.log(f"[{self.edge.name}] potentials: computing from {len(sym)} atoms around atom {atom} of frame {frame_index} "
                 f"(self-consistency {'radius ' + format(self.edge.scf_radius, 'g') + ' A' if self.edge.scf_radius > 0 else 'off'})")
        t0 = time.time()
        phase, binfo = build_phase(str(inp), xkmax=self.edge.phase_kmax_resolved, device=self.device)
        info = {"mode": self.edge.potentials, "key": key, "frame": int(frame_index), "atom": atom, "cluster_atoms": int(len(sym)),
                "cluster_radius": float(self.edge.cluster_radius), "ipot_of": ipot_of, "potentials": binfo.get("potentials"),
                "Z": binfo.get("Z"), "seconds": round(time.time() - t0, 2), "settings": self._settings(frame_index)}
        _atomic_write(pkl, pickle.dumps(phase), binary=True)
        _atomic_write(meta, json.dumps(info, indent=1))
        self.log(f"[{self.edge.name}] potentials: done in {info['seconds']:.1f} s -> {pkl.relative_to(self.output)}")
        return phase, ipot_of, info

    def _settings(self, frame_index):
        e = self.edge
        return {"element": e.element, "edge": e.edge, "frame": int(frame_index), "reference_atom": e.reference_atom,
                "cluster_radius": float(e.cluster_radius), "scf_radius": float(e.scf_radius), "phase_kmax": float(e.phase_kmax_resolved),
                "cards": list(e.cards), "trajectory": str(self.reader.path)}
