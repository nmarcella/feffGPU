"""Reading molecular dynamics trajectories frame by frame, without OVITO.

A trajectory is read as a stream of frames. A frame (``Frame``) is the element symbol of every atom, the Cartesian
positions (A) and the periodic cell (3 x 3, rows the lattice vectors; None for a cluster). Frames are produced one at
a time and the whole trajectory is never held in memory, so files with tens of thousands of frames can be used.

Two readers:

``"ase"`` (default)
    ``ase.io.iread`` for every format that ASE reads frame by frame: extended xyz, plain and CP2K xyz (both read by
    ASE's extended xyz reader), LAMMPS text dumps, ASE ``.traj`` files, and the others. ASE's own VASP XDATCAR reader
    reads every frame at once, so XDATCAR files are read by a streaming reader of this module instead.
``"mdanalysis"``
    MDAnalysis (the optional extra ``pip install 'feffgpu[mda]'``) for large binary formats such as DCD, XTC, TRR or
    NetCDF, usually with a topology file that names the atoms.

Numeric atom types (LAMMPS) are turned into elements with a type map (``{1: "Zn", 2: "C"}``). The cell of a frame is
either a static cell given by the user (one cell for every frame, NVT), the cell stored in each frame (NPT), or none
(a cluster). Frames are addressed by their position in the file (0 for the first frame); a selection is an increasing
list of positions, so the frames a run uses depend only on the selection, never on how the run is split.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .structure import Z_OF, cell_from_parameters

__all__ = ["Frame", "TrajectoryReader", "guess_format", "count_frames", "static_cell", "frame_selection", "arithmetic_slice",
           "iter_xdatcar", "write_extxyz_frame", "write_lammps_dump_frame", "element_counts"]


@dataclass
class Frame:
    """One frame of a trajectory.

    Attributes
    ----------
    index : int
        Position of the frame in the file (0 for the first frame).
    symbols : numpy.ndarray of str, shape (n,)
        Element symbol of every atom.
    positions : numpy.ndarray, shape (n, 3)
        Cartesian positions (A).
    cell : numpy.ndarray, shape (3, 3), or None
        Lattice vectors as rows (A), or None for a cluster.
    """
    index: int
    symbols: np.ndarray
    positions: np.ndarray
    cell: np.ndarray | None

    @property
    def n_atoms(self):
        """Number of atoms."""
        return int(len(self.symbols))


def static_cell(a=None, b=None, c=None, alpha=90.0, beta=90.0, gamma=90.0, matrix=None):
    """A cell from lengths and angles, from three lattice vectors, or from a 3 x 3 matrix.

    Parameters
    ----------
    a, b, c : float or sequence of 3 floats, optional
        Either the three lengths (A), used with ``alpha``, ``beta``, ``gamma``, or the three lattice vectors (A).
    alpha, beta, gamma : float
        Angles (degrees) between b and c, a and c, a and b, when a, b and c are lengths.
    matrix : array, shape (3, 3), optional
        Lattice vectors as rows (A). Takes precedence over a, b and c.

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        Lattice vectors as rows (A).
    """
    if matrix is not None:
        m = np.asarray(matrix, float)
        if m.shape != (3, 3):
            raise ValueError(f"a cell matrix must be 3 x 3 (got shape {m.shape})")
        out = m
    else:
        if a is None or b is None or c is None:
            raise ValueError("a static cell needs a, b and c (lengths or vectors), or a 3 x 3 matrix")
        va, vb, vc = (np.asarray(v, float) for v in (a, b, c))
        if va.ndim == vb.ndim == vc.ndim == 0:
            out = cell_from_parameters(float(va), float(vb), float(vc), alpha, beta, gamma)
        elif va.shape == vb.shape == vc.shape == (3,):
            out = np.array([va, vb, vc])
        else:
            raise ValueError("a, b and c must all be lengths or all be vectors of three numbers")
    if abs(np.linalg.det(out)) < 1e-9:
        raise ValueError("the cell has zero volume")
    return out


def guess_format(path, format=None):
    """The ASE format name of a trajectory file: ``format`` if given, else guessed from the name and the content.

    ``XDATCAR`` file names give ``"vasp-xdatcar"``, ``.lammpstrj`` and ``.dump`` give ``"lammps-dump-text"``; other
    names are passed to ``ase.io.formats.filetype``. The plain ``"xyz"`` format is read with the extended xyz reader,
    which reads plain and CP2K xyz files frame by frame (ASE's plain xyz reader reads the whole file)."""
    if format:
        return "extxyz" if format == "xyz" else str(format)
    p = Path(path)
    name = p.name.lower()
    if "xdatcar" in name:
        return "vasp-xdatcar"
    if name.endswith((".lammpstrj", ".dump", ".lammpsdump")):
        return "lammps-dump-text"
    if name.endswith((".xtc", ".trr", ".nc", ".ncdf", ".netcdf")):
        raise ValueError(f"{p.name}: binary trajectories of this kind are read with reader = \"mdanalysis\" (pip install 'feffgpu[mda]')")
    from ase.io.formats import filetype
    try:
        fmt = filetype(str(p), guess=True)
    except Exception as err:
        raise ValueError(f"{p.name}: the format could not be guessed ({err}); give it with the format setting, for example "
                         "\"extxyz\", \"lammps-dump-text\", \"vasp-xdatcar\" or \"traj\"") from err
    return "extxyz" if fmt == "xyz" else fmt


def _count_extxyz(path):
    n = 0
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line or not line.strip():
                break
            try:
                natoms = int(line.split()[0])
            except ValueError as err:
                raise ValueError(f"{path}: frame {n} does not start with a number of atoms (line {line.strip()[:40]!r})") from err
            for _ in range(natoms + 1):
                if not f.readline():
                    raise ValueError(f"{path}: frame {n} is incomplete")
            n += 1
    return n


def _count_lines_with(path, marker):
    n = 0
    with open(path, "r") as f:
        for line in f:
            if marker in line:
                n += 1
    return n


def count_frames(path, format=None):
    """The number of frames in a trajectory file, by a scan that does not build the frames where possible.

    Extended and plain xyz files, LAMMPS text dumps, VASP XDATCAR files and ASE ``.traj`` files are counted by a fast
    scan; other formats by reading them frame by frame with ASE."""
    fmt = guess_format(path, format)
    if fmt in ("extxyz", "xyz"):
        return _count_extxyz(path)
    if fmt == "lammps-dump-text":
        return _count_lines_with(path, "ITEM: TIMESTEP")
    if fmt == "vasp-xdatcar":
        return _count_lines_with(path, "configuration")
    if fmt == "traj":
        from ase.io.trajectory import Trajectory
        with Trajectory(str(path), "r") as t:
            return len(t)
    import ase.io
    return sum(1 for _ in ase.io.iread(str(path), index=":", format=fmt))


def arithmetic_slice(indices):
    """``slice(start, stop, step)`` if the increasing integers ``indices`` form an arithmetic progression, else None."""
    idx = [int(i) for i in indices]
    if not idx:
        return slice(0, 0, 1)
    if len(idx) == 1:
        return slice(idx[0], idx[0] + 1, 1)
    step = idx[1] - idx[0]
    if step <= 0 or any(b - a != step for a, b in zip(idx, idx[1:])):
        return None
    return slice(idx[0], idx[-1] + 1, step)


def frame_selection(n_frames, start=0, stop=None, step=1):
    """The positions of the selected frames: ``range(start, stop, step)`` clipped to the file.

    Parameters
    ----------
    n_frames : int
        Number of frames in the file.
    start : int
        First frame (0-based, not negative).
    stop : int, optional
        End of the range (excluded); None for the end of the file. A stop beyond the end is clipped.
    step : int
        Positive step.

    Returns
    -------
    list of int
    """
    if start < 0 or step < 1:
        raise ValueError(f"the frame range needs start >= 0 and step >= 1 (got start {start}, step {step})")
    if start >= n_frames:
        raise ValueError(f"the first selected frame ({start}) is beyond the end of the trajectory ({n_frames} frames)")
    end = n_frames if stop is None else min(int(stop), n_frames)
    if end <= start:
        raise ValueError(f"the frame range {start} to {stop} selects no frame")
    return list(range(int(start), end, int(step)))


def iter_xdatcar(path, indices=None):
    """Stream the frames of a VASP XDATCAR file (VASP 5 or later, with the line of element symbols).

    Constant-cell files (one header) and variable-cell files (a header before every frame) are both read. Frames that
    are not selected are skipped without being parsed.

    Parameters
    ----------
    path : str or Path
    indices : iterable of int, optional
        Increasing positions of the frames to yield (default every frame).

    Yields
    ------
    (index, symbols, positions, cell)
        Positions are Cartesian (A); the cell has the lattice vectors as rows (A).
    """
    wanted = None if indices is None else iter(sorted(int(i) for i in indices))
    target = None if wanted is None else next(wanted, None)
    cell, symbols, natoms, scale = None, None, 0, 1.0
    i = 0
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                return
            if not line.strip():
                continue
            low = line.strip().lower()
            if "configuration" in low:
                if cell is None:
                    raise ValueError(f"{path}: a frame comes before the cell header")
                cartesian = low[0] in "ck"
                take = wanted is None or target == i
                rows = [f.readline() for _ in range(natoms)]
                if take:
                    frac = np.array([[float(v) for v in r.split()[:3]] for r in rows])
                    pos = frac * scale if cartesian else frac @ cell
                    yield i, symbols.copy(), pos, cell.copy()
                    if wanted is not None:
                        target = next(wanted, None)
                        if target is None:
                            return
                i += 1
                continue
            # a header: comment (this line), scale, three lattice vectors, element symbols, counts
            scale = float(f.readline().split()[0])
            vecs = np.array([[float(v) for v in f.readline().split()[:3]] for _ in range(3)])
            if scale < 0:                                                    # a negative scale is the cell volume
                scale = (-scale / abs(np.linalg.det(vecs))) ** (1.0 / 3.0)
            cell = vecs * scale
            names = f.readline().split()
            if all(t.lstrip("-").isdigit() for t in names):
                counts, names = [int(t) for t in names], [str(j + 1) for j in range(len(names))]
            else:
                counts = [int(t) for t in f.readline().split()]
            symbols = np.array([s for s, c in zip(names, counts) for _ in range(c)])
            natoms = int(sum(counts))


def write_extxyz_frame(handle, symbols, positions, cell=None, info=""):
    """Write one frame in extended xyz format to an open text file (for examples and tests).

    Parameters
    ----------
    handle : file object
    symbols : sequence of str, shape (n,)
    positions : array, shape (n, 3)
        Cartesian (A).
    cell : array, shape (3, 3), optional
        Lattice vectors as rows (A); written as the ``Lattice`` key with periodic boundaries.
    info : str
        Extra ``key=value`` text for the comment line.
    """
    pos = np.asarray(positions, float)
    head = 'Properties=species:S:1:pos:R:3'
    if cell is not None:
        c = np.asarray(cell, float).ravel()
        head = 'Lattice="' + " ".join(f"{v:.10f}" for v in c) + '" ' + head + ' pbc="T T T"'
    lines = [str(len(pos)), (head + (" " + info if info else "")).strip()]
    lines += [f"{s} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}" for s, p in zip(symbols, pos)]
    handle.write("\n".join(lines) + "\n")


def write_lammps_dump_frame(handle, types, positions, box, timestep=0):
    """Write one frame of a LAMMPS text dump with numeric types and an orthogonal box (for examples and tests).

    Parameters
    ----------
    handle : file object
    types : sequence of int, shape (n,)
    positions : array, shape (n, 3)
        Cartesian (A).
    box : float or sequence of 3 floats
        Box lengths (A); the box spans 0 to the length in each direction.
    timestep : int
    """
    pos = np.asarray(positions, float)
    L = np.broadcast_to(np.asarray(box, float), (3,))
    lines = ["ITEM: TIMESTEP", str(int(timestep)), "ITEM: NUMBER OF ATOMS", str(len(pos)), "ITEM: BOX BOUNDS pp pp pp"]
    lines += [f"0.0 {L[d]:.10f}" for d in range(3)]
    lines += ["ITEM: ATOMS id type x y z"]
    lines += [f"{j + 1} {int(t)} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}" for j, (t, p) in enumerate(zip(types, pos))]
    handle.write("\n".join(lines) + "\n")


def _normalize_type_map(type_map):
    """{type label (str): element symbol}; keys may be integers or strings of integers."""
    if not type_map:
        return {}
    out = {}
    for key, value in type_map.items():
        k = str(key).strip()
        v = str(value).strip()
        if v not in Z_OF or v == "X":
            raise ValueError(f"atom type {k}: {v!r} is not an element symbol")
        out[k] = v
    return out


class TrajectoryReader:
    """A trajectory file read frame by frame.

    Parameters
    ----------
    path : str or Path
        The trajectory file.
    format : str, optional
        ASE's format name (``"extxyz"``, ``"lammps-dump-text"``, ``"vasp-xdatcar"``, ``"traj"``, ...); guessed from
        the name and the content if not given. For ``reader = "mdanalysis"``, MDAnalysis's format name (optional).
    reader : {"ase", "mdanalysis"}
        Which library reads the file.
    topology : str or Path, optional
        Topology file for MDAnalysis (for example a PDB or a LAMMPS data file).
    type_map : dict, optional
        ``{numeric type: element symbol}`` for files with numeric atom types. For LAMMPS dumps the map applies to the
        ``type`` column; for other files to the atomic numbers ASE read (a type 1 is read as H).
    cell : array, shape (3, 3), optional
        A static cell used for every frame (lattice vectors as rows, A). The cells stored in the file are ignored.
    dynamic : bool
        Use the cell stored in every frame (NPT). A frame without a cell is an error.
    periodic : bool
        False: the frames are clusters (no cell, no periodic images).

    Notes
    -----
    Every frame must hold the same atoms in the same order (the same element sequence); a change of composition is
    an error. LAMMPS dumps are sorted by atom id by ASE, so the order does not depend on how LAMMPS wrote them.
    """

    def __init__(self, path, format=None, reader="ase", topology=None, type_map=None, cell=None, dynamic=False, periodic=True):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"no such trajectory file: {self.path}")
        self.reader = str(reader)
        if self.reader not in ("ase", "mdanalysis"):
            raise ValueError(f"reader must be \"ase\" or \"mdanalysis\" (got {reader!r})")
        self.topology = Path(topology) if topology else None
        self.format = guess_format(self.path, format) if self.reader == "ase" else (format or None)
        self.type_map = _normalize_type_map(type_map)
        self.periodic = bool(periodic)
        self.dynamic = bool(dynamic) and self.periodic
        self.cell = None if (cell is None or not self.periodic) else np.asarray(cell, float)
        if self.periodic and self.cell is None and not self.dynamic:
            raise ValueError("give a static cell, or dynamic = true to read the cell of every frame, or periodic = false for clusters")
        if self.cell is not None and self.dynamic:
            raise ValueError("a static cell and dynamic = true exclude each other")
        self._count = None
        self._symbols = None

    # -- counting and selection ---------------------------------------------------------------------------------
    def count(self):
        """Number of frames in the file (computed once)."""
        if self._count is None:
            if self.reader == "mdanalysis":
                self._count = len(self._universe().trajectory)
            else:
                self._count = count_frames(self.path, self.format)
        return self._count

    def selection(self, start=0, stop=None, step=1):
        """The positions of the frames ``range(start, stop, step)``, clipped to the file (see ``frame_selection``)."""
        return frame_selection(self.count(), start, stop, step)

    # -- frames ---------------------------------------------------------------------------------------------------
    def _symbols_of(self, atoms, index):
        if "type" in atoms.arrays and self.type_map:
            labels = [str(int(t)) for t in atoms.arrays["type"]]
        elif self.type_map and all(k.lstrip("-").isdigit() for k in self.type_map):
            labels = [str(int(z)) for z in atoms.get_atomic_numbers()]
        else:
            return np.array(atoms.get_chemical_symbols())
        missing = sorted(set(labels) - set(self.type_map), key=lambda s: int(s) if s.lstrip("-").isdigit() else 0)
        if missing:
            raise ValueError(f"frame {index}: atom types {missing} are not in the type map {self.type_map} ([atoms] in the configuration)")
        return np.array([self.type_map[t] for t in labels])

    def _cell_of(self, file_cell, index):
        if not self.periodic:
            return None
        if self.cell is not None:
            return self.cell.copy()
        c = None if file_cell is None else np.asarray(file_cell, float)
        if c is None or c.shape != (3, 3) or abs(np.linalg.det(c)) < 1e-9:
            raise ValueError(f"frame {index} has no periodic cell: give the cell in [lattice] (a, b, c or matrix), or periodic = false "
                             "for clusters")
        return c

    def _check_composition(self, symbols, index):
        if self._symbols is None:
            self._symbols = symbols
        elif len(symbols) != len(self._symbols) or not np.array_equal(symbols, self._symbols):
            raise ValueError(f"frame {index} has a different number, order or kind of atoms than the first frame read; MD-EXAFS needs "
                             "the same atoms in the same order in every frame")

    def _universe(self):
        try:
            import MDAnalysis as mda
        except ImportError as err:
            raise ImportError("reader = \"mdanalysis\" needs MDAnalysis, which is not installed: pip install 'feffgpu[mda]'") from err
        if not hasattr(self, "_mda_universe"):
            args = [str(self.topology), str(self.path)] if self.topology else [str(self.path)]
            kw = {"format": self.format} if self.format else {}
            self._mda_universe = mda.Universe(*args, **kw)
        return self._mda_universe

    def _iter_mdanalysis(self, indices):
        u = self._universe()
        from MDAnalysis.lib.mdamath import triclinic_vectors
        atoms = u.atoms
        if self.type_map:
            labels = [str(t) for t in atoms.types]
            missing = sorted(set(labels) - set(self.type_map))
            if missing:
                raise ValueError(f"atom types {missing} are not in the type map {self.type_map} ([atoms] in the configuration)")
            symbols = np.array([self.type_map[t] for t in labels])
        else:
            try:
                symbols = np.array([str(e).capitalize() for e in atoms.elements])
            except Exception as err:
                raise ValueError("the topology names no elements: give a type map ([atoms]) from the atom types to elements") from err
        bad = sorted({s for s in symbols if s not in Z_OF})
        if bad:
            raise ValueError(f"the topology's element names {bad} are not element symbols: give a type map ([atoms])")
        sl = arithmetic_slice(indices)
        wanted = set(int(i) for i in indices)
        frames = u.trajectory[sl] if sl is not None else u.trajectory[min(wanted):max(wanted) + 1]
        for ts in frames:
            if int(ts.frame) not in wanted:
                continue
            dims = ts.dimensions
            fc = None if dims is None or not np.all(np.asarray(dims[:3]) > 0) else np.asarray(triclinic_vectors(dims), float)
            yield Frame(int(ts.frame), symbols.copy(), np.asarray(atoms.positions, float).copy(), self._cell_of(fc, int(ts.frame)))

    def iter_frames(self, indices):
        """Stream the frames at the given positions, in increasing order.

        Parameters
        ----------
        indices : sequence of int
            Increasing positions in the file. An arithmetic progression is read with one slice; any other list is
            read by stepping through the covering range and skipping what is not wanted (still frame by frame).

        Yields
        ------
        Frame
        """
        idx = [int(i) for i in indices]
        if not idx:
            return
        if any(b <= a for a, b in zip(idx, idx[1:])):
            raise ValueError("frame positions must be increasing")
        if self.reader == "mdanalysis":
            for fr in self._iter_mdanalysis(idx):
                self._check_composition(fr.symbols, fr.index)
                yield fr
            return
        wanted = set(idx)
        if self.format == "vasp-xdatcar":
            for i, sym, pos, cell in iter_xdatcar(self.path, idx):
                if self.type_map and all(s.isdigit() for s in sym):
                    missing = sorted(set(sym) - set(self.type_map))
                    if missing:
                        raise ValueError(f"frame {i}: atom types {missing} are not in the type map ([atoms])")
                    sym = np.array([self.type_map[s] for s in sym])
                self._check_composition(sym, i)
                yield Frame(i, sym, pos, self._cell_of(cell, i))
            return
        sl = arithmetic_slice(idx)
        positions = range(idx[0], idx[-1] + 1, sl.step) if sl is not None else range(idx[0], idx[-1] + 1)
        read_slice = sl if sl is not None else slice(idx[0], idx[-1] + 1, 1)
        for i, atoms in zip(positions, self._ase_stream(read_slice)):
            if i not in wanted:
                continue
            sym = self._symbols_of(atoms, i)
            self._check_composition(sym, i)
            cell = np.asarray(atoms.get_cell(), float)
            cell = cell if abs(np.linalg.det(cell)) > 1e-9 else None
            yield Frame(i, sym, np.asarray(atoms.get_positions(), float), self._cell_of(cell, i))

    def _ase_stream(self, read_slice):
        """ASE atoms of a slice of the file, one at a time. LAMMPS text dumps use ASE's chunk iterator directly:
        ``ase.io.iread`` of that format builds every frame of the slice before yielding the first."""
        if self.format == "lammps-dump-text":
            from ase.io.lammpsrun import iread_lammps_dump_text
            with open(self.path, "r") as fh:
                yield from iread_lammps_dump_text(fh, index=read_slice)
            return
        import ase.io
        yield from ase.io.iread(str(self.path), index=read_slice, format=self.format)

    def read_frame(self, index):
        """One frame by its position in the file."""
        for fr in self.iter_frames([int(index)]):
            return fr
        raise IndexError(f"frame {index} is beyond the end of {self.path.name}")

    def first_symbols(self, index=0):
        """The element of every atom, from the frame at ``index`` (default the first frame)."""
        return self.read_frame(index).symbols


def element_counts(symbols):
    """{element: number of atoms} in the order of first appearance."""
    out = {}
    for s in symbols:
        out[str(s)] = out.get(str(s), 0) + 1
    return out
