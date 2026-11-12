"""Structures for the EXAFS tools: element symbols, cells, readers (POSCAR, xyz, and every format that ASE reads),
supercells, and the potential indices of a phase file.

A structure is three arrays: ``symbols`` (n,) of element symbols, ``positions`` (n, 3) Cartesian in A, and ``cell``
(3, 3) with the lattice vectors as rows (or None for a cluster).
"""
from __future__ import annotations
from pathlib import Path
import math
import numpy as np

SYMBOLS = ['X', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
Z_OF = {s: i for i, s in enumerate(SYMBOLS)}


def cell_from_parameters(a, b, c, alpha=90.0, beta=90.0, gamma=90.0):
    """Lattice vectors (3, 3), rows: a along x, b in the xy plane (the usual crystallographic convention)."""
    al, be, ga = (math.radians(v) for v in (alpha, beta, gamma))
    va = np.array([a, 0.0, 0.0])
    vb = np.array([b * math.cos(ga), b * math.sin(ga), 0.0])
    cx = c * math.cos(be)
    cy = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
    cz = math.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    return np.array([va, vb, [cx, cy, cz]])


def read_poscar(path):
    """VASP POSCAR / CONTCAR (with the species line): symbols (n,), Cartesian positions (n, 3), cell (3, 3)."""
    lines = [ln for ln in Path(path).read_text().splitlines()]
    scale = float(lines[1].split()[0])
    cell = scale * np.array([[float(v) for v in lines[i].split()[:3]] for i in (2, 3, 4)])
    species = lines[5].split()
    counts = [int(v) for v in lines[6].split()]
    i = 7
    if lines[i].strip()[0].lower() == "s":
        i += 1
    cartesian = lines[i].strip()[0].lower() in "ck"
    i += 1
    n = sum(counts)
    frac = np.array([[float(v) for v in lines[i + j].split()[:3]] for j in range(n)])
    pos = frac if cartesian else frac @ cell
    if cartesian:
        pos = pos * scale
    sym = np.array([s for s, c in zip(species, counts) for _ in range(c)])
    return sym, pos, cell


def supercell(sym, pos, cell, n):
    """Replicate (nx, ny, nz) times; the images follow the originals in blocks."""
    n = [int(v) for v in n]
    shifts = [np.array([i, j, k]) @ cell for i in range(n[0]) for j in range(n[1]) for k in range(n[2])]
    P = np.concatenate([pos + s for s in shifts])
    S = np.concatenate([sym] * len(shifts))
    return S, P, np.diag(n) @ cell


def read_xyz(path):
    """A plain xyz file (also EvAX's ``*_no_PBC.xyz``): the atom count, a comment line, then ``El x y z`` per atom.
    Returns symbols (n,) and positions (n, 3)."""
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].split()[0])
    sym, pos = [], []
    for ln in lines[2:2 + n]:
        t = ln.split(); sym.append(t[0]); pos.append([float(t[1]), float(t[2]), float(t[3])])
    return np.array(sym), np.array(pos, float)


def read_structure(path, format=None, index=-1):
    """Read one structure with ASE (POSCAR/CONTCAR, CIF, extended xyz, LAMMPS data, and the other formats of
    ``ase.io.read``).

    Parameters
    ----------
    path : str or Path
    format : str, optional
        ASE's format name; guessed from the file name if not given.
    index : int
        Which image of a file with several (default the last).

    Returns
    -------
    symbols : numpy.ndarray of str, shape (n,)
    positions : numpy.ndarray, shape (n, 3)
        Cartesian, A.
    cell : numpy.ndarray, shape (3, 3), or None
        Lattice vectors as rows; None if the file has no periodic cell.
    """
    from ase.io import read
    atoms = read(str(path), index=index, format=format)
    cell = np.array(atoms.get_cell(), float)
    has_cell = bool(np.abs(np.linalg.det(cell)) > 1e-9)
    return np.array(atoms.get_chemical_symbols()), np.array(atoms.get_positions(), float), (cell if has_cell else None)


def read_potentials(feff_inp):
    """POTENTIALS block of a feff.inp: {ipot: (Z, label)}."""
    pots, inside = {}, False
    for ln in Path(feff_inp).read_text().splitlines():
        t = ln.split()
        if not t or t[0].startswith("*"):
            continue
        if t[0].upper() == "POTENTIALS":
            inside = True; continue
        if t[0].upper() == "ATOMS":
            break
        if inside and t[0].lstrip("-").isdigit():
            pots[int(t[0])] = (int(t[1]), t[2] if len(t) > 2 else "")
    return pots


def ipot_table(pots):
    """{Z: scatterer potential index} from the POTENTIALS block ({ipot: (Z, label)}): the first potential >= 1 of each Z."""
    out = {}
    for ip, (z, _) in sorted(pots.items()):
        if ip >= 1 and z not in out:
            out[z] = ip
    return out


def parse_ipots(spec):
    """'Ti:1,O:2' -> {Z: ipot}."""
    out = {}
    for item in filter(None, spec.split(",")):
        el, ip = item.split(":")
        out[Z_OF[el]] = int(ip)
    return out
