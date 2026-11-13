"""Readers for the files of an EvAX run (Timoshenko and Kuzmin's reverse Monte Carlo program), so that an EvAX input
can be used here and an EvAX result compared with a result of this package."""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np

def read_cell_file(path):
    """EvAX's temp_P1.p1 (POSCAR-like): the unit cell's lattice vectors (3, 3), rows the vectors."""
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    scale = float(lines[1].split()[0])
    return scale * np.array([[float(v) for v in lines[i].split()[:3]] for i in (2, 3, 4)])


def read_parameters(path):
    """The keywords of an EvAX parameter file: ``{keyword: [values]}``.

    Both layouts are read: ``Keyword value value`` and ``Keyword: value``. Lines that start with ``#`` and the marks
    ``BEGIN`` and ``END`` are skipped; a later line with the same keyword replaces an earlier one."""
    out = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line in ("BEGIN", "END"):
            continue
        key, _, value = line.partition(" ") if not re.match(r"^\S+:\s", line + " ") else line.partition(":")
        key = key.strip().strip("'\"").rstrip(":").strip("'\"")
        if key:
            out[key] = value.replace(":", " ").split()
    return out


def read_evax_params(path):
    """An EvAX parameter file (``start.dat``) as ``{keyword: value string}`` (``read_parameters`` with the values of
    a keyword joined by a blank; lines with a keyword and no value are left out)."""
    return {key: " ".join(values) for key, values in read_parameters(path).items() if values}


def read_evax_rdf(path):
    """rdf.dat: header 'R A-A A-B ...' (absorber-element pairs), rows until '####'. Returns {pair: g (nbin,)} and r (nbin,)."""
    lines = Path(path).read_text().splitlines()
    names = lines[0].split()
    rows = []
    for ln in lines[1:]:
        if ln.startswith("#"):
            break
        t = ln.split()
        if len(t) == len(names):
            rows.append([float(v) for v in t])
    a = np.array(rows)
    return a[:, 0], {names[i]: a[:, i] for i in range(1, len(names))}
