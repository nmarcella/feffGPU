"""Helpers of the RMC-EXAFS tests: paths of the phase files, structures built in code, synthetic targets.

Every structure is made here; the only data files used are FEFF 8.5L's phase files of the repository
(``tests/data/W/phase.pad``: bcc W, K edge, one scattering potential; ``data/validation/A000/phase.pad``: potentials
0 C absorber, 1 Zn, 2 C). Where a test needs more elements than a phase file has, two elements share a potential:
these are code tests, not chemistry.
"""
from __future__ import annotations
import itertools
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
PHASE_W = ROOT / "tests/data/W/phase.pad"
PHASE_ZNC = ROOT / "data/validation/A000/phase.pad"
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
A_W = 3.1652               # bcc W (the structure of tests/data/W)
A_ZNC = 4.3765             # the Zn-C rock salt of data/validation/A000


def write_poscar(path, sym, pos, cell, title="test structure"):
    """A POSCAR with Cartesian coordinates, the atoms grouped by element in order of first appearance."""
    sym = np.asarray(sym).astype(str)
    species = list(dict.fromkeys(sym.tolist()))
    lines = [title, "1.0"] + [" ".join(f"{v:.10f}" for v in row) for row in np.asarray(cell, float)]
    lines += [" ".join(species), " ".join(str(int((sym == s).sum())) for s in species), "Cartesian"]
    for s in species:
        for p in np.asarray(pos, float)[sym == s]:
            lines.append(" ".join(f"{v:.10f}" for v in p))
    Path(path).write_text("\n".join(lines) + "\n")


def write_chi(path, k, chi):
    """Two columns k, chi with full precision (k on the model's grid)."""
    with open(path, "w") as f:
        f.write("# k chi (synthetic)\n")
        for a, b in zip(k, chi):
            f.write(f"{float(a)!r} {float(b)!r}\n")


def bcc(L, a=A_W, element="W"):
    """An L x L x L bcc block: symbols, positions, cell (3, 3)."""
    from feffgpu.io import bcc_block
    sites, box = bcc_block(L, a)
    return np.array([element] * len(sites)), sites, np.eye(3) * box


def rock_salt(L=2, a=A_ZNC):
    """An L x L x L block of the Zn-C rock salt (C on the anion sites): symbols, positions, cell."""
    sites, sym = [], []
    for i, j, l in itertools.product(range(L), repeat=3):
        for d, s in (((0, 0, 0), "C"), ((0.5, 0, 0), "Zn"), ((0, 0.5, 0), "Zn"), ((0, 0, 0.5), "Zn"),
                     ((0.5, 0.5, 0), "C"), ((0.5, 0, 0.5), "C"), ((0, 0.5, 0.5), "C"), ((0.5, 0.5, 0.5), "Zn")):
            sites.append(((i + d[0]) * a, (j + d[1]) * a, (l + d[2]) * a))
            sym.append(s)
    return np.array(sym), np.array(sites), np.eye(3) * L * a


# -- the toy of the equality test (tests/rmc/test_equals_reference.py) ------------------------------------------------
TOY_A, TOY_NSC = 5.5, 2                                         # every coordinate an exact binary fraction
TOY_EDGES = {"C": dict(phase=PHASE_ZNC, potentials={"Zn": 1, "Cu": 1, "C": 2}, s02=0.9, de0=1.0),
             "Zn": dict(phase=PHASE_W, potentials={"Zn": 1, "Cu": 1, "C": 1}, s02=0.85, de0=-1.5)}
TOY_WINDOW = dict(kmin=3.0, kmax=12.0, dk=2.0, kweight=2, rmin=1.0, rmax=4.0)


def fluorite_toy(seed, n_vac):
    """A fluorite-type cell of 2 x 2 x 2 conventional cells (a = 5.5 A): 16 Zn and 16 Cu on the cation sites (both
    scatter with the Zn potential of the A000 phase file), C on the anion sites with ``n_vac`` of them vacant."""
    fcc = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    cat = np.concatenate([(fcc + np.array(ijk)) * TOY_A for ijk in itertools.product(range(TOY_NSC), repeat=3)])
    an = np.concatenate([(np.concatenate([fcc + .25, fcc + .75]) + np.array(ijk)) * TOY_A for ijk in itertools.product(range(TOY_NSC), repeat=3)])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(cat))
    keep = np.sort(rng.choice(len(an), len(an) - n_vac, replace=False))
    pos = np.concatenate([cat[np.sort(perm[:16])], cat[np.sort(perm[16:])], an[keep]])
    sym = np.array(["Zn"] * 16 + ["Cu"] * 16 + ["C"] * len(keep))
    return sym, pos, np.eye(3) * TOY_A * TOY_NSC


def toy_truth(pos, seed, sd=0.05):
    return pos + np.random.default_rng(seed).normal(0, sd, pos.shape)


def toy_target(edge, sym, pos_true, cell):
    """The synthetic spectrum of one edge of the true toy structure, computed on the CPU and rounded to 11 significant
    digits (so that it is the same in every process). The number of threads is not changed: in some PyTorch builds
    torch.set_num_threads() before the first linear-algebra call breaks batched solves for the rest of the process."""
    from feffgpu.exafs import k_grid, FitWindow
    from feffgpu.exafs.edge import EdgeModel
    spec = TOY_EDGES[edge]
    k = k_grid(24.0)
    e = EdgeModel(edge, k, np.zeros(len(k)), FitWindow(**TOY_WINDOW), str(spec["phase"]), sym, cell, edge, potentials=spec["potentials"],
                  rpath=4.0, nleg=3, criteria=2.5, device="cpu")
    chi = e.spectra(np.asarray(pos_true)[None])
    model = e.model(chi, s02=spec["s02"], de0=spec["de0"])[0].numpy()
    return k, np.array([float(f"{v:.10e}") for v in model])


def synthetic_edge_data(edge_kwargs, positions, s02=1.0, de0=0.0, device=DEVICE):
    """A synthetic measured spectrum: S0^2 x the calculated chi of ``positions`` (S, n, 3) (mean over the
    configurations) at the shifted k, from an EdgeModel built with ``edge_kwargs`` (zero data)."""
    from feffgpu.exafs import k_grid
    from feffgpu.exafs.edge import EdgeModel
    k = k_grid(24.0)
    e = EdgeModel(data=np.zeros(len(k)), k=k, device=device, **edge_kwargs)
    with torch.no_grad():
        chi = e.spectra(np.asarray(positions)).mean(0, keepdim=True)
        return k, e.model(chi, s02=s02, de0=de0)[0].cpu().numpy()
