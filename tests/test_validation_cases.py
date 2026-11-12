"""Regression cases from the random-environment validation (tools/validate_random.py): each
directory under data/validation holds a feff.inp with FEFF's own chi.dat, pot.pad, phase.pad and paths.dat.
The whole GPU chain runs from feff.inp and must match chi.dat in the analysis window (R-space, k 3 to 23, R 1 to the
path cutoff) to 2e-5, with the same degeneracy-weighted path counts. The cases cover what the first validation
round found wrong and fixed: symmetry-degenerate paths on an ideal lattice, ground-state exchange (the matrix
elements use the EXCHANGE card's fourth value), an ionised absorber (two free-atom passes), and an M5 edge
(rotation matrices to |m| = 3 and FEFF's truncated lambda set)."""
import os
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CASES = sorted(p for p in (ROOT / "data/validation").glob("*") if (p / "chi.dat").exists())


def _chi(path):
    rows = [l.split() for l in Path(path).read_text().splitlines() if l.strip() and not l.startswith("#")]
    tab = np.array([[float(r[0]), float(r[1])] for r in rows if len(r) >= 2 and r[0].replace(".", "", 1).isdigit()])
    return tab[:, 0], tab[:, 1]


def _feff_degeneracies(path):
    out = {}
    for l in Path(path).read_text().splitlines():
        if "index, nleg" in l:
            w = l.split()
            out[int(w[1])] = out.get(int(w[1]), 0.0) + float(w[2])
    return out


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_validation_case(case):
    from feffgpu.chain import run
    from feffgpu.inp import read_feff_inp
    from feffgpu.compare import r_factor
    os.environ.pop("FEFF_LMAX", None)
    P = read_feff_inp(case / "feff.inp")
    out = run(case / "feff.inp", device="cuda")
    kf, chi_ref = _chi(case / "chi.dat")
    idx = np.round(kf / 0.05).astype(int) - 1
    ok = (idx >= 0) & (idx < len(out["chi"]))
    kf, chi_ref, idx = kf[ok], chi_ref[ok], idx[ok]
    k0 = np.concatenate([[0.0], kf]); z0 = lambda c: np.concatenate([[0.0], c])
    R = r_factor(k0, z0(chi_ref), z0(out["chi"][idx]), kmin=3.0, kmax=float(min(23.0, kf.max() - 1.0)), rmin=1.0, rmax=float(P["rmax"]))
    assert R < 2e-5, R
    deg = {nleg: float(v[2].sum()) for nleg, v in out["kept"].items() if v[0].shape[0]}
    ref = _feff_degeneracies(case / "paths.dat")
    assert {n: round(d) for n, d in deg.items()} == {n: round(d) for n, d in ref.items()}, (deg, ref)
