"""The GPU path finder reproduces FEFF's own path finder (rdinp + pathfinder with the pinned W phase file) on a
disordered cluster: identical 3- and 4-leg sets after the importance filter, and the single-scattering set up to
FEFF's accidental merging of neighbours at coincident distances."""
import os, shutil, subprocess
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
FEFF = Path(os.environ.get("FEFF_HOME", ROOT / "fortran"))
PATH_SET = ROOT / "tests/data/W"


def _parse_paths(fname):
    lines = Path(fname).read_text().splitlines()
    out, i = [], 2
    while i < len(lines):
        if "index, nleg" not in lines[i]:
            i += 1
            continue
        w = lines[i].split()
        nleg, deg = int(w[1]), float(w[2])
        pts = np.array([[float(v) for v in lines[i + 2 + j].split()[:3]] for j in range(nleg)])
        out.append((nleg, deg, pts[:-1]))
        i += 2 + nleg
    return out


@pytest.mark.gpu
@pytest.mark.oracle
@pytest.mark.skipif(not torch.cuda.is_available() or not (FEFF / "bin/feff8l_pathfinder").exists(), reason="needs GPU and FEFF")
def test_pathfinder_matches_feff(tmp_path):
    def cluster_input(pos, box, absorber, rcut, pwcrit):
        """feff.inp of one absorber of a periodic W block: the atoms within rcut, minimum image, the W set's cards."""
        rel = pos - pos[absorber]
        rel -= box * np.round(rel / box)
        r = np.linalg.norm(rel, axis=1)
        order = np.argsort(r)
        keep = order[r[order] < rcut]
        lines = ["TITLE W block, one absorber", "CONTROL 1 1 1 1 1 1", "PRINT 0 0 0 0 0 0", "EDGE K", "S02 1.0", "SCF 4.0 0 30 .2 1",
                 "RPATH 9.4", "EXAFS 38.0", "AFOLP 1.15", "NLEG 4", f"CRITERIA 0.0 {pwcrit}", "", "POTENTIALS", "    0   74   W", "    1   74   W", "", "ATOMS"]
        for i in keep:
            x, y, z = rel[i]
            lines.append(f"  {x:10.5f} {y:10.5f} {z:10.5f}  {0 if i == absorber else 1}  W  {r[i]:8.5f}")
        return "\n".join(lines + ["END", ""])
    from feffgpu.pathfinder import GPUPathFinder
    rng = np.random.default_rng(11)
    a, L = 3.1652, 7
    c = np.stack(np.meshgrid(*[np.arange(L)] * 3, indexing="ij"), -1).reshape(-1, 3)
    sites = np.concatenate([c, c + 0.5]) * a
    pos = sites + rng.normal(0, 0.06, sites.shape)
    box, absorber = L * a, 171
    work = tmp_path / "feff"
    work.mkdir()
    (work / "feff.inp").write_text(cluster_input(pos, box, absorber, 10.5, 0.1))
    for f in ("pot.pad", "phase.pad", "xsect.json"):
        shutil.copy(PATH_SET / f, work / f)
    env = dict(os.environ, LD_LIBRARY_PATH=str(FEFF / "lib"), FEFF_LMAX="56")
    for m in ("feff8l_rdinp", "feff8l_pathfinder"):
        r = subprocess.run([str(FEFF / "bin" / m)], cwd=work, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-300:]
    rel = pos - pos[absorber]
    rel -= box * np.round(rel / box)

    def atoms_of(pts):
        idx = []
        for p in pts:
            d = np.linalg.norm(rel - p, axis=1)
            j = int(np.argmin(d))
            assert d[j] < 2e-3
            idx.append(j)
        t = tuple(idx)
        return min(t, tuple(reversed(t)))

    feff = {}
    for nleg, deg, pts in _parse_paths(work / "paths.dat"):
        feff.setdefault(nleg, set()).add(atoms_of(pts))
    pf = GPUPathFinder(str(PATH_SET / "phase.pad"), device="cuda", rmax=9.4, nlegmax=4, critpw=0.1)
    kept, ref = pf.enumerate(pos, box, absorber, np.ones(len(pos), int))
    mine = {nleg: set(min(tuple(r), tuple(reversed(r))) for r in v[4].cpu().numpy().tolist()) for nleg, v in kept.items()}
    for nleg in (3, 4):
        assert feff[nleg] == mine[nleg], (nleg, len(feff[nleg] - mine[nleg]), len(mine[nleg] - feff[nleg]))
    # single scattering: FEFF merges equal-length pairs; the representative it keeps comes from an unstable sort,
    # so compare the multiset of lengths (0.001 A) and the count
    r_feff = sorted(round(float(np.linalg.norm(rel[t[0]])), 3) for t in feff[2])
    r_mine = sorted(round(float(np.linalg.norm(rel[t[0]])), 3) for t in mine[2])
    assert len(r_feff) == len(r_mine) and max(abs(a - b) for a, b in zip(r_feff, r_mine)) <= 0.001


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_symmetry_degeneracy_totals_on_ideal_lattice():
    """FEFF merges symmetry-equivalent multiple-scattering paths before its importance filter: on an ideal simple
    cubic lattice the right-angle triangles through two first neighbours have total degeneracy 24 and the
    first-second neighbour triangles 48 (FEFF's paths.dat for such a cluster), while every member stays listed."""
    from feffgpu.pathfinder import GPUPathFinder
    a = 2.823
    m = 3
    cells = np.stack(np.meshgrid(*[np.arange(-m, m + 1)] * 3, indexing="ij"), -1).reshape(-1, 3) * a
    pos = cells[np.argsort(np.linalg.norm(cells, axis=1), kind="stable")]
    pf = GPUPathFinder(str(PATH_SET / "phase.pad"), device="cuda", rmax=5.32, nlegmax=3, critpw=0.0)
    kept, ref = pf.enumerate(pos, None, 0, np.ones(len(pos), int))
    coords, ipots, deg, xp, atoms = kept[2]
    assert sorted(deg.cpu().numpy().tolist()) == [6.0, 8.0, 12.0]                  # the three shells, merged as FEFF does
    coords, ipots, deg, xp, atoms = kept[3]
    tot = pf.degenerate_total(coords, deg).cpu().numpy()
    d1 = np.linalg.norm(coords[:, 1].cpu().numpy(), axis=1); d2 = np.linalg.norm(coords[:, 2].cpu().numpy(), axis=1)
    first_first = (np.abs(d1 - a) < 1e-3) & (np.abs(d2 - a) < 1e-3)
    first_second = ((np.abs(d1 - a) < 1e-3) & (np.abs(d2 - a * 2 ** 0.5) < 1e-3)) | ((np.abs(d2 - a) < 1e-3) & (np.abs(d1 - a * 2 ** 0.5) < 1e-3))
    assert first_first.sum() == 12 and np.all(tot[first_first] == 24.0)
    assert first_second.sum() == 24 and np.all(tot[first_second] == 48.0)
