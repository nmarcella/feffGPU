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


def _compact_cluster(seed=3):
    """The absorber and six atoms 1.7 A from it along the axes, slightly displaced: paths of up to eight legs fit inside
    a cutoff of 7.4 A."""
    pos = np.vstack([np.zeros(3), 1.7 * np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])])
    pos[1:] += np.random.default_rng(seed).normal(0, 0.05, (6, 3))
    return np.round(pos, 5)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_paths_of_up_to_eight_legs_equal_a_plain_enumeration():
    """Every sequence of scatterers within the cutoff, built without the finder: the first and the last atom not the
    absorber, no atom following itself, a path and its reverse counted once with degeneracy 2 (1 if they are the same)."""
    from feffgpu.pathfinder import GPUPathFinder, NLEG_MAX
    pos, rmax = _compact_cluster(), 7.4
    pf = GPUPathFinder(str(PATH_SET / "phase.pad"), device="cuda", rmax=rmax, nlegmax=9, critpw=0.0, merge_degenerate=False)
    kept, _ = pf.enumerate(pos, None, 0, np.ones(len(pos), int))
    mine = {}
    for n, (coords, ipots, deg, xp, atoms) in kept.items():
        assert bool(((atoms == 0) == (ipots[:, 1:] == 0)).all())           # the absorber inside a path keeps potential 0
        for a, d in zip(atoms.cpu().numpy().tolist(), deg.cpu().numpy().tolist()):
            key = min(tuple(a), tuple(reversed(a)))
            assert key not in mine.setdefault(n, {})                       # no path twice
            mine[n][key] = d
    D = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    plain = {}

    def grow(seq, length):
        last = seq[-1]
        if last != 0 and length + D[last, 0] <= 2 * rmax:
            plain.setdefault(len(seq) + 1, {})[min(seq, tuple(reversed(seq)))] = 1.0 if seq == tuple(reversed(seq)) else 2.0
        if len(seq) == 8:
            return
        for nxt in range(len(pos)):
            if nxt != last and length + D[last, nxt] + D[nxt, 0] <= 2 * rmax:
                grow(seq + (nxt,), length + D[last, nxt])
    for first in range(1, len(pos)):
        grow((first,), D[0, first])
    assert set(plain) == set(mine) == {2, 3, 4, 5, 6, 7, 8} and len(plain[8]) > 500 and len(plain[6]) > 2000
    for n in plain:
        assert plain[n] == mine[n], n
    assert NLEG_MAX == 9
    with pytest.raises(ValueError, match="between 2 and 9"):
        GPUPathFinder(str(PATH_SET / "phase.pad"), device="cuda", rmax=rmax, nlegmax=10)


@pytest.mark.gpu
@pytest.mark.oracle
@pytest.mark.skipif(not torch.cuda.is_available() or not (FEFF / "bin/feff8l_pathfinder").exists(), reason="needs GPU and FEFF")
def test_paths_of_up_to_eight_legs_match_feff(tmp_path):
    """FEFF's own path finder with NLEG 8 on the compact cluster: the same paths with the same degeneracies after the
    importance filter, for every number of legs from two to eight."""
    from feffgpu.pathfinder import GPUPathFinder
    pos, rmax, pwcrit = _compact_cluster(), 7.4, 0.1
    lines = ["TITLE compact cluster", "CONTROL 1 1 1 1 1 1", "PRINT 0 0 0 0 0 0", "EDGE K", "S02 1.0", "SCF 4.0 0 30 .2 1", f"RPATH {rmax}", "EXAFS 38.0",
             "AFOLP 1.15", "NLEG 8", f"CRITERIA 0.0 {pwcrit}", "", "POTENTIALS", "    0   74   W", "    1   74   W", "", "ATOMS"]
    lines += [f"  {p[0]:10.5f} {p[1]:10.5f} {p[2]:10.5f}  {0 if i == 0 else 1}  W  {np.linalg.norm(p):8.5f}" for i, p in enumerate(pos)]
    work = tmp_path / "feff"
    work.mkdir()
    (work / "feff.inp").write_text("\n".join(lines + ["END", ""]))
    for f in ("pot.pad", "phase.pad", "xsect.json"):
        shutil.copy(PATH_SET / f, work / f)
    env = dict(os.environ, LD_LIBRARY_PATH=str(FEFF / "lib"), FEFF_LMAX="56")
    for m in ("feff8l_rdinp", "feff8l_pathfinder"):
        r = subprocess.run([str(FEFF / "bin" / m)], cwd=work, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-300:]
    feff = {}
    for nleg, deg, pts in _parse_paths(work / "paths.dat"):
        idx = tuple(int(np.argmin(np.linalg.norm(pos - p, axis=1))) for p in pts)
        feff.setdefault(nleg, {})[min(idx, tuple(reversed(idx)))] = deg
    pf = GPUPathFinder(str(PATH_SET / "phase.pad"), device="cuda", rmax=rmax, nlegmax=8, critpw=pwcrit)
    kept, _ = pf.enumerate(pos, None, 0, np.ones(len(pos), int))
    mine = {n: {min(tuple(a), tuple(reversed(a))): d for a, d in zip(v[4].cpu().numpy().tolist(), v[2].cpu().numpy().tolist())} for n, v in kept.items()}
    assert set(feff) == set(mine) == {2, 3, 4, 5, 6, 7, 8}
    for n in feff:
        assert feff[n] == mine[n], n


def _lattice_walk(rng, nleg, a=3.1652, sd=0.06):
    """A path of ``nleg`` legs over first and second neighbors of a bcc lattice, its atoms displaced at random (an atom
    met twice keeps its displacement; the absorber may be met again and stays at the origin)."""
    steps = np.vstack([np.array([[i, j, l] for i in (-.5, .5) for j in (-.5, .5) for l in (-.5, .5)]),
                       np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])]) * a
    while True:
        site, sites = np.zeros(3), []
        for _ in range(nleg - 1):
            site = site + steps[rng.integers(len(steps))]
            sites.append(site.copy())
        if np.linalg.norm(sites[-1]) > 1e-6:
            break
    moved, shift = [np.zeros(3)], {}
    for s in sites:
        key = tuple(np.round(s, 3))
        if np.linalg.norm(s) > 1e-6 and key not in shift:
            shift[key] = rng.normal(0, sd, 3)
        moved.append(s + shift.get(key, 0.0))
    return np.array(moved), [0] + [0 if np.linalg.norm(s) < 1e-6 else 1 for s in sites]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_the_path_stage_of_five_to_eight_legs_equals_the_literal_port():
    """The batched GPU path stage against the literal port of FEFF's genfmt, path by path, for five to eight legs (and
    two to four as before), up to k = 20 1/A: there both keep every partial wave."""
    from feffgpu.stage import GPUPathStage
    from feffgpu.genfmt import PathStage
    stage, port = GPUPathStage(str(PATH_SET / "phase.pad"), device="cuda", dtype=torch.complex128), PathStage(str(PATH_SET / "phase.pad"))
    rng = np.random.default_rng(5)
    for nleg in range(2, 9):
        for _ in range(2):
            coords, ipots = _lattice_walk(rng, nleg)
            cc, reff = stage.cchi(torch.as_tensor(coords, device="cuda", dtype=torch.float64)[None], torch.as_tensor(ipots, device="cuda")[None])
            ref = port.compute(coords, ipots)
            low = ref["k"] < 20.0
            assert np.abs(cc[0].cpu().numpy() - ref["cchi"])[low].max() < 1e-8 * np.abs(ref["cchi"][low]).max(), nleg
            assert float(reff[0]) == pytest.approx(ref["reff"], rel=1e-10)


@pytest.mark.gpu
@pytest.mark.oracle
@pytest.mark.skipif(not torch.cuda.is_available() or not (FEFF / "lib").exists(), reason="needs GPU and the FEFF library")
def test_the_path_stage_of_five_to_eight_legs_matches_feff():
    """FEFF's own path routine (the rebuilt library) for explicit paths of two to eight legs: the amplitude of the GPU
    path stage agrees above k = 3 1/A to the precision of FEFF's columns."""
    onepath = pytest.importorskip("feffgpu.onepath")
    try:
        feff = onepath.OnePath(str(PATH_SET / "phase.pad"))
    except Exception as err:                                                  # the library was not built
        pytest.skip(f"FEFF's path library is not available: {err}")
    from feffgpu.stage import GPUPathStage
    from feffgpu.genfmt import PathStage
    from feffgpu.pad import BOHR
    stage, port = GPUPathStage(str(PATH_SET / "phase.pad"), device="cuda", dtype=torch.complex128), PathStage(str(PATH_SET / "phase.pad"))
    rng = np.random.default_rng(5)
    for nleg in range(2, 9):
        coords, ipots = _lattice_walk(rng, nleg)
        cc, reff = stage.cchi(torch.as_tensor(coords, device="cuda", dtype=torch.float64)[None], torch.as_tensor(ipots, device="cuda")[None])
        mine = port._columns(cc[0].cpu().numpy(), float(reff[0]) / BOHR, range(port.ne1))
        ref = feff.compute(coords, ipots)
        n = min(len(ref["k"]), len(mine["k"]))
        sel = ref["k"][:n] > 3.0
        assert ref["nleg"] == nleg
        assert np.abs(mine["mag"][:n] - ref["mag"][:n])[sel].max() < 5e-5 * np.abs(ref["mag"][:n]).max(), nleg
