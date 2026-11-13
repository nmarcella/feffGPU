"""The layer shared by MD-EXAFS and RMC-EXAFS: conventions (k window, Fourier transform, R-factor), the forward model of
configurations (pair tables against the path stage), pair analysis, shards and keyed random streams."""
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PHASE_W = ROOT / "tests/data/W/phase.pad"                      # bcc W, K edge: FEFF 8.5L's own phase file
PHASE_ZNC = ROOT / "data/validation/A000/phase.pad"            # Zn-C rock salt, K edge of C: potentials 0 C, 1 Zn, 2 C
GPU = pytest.mark.gpu


def test_r_factor_forms_agree_and_window_is_larch_like():
    from feffgpu.exafs import FitWindow, k_grid, r_factor, r_factor_torch, window_tensor
    k = k_grid(20.0)
    rng = np.random.default_rng(0)
    data = np.sin(2 * 2.5 * k) * np.exp(-0.01 * k ** 2) / np.maximum(k, 1.0)
    model = data + 0.02 * rng.normal(size=len(k))
    w = FitWindow(kmin=3.0, kmax=16.0, dk=2.0, kweight=2, rmin=1.0, rmax=5.0)
    a = r_factor(k, data, model, kmin=w.kmin, kmax=w.kmax, rmin=w.rmin, rmax=w.rmax, kweight=w.kweight, dk=w.dk)
    b = float(r_factor_torch(torch.tensor(model)[None], torch.tensor(data), torch.tensor(k), w)[0])
    assert abs(a - b) < 1e-9 * max(a, 1e-30) + 1e-12, (a, b)
    win = window_tensor(torch.tensor(k), w).numpy()
    assert win[np.isclose(k, 3.0)][0] == pytest.approx(0.5) and win[np.isclose(k, 16.0)][0] == pytest.approx(0.5)   # sills centered on kmin, kmax
    assert np.all(win[(k >= 4.0) & (k <= 15.0)] == pytest.approx(1.0)) and np.all(win[(k <= 2.0) | (k >= 17.0)] == 0.0)
    assert w.independent_points == pytest.approx(2 * 13 * 4 / np.pi + 2)
    with pytest.raises(ValueError):
        FitWindow(kmin=5.0, kmax=4.0)


def test_a_window_without_sills_is_rectangular_in_both_forms():
    """dk = 0 (EvAX's default window): the numpy and the torch form give 1 from kmin to kmax and 0 outside, and the
    R-factor with it is a number (the numpy form used to divide by zero)."""
    import torch
    from feffgpu.compare import hanning, r_factor
    from feffgpu.exafs import hanning_window, k_grid
    k = k_grid(16.0)
    w = hanning(k, 3.0, 13.0, 0.0)
    assert np.isfinite(w).all() and w[(k >= 3.0) & (k <= 13.0)].min() == 1.0 and w[(k < 3.0) | (k > 13.0)].max() == 0.0
    wt = hanning_window(torch.as_tensor(k), torch.tensor([3.0]), torch.tensor([13.0]), dk=0.0)[0].numpy()
    assert np.array_equal(w, wt)
    chi = np.sin(2 * 2.5 * k) * np.exp(-0.01 * k ** 2)
    assert r_factor(k, chi, chi, kmin=3.0, kmax=13.0, dk=0.0) == 0.0
    assert 0.0 < r_factor(k, chi, 0.9 * chi, kmin=3.0, kmax=13.0, dk=0.0) < 0.02


def test_evax_window_is_the_hanning_window_shifted_by_half_a_sill():
    from feffgpu.exafs import FitWindow, evax_window, k_grid, window_tensor
    k = k_grid(20.0)
    ev = evax_window(k, 3.0, 15.0, 1.0)
    la = window_tensor(torch.tensor(k), FitWindow(kmin=3.5, kmax=14.5, dk=1.0)).numpy()
    assert np.abs(ev - la).max() < 1e-6


def test_read_chi_and_grid(tmp_path):
    from feffgpu.exafs import check_window_inside_data, k_grid, read_chi, to_grid
    f = tmp_path / "x.chi"
    f.write_text("# a header\n# k chi\n 2.0 0.5\n 2.5, 0.25\nnot numbers\n 3.0 0.0\n")
    kd, cd = read_chi(f)
    assert list(kd) == [2.0, 2.5, 3.0] and list(cd) == [0.5, 0.25, 0.0]
    g = to_grid(k_grid(4.0), kd, cd)
    assert g[int(round(2.25 / 0.05))] == pytest.approx(0.375) and g[0] == 0.0 and g[-1] == 0.0
    with pytest.raises(ValueError):
        check_window_inside_data(kd, 1.0, 3.0)


def test_pair_analysis_on_an_ideal_lattice():
    from feffgpu.analysis import bond_angles, partial_rdfs, shell_table, short_pairs
    from feffgpu.io import bcc_block
    a = 3.0
    sites, box = bcc_block(3, a)
    sym = np.array(["W"] * len(sites))
    rows = shell_table(sites, sym, box, [("W", "W", 2.0, 2.8), ("W", "W", 2.8, 3.2)])
    assert rows[0]["N"] == pytest.approx(8.0) and rows[0]["R"] == pytest.approx(a * np.sqrt(3) / 2) and rows[0]["sigma2"] == pytest.approx(0.0, abs=1e-12)
    assert rows[1]["N"] == pytest.approx(6.0) and rows[1]["R"] == pytest.approx(a)
    for cell in (box, [box] * 3, np.eye(3) * box):                                    # the three ways to give a cell
        assert shell_table(sites, sym, cell, [("W", "W", 2.0, 2.8)])[0]["N"] == pytest.approx(8.0)
    r, g, dist = partial_rdfs(sites[None], box, np.arange(len(sites)), sym, ["W"], rmax=3.2, dr=0.01)
    assert g["W"].sum() * 0.01 == pytest.approx(14.0)                                # 8 + 6 neighbors below 3.2 A
    sp = short_pairs(sites, sym, box, {"W-W": 2.7})
    assert sp["W-W"]["per_center"] == pytest.approx(8.0) and sp["W-W"]["closest"] == pytest.approx(a * np.sqrt(3) / 2)
    ang = bond_angles(sites, sym, box, "W", ("W", "W"), ((2.0, 2.8), (2.0, 2.8)))
    assert len(ang) == len(sites) * 28                                                 # 8 neighbors: 28 unordered pairs per center
    assert set(np.round(ang, 2)) == {70.53, 109.47, 180.0}                            # the cube's body diagonals


def test_shards_and_keyed_streams():
    from feffgpu.parallel import Shard, keyed_rng, resolve_devices, run_items
    items = list(range(23))
    parts = [Shard(i, 4).select(items) for i in range(1, 5)]
    assert sorted(sum(parts, [])) == items and Shard.parse("3/16") == Shard(3, 16) and Shard.parse(None) == Shard()
    with pytest.raises(ValueError):
        Shard.parse("0/4")
    one = [keyed_rng(7, "frame", i).random() for i in items]
    other = [keyed_rng(7, "frame", i).random() for i in reversed(items)][::-1]         # order of computation does not matter
    assert one == other and len(set(one)) == len(items) and keyed_rng(8, "frame", 0).random() != one[0]
    assert resolve_devices("cpu") == ["cpu"]
    assert run_items(lambda item, device: (item * 2, device), [1, 2, 3], devices="cpu") == [(2, "cpu"), (4, "cpu"), (6, "cpu")]


def _rock_salt(L=3, a=4.3765, seed=0):
    rng = np.random.default_rng(seed)
    sites, sym = [], []
    for i in range(L):
        for j in range(L):
            for l in range(L):
                for d, s in (((0, 0, 0), "C"), ((0.5, 0, 0), "Zn"), ((0, 0.5, 0), "Zn"), ((0, 0, 0.5), "Zn"),
                             ((0.5, 0.5, 0), "C"), ((0.5, 0, 0.5), "C"), ((0, 0.5, 0.5), "C"), ((0.5, 0.5, 0.5), "Zn")):
                    sites.append(((i + d[0]) * a, (j + d[1]) * a, (l + d[2]) * a)); sym.append(s)
    sites = np.array(sites); sym = np.array(sym)
    return sites + rng.normal(0, 0.05, sites.shape), sym, L * a


@GPU
def test_pair_tables_equal_the_two_leg_paths_of_the_path_stage():
    """One element, every atom an absorber: the histogram of pair distances times the table equals the two-leg paths of
    every absorber through the path finder and the path stage (to the table's 0.002 A grid)."""
    from feffgpu.exafs import PairTables, PathSpectra, k_grid
    from feffgpu.io import bcc_block
    sites, box = bcc_block(4, 3.1652)
    pos = sites + np.random.default_rng(0).normal(0, 0.05, sites.shape)
    k = k_grid(24.0)
    pairs = PairTables(str(PHASE_W), k, np.zeros(len(k)), 2.0, window=(3.0, 20.0), r_range=(1.0, 5.5), box=box, rmax=6.0, ms=False)
    P = torch.tensor(pos, device="cuda", dtype=torch.float32)[None]
    chi = pairs.single_scattering(P)
    ref = PathSpectra(str(PHASE_W), k, np.zeros(len(k)), 2.0, window=(3.0, 20.0), r_range=(1.0, 5.5), box=box, rmax=6.0, nlegmax=2, critpw=0.0)
    ref.refresh(pos[None], np.arange(len(pos))[None])
    with torch.no_grad():
        chi_ref = ref.spectra(P.reshape(-1, 3))[0]
    sel = (k > 3) & (k < 20)
    assert float((chi - chi_ref)[sel].abs().max() / chi_ref[sel].abs().max()) < 2e-3
    # differentiable in the positions
    Pg = P.clone().requires_grad_(True)
    pairs.single_scattering(Pg).pow(2).sum().backward()
    assert torch.isfinite(Pg.grad).all() and float(Pg.grad.abs().max()) > 0


@GPU
def test_multiple_scattering_rows_follow_the_absorbers_in_a_sheared_cell():
    """Two elements, an absorber subset, a monoclinic cell: one row per absorber, the subset evaluation agrees with the
    full pass, and the exact gradient is finite."""
    from feffgpu.exafs import MultipleScattering, k_grid
    pos, sym, box = _rock_salt(L=2)
    cell = np.array([[box, 0, 0], [0.15 * box, box, 0], [0, 0, box]])
    ipot = np.where(sym == "Zn", 1, 2)
    absorbers = np.nonzero(sym == "C")[0]
    k = k_grid(24.0)
    ms = MultipleScattering(str(PHASE_ZNC), k, 1.0, cell, rmax=4.0, critpw=1.0, nlegmax=3)
    P = torch.tensor(np.stack([pos, pos + 0.01]), device="cuda", dtype=torch.float32)
    counts = ms.enumerate(P, ipot, absorbers)
    assert 3 in counts and 4 not in counts
    rows = ms.full_pass(P.reshape(-1, 3))
    assert rows.shape == (2 * len(absorbers), len(k))
    sub = ms.abs_atoms[::5]
    assert float((ms.chi_absorbers(P.reshape(-1, 3), sub) - rows[ms.row_of[sub]]).abs().max()) < 1e-5
    g = ms.full_gradient(P.reshape(-1, 3), lambda chi: (chi ** 2).sum())
    assert torch.isfinite(g).all() and float(g.abs().max()) > 0


def _toy_spectrum(k, r0=2.5, amp=1.0):
    return amp * np.sin(2 * r0 * k + 0.3) * np.exp(-0.006 * k ** 2) / np.maximum(k, 0.5)


def test_s02_and_e0_shift_are_recovered_and_shared_between_samples(tmp_path):
    """Two samples of one edge share S0^2 and the shift: the joint fit recovers the planted values, the intervals contain
    them, and a comparison that is not free keeps its values."""
    from feffgpu.exafs import ETOK, FitWindow, SpectrumComparison, fit_shared, k_grid, plot_comparison, shared_intervals
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    k = k_grid(20.0)
    w = FitWindow(kmin=3.0, kmax=16.0, dk=2.0, kweight=2, rmin=1.0, rmax=4.0)
    s02_true, de0_true = 0.85, 3.0
    rng = np.random.default_rng(0)
    groups, calcs = [], []
    for r0 in (2.4, 2.9):
        calc = _toy_spectrum(k, r0)
        kshift = np.sqrt(np.clip(k ** 2 - ETOK * de0_true, 0, None))
        data = s02_true * _toy_spectrum(kshift, r0) + 2e-4 * rng.normal(size=len(k))          # the analytic function at the shifted k
        cmp = SpectrumComparison("A", k, data, w, s02=1.0, e0=0.0, fit_s02=True, fit_e0=True, device=dev)
        calcs.append(calc)
        groups.append([(cmp, torch.tensor(calc, device=dev, dtype=torch.float32), 1.0)])
    held = SpectrumComparison("B", k, _toy_spectrum(k, 2.0), w, s02=0.9, e0=1.0, device=dev)
    groups[0].append((held, torch.tensor(_toy_spectrum(k, 2.0), device=dev, dtype=torch.float32), 1.0))
    before = [float(g[0][0].residual(g[0][1][None])[0]) for g in groups]
    fit = fit_shared(groups, s02_bounds=(0.5, 1.5), de0_bounds=(-8.0, 8.0))
    assert fit["s02_names"] == ["A"] and fit["de0_names"] == ["A"] and not fit["at_bound"]
    assert fit["s02"]["A"] == pytest.approx(s02_true, rel=0.01) and fit["de0"]["A"] == pytest.approx(de0_true, abs=0.05)
    assert groups[0][0][0].s02 == groups[1][0][0].s02 and groups[0][0][0].de0 == groups[1][0][0].de0     # shared
    assert held.s02 == 0.9 and held.de0 == 1.0
    assert all(fit["ratios"][i][0] < 1e-3 * before[i] for i in range(2))
    iv = shared_intervals(groups)
    assert iv["positive_definite"] and iv["s02_lo"]["A"] < s02_true < iv["s02_hi"]["A"] and iv["de0_lo"]["A"] < de0_true < iv["de0_hi"]["A"]
    assert iv["s02_hi"]["A"] - iv["s02_lo"]["A"] < 0.1 and "standard deviations" in iv["method"]
    out = tmp_path / "fit.png"
    plot_comparison([g[0][0] for g in groups[:1]], [{"A": calcs[0]}], out=out)
    assert out.stat().st_size > 10000


def test_comparison_shift_matches_direct_evaluation():
    from feffgpu.exafs import ETOK, FitWindow, SpectrumComparison, k_grid
    k = k_grid(20.0)
    cmp = SpectrumComparison("A", k, _toy_spectrum(k), FitWindow(kmin=3.0, kmax=16.0, rmin=1.0, rmax=4.0), device="cpu")
    chi = torch.tensor(_toy_spectrum(k), dtype=torch.float32)[None]
    shifted = cmp.shift(chi, 4.0)[0].numpy()
    kshift = np.sqrt(np.clip(k ** 2 - ETOK * 4.0, 0, None))
    direct = np.sin(2 * 2.5 * kshift + 0.3) * np.exp(-0.006 * kshift ** 2) / np.maximum(kshift, 0.5)
    sel = (k > 3) & (k < 16)
    assert np.abs(shifted - direct)[sel].max() < 2e-3 * np.abs(direct[sel]).max()
    assert cmp.r_factor(_toy_spectrum(k)) < 1e-12 and cmp.r_factor(1.1 * _toy_spectrum(k)) == pytest.approx(0.01, rel=1e-3)
    with pytest.raises(ValueError):
        SpectrumComparison("A", k[::2], _toy_spectrum(k[::2]), FitWindow(), device="cpu")
