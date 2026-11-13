"""The spectrum of a frame: against an independent calculation with the path finder and path stage of every absorber,
the decomposition by path type, and the damping of a trajectory of Gaussian-displaced frames."""
import numpy as np
import pytest
import torch
from mdexafs_testing import PHASE_W, PHASE_ZNC, bcc_sites, gaussian_frames, gpu, rock_salt, write_config, write_extxyz


def test_path_type_names_merge_a_path_with_its_time_reverse():
    from feffgpu.md.paths import PathTypes, path_type_name
    assert path_type_name("Zn", ["Zn", "C"]) == path_type_name("Zn", ["C", "Zn"]) == "Zn-C-Zn"
    assert path_type_name("Zn", ["C", "Zn", "Zn"]) == "Zn-C-Zn-Zn"
    t = PathTypes("C", ["Zn", "C"], ["C", "Zn"], 4, {1: "Zn", 2: "C"})
    assert t.ss == ["C-Zn", "C-C"] and t.ms[:3] == ["C-C-C", "C-C-Zn", "C-Zn-Zn"]
    assert len(t.ms) == 3 + 6                                                       # 3 three-leg and 6 four-leg types of two elements
    assert t.normalize("C-Zn-C") == "C-C-Zn" and t.normalize("C-Zn") == "C-Zn"
    with pytest.raises(KeyError, match="known types"):
        t.normalize("Zn-C")
    lut = t.lookup(3, 3).numpy()
    assert lut[1, 2] == lut[2, 1] == t.ms.index("C-C-Zn")                             # both orientations, one type
    lut4 = t.lookup(4, 3).numpy()
    assert lut4[1, 0, 1] == t.ms.index("C-Zn-C-Zn")                                   # the absorber itself in the middle
    # five and more legs: one type per number of legs
    long = PathTypes("C", ["Zn", "C"], ["C", "Zn"], 7, {1: "Zn", 2: "C"})
    assert long.ms[:9] == t.ms and long.ms[9:] == ["C-(5 legs)", "C-(6 legs)", "C-(7 legs)"] == long.long
    assert long.legs_of("C-(6 legs)") == 6 and long.legs_of("C-Zn-C-Zn") == 4 and long.long_index(5) == 9
    assert long.normalize("C-(5 legs)") == "C-(5 legs)"
    with pytest.raises(KeyError, match="known types"):
        t.normalize("C-(5 legs)")                                                    # not a type of an edge with four legs
    with pytest.raises(ValueError, match="between 2 and 8"):
        PathTypes("C", ["Zn", "C"], ["C", "Zn"], 9, {1: "Zn", 2: "C"})


@gpu
def test_paths_of_five_legs_are_one_type_and_match_the_shared_classes():
    from feffgpu.exafs import MultipleScattering, k_grid
    from feffgpu.md.spectrum import EdgeSpectra
    k = k_grid(16.0)
    sites, sym, box = rock_salt(3)
    pos = gaussian_frames(sites, 1, 0.04, seed=2)[0]
    calc = EdgeSpectra(str(PHASE_ZNC), "C", {"Zn": 1, "C": 2}, ["C", "Zn"], k, rpath=6.0, nleg=5, criteria=0.0, device="cuda")
    assert calc.types.names[-1] == "C-(5 legs)"
    ab = np.nonzero(sym == "C")[0][::27]
    s = calc.spectrum(pos, np.eye(3) * box, ab, sym)
    assert s.n_paths.get(5, 0) > 0
    part = s.chi_types[calc.types.names.index("C-(5 legs)")]
    scale = np.abs(s.chi).max()
    assert np.abs(part).max() > 0 and np.abs(s.chi_types.sum(0) - s.chi).max() < 1e-12 * scale + 1e-15
    ipot = torch.as_tensor(np.where(sym == "Zn", 1, 2), device="cuda")
    P = torch.tensor(pos, device="cuda", dtype=torch.float32)
    ms = MultipleScattering(str(PHASE_ZNC), k, 0.0, box, rmax=6.0, critpw=0.0, nlegmax=5)
    ms.enumerate(P[None], ipot, torch.as_tensor(ab, device="cuda"))
    assert ms.counts.get(5, 0) == s.n_paths[5]
    ref = ms.full_pass(P.reshape(-1, 3)).mean(0).double().cpu().numpy()
    assert np.abs(s.chi_ms - ref).max() < 2e-4 * np.abs(ref).max()
    four = MultipleScattering(str(PHASE_ZNC), k, 0.0, box, rmax=6.0, critpw=0.0, nlegmax=4)
    four.enumerate(P[None], ipot, torch.as_tensor(ab, device="cuda"))
    only_five = ref - four.full_pass(P.reshape(-1, 3)).mean(0).double().cpu().numpy()
    assert np.abs(part - only_five).max() < 2e-4 * np.abs(ref).max()                  # the type holds the five-leg paths and nothing else


def test_absorber_subsets_depend_only_on_seed_frame_and_element():
    from feffgpu.md.spectrum import choose_absorbers
    cand = np.arange(10, 110)
    a = choose_absorbers(cand, 12, seed=5, frame_index=40, element="Zn")
    assert len(a) == 12 and np.all(np.diff(a) > 0) and set(a) <= set(cand)
    assert np.array_equal(a, choose_absorbers(cand, 12, 5, 40, "Zn"))
    assert not np.array_equal(a, choose_absorbers(cand, 12, 5, 41, "Zn"))
    assert not np.array_equal(a, choose_absorbers(cand, 12, 6, 40, "Zn"))
    assert np.array_equal(choose_absorbers(cand, "all", 5, 40, "Zn"), cand) and len(choose_absorbers(cand, 500, 5, 40, "Zn")) == 100


def test_kmax_beyond_the_phase_file_is_refused():
    from feffgpu.exafs import k_grid
    from feffgpu.md.spectrum import EdgeSpectra
    with pytest.raises(ValueError, match="beyond the phase file's k range"):
        EdgeSpectra(str(PHASE_ZNC), "C", {"Zn": 1, "C": 2}, ["C", "Zn"], k_grid(30.0), device="cpu")


def _independent(phase, pos, cell, absorbers, ipot, k, rpath, nleg, criteria):
    """Single scattering: every pair through the path finder (no filter) and the path stage; multiple scattering:
    the path finder with the importance filter and the path stage. PathSpectra, one call per kind of path."""
    from feffgpu.exafs import PathSpectra
    P = torch.tensor(pos, device="cuda", dtype=torch.float32)
    ss = PathSpectra(str(phase), k, np.zeros(len(k)), 0.0, box=cell, rmax=rpath, nlegmax=2, critpw=0.0)
    ss.refresh(pos[None], absorbers[None], ipot)
    ms = PathSpectra(str(phase), k, np.zeros(len(k)), 0.0, box=cell, rmax=rpath, nlegmax=nleg, critpw=criteria)
    counts = ms.refresh(pos[None], absorbers[None], ipot)
    with torch.no_grad():
        chi_ss = ss.spectra(P.reshape(-1, 3))[0].double().cpu().numpy()
        chi_ms = ms.spectra(P.reshape(-1, 3), nlegs=tuple(range(3, nleg + 1)))[0].double().cpu().numpy()
    return chi_ss, chi_ms, counts


@gpu
def test_frame_spectrum_equals_the_path_stage_of_every_absorber():
    """bcc W (one element) and the Zn-C rock salt in a sheared cell (two elements, a subset of C absorbers): the
    spectrum of the frame equals the independent path-stage calculation, to the pair table's interpolation error
    in single scattering and to single precision in multiple scattering."""
    from feffgpu.exafs import k_grid
    from feffgpu.md.spectrum import EdgeSpectra
    k = k_grid(18.0)
    sel = (k > 3) & (k < 17)
    sites, box = bcc_sites(4)
    pos = gaussian_frames(sites, 1, 0.06, seed=0)[0]
    sym = np.array(["W"] * len(pos))
    calc = EdgeSpectra(str(PHASE_W), "W", {"W": 1}, ["W"], k, rpath=5.0, nleg=3, criteria=0.0, device="cuda")
    ab = np.arange(0, len(pos), 16)
    spec = calc.spectrum(pos, np.eye(3) * box, ab, sym)
    chi_ss, chi_ms, counts = _independent(PHASE_W, pos, box, ab, None, k, 5.0, 3, 0.0)
    assert counts.get(3, 0) == spec.n_paths[3] > 0
    assert np.abs(spec.chi_ss - chi_ss)[sel].max() < 1e-3 * np.abs(chi_ss[sel]).max()
    assert np.abs(spec.chi_ms - chi_ms)[sel].max() < 1e-4 * np.abs(chi_ms[sel]).max()
    assert np.abs(spec.chi - (chi_ss + chi_ms))[sel].max() < 1e-3 * np.abs((chi_ss + chi_ms)[sel]).max()

    pos2, sym2, box2 = rock_salt(3)
    pos2 = pos2 + np.random.default_rng(1).normal(0, 0.05, pos2.shape)
    cell = np.array([[box2, 0, 0], [0.12 * box2, box2, 0], [0, 0, box2]])                    # monoclinic: fractional minimum image
    calc2 = EdgeSpectra(str(PHASE_ZNC), "C", {"Zn": 1, "C": 2}, ["C", "Zn"], k, rpath=5.0, nleg=3, criteria=2.0, device="cuda")
    ab2 = np.nonzero(sym2 == "C")[0][::9]
    spec2 = calc2.spectrum(pos2, cell, ab2, sym2)
    ipot2 = np.where(sym2 == "Zn", 1, 2)
    c_ss, c_ms, counts2 = _independent(PHASE_ZNC, pos2, cell, ab2, ipot2, k, 5.0, 3, 2.0)
    assert counts2.get(3, 0) == spec2.n_paths[3] > 0
    assert np.abs(spec2.chi_ss - c_ss)[sel].max() < 1e-3 * np.abs(c_ss[sel]).max()
    assert np.abs(spec2.chi_ms - c_ms)[sel].max() < 1e-4 * np.abs(c_ms[sel]).max()


@gpu
def test_parts_by_path_type_sum_to_the_total_and_match_the_shared_classes():
    from feffgpu.exafs import MultipleScattering, PairTables, k_grid
    from feffgpu.md.spectrum import EdgeSpectra
    k = k_grid(18.0)
    sites, box = bcc_sites(4)
    pos = gaussian_frames(sites, 1, 0.05, seed=4)[0]
    sym = np.array(["W"] * len(pos))
    calc = EdgeSpectra(str(PHASE_W), "W", {"W": 1}, ["W"], k, rpath=5.6, nleg=4, criteria=0.5, ms_bin_width=0.5, device="cuda")
    ab = np.arange(3, len(pos), 21)
    s = calc.spectrum(pos, np.eye(3) * box, ab, sym)
    assert s.n_paths.get(4, 0) > 0 and s.chi_types.shape == (len(calc.types.names), len(k))
    scale = np.abs(s.chi).max()
    assert np.abs(s.chi_types.sum(0) - s.chi).max() < 1e-12 * scale + 1e-15                       # types sum to the total
    assert np.abs(s.chi_ss + s.chi_ms - s.chi).max() < 1e-12 * scale + 1e-15
    assert np.abs(s.ms_binned.sum(1) - s.chi_types[len(calc.types.ss):]).max() < 1e-12 * scale + 1e-15   # bins sum to their type
    P = torch.tensor(pos, device="cuda", dtype=torch.float32)
    ms = MultipleScattering(str(PHASE_W), k, 0.0, box, rmax=5.6, critpw=0.5, nlegmax=4)
    ms.enumerate(P[None], None, torch.as_tensor(ab, device="cuda"))
    ref = ms.full_pass(P.reshape(-1, 3)).mean(0).double().cpu().numpy()
    assert np.abs(s.chi_ms - ref).max() < 2e-4 * np.abs(ref).max()                                 # single-precision summation order
    every = calc.spectrum(pos, np.eye(3) * box, np.arange(len(pos)), sym)                          # one element, every atom an absorber
    pt = PairTables(str(PHASE_W), k, np.zeros(len(k)), 0.0, box=box, rmax=5.6, r_min=1.0, ms=False)
    ss_ref = pt.single_scattering(P[None]).double().cpu().numpy()
    assert np.abs(every.chi_ss - ss_ref).max() < 1e-5 * np.abs(ss_ref).max()


def _envelope(k, chi, kmin=1.5, kmax=18.0, dk=3.0, rlo=1.6, rhi=3.9, dr=1.0):
    """|chi(k)| of a single shell by Fourier filtering (smooth windows in k and in R)."""
    from feffgpu.compare import hanning
    n = 4096
    x = np.zeros(n)
    x[:len(k)] = chi * hanning(k, kmin, kmax, dk)
    r = np.arange(n) * np.pi / (n * (k[1] - k[0]))
    y = np.fft.ifft(np.fft.fft(x) * hanning(r, rlo, rhi, dr) * (r < r.max() / 2))[:len(k)] * 2
    return np.abs(y)


@gpu
def test_gaussian_frames_damp_the_first_shell_like_exp_minus_2_k2_sigma2(tmp_path):
    """Frames of independent Gaussian displacements (standard deviation s per component): the first-shell part of
    the averaged spectrum, divided by that of the ideal lattice, decays like exp(-2 k^2 sigma^2) with
    sigma^2 = 2 s^2 within the sampling error, and agrees with the MSRD of the frames' pair distances."""
    from feffgpu.analysis import shell_table
    from feffgpu.md import average_cache, load_config, run
    from feffgpu.md.config import AverageConfig
    s = 0.04
    sites, box = bcc_sites(4)
    sym = np.array(["W"] * len(sites))
    frames = gaussian_frames(sites, 20, s, seed=1)
    write_extxyz(tmp_path / "t.extxyz", sym, frames, np.eye(3) * box)
    write_extxyz(tmp_path / "ideal.extxyz", sym, [sites], np.eye(3) * box)
    for name, traj in (("avg", "t.extxyz"), ("ideal", "ideal.extxyz")):
        write_config(tmp_path / f"{name}.toml", {"system": {"trajectory": traj, "output": f"out_{name}"}, "lattice": {"dynamic": True},
                                                 "edge": [{"element": "W", "potentials": "file", "phase": str(PHASE_W), "rpath": 4.0,
                                                           "nleg": 2, "kmax": 18.0}]})
        run(load_config(tmp_path / f"{name}.toml"), log=lambda m: None)
    first = AverageConfig(path_types=["W-W"], max_distance=2.95)                                  # the first shell, from the cache
    avg = average_cache(tmp_path / "out_avg/chi.h5", first)["W"]
    ideal = average_cache(tmp_path / "out_ideal/chi.h5", first)["W"]
    k = avg["k"]
    m = (k >= 5.0) & (k <= 14.0)
    y = np.log(_envelope(k, avg["mean"])[m] / _envelope(k, ideal["mean"])[m])
    A = np.vstack([np.ones(m.sum()), k[m] ** 2]).T
    (c, slope), *_ = np.linalg.lstsq(A, y, rcond=None)
    sigma2_spectrum = -slope / 2
    per_frame = [shell_table(f, sym, box, [("W", "W", 2.3, 2.95)])[0]["sigma2"] for f in frames]
    sigma2_frames = shell_table(np.array(frames), sym, box, [("W", "W", 2.3, 2.95)])[0]["sigma2"]
    se = np.std(per_frame, ddof=1) / np.sqrt(len(per_frame))                                       # sampling error of the MSRD
    assert np.abs(y - A @ np.array([c, slope])).max() < 0.05                                       # the log-ratio is a line in k^2
    assert abs(c) < 0.03                                                                            # same number of neighbors
    assert abs(sigma2_spectrum - sigma2_frames) < 0.03 * sigma2_frames
    assert abs(sigma2_spectrum - 2 * s * s) < 3 * se + 0.02 * 2 * s * s
