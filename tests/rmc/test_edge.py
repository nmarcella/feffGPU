"""The forward model of one edge (feffgpu.exafs.edge.EdgeModel) and the optional wavelet objectives: single scattering equals
the two-leg paths of the path stage for any element mix and cell, the tables are the pair tables themselves (no thermal
widths), ignored elements are invisible, the E0 shift on the model equals tables built at the shifted energy, S0^2 and
the E0 shift are recovered with intervals and shared between samples within the configured bounds; the Fourier
transform of the shared layer equals xraylarch's (CPU; skipped where xraylarch is not installed)."""
import math
import numpy as np
import pytest
import torch
from rmc_helpers import GPU, DEVICE, PHASE_W, PHASE_ZNC, bcc, rock_salt
from feffgpu.exafs import FitWindow, PairTables, PathSpectra, k_grid, fit_shared, shared_intervals, likelihood_weights
from feffgpu.exafs.edge import EdgeModel


def on_gpu(test):
    """Every test of this file but the comparison of the transform with xraylarch (CPU) runs on the GPU."""
    return pytest.mark.gpu(GPU(test))


K = k_grid(24.0)
WIN = FitWindow(kmin=3.0, kmax=16.0, dk=1.0, kweight=2, rmin=1.0, rmax=5.0)


def w_edge(data=None, sym=None, cell=None, **kw):
    s, p, c = bcc(4)
    return EdgeModel("W", K, np.zeros(len(K)) if data is None else data, kw.pop("window", WIN), str(PHASE_W), s if sym is None else sym,
                     c if cell is None else cell, "W", rpath=kw.pop("rpath", 5.0), nleg=kw.pop("nleg", 2), device=DEVICE, **kw), p


@on_gpu
def test_batched_single_scattering_matches_the_pair_tables():
    e, pos = w_edge()
    rng = np.random.default_rng(0)
    states = pos[None] + (rng.random((4, len(pos), 3)) * 2 - 1) * 0.02 / np.sqrt(2)
    chi = e.spectrum(e.static_histograms(states))
    pairs = PairTables(str(PHASE_W), K, np.zeros(len(K)), 0.0, window=(3.0, 16.0), r_range=(1.0, 5.0), box=bcc(4)[2], rmax=5.0, ms=False)
    for s in range(4):
        ref = pairs.single_scattering(torch.tensor(states[s:s + 1], device=DEVICE, dtype=torch.float32))
        assert torch.allclose(chi[s], ref, atol=1e-6, rtol=1e-4)


@on_gpu
@pytest.mark.parametrize("cell_kind", ["number", "matrix"])
def test_multi_element_single_scattering_matches_the_path_stage(cell_kind):
    """C absorbers of the Zn-C rock salt (an absorber subset, two scattering potentials): the histogram of the pairs times the
    tables equals the two-leg paths of those absorbers through the path finder and the path stage."""
    sym, pos, cell = rock_salt(3)
    pos = pos + np.random.default_rng(0).normal(0, 0.05, pos.shape)
    box = float(cell[0, 0]) if cell_kind == "number" else cell
    e = EdgeModel("C", K, np.zeros(len(K)), FitWindow(kmin=3.0, kmax=12.0, rmin=1.0, rmax=5.0), str(PHASE_ZNC), sym, box, "C",
                  potentials={"Zn": 1, "C": 2}, rpath=5.0, nleg=2, device=DEVICE)
    chi = e.spectrum(e.static_histograms(pos[None]))[0]
    ref = PathSpectra(str(PHASE_ZNC), K, np.zeros(len(K)), 0.0, window=(3.0, 12.0), r_range=(1.0, 5.0), box=box, rmax=5.0, nlegmax=2, critpw=0.0)
    absorbers = np.nonzero(sym == "C")[0]
    ref.refresh(pos[None], absorbers[None], ipot=np.where(sym == "Zn", 1, 2))
    with torch.no_grad():
        chi_ref = ref.spectra(torch.tensor(pos, device=DEVICE, dtype=torch.float32))[0]
    sel = (K > 3) & (K < 12)
    assert float((chi - chi_ref)[sel].abs().max() / chi_ref[sel].abs().max()) < 2e-3     # the tables' 0.002 A grid against the exact path


@on_gpu
def test_the_tables_are_the_pair_tables_and_windows_do_not_change_the_spectrum():
    e, pos = w_edge(pair_windows={"W": [2.0, 3.0, 4.0, 5.0]})
    assert len(e.types) == 3 and all(torch.equal(e.tables[t], e.pairs.tables[1]) for t in range(3))   # no convolution: the positions carry all disorder
    one, _ = w_edge()
    p = torch.tensor(pos[None] + np.random.default_rng(1).normal(0, 0.05, pos.shape), device=DEVICE, dtype=torch.float32)
    a = e.spectrum(e.static_histograms(p))
    b = one.spectrum(one.static_histograms(p))
    assert float((a - b).abs().max()) < 1e-6 * float(b.abs().max())
    with pytest.raises(ValueError):
        w_edge(pair_windows={"W": [2.0, 4.0]})                                            # the last boundary must reach rpath
    with pytest.raises(ValueError):
        w_edge(pair_windows={"Mo": [2.0, 5.0]})                                           # an element the structure does not have


@on_gpu
def test_ignored_elements_are_invisible():
    """Hydrogen atoms next to the C atoms of the rock salt, ignored by the edge: the same single scattering, the same
    multiple-scattering paths and the same spectrum as without them."""
    sym, pos, cell = rock_salt(2)
    c = np.nonzero(sym == "C")[0][:6]
    sym_h = np.concatenate([sym, ["H"] * len(c)])
    pos_h = np.concatenate([pos, pos[c] + np.array([0.0, 0.0, 0.98])])
    kw = dict(rpath=4.3, nleg=3, criteria=2.5, device=DEVICE)
    win = FitWindow(kmin=3.0, kmax=12.0, rmin=1.0, rmax=4.0)
    e_h = EdgeModel("C", K, np.zeros(len(K)), win, str(PHASE_ZNC), sym_h, cell, "C", potentials={"Zn": 1, "C": 2}, ignore=("H",), **kw)
    e_0 = EdgeModel("C", K, np.zeros(len(K)), win, str(PHASE_ZNC), sym, cell, "C", potentials={"Zn": 1, "C": 2}, **kw)
    p_h = torch.tensor(pos_h[None], device=DEVICE, dtype=torch.float32)
    p_0 = torch.tensor(pos[None], device=DEVICE, dtype=torch.float32)
    chi_h, chi_0 = e_h.spectra(p_h), e_0.spectra(p_0)
    assert e_h.ms.counts == e_0.ms.counts
    assert float((chi_h - chi_0).abs().max()) < 1e-6 * float(chi_0.abs().max()) + 1e-9
    with pytest.raises(ValueError):
        EdgeModel("C", K, np.zeros(len(K)), win, str(PHASE_ZNC), sym, cell, "C", potentials={"Zn": 1}, **kw)   # a potential missing


@on_gpu
def test_the_energy_shift_equals_tables_built_at_the_shifted_energy():
    """The interpolated shift of a model calculated at e0 = 0 equals the model calculated at e0 = -5 eV (xraylarch's
    convention k' = sqrt(k^2 - 0.2625 dE0)), to about 1e-4 of the signal inside the window."""
    pos = torch.tensor(bcc(4)[1][None], device=DEVICE, dtype=torch.float32)
    chis, edges = {}, {}
    for e0 in (0.0, -5.0):
        e, _ = w_edge(e0=e0)
        chis[e0], edges[e0] = e.spectrum(e.static_histograms(pos)), e
    shifted = edges[0.0].shift(chis[0.0], -5.0)
    k = torch.tensor(K, device=DEVICE, dtype=torch.float32)
    sel = (k >= 3) & (k <= 16)
    err = float(((k ** 2 * (shifted - chis[-5.0]))[:, sel] ** 2).sum() / ((k ** 2 * chis[-5.0])[:, sel] ** 2).sum())
    assert err < 1e-4, err
    assert torch.allclose(edges[0.0].shift(chis[0.0], 0.0), chis[0.0])
    d = torch.tensor(-2.0, device=DEVICE, requires_grad=True)
    edges[0.0].shift(chis[0.0], d)[:, sel].sum().backward()
    assert torch.isfinite(d.grad) and float(d.grad.abs()) > 0


def kicked(n_states, seed, kick=0.05):
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed)
    return torch.tensor(bcc(4)[1], device=DEVICE, dtype=torch.float32)[None] + kick * torch.randn(n_states, 128, 3, device=DEVICE, generator=g)


@on_gpu
def test_s02_and_the_e0_shift_are_recovered_with_intervals():
    pos = kicked(1, 0)
    e, _ = w_edge(window=WIN)
    s02_true, de0_true = 0.8, -4.0
    chi_t = e.model(e.spectrum(e.static_histograms(pos)), s02_true, de0_true)[0].cpu().numpy()
    noise = 0.01 * np.random.default_rng(3).normal(size=len(K)) / np.maximum(K, 1.0) ** 2 * (K > 0)
    e2, _ = w_edge(data=chi_t + noise, fit_s02=True, fit_e0=True)
    groups = [[(e2, e2.spectrum(e2.static_histograms(pos)), 1.0)]]
    res = fit_shared(groups, s02_bounds=(0.5, 1.5), de0_bounds=(-8.0, 8.0), iters=150)
    assert abs(res["s02"]["W"] - s02_true) < 0.03 and abs(res["de0"]["W"] - de0_true) < 0.3, res
    iv = shared_intervals(groups)
    assert iv["positive_definite"] and 0 < iv["s02_sd_log"]["W"] < 0.2 and 0 < iv["de0_sd"]["W"] < 2
    assert abs(math.log(res["s02"]["W"] / s02_true)) <= 3 * iv["s02_sd_log"]["W"] and abs(res["de0"]["W"] - de0_true) <= 3 * iv["de0_sd"]["W"]
    assert abs(iv["de0_s02_correlation"]["W"]) < 1
    e3, _ = w_edge(data=chi_t + noise)                                                        # nothing free: the values stay
    res3 = fit_shared([[(e3, e3.spectrum(e3.static_histograms(pos)), 1.0)]])
    assert res3["s02_names"] == [] and e3.s02 == 1.0 and e3.de0 == 0.0


@on_gpu
def test_the_joint_fit_shares_s02_and_the_e0_shift_within_the_configured_bounds():
    s02_true, de0_true = 0.8, -4.0
    rng = np.random.default_rng(5)
    groups = []
    for seed, kick in ((1, 0.04), (2, 0.07)):                                                 # two samples: one lattice, different disorder
        pos = kicked(1, seed, kick)
        e, _ = w_edge()
        chi_t = e.model(e.spectrum(e.static_histograms(pos)), s02_true, de0_true)[0].cpu().numpy()
        e2, _ = w_edge(data=chi_t + 0.01 * rng.normal(size=len(K)) / np.maximum(K, 1.0) ** 2 * (K > 0), fit_s02=True, fit_e0=True)
        groups.append([(e2, e2.spectrum(e2.static_histograms(pos)), 1.0)])
    res = fit_shared(groups, s02_bounds=(0.5, 1.5), de0_bounds=(-8.0, 8.0), iters=150)
    assert abs(res["s02"]["W"] - s02_true) < 0.03 and abs(res["de0"]["W"] - de0_true) < 0.3, res
    assert groups[0][0][0].s02 == groups[1][0][0].s02 == res["s02"]["W"]                     # one value for every spectrum of the name
    shared = shared_intervals(groups, W=likelihood_weights(groups))
    assert shared["positive_definite"] and abs(math.log(res["s02"]["W"] / s02_true)) <= 3 * shared["s02_sd_log"]["W"]
    for g in groups:                                                                          # the bounds hold: a range that excludes the truth stops S0^2 on its edge
        g[0][0].set_s02(1.0)
    res = fit_shared(groups, s02_bounds=(0.9, 1.5), de0_bounds=(-8.0, 8.0), iters=150)
    assert res["s02"]["W"] == pytest.approx(0.9, abs=1e-4) and res["at_bound"] == ["W"]


@on_gpu
def test_wavelet_residuals_are_zero_at_the_target_and_positive_elsewhere():
    pos = kicked(2, 7, 0.02)
    e, _ = w_edge(window=FitWindow(kmin=3.0, kmax=18.0, dk=1.0, kweight=2, rmin=1.0, rmax=5.5), rpath=6.0)
    chi = e.spectrum(e.static_histograms(pos))
    for objective in ("wavelet", "cauchy_wavelet"):
        e2, _ = w_edge(data=chi[0].cpu().numpy(), window=FitWindow(kmin=3.0, kmax=18.0, dk=1.0, kweight=2, rmin=1.0, rmax=5.5), rpath=6.0,
                       objective=objective)
        r = e2.residual(chi)
        assert float(r[0]) < 1e-8 and float(r[1]) > 1e-3, (objective, r)
        W = e2.wavelet.transform(chi[:1])
        assert W.dim() == 3 and torch.isfinite(W).all()


def test_the_transform_matches_xraylarch():
    pytest.importorskip("larch")
    from larch import Group
    from larch.xafs import xftf
    from feffgpu.exafs import ft, hanning_window, r_grid
    k = 0.05 * np.arange(481)
    rng = np.random.default_rng(0)
    chi = np.exp(-0.004 * k ** 2) * (np.sin(2 * 2.4 * k + 1.0) / np.maximum(k, 0.5) ** 2 + 0.3 * np.sin(2 * 3.9 * k) / np.maximum(k, 0.5) ** 2) + 0.002 * rng.normal(size=len(k))
    for kmin, kmax, dk, kw in ((2.0, 15.0, 2.0, 2), (3.0, 14.0, 1.0, 2), (3.0, 12.0, 1.0, 3)):
        g = Group(k=k, chi=chi)
        xftf(g, kmin=kmin, kmax=kmax, dk=dk, kweight=kw, window="hanning", rmax_out=10.0)
        kt = torch.tensor(k, dtype=torch.float64)
        w = hanning_window(kt, torch.tensor([kmin], dtype=torch.float64), torch.tensor([kmax], dtype=torch.float64), dk=dk)[0]
        c = ft(torch.tensor(chi, dtype=torch.float64)[None], kt, w, kweight=kw)[0].numpy()
        r = r_grid().numpy()
        sel = (r >= 0.5) & (r <= 6)
        mag, re = np.interp(r, g.r, g.chir_mag), np.interp(r, g.r, g.chir_re)
        assert np.abs(mag[sel] - np.abs(c[sel])).max() / mag[sel].max() < 2e-3
        assert np.abs(re[sel] - c[sel].real).max() / mag[sel].max() < 2e-3
