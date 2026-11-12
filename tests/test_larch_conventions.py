"""The conventions of feffgpu.exafs against xraylarch itself: the Hanning window, the Fourier transform to R space and
the R-factor on the R window. These tests need xraylarch (`pip install "feffgpu[larch]"`) and are skipped without it."""
import numpy as np
import pytest
import torch

larch = pytest.importorskip("larch", reason="needs xraylarch: pip install 'feffgpu[larch]'")

CASES = [(2.0, 15.0, 2.0, 2), (3.0, 14.0, 1.0, 2), (3.0, 12.0, 1.0, 3), (2.5, 17.5, 3.0, 1)]       # kmin, kmax, dk, kweight


def _spectrum(seed=0):
    k = 0.05 * np.arange(481)
    rng = np.random.default_rng(seed)
    chi = np.exp(-0.004 * k ** 2) * (np.sin(2 * 2.4 * k + 1.0) + 0.3 * np.sin(2 * 3.9 * k)) / np.maximum(k, 0.5) ** 2 + 0.002 * rng.normal(size=len(k))
    return k, chi


@pytest.mark.parametrize("kmin,kmax,dk,kw", CASES)
def test_the_window_is_xraylarchs_hanning_window(kmin, kmax, dk, kw):
    from larch.xafs import ftwindow
    from feffgpu.compare import hanning
    from feffgpu.exafs import FitWindow, window_tensor
    k, _ = _spectrum()
    ref = ftwindow(k, xmin=kmin, xmax=kmax, dx=dk, window="hanning")
    mine = window_tensor(torch.tensor(k, dtype=torch.float64), FitWindow(kmin=kmin, kmax=kmax, dk=dk, kweight=kw)).numpy()
    assert np.abs(mine - ref).max() < 1e-9
    assert np.abs(hanning(k, kmin, kmax, dk) - ref).max() < 1e-9                        # the numpy form used by feffgpu.compare


@pytest.mark.parametrize("kmin,kmax,dk,kw", CASES)
def test_the_transform_is_xraylarchs_xftf(kmin, kmax, dk, kw):
    from larch import Group
    from larch.xafs import xftf
    from feffgpu.exafs import ft, hanning_window, r_grid
    k, chi = _spectrum()
    g = Group(k=k, chi=chi)
    xftf(g, kmin=kmin, kmax=kmax, dk=dk, kweight=kw, window="hanning", rmax_out=10.0)
    kt = torch.tensor(k, dtype=torch.float64)
    w = hanning_window(kt, torch.tensor([kmin], dtype=torch.float64), torch.tensor([kmax], dtype=torch.float64), dk=dk)[0]
    c = ft(torch.tensor(chi, dtype=torch.float64)[None], kt, w, kweight=kw)[0].numpy()
    r = r_grid().numpy()
    n = min(len(r), len(g.r))
    assert np.abs(r[:n] - g.r[:n]).max() < 1e-6                                         # the same R grid
    sel = r[:n] <= 8.0
    assert np.abs(g.chir_re[:n][sel] - c[:n][sel].real).max() / g.chir_mag.max() < 1e-6
    assert np.abs(g.chir_im[:n][sel] - c[:n][sel].imag).max() / g.chir_mag.max() < 1e-6


@pytest.mark.parametrize("kmin,kmax,dk,kw", CASES[:3])
def test_the_r_factor_is_the_one_computed_from_xraylarchs_transform(kmin, kmax, dk, kw):
    """sum |chi_R(model) - chi_R(data)|^2 / sum |chi_R(data)|^2 over the R window, real and imaginary parts, from
    xraylarch's own chi(R), equals SpectrumComparison's R-factor and feffgpu.compare.r_factor."""
    from larch import Group
    from larch.xafs import xftf
    from feffgpu.compare import r_factor
    from feffgpu.exafs import FitWindow, SpectrumComparison
    k, data = _spectrum(0)
    model = 0.93 * np.interp(k * 1.004, k, data) + 0.004 * np.sin(5.3 * k) / np.maximum(k, 0.5)
    rmin, rmax = 1.0, 4.5
    gd, gm = Group(k=k, chi=data), Group(k=k, chi=model)
    for g in (gd, gm):
        xftf(g, kmin=kmin, kmax=kmax, dk=dk, kweight=kw, window="hanning", rmax_out=10.0)
    sel = (gd.r >= rmin) & (gd.r <= rmax)
    ref = (((gm.chir_re - gd.chir_re) ** 2 + (gm.chir_im - gd.chir_im) ** 2)[sel].sum() / (gd.chir_re ** 2 + gd.chir_im ** 2)[sel].sum())
    cmp = SpectrumComparison("x", k, data, FitWindow(kmin=kmin, kmax=kmax, dk=dk, kweight=kw, rmin=rmin, rmax=rmax), device="cpu")
    assert cmp.r_factor(model) == pytest.approx(ref, rel=1e-5)
    assert r_factor(k, data, model, kmin=kmin, kmax=kmax, rmin=rmin, rmax=rmax, kweight=kw, dk=dk) == pytest.approx(ref, rel=1e-6)
