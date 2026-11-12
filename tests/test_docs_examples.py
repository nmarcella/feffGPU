"""The code of the documentation pages docs/feff/python_api.md and docs/feff/differentiability.md, run as written (with
the tungsten test set in place of the reader's own files), so that the pages cannot drift from the code."""
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "tests/data/W"
pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def setup():
    from feffgpu import read_feff_inp, read_phase_pad
    from feffgpu.io import bcc_block
    phase = read_phase_pad(W / "phase.pad")
    P = read_feff_inp(W / "feff.inp")
    sites, box = bcc_block(4, 3.1652)
    positions = sites + np.random.default_rng(0).normal(0, 0.04, sites.shape)
    return phase, P, positions, box


def test_paths_of_one_absorber_and_their_signal(setup):
    from feffgpu import GPUPathFinder, GPUPathStage
    phase, P, positions, cell = setup
    ipot = np.ones(len(positions), dtype=int)
    finder = GPUPathFinder(phase, device="cuda", rmax=6.0, nlegmax=4, critpw=2.0)
    kept, ref = finder.enumerate(positions, cell, 0, ipot)
    assert set(kept) == {2, 3, 4} and kept[2][0].shape[1:] == (2, 3)
    stage = GPUPathStage(phase, device="cuda")
    k = torch.arange(0, 20.0001, 0.05)
    S = stage.linear_matrix(k.numpy())
    chi = torch.zeros(len(k), device="cuda")
    for nleg, (coords, ipots, deg, _imp, _atoms) in kept.items():
        cchi, reff = stage.cchi(coords, ipots)
        chi += (deg[:, None] * stage.chi_fine(cchi, reff, k.numpy(), S, "feff")).sum(0)
    assert torch.isfinite(chi).all() and float(chi.abs().max()) > 1e-3


def test_spectrum_of_a_configuration_comparison_and_gradient(setup):
    from feffgpu.analysis import shell_table
    from feffgpu.exafs import (FitWindow, MultipleScattering, PairTables, SpectrumComparison, fit_shared, k_grid, r_factor_torch,
                               shared_intervals)
    phase, P, positions, cell = setup
    k = k_grid(20.0)
    pairs = PairTables(phase, k, np.zeros(len(k)), e0=0.0, box=cell, rmax=6.0, ms=False, device="cuda")
    pos = torch.tensor(positions, device="cuda", dtype=torch.float32)[None]
    chi_ss = pairs.single_scattering(pos)
    ms = MultipleScattering(phase, k, 0.0, cell, rmax=6.0, critpw=2.0, nlegmax=4, device="cuda")
    ms.enumerate(pos)
    chi_ms = ms.full_pass(pos.reshape(-1, 3)).mean(0)
    chi = (chi_ss + chi_ms).float()
    assert chi.shape == (len(k),)
    # a "measured" spectrum made from the model with a known amplitude
    window = FitWindow(kmin=3.0, kmax=14.0, dk=2.0, kweight=2, rmin=1.0, rmax=5.0)
    cmp = SpectrumComparison("W", k, 0.9 * chi.cpu().numpy(), window, fit_s02=True, fit_e0=True, device="cuda")
    groups = [[(cmp, chi, 1.0)]]
    fit_shared(groups, s02_bounds=(0.5, 1.5), de0_bounds=(-8.0, 8.0))
    intervals = shared_intervals(groups)
    assert cmp.s02 == pytest.approx(0.9, rel=2e-3) and abs(cmp.de0) < 0.05 and cmp.r_factor(chi) < 1e-5 and intervals["method"]
    # the gradient of a misfit with respect to every coordinate
    target = torch.tensor(0.9 * chi.cpu().numpy(), device="cuda")
    posg = pos.clone().requires_grad_(True)
    misfit = r_factor_torch(pairs.single_scattering(posg)[None], target, pairs.kt, FitWindow(kmin=3.0, kmax=16.0, dk=1.0, kweight=2, rmin=1.0, rmax=5.0))[0]
    misfit.backward()
    assert torch.isfinite(posg.grad).all() and float(posg.grad.abs().max()) > 0
    rows = shell_table(positions, np.array(["W"] * len(positions)), cell, [("W", "W", 2.3, 3.0)])
    assert rows[0]["N"] == pytest.approx(8.0, abs=0.2)
