"""The gradient refinement (GpuRmc): from a block with too little disorder, gradient descent on the exact spectrum
(single scattering for every pair, multiple scattering of every absorber with a rotating-subset gradient) recovers the
pair variance of a synthetic target and drives the misfit down; the command runs it from a TOML file."""
import json
import numpy as np
import pytest
import torch
from rmc_helpers import GPU, DEVICE, PHASE_W, A_W, bcc, write_chi, write_poscar
from feffgpu.analysis import registered_pair_distances, cumulants
from feffgpu.exafs import k_grid

pytestmark = [GPU, pytest.mark.gpu]
SHELL1 = [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]


def case(ms, L=5):
    from feffgpu.rmc import GpuRmc
    from feffgpu.io import bcc_block
    C = 4
    sites, box = bcc_block(L, A_W)
    rng = np.random.default_rng(0)
    truth = sites[None] + rng.normal(0, 0.05, (C, len(sites), 3))          # pair variance 2 x 0.05^2 = 0.0050 A^2
    start = sites[None] + rng.normal(0, 0.02, (C, len(sites), 3))          # 0.0008 A^2
    k = k_grid(24.0)
    rmc = GpuRmc(str(PHASE_W), k, np.zeros(len(k)), e0=2.0, box=box, window=(3.0, 23.0), r_range=(1.0, 5.5), rmax=6.0, critpw=2.0, ms=ms, chunk=1024,
                 device=DEVICE)
    chi_t, _ = rmc.spectrum_only(truth)
    rmc.pairs.base.set_target(chi_t.cpu().numpy())
    return rmc, sites, box, truth, start


def ss2(pos, sites, box):
    return cumulants(registered_pair_distances(pos, sites, A_W, box, {1: SHELL1})[1])["sigma2"]


def test_single_scattering_refinement_recovers_the_pair_variance():
    rmc, sites, box, truth, start = case(ms=False)
    _, M0 = rmc.spectrum_only(start)
    assert M0 > 0.2
    final, hist = rmc.refine(start, steps=250, lr=0.005, optimizer="sgd", log_every=50, verbose=False)
    assert hist[-1]["misfit"] < 0.005 and hist[-1]["core"] == 0.0
    assert abs(ss2(final, sites, box) - ss2(truth, sites, box)) < 0.1 * ss2(truth, sites, box)
    final2, _ = rmc.refine(start, steps=1, lr=0.002, optimizer="sgd", lr_final=1.0, log_every=1, verbose=False)
    assert abs(np.sqrt(((final2 - start) ** 2).mean()) - 0.002) < 1e-5                       # the step of the normalized descent is what it says


@pytest.mark.slow
def test_multiple_scattering_refinement_recovers_the_pair_variance():
    rmc, sites, box, truth, start = case(ms=True, L=4)                                        # 128 atoms x 4 copies: 512 absorbers
    chi_start, M0 = rmc.spectrum_only(start)
    assert M0 > 0.2
    pos = torch.tensor(start, device=DEVICE, dtype=torch.float32).reshape(-1, 3)
    sub = torch.arange(0, rmc.ms.n_abs, 7, device=DEVICE)
    assert torch.allclose(rmc.ms.chi_absorbers(pos, sub), rmc.ms.rows[sub], atol=1e-6, rtol=1e-4)   # the rotating rows reproduce the full pass
    final, hist = rmc.refine(start, steps=100, lr=0.004, ms_absorbers=64, value_every=10, refresh_every=100, optimizer="sgd", log_every=50, verbose=False)
    chi_f, M_exact = rmc.spectrum_only(final)
    assert M_exact < 0.01, M_exact
    assert rmc.steps_info["rejected_steps"] <= 5
    assert abs(ss2(final, sites, box) - ss2(truth, sites, box)) < 0.1 * ss2(truth, sites, box)


def test_the_refine_command_runs_from_a_toml_file(tmp_path):
    from feffgpu.rmc.cli import main
    from feffgpu.exafs.edge import EdgeModel
    from feffgpu.exafs import FitWindow
    sym, pos, cell = bcc(3)
    write_poscar(tmp_path / "w.vasp", sym, pos, cell)
    truth = pos[None] + np.random.default_rng(2).normal(0, 0.05, (2, len(pos), 3))
    k = k_grid(24.0)
    e = EdgeModel("W", k, np.zeros(len(k)), FitWindow(kmin=3.0, kmax=16.0, rmin=1.0, rmax=4.5), str(PHASE_W), sym, cell, "W", rpath=4.5, nleg=2,
                  device=DEVICE)
    chi = e.spectra(truth).mean(0).cpu().numpy()
    write_chi(tmp_path / "w.chi", k, chi)
    (tmp_path / "r.toml").write_text(f"""
[run]
output = "out"
[[sample]]
label = "w"
structure = "w.vasp"
  [[sample.edge]]
  name = "W"
  absorber = "W"
  chi = "w.chi"
  phase = "{PHASE_W}"
  window = {{ kmin = 3.0, kmax = 16.0, dk = 1.0, kweight = 2, rmin = 1.0, rmax = 4.5 }}
  rpath = 4.5
  nleg = 2
[refine]
copies = 2
steps = 60
step_size = 0.004
core_distance = 2.2
log_every = 30
""")
    assert main(["refine", str(tmp_path / "r.toml"), "--json"]) == 0
    rep = json.loads((tmp_path / "out/w/refine.json").read_text())
    assert rep["misfit_final"] < 0.2 * rep["misfit_start"]
    z = np.load(tmp_path / "out/w/refine.npz")
    assert z["positions"].shape == (2, len(pos), 3) and (tmp_path / "out/config.resolved.toml").exists()
    assert rep["method"] == "feffgpu.rmc: gradient refinement"                                # one element, one R-factor spectrum: GpuRmc
    # the analysis commands read the refinement: the copies are the cells of one state
    from feffgpu.rmc.analysis import load_states, plot_run
    d = load_states(tmp_path / "out/w")
    assert d["population"].shape == (1, 2, len(pos), 3) and np.allclose(d["best"], z["positions"])
    assert main(["shells", str(tmp_path / "out"), "--shell", "W-W:2.2:3.0", "--json"]) == 0
    assert {f.name for f in plot_run(tmp_path / "out/w")} == {"fits.png", "pairs.png"}
    with np.load(tmp_path / "out/w/spectra.npz") as sp:                                       # the data as measured, the models as compared with it
        assert np.allclose(sp["data_W"], chi, atol=1e-6) and sp["model_refit_W"].shape == chi.shape
