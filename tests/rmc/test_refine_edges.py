"""The gradient refinement of structures with several elements and several spectra (``EdgeRefinement``, the route of
``rmc-exafs refine`` for everything but one element with one R-factor spectrum): the misfit falls, the displacement
limit and the closest distances hold, several spectra are lowered together, and it converges where the gradient steps
of the search (multiple scattering held) do not."""
import json
import numpy as np
import pytest
import torch
from rmc_helpers import GPU, DEVICE, PHASE_W, PHASE_ZNC, bcc, rock_salt, write_chi, write_poscar
from feffgpu.exafs import EdgeModel, FitWindow, k_grid
from feffgpu.rmc import EdgeRefinement

pytestmark = [GPU, pytest.mark.gpu]
K = k_grid(24.0)
WINDOW = FitWindow(kmin=3.0, kmax=12.0, dk=1.0, kweight=2, rmin=1.0, rmax=4.0)


def carbon_edge(sym, cell, data=None, name="C", rule="evax", window=WINDOW, objective="r_factor", nleg=3):
    return EdgeModel(name, K, np.zeros(len(K)) if data is None else data, window, str(PHASE_ZNC), sym, cell, "C", potentials={"Zn": 1, "C": 2}, rpath=4.3,
                     nleg=nleg, criteria=1.0, device=DEVICE, criteria_rule=rule, objective=objective)


def truth_and_target(seed=7, sd=0.06, **kw):
    sym, pos, cell = rock_salt(2)
    truth = pos + np.random.default_rng(seed).normal(0, sd, pos.shape)
    maker = EdgeRefinement([carbon_edge(sym, cell, **kw)], pos, sym, cell)
    chis, _ = maker.spectra(truth[None])
    return sym, pos, cell, truth, chis[0].double().cpu().numpy()


def test_two_elements_are_refined_and_the_limits_hold():
    sym, pos, cell, truth, target = truth_and_target()
    edge = carbon_edge(sym, cell, target)
    ref = EdgeRefinement([edge], pos, sym, cell, max_displacement=0.12, closest_distance={"C-Zn": 1.9}, core_distance=1.5)
    assert ref.closest.as_dict() == {"C-C": 1.5, "C-Zn": 1.9, "Zn-Zn": 1.5}          # the named pair keeps its own distance
    start = pos[None] + np.random.default_rng(1).normal(0, 0.01, (1, *pos.shape))
    _, m0 = ref.spectra(start)
    final, history = ref.refine(start, 40, step_size=0.004, ms_absorbers=16, value_every=5, log_every=10, log=None)
    _, m1 = ref.spectra(final)
    assert m1[0] < 0.1 * m0[0] and history[-1]["misfit"][0] == pytest.approx(m1[0], rel=1e-4)      # the last record is the exact misfit
    assert [h["step"] for h in history] == [0, 10, 20, 30, 40]
    d = final[0] - pos
    assert np.linalg.norm(d, axis=1).max() <= 0.12 + 1e-5                           # every atom within the limit of its site
    assert ref.closest.below(torch.as_tensor(final, device=DEVICE, dtype=torch.float32))[0] == 0
    assert ref.steps_info["rejected_steps"] == 0 and set(ref.timing) == {"paths", "full_pass", "full_gradient", "steps"}


def test_two_spectra_fall_together_and_feff_s_rule_finds_its_paths_again():
    sym, pos, cell = rock_salt(2)
    truth = pos + np.random.default_rng(3).normal(0, 0.06, pos.shape)
    short = FitWindow(kmin=3.0, kmax=10.0, dk=1.0, kweight=2, rmin=1.0, rmax=3.0)
    make = lambda data=None: [carbon_edge(sym, cell, None if data is None else data[0], name="C", rule="feff"),                  # noqa: E731
                              carbon_edge(sym, cell, None if data is None else data[1], name="C_wavelet", rule="feff", window=short, objective="wavelet")]
    chis, _ = EdgeRefinement(make(), pos, sym, cell).spectra(truth[None])
    ref = EdgeRefinement(make([c.double().cpu().numpy() for c in chis]), pos, sym, cell)
    start = pos[None] + np.random.default_rng(1).normal(0, 0.01, (1, *pos.shape))
    _, m0 = ref.spectra(start)
    final, history = ref.refine(start, 30, step_size=0.004, ms_absorbers=16, value_every=5, refresh_every=10, log_every=30, log=None)
    _, m1 = ref.spectra(final)
    assert len(m1) == 2 and m1[0] < 0.2 * m0[0] and m1[1] < 0.2 * m0[1]
    assert ref.timing["paths"] > 0 and len(history[-1]["misfit"]) == 2


def test_single_scattering_only_and_several_copies():
    sym, pos, cell, truth, target = truth_and_target(nleg=2)
    edge = carbon_edge(sym, cell, target, nleg=2)
    assert edge.ms is None
    ref = EdgeRefinement([edge], pos, sym, cell)
    start = np.repeat(pos[None], 2, 0) + np.random.default_rng(1).normal(0, 0.01, (2, *pos.shape))
    _, m0 = ref.spectra(start)
    final, _ = ref.refine(start, 30, step_size=0.004, log_every=30, log=None)
    assert final.shape == (2, len(pos), 3) and ref.spectra(final)[1][0] < 0.1 * m0[0]
    with pytest.raises(ValueError, match="atoms"):
        EdgeRefinement([edge], pos[:-1], sym[:-1], cell)


def test_the_refine_command_with_two_elements(tmp_path):
    from feffgpu.rmc.cli import main
    sym, pos, cell, truth, target = truth_and_target()
    write_poscar(tmp_path / "start.vasp", sym, pos, cell)
    write_chi(tmp_path / "c.chi", K, target)
    text = f"""
[run]
output = "out"
seed = 2
[[sample]]
label = "rs"
structure = "start.vasp"
  [[sample.edge]]
  name = "C"
  absorber = "C"
  chi = "c.chi"
  phase = "{PHASE_ZNC}"
  potentials = {{ Zn = 1, C = 2 }}
  window = {{ kmin = 3.0, kmax = 12.0, dk = 1.0, kweight = 2, rmin = 1.0, rmax = 4.0 }}
  rpath = 4.3
  nleg = 3
  criteria = 1.0
  criteria_rule = "evax"
[amplitude]
mode = "fixed"
[constraints]
max_displacement = 0.3
[refine]
copies = 1
steps = 30
step_size = 0.004
ms_absorbers = 16
value_every = 5
core_distance = 1.5
start_kick = 0.01
log_every = 10
"""
    (tmp_path / "r.toml").write_text(text)
    assert main(["refine", str(tmp_path / "r.toml"), "--json"]) == 0
    rep = json.loads((tmp_path / "out/rs/refine.json").read_text())
    assert rep["edges"] == ["C"] and rep["objective"] == {"C": "r_factor"} and rep["criteria_rule"] == {"C": "evax"}
    assert rep["misfit_final"]["C"] < 0.1 * rep["misfit_start"]["C"] and rep["max_displacement"] == 0.3
    assert any(t["kept"] and t["legs"] == 3 for t in rep["path_types"]["C"])
    z = np.load(tmp_path / "out/rs/refine.npz")
    assert z["positions"].shape == (1, len(pos), 3) and {"start", "sites", "k", "cell", "symbols", "target_C", "model_start_C", "model_final_C"} <= set(z.files)
    assert np.allclose(z["target_C"], target, atol=1e-6)
    # the analysis commands read a refinement like a search with one state
    assert {"state.npz", "spectra.npz", "report.json", "refine.json", "refine.npz"} <= {f.name for f in (tmp_path / "out/rs").iterdir()}
    assert json.loads((tmp_path / "out/run.json").read_text())["samples"] == ["rs"]
    from feffgpu.rmc.analysis import shell_statistics, plot_run, load_states
    d = load_states(tmp_path / "out/rs")
    assert d["population"].shape == (1, 1, len(pos), 3) and np.allclose(d["best"], z["positions"]) and np.allclose(d["ideal"][0], z["sites"])
    shells = shell_statistics([tmp_path / "out/rs"], [("C", "Zn", 1.6, 2.8)])["shells"][0]
    assert shells["N"]["best"] == pytest.approx(6.0) and 0.001 < shells["sigma2"]["best"] < 0.02
    assert main(["shells", str(tmp_path / "out"), "--shell", "C-Zn:1.6:2.8", "--json"]) == 0
    assert main(["short-pairs", str(tmp_path / "out"), "--json"]) == 0
    assert {f.name for f in plot_run(tmp_path / "out/rs")} == {"fits.png", "pairs.png"}
    (tmp_path / "adam.toml").write_text(text.replace("step_size = 0.004", 'step_size = 0.004\noptimizer = "adam"'))
    assert main(["refine", str(tmp_path / "adam.toml")]) == 2


def test_it_converges_where_the_gradient_steps_of_the_search_stall():
    """A metal with strong multiple scattering (bcc W): from the same start, the gradient steps of the search hold the
    multiple scattering fixed while the atoms move and stop improving; the refinement keeps lowering the misfit."""
    from feffgpu.rmc import GradientObjective, fire_relax
    sym, pos, cell = bcc(4)
    truth = pos + np.random.default_rng(5).normal(0, 0.06, pos.shape)
    window = FitWindow(kmin=3.0, kmax=16.0, dk=1.0, kweight=2, rmin=1.0, rmax=5.5)

    def tungsten(data=None):
        return EdgeModel("W", K, np.zeros(len(K)) if data is None else data, window, str(PHASE_W), sym, cell, "W", rpath=5.6, nleg=4, criteria=1.0,
                         device=DEVICE, criteria_rule="evax")

    chis, _ = EdgeRefinement([tungsten()], pos, sym, cell).spectra(truth[None])
    edge = tungsten(chis[0].double().cpu().numpy())
    ref = EdgeRefinement([edge], pos, sym, cell)
    start = pos[None] + np.random.default_rng(1).normal(0, 0.01, (1, *pos.shape))
    _, m0 = ref.spectra(start)
    final, _ = ref.refine(start, 40, step_size=0.004, ms_absorbers=32, value_every=5, log_every=40, log=None)
    refined = ref.spectra(final)[1][0]
    # the search's gradient steps: the multiple scattering of the start held, the atoms relaxed on single scattering
    P = torch.as_tensor(start, device=DEVICE, dtype=torch.float32)
    held = edge.ms_pass(P)
    objective = GradientObjective([edge], 1, scale=1.0)
    objective.weights = np.array([1.0])
    objective.ms = [held]
    g = objective.energy_forces(P)[1].abs().max()
    objective.scale = float(1.0 / g)
    relaxed, *_ = fire_relax(objective, P, 200)
    with_held_ms = float(edge.residual(edge.spectra(relaxed.view(1, -1, 3)))[0])
    assert refined < 0.05 * m0[0]
    assert with_held_ms > 3 * refined                                  # the held multiple scattering is no longer that of the moved atoms
