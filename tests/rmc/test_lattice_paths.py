"""EvAX's rule for choosing the multiple-scattering paths (``criteria_rule = "evax"``): the paths are chosen once on
the lattice, whole types at a time, by the size of their signal relative to the lattice's whole spectrum; the signals
still follow the atoms; the refinement and the search run with it."""
import json
import numpy as np
import pytest
import torch
from rmc_helpers import GPU, DEVICE, PHASE_W, PHASE_ZNC, bcc, rock_salt, write_chi, write_poscar
from feffgpu.exafs import FitWindow, MultipleScattering, PairTables, k_grid

pytestmark = [GPU, pytest.mark.gpu]
K = k_grid(20.0)
RMAX, CRIT = 5.6, 1.0


def size(chi, kweight=2):
    chi = np.asarray(chi, float)
    return float(np.sqrt(((chi * K ** kweight) ** 2).sum()))


@pytest.fixture(scope="module")
def lattice():
    sym, pos, cell = bcc(4)                                           # 128 W atoms, 12.66 A wide
    ms = MultipleScattering(str(PHASE_W), K, 0.0, cell, rmax=RMAX, critpw=CRIT, device=DEVICE, criteria_rule="evax")
    counts = ms.set_lattice(pos, kweight=2)
    return ms, pos, cell, counts


def test_types_are_kept_or_dropped_as_a_whole_for_every_atom(lattice):
    ms, pos, cell, counts = lattice
    n = len(pos)
    kept = [t for t in ms.types if t["kept"]]
    assert kept and all(t["importance"] >= CRIT for t in kept) and all(t["importance"] < CRIT for t in ms.types if not t["kept"])
    assert all(t["directions"] in (1, 2) and t["legs"] in (3, 4) for t in ms.types)
    for nleg, L in ms.lattice.items():
        per_atom = torch.bincount(L["absorber"], minlength=n)
        assert int(per_atom.min()) == int(per_atom.max()) > 0             # the same paths for every atom of the lattice
        for j, t in enumerate(ms.types):                                  # the degeneracy of a type is that of its paths
            if t["kept"] and t["legs"] == nleg:
                assert float(L["deg"][L["type"] == j].sum()) / n == pytest.approx(t["degeneracy"])
    assert counts == {nleg: int(L["atoms"].shape[0]) for nleg, L in ms.lattice.items()}


def test_the_importance_is_the_size_of_the_signal_relative_to_the_whole_lattice_spectrum(lattice):
    """Against a count that does not use the types: every path of the lattice from FEFF's path finder without a filter,
    single scattering from the pair tables."""
    ms, pos, cell, _ = lattice
    P = torch.as_tensor(pos, device=DEVICE, dtype=torch.float32)
    everything = MultipleScattering(str(PHASE_W), K, 0.0, cell, rmax=RMAX, critpw=0.0, device=DEVICE)
    everything.enumerate(P[None])
    pairs = PairTables(str(PHASE_W), K, np.zeros(len(K)), 0.0, window=(3.0, 16.0), r_range=(1.0, 5.5), box=cell, rmax=RMAX, ms=False, device=DEVICE)
    whole = pairs.single_scattering(P[None]).double().cpu().numpy() + everything.full_pass(P).mean(0).double().cpu().numpy()
    ms.enumerate(P[None])
    signals = ms.type_signals(P).cpu().numpy()
    checked = 0
    for j, t in enumerate(ms.types):
        if t["kept"]:
            assert 100 * size(signals[j]) / t["directions"] / size(whole) == pytest.approx(t["importance"], rel=2e-3)
            checked += 1
        else:
            assert not signals[j].any()                                   # a dropped type has no path in the lists
    assert checked >= 3
    assert any(t["directions"] == 2 for t in ms.types if t["kept"])       # a path and its reverse: two types of half the degeneracy
    assert np.allclose(signals.sum(0), ms.full_pass(P).mean(0).double().cpu().numpy(), atol=1e-9)


def test_the_paths_do_not_depend_on_the_positions_and_the_signal_follows_the_atoms(lattice):
    ms, pos, cell, _ = lattice
    g = torch.Generator(device=DEVICE)
    g.manual_seed(3)
    P = torch.as_tensor(pos, device=DEVICE, dtype=torch.float32)
    moved = P[None] + 0.08 * torch.randn(2, len(pos), 3, device=DEVICE, generator=g)
    counts = ms.enumerate(moved)                                      # two copies: the lattice list twice, with shifted indices
    assert counts == {nleg: 2 * int(L["atoms"].shape[0]) for nleg, L in ms.lattice.items()}
    for nleg, L in ms.lattice.items():
        m = L["atoms"].shape[0]
        assert torch.equal(ms.lists[nleg]["atoms"][:m], L["atoms"]) and torch.equal(ms.lists[nleg]["atoms"][m:], L["atoms"] + len(pos))
        assert torch.equal(ms.lists[nleg]["absorber"][m:], L["absorber"] + len(pos))
    spectrum_moved = ms.full_pass(moved.reshape(-1, 3)).mean(0)
    ms.enumerate(P[None])
    spectrum_ideal = ms.full_pass(P).mean(0)
    assert size((spectrum_moved - spectrum_ideal).cpu().numpy()) > 0.1 * size(spectrum_ideal.cpu().numpy())
    # FEFF's rule on the same moved atoms chooses other paths: every path alone, relative to the shortest single-scattering path
    feff = MultipleScattering(str(PHASE_W), K, 0.0, cell, rmax=RMAX, critpw=CRIT, device=DEVICE)
    assert feff.enumerate(moved) != counts
    with pytest.raises(RuntimeError, match="set_lattice"):
        MultipleScattering(str(PHASE_W), K, 0.0, cell, rmax=RMAX, critpw=CRIT, device=DEVICE, criteria_rule="evax").enumerate(moved)
    with pytest.raises(ValueError, match="criteria_rule"):
        MultipleScattering(str(PHASE_W), K, 0.0, cell, rmax=RMAX, critpw=CRIT, device=DEVICE, criteria_rule="other")


def test_the_refinement_runs_with_the_paths_of_the_lattice(tmp_path):
    from feffgpu.rmc.cli import main
    from feffgpu.rmc import GpuRmc
    sym, pos, cell = bcc(4)
    write_poscar(tmp_path / "w.vasp", sym, pos, cell)
    truth = pos[None] + np.random.default_rng(2).normal(0, 0.05, (1, len(pos), 3))
    maker = GpuRmc(str(PHASE_W), K, np.zeros(len(K)), 0.0, cell, window=(3.0, 16.0), r_range=(1.0, 5.0), rmax=RMAX, critpw=CRIT, device=DEVICE,
                   criteria_rule="evax", sites=pos)
    chi, _ = maker.spectrum_only(truth)
    write_chi(tmp_path / "w.chi", K, chi.cpu().numpy())
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
  window = {{ kmin = 3.0, kmax = 16.0, dk = 1.0, kweight = 2, rmin = 1.0, rmax = 5.0 }}
  rpath = {RMAX}
  criteria = {CRIT}
  criteria_rule = "evax"
[refine]
copies = 1
steps = 12
step_size = 0.004
core_distance = 2.2
start_kick = 0.02
ms_absorbers = 32
log_every = 6
""")
    assert main(["refine", str(tmp_path / "r.toml"), "--json"]) == 0
    rep = json.loads((tmp_path / "out/w/refine.json").read_text())
    assert rep["criteria_rule"] == "evax" and any(t["kept"] for t in rep["path_types"])
    assert rep["misfit_final"] < 0.5 * rep["misfit_start"]


def test_the_search_runs_with_the_paths_of_the_lattice_and_refuses_rearrangements(tmp_path):
    from feffgpu.exafs.edge import EdgeModel
    from feffgpu.rmc import run_search, StatePaths
    from feffgpu.rmc import config as C
    sym, pos, cell = rock_salt(2)
    write_poscar(tmp_path / "start.vasp", sym, pos, cell)
    window = FitWindow(kmin=3.0, kmax=12.0, dk=1.0, kweight=2, rmin=1.0, rmax=4.0)
    e = EdgeModel("C", K, np.zeros(len(K)), window, str(PHASE_ZNC), sym, cell, "C", potentials={"Zn": 1, "C": 2}, rpath=4.3, nleg=3, criteria=1.0,
                  device=DEVICE, criteria_rule="evax")
    e.ms.set_lattice(torch.as_tensor(pos, dtype=torch.float32)[e.keep.cpu()], e.ipot[e.keep], e.abs_keep, kweight=2)
    assert {t["potentials"][0] for t in e.ms.types} <= {1, 2}                # the potentials of the scatterers are part of a type
    sp = StatePaths(e)
    truth = torch.as_tensor(pos + np.random.default_rng(7).normal(0, 0.05, pos.shape), device=DEVICE, dtype=torch.float32)
    L1, L2 = sp.find(truth), sp.find(torch.as_tensor(pos, device=DEVICE, dtype=torch.float32))
    assert all(torch.equal(L1[n]["atoms"], L2[n]["atoms"]) for n in L1) and L1                       # the same list for every state
    some = e.abs_keep[:5]
    part = sp.find(truth, some)
    assert all(set(part[n]["absorber"].tolist()) <= set(some.tolist()) for n in part)
    chi = (e.spectrum(e.static_histograms(truth[None]))[0] + sp.signal([L1], truth[None])[0]).cpu().numpy()
    write_chi(tmp_path / "target.chi", K, chi)
    text = f"""
[run]
output = "out"
seed = 3
[[sample]]
label = "rs"
structure = "start.vasp"
  [[sample.edge]]
  name = "C"
  absorber = "C"
  chi = "target.chi"
  phase = "{PHASE_ZNC}"
  potentials = {{ Zn = 1, C = 2 }}
  window = {{ kmin = 3.0, kmax = 12.0, dk = 1.0, kweight = 2, rmin = 1.0, rmax = 4.0 }}
  rpath = 4.3
  nleg = 3
  criteria = 1.0
  criteria_rule = "evax"
[amplitude]
mode = "fixed"
[search]
states = 4
generations = 5
gradient_steps = 10
froze_in = 5
paths_every = 2
[constraints]
closest_distance = {{ "C-Zn" = 1.8 }}
closest_tolerance = 0.05
"""
    (tmp_path / "c.toml").write_text(text)
    run_search(tmp_path / "c.toml", log=None)
    best = [g["best"] for g in json.loads((tmp_path / "out/rs/history.json").read_text())["generations"]]
    assert best[-1] < 0.2 * best[0]
    (tmp_path / "moves.toml").write_text(text + '\n[moves]\nexchange = ["Zn", "C"]\n')
    problems = C.validate_file(tmp_path / "moves.toml")
    assert any("criteria_rule" in p and "exchanges or hops" in p for p in problems)
