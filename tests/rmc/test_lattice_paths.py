"""EvAX's rule for choosing the multiple-scattering paths (``criteria_rule = "evax"``): the paths are chosen once on
the lattice, whole groups at a time, by the size of their signal relative to the lattice's whole spectrum; the groups
are formed as EvAX's path finder forms them; the signals still follow the atoms; the refinement and the search run
with it."""
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


def fcc(m, d=2.74):
    """m x m x m cubic cells of an fcc lattice with nearest-neighbor distance d (A): positions and the cell length."""
    a = d * np.sqrt(2.0)
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    cells = np.array([[i, j, l] for i in range(m) for j in range(m) for l in range(m)])
    return ((cells[:, None, :] + basis[None]).reshape(-1, 3) * a), m * a


@pytest.fixture(scope="module")
def lattice():
    sym, pos, cell = bcc(4)                                           # 128 W atoms, 12.66 A wide
    ms = MultipleScattering(str(PHASE_W), K, 0.0, cell, rmax=RMAX, critpw=CRIT, device=DEVICE, criteria_rule="evax")
    counts = ms.set_lattice(pos, kweight=2, tolerance=1e-3)           # groups of identical geometry only
    return ms, pos, cell, counts


def test_groups_are_kept_or_dropped_as_a_whole_for_every_atom(lattice):
    ms, pos, cell, counts = lattice
    n = len(pos)
    kept = [t for t in ms.types if t["kept"]]
    assert kept and all(t["importance"] > CRIT for t in kept) and all(t["importance"] <= CRIT for t in ms.types if not t["kept"])
    assert all(t["legs"] in (2, 3, 4) and len(t["potentials"]) == t["legs"] - 1 for t in ms.types)
    assert set(ms.lattice) <= {3, 4}                                      # single scattering is not in the lists: the pair tables give it
    for nleg, L in ms.lattice.items():
        per_atom = torch.bincount(L["absorber"], minlength=n)
        assert int(per_atom.min()) == int(per_atom.max()) > 0             # the same paths for every atom of the lattice
        assert set(torch.unique(L["deg"]).tolist()) <= {1.0, 2.0}         # a path counts once per kept direction of travel
        assert torch.equal(L["deg"].long(), (L["type"] >= 0).long() + (L["type_back"] >= 0).long())
        for j, t in enumerate(ms.types):                                  # the degeneracy of a group is the number of its chains per atom
            if t["kept"] and t["legs"] == nleg:
                assert (int((L["type"] == j).sum()) + int((L["type_back"] == j).sum())) / n == pytest.approx(t["degeneracy"])
    assert counts == {nleg: int(L["atoms"].shape[0]) for nleg, L in ms.lattice.items()}


def test_the_importance_is_the_size_of_the_signal_relative_to_the_whole_lattice_spectrum(lattice):
    """Against a count that does not use the groups: every path of the lattice from FEFF's path finder without a filter,
    single scattering from the pair tables."""
    ms, pos, cell, _ = lattice
    P = torch.as_tensor(pos, device=DEVICE, dtype=torch.float32)
    everything = MultipleScattering(str(PHASE_W), K, 0.0, cell, rmax=RMAX, critpw=0.0, device=DEVICE)
    everything.enumerate(P[None])
    pairs = PairTables(str(PHASE_W), K, np.zeros(len(K)), 0.0, window=(3.0, 16.0), r_range=(1.0, 5.5), box=cell, rmax=RMAX, ms=False, device=DEVICE)
    single = pairs.single_scattering(P[None]).double().cpu().numpy()
    whole = single + everything.full_pass(P).mean(0).double().cpu().numpy()
    ms.enumerate(P[None])
    signals = ms.type_signals(P).cpu().numpy()
    checked = 0
    for j, t in enumerate(ms.types):
        if t["kept"] and t["legs"] > 2:
            assert 100 * size(signals[j]) / size(whole) == pytest.approx(t["importance"], rel=2e-3)
            checked += 1
        else:
            assert not signals[j].any()                                   # a dropped group, or single scattering: no path in the lists
    assert checked >= 3
    shells = [t for t in ms.types if t["legs"] == 2]                      # the groups of two legs are the neighbor shells
    assert [round(t["degeneracy"]) for t in shells][:3] == [8, 6, 12]
    assert sum(t["importance"] ** 2 for t in shells) > 0 and all(t["kept"] for t in shells[:3])
    twice = [(t["legs"], round(t["half_length"], 3), round(t["degeneracy"], 3)) for t in ms.types if t["kept"] and t["legs"] > 2]
    assert len(twice) > len(set(twice))                                   # a path and its reverse: two groups of the same size
    assert np.allclose(signals.sum(0), ms.full_pass(P).mean(0).double().cpu().numpy(), atol=1e-9)


def test_the_importance_does_not_depend_on_how_far_the_k_grid_reaches(lattice):
    """EvAX sums the sizes over the k range of its FEFF runs, 0 to 20 1/A. A model grid that reaches further (the
    commands use 24 1/A) must give the same groups and the same importances."""
    ms, pos, cell, _ = lattice
    longer = MultipleScattering(str(PHASE_W), k_grid(24.0), 0.0, cell, rmax=RMAX, critpw=CRIT, device=DEVICE, criteria_rule="evax")
    assert longer.set_lattice(pos, kweight=2, tolerance=1e-3) == {nleg: int(L["atoms"].shape[0]) for nleg, L in ms.lattice.items()}
    assert len(longer.types) == len(ms.types)
    for a, b in zip(longer.types, ms.types):
        assert (a["legs"], a["kept"]) == (b["legs"], b["kept"]) and a["importance"] == pytest.approx(b["importance"], rel=1e-4)


def test_paths_of_nearly_the_same_geometry_form_one_group():
    """A lattice strained by one percent along z has more distinct geometries than the cubic one. With EvAX's tolerance of
    0.5 A they fall into the groups of the cubic lattice; with a tolerance of 0.001 A every geometry is its own group.
    (At 0.5 A the two nearest shells of bcc, 0.42 A apart, are one group of fourteen, as in EvAX.)"""
    sym, pos, cell = bcc(4)
    stretch = np.array([1.0, 1.0, 1.01])

    def groups(sites, box, tolerance):
        ms = MultipleScattering(str(PHASE_W), K, 0.0, box, rmax=RMAX, critpw=CRIT, device=DEVICE, criteria_rule="evax")
        ms.set_lattice(sites, kweight=2, tolerance=tolerance)
        return ms

    cubic, coarse = groups(pos, cell, 0.5), groups(pos * stretch, cell * stretch, 0.5)
    cubic_fine, fine = groups(pos, cell, 1e-3), groups(pos * stretch, cell * stretch, 1e-3)
    shells = lambda ms: [round(t["degeneracy"]) for t in ms.types if t["legs"] == 2]                                  # noqa: E731
    inside = lambda ms: sorted((t["legs"], round(t["degeneracy"])) for t in ms.types if t["kept"] and t["legs"] > 2 and t["half_length"] < RMAX - 0.05)   # noqa: E731
    assert shells(cubic) == shells(coarse) == [14, 12, 32]
    assert shells(cubic_fine) == [8, 6, 12, 24, 8] and len(shells(fine)) > len(shells(cubic_fine))
    assert inside(coarse) == inside(cubic) and inside(coarse)             # the same groups of multiple-scattering paths (away from the cutoff)
    assert inside(fine) != inside(cubic_fine)


def test_chain_coordinates_are_evax_s():
    """EvAX's coordinates of a chain: distance, then (x, y) in the plane, then (x, y, z); atoms met again carry their
    position in the chain; with the first two scatterers on a line with the absorber the frame of the third atom
    depends on how the path lies in the structure."""
    f = MultipleScattering._chain_coordinates
    a, b, c = np.array([2.0, 0.0, 0.0]), np.array([1.0, 2.0, 0.0]), np.array([1.0, 1.0, 1.5])
    turn = np.array([[0.36, 0.48, -0.8], [-0.8, 0.6, 0.0], [0.48, 0.64, 0.6]])            # a rotation of the whole path
    chains = torch.tensor(np.array([[a, b, c], [turn @ a, turn @ b, turn @ c]]), dtype=torch.float64)
    code, coord = f(chains)
    assert torch.all(code == -1)
    assert torch.allclose(coord[0], coord[1], atol=1e-9)                                  # the same path turned: the same coordinates
    assert torch.allclose(coord[0, 0], torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64))
    assert coord[0, 1, 0].item() == pytest.approx(1.0) and coord[0, 1, 1].item() == pytest.approx(2.0) and coord[0, 1, 2].item() == 0.0
    assert coord[0, 2].tolist() == pytest.approx([1.0, 1.0, 1.5])
    # an atom met again, and the absorber met again
    again = torch.tensor(np.array([[a, b, a], [a, np.zeros(3), b]]), dtype=torch.float64)
    code, coord = f(again)
    assert code.tolist() == [[-1, -1, 2], [-1, 1, -1]]
    assert coord[1, 2, :2].tolist() == pytest.approx([1.0, 2.0])                          # b is the second new atom of the second chain
    # the first two scatterers on a line with the absorber: the third atom's (y, z) are not turned into a fixed frame
    line = np.array([[a, 2 * a, c], [turn @ a, turn @ (2 * a), turn @ c]])
    code, coord = f(torch.tensor(line, dtype=torch.float64))
    assert torch.allclose(coord[:, 2, 0], torch.tensor([1.0, 1.0], dtype=torch.float64), atol=1e-9)            # along the line: the same
    assert torch.allclose(coord[:, 2, 1:].norm(dim=1), torch.tensor([1.25 ** 0.5 * 1.3416407864998738] * 2, dtype=torch.float64) * 0 + coord[0, 2, 1:].norm(), atol=1e-9)
    assert not torch.allclose(coord[0, 2, 1:], coord[1, 2, 1:], atol=1e-3)                # but in another direction around it


def test_paths_through_two_atoms_on_a_line_count_in_one_direction_only():
    """EvAX's quirk, reproduced: in an fcc lattice the four-leg paths absorber -> nearest neighbor -> the atom behind it ->
    a third atom split into small groups in that direction of travel (the frame of the third atom is not fixed), each
    below the threshold, while the opposite direction of the same paths is one group and is kept."""
    pos, cell = fcc(4)                                                # 256 atoms, 15.5 A wide
    d = 2.74
    ms = MultipleScattering(str(PHASE_W), K, 0.0, cell, rmax=6.6, critpw=0.2, device=DEVICE, criteria_rule="evax")
    ms.set_lattice(pos, kweight=2)
    half = (3 + np.sqrt(3.0)) * d / 2                                 # three nearest-neighbor legs and one to the third shell
    here = [(j, t) for j, t in enumerate(ms.types) if t["legs"] == 4 and abs(t["half_length"] - half) < 0.01]
    whole = [(j, t) for j, t in here if t["kept"]]
    assert len(whole) == 2 and all(round(t["degeneracy"]) == 48 for _, t in whole)
    pieces = [t for _, t in here if not t["kept"] and round(t["degeneracy"]) == 8]
    strongest = max((t for _, t in whole), key=lambda t: t["importance"])
    # its other direction: pieces of eight (and of four) chains with the same signal each, a sixth of the importance each
    assert any(6 * q["importance"] == pytest.approx(strongest["importance"], rel=1e-3) for q in pieces)
    L = ms.lattice[4]
    member = torch.zeros(L["deg"].shape[0], dtype=torch.bool, device=L["deg"].device)
    for j, _ in whole:
        member |= (L["type"] == j) | (L["type_back"] == j)
    assert int(member.sum()) == 2 * 48 * len(pos)
    assert bool((L["deg"][member] == 1).all())                        # these paths count once: FEFF's degeneracy would be two
    assert bool((L["deg"] == 2).any())                                # other paths count with both directions


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


def test_paths_of_more_than_four_legs_in_both_rules():
    """A rock salt with a cutoff of 6.0 A has five-leg paths (four bonds and one leg between like atoms). With nleg = 6
    both rules carry them: with no
    threshold EvAX's rule keeps every direction of every path, so its lists give the spectrum of FEFF's finder without
    a filter; with a threshold the groups of five legs are listed like the others."""
    from feffgpu.exafs.edge import EdgeModel
    sym, pos, cell = rock_salt(3)                                     # 216 atoms, 13.1 A wide
    window = FitWindow(kmin=3.0, kmax=12.0, dk=1.0, kweight=2, rmin=1.0, rmax=5.0)
    P = torch.as_tensor(pos + np.random.default_rng(4).normal(0, 0.04, pos.shape), device=DEVICE, dtype=torch.float32)
    some = None

    def model(rule, criteria, nleg):
        return EdgeModel("C", K, np.zeros(len(K)), window, str(PHASE_ZNC), sym, cell, "C", potentials={"Zn": 1, "C": 2}, rpath=6.0, nleg=nleg,
                         criteria=criteria, device=DEVICE, criteria_rule=rule)

    everything = model("evax", 0.0, 6)
    some = everything.abs_keep[:6]
    counts = everything.ms.set_lattice(torch.as_tensor(pos, dtype=torch.float32), everything.ipot, some, kweight=2)
    assert set(counts) == {3, 4, 5} and counts[5] > 0                 # five legs fit, six do not (6 x 2.19 A > 2 x 6.0 A)
    everything.ms.enumerate(P[None], everything.ipot, some)
    by_groups = everything.ms.full_pass(P).mean(0).double().cpu().numpy()
    S = torch.as_tensor(pos, device=DEVICE, dtype=torch.float32)
    finder = model("feff", 0.0, 6)
    assert finder.ms.enumerate(S[None], finder.ipot, some) == counts  # FEFF's finder without a filter on the same sites: the same paths
    by_finder = finder.ms.full_pass(P).mean(0).double().cpu().numpy() # ... evaluated on the moved atoms
    assert size(by_groups - by_finder) < 1e-5 * size(by_finder)
    four = model("feff", 0.0, 4)
    four.ms.enumerate(S[None], four.ipot, some)
    assert set(four.ms.counts) == {3, 4}
    longer = size(by_finder - four.ms.full_pass(P).mean(0).double().cpu().numpy())
    assert 1e-4 * size(by_finder) < longer < 0.2 * size(by_finder)   # the five-leg paths are there and they are small
    kept = model("evax", 0.3, 6)
    kept.ms.set_lattice(torch.as_tensor(pos, dtype=torch.float32), kept.ipot, some, kweight=2)
    five = [t for t in kept.ms.types if t["legs"] == 5]
    assert five and all(len(t["potentials"]) == 4 for t in five) and any(t["kept"] for t in five)
    assert 5 in kept.ms.lattice and bool((kept.ms.lattice[5]["type"] >= 0).any())
    with pytest.raises(ValueError, match="between 2 and 8"):
        model("feff", 1.0, 9)
