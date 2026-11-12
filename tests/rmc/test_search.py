"""The search end to end on the GPU: the simplest case from a TOML file, the recovery of a planted disorder, the rules
for continuing a run and for taking S0^2 and the E0 shift from an earlier run, the configured bounds, rearrangements on
a general structure, EvAX's limits without gradient steps, two edges, and the analysis commands on a finished run."""
import json
import numpy as np
import pytest
from rmc_helpers import GPU, PHASE_W, PHASE_ZNC, bcc, rock_salt, synthetic_edge_data, write_chi, write_poscar
from feffgpu.exafs import FitWindow
from feffgpu.analysis import shell_table

pytestmark = [GPU, pytest.mark.gpu]
WIN_ZNC = FitWindow(kmin=3.0, kmax=12.0, dk=1.0, kweight=2, rmin=1.0, rmax=4.0)
WIN_TOML = "{ kmin = 3.0, kmax = 12.0, dk = 1.0, kweight = 2, rmin = 1.0, rmax = 4.0 }"


def rock_salt_inputs(d, seed=7, sd=0.05, s02=0.9, de0=1.5):
    """The Zn-C rock salt (2 x 2 x 2 cells, 64 atoms): start.vasp and a synthetic C-edge spectrum of a disordered copy."""
    sym, pos, cell = rock_salt(2)
    write_poscar(d / "start.vasp", sym, pos, cell)
    truth = pos + np.random.default_rng(seed).normal(0, sd, pos.shape)
    k, chi = synthetic_edge_data(dict(name="C", window=WIN_ZNC, phase=str(PHASE_ZNC), symbols=sym, cell=cell, absorber="C", potentials={"Zn": 1, "C": 2},
                                      rpath=4.3, nleg=3, criteria=2.5), truth[None], s02=s02, de0=de0)
    write_chi(d / "target.chi", k, chi)
    return sym, pos, cell, truth


def one_edge_toml(d, output="out", generations=12, extra="", amplitude='refit_from = 4\nrefit_every = 4', search_extra="", name="config.toml"):
    text = f"""
[run]
output = "{output}"
seed = 3
save_every = 5

[[sample]]
label = "rs"
structure = "start.vasp"

  [[sample.edge]]
  name = "C"
  absorber = "C"
  chi = "target.chi"
  phase = "{PHASE_ZNC}"
  potentials = {{ Zn = 1, C = 2 }}
  window = {WIN_TOML}
  rpath = 4.3
  nleg = 3

[amplitude]
{amplitude}

[search]
states = 8
generations = {generations}
gradient_steps = 10
froze_in = {generations}
paths_every = 5
{search_extra}

[constraints]
closest_distance = {{ "C-Zn" = 1.8, "C-C" = 2.6, "Zn-Zn" = 2.6 }}
closest_tolerance = 0.05
{extra}
"""
    f = d / name
    f.write_text(text)
    return f


@pytest.fixture(scope="module")
def one_edge_run(tmp_path_factory):
    """The simplest case run once through the command: one sample, one edge, no [moves] section."""
    from feffgpu.rmc.cli import main
    d = tmp_path_factory.mktemp("one_edge")
    rock_salt_inputs(d)
    cfg = one_edge_toml(d)
    assert main(["validate", str(cfg)]) == 0
    assert main(["run", str(cfg)]) == 0
    return d


def test_one_sample_one_edge_from_a_toml_file(one_edge_run):
    from feffgpu.rmc import load_config, validate_file
    out = one_edge_run / "out"
    run = json.loads((out / "run.json").read_text())
    assert run["samples"] == ["rs"] and set(run["shared"]["s02"]) == {"C"} and run["amplitude_mode"] == "fit"
    iv = run["intervals"]
    assert iv["s02_lo"]["C"] < run["shared"]["s02"]["C"] < iv["s02_hi"]["C"] and "standard deviations" in iv["method"]
    rep = json.loads((out / "rs/report.json").read_text())
    hist = json.loads((out / "rs/history.json").read_text())
    assert len(hist["generations"]) == 12 and all(g["weights"] == [1.0] for g in hist["generations"])       # one spectrum: weight 1
    assert rep["final_refit"]["misfit"] < 0.1 * rep["start"]["misfit"]
    assert len(hist["refits"]) == 3 and rep["search"]["generations_done"] == 12
    assert rep["arrangement"]["final_best"]["atoms_off_start_site"] == 0                                   # no exchanges, no hops
    z = np.load(out / "rs/state.npz")
    assert z["population"].shape == (8, 1, 64, 3) and z["site_of"].shape == (8, 1, 64) and list(z["edges"]) == ["C"]
    sp = np.load(out / "rs/spectra.npz")
    for key in ("k", "data_C", "window_C", "model_start_C", "model_final_C", "model_population_C", "model_refit_C", "calc_refit_C"):
        assert key in sp.files, key
    assert sp["model_population_C"].shape == (8, len(sp["k"]))
    # the stored windows reproduce the reported R-factor from the stored spectra
    from feffgpu.exafs import r_factor
    kmin, kmax, dk, kw, rmin, rmax = sp["window_C"]
    R = r_factor(sp["k"], sp["data_C"], sp["model_refit_C"], kmin=kmin, kmax=kmax, rmin=rmin, rmax=rmax, kweight=int(kw), dk=dk)
    assert R == pytest.approx(rep["final_refit"]["ratios"]["C"], rel=1e-3)
    assert (out / "rs/checkpoint.npz").exists()
    resolved = out / "config.resolved.toml"
    assert validate_file(resolved) == [] and load_config(resolved).search.generations == 12


def test_a_planted_disorder_is_recovered(tmp_path):
    """bcc W with random displacements of 0.05 A per coordinate: the search, started from the ideal lattice, finds the
    pair distances of the first two shells (the numbers of neighbors are inputs of the lattice)."""
    from feffgpu.rmc import run_search
    from feffgpu.rmc.analysis import load_states
    d = tmp_path
    sym, pos, cell = bcc(4)
    write_poscar(d / "start.vasp", sym, pos, cell)
    truth = pos + np.random.default_rng(0).normal(0, 0.05, pos.shape)
    win = FitWindow(kmin=3.0, kmax=18.0, dk=1.0, kweight=2, rmin=1.0, rmax=5.5)
    k, chi = synthetic_edge_data(dict(name="W", window=win, phase=str(PHASE_W), symbols=sym, cell=cell, absorber="W", rpath=6.0, nleg=2,
                                      multiple_scattering=False), truth[None])
    write_chi(d / "target.chi", k, chi)
    shells = [("W", "W", 2.4, 3.0), ("W", "W", 3.0, 3.4)]
    true = shell_table(truth, sym, cell, shells)
    for steps, gens, acc in ((20, 30, 0.8), (0, 300, 0.0)):                                   # with gradient steps; EvAX's greedy rules alone
        (d / "c.toml").write_text(f"""
[run]
output = "out_{steps}"
[[sample]]
label = "w"
structure = "start.vasp"
  [[sample.edge]]
  name = "W"
  absorber = "W"
  chi = "target.chi"
  phase = "{PHASE_W}"
  window = {{ kmin = 3.0, kmax = 18.0, dk = 1.0, kweight = 2, rmin = 1.0, rmax = 5.5 }}
  rpath = 6.0
  nleg = 2
[amplitude]
mode = "fixed"
[search]
states = 8
generations = {gens}
gradient_steps = {steps}
acceptance = {acc}
froze_in = {gens}
start_kick = {0.01 if steps else 0.02}
""")
        run_search(d / "c.toml", log=None)
        st = load_states(d / f"out_{steps}" / "w")
        rep = json.loads((d / f"out_{steps}/w/report.json").read_text())
        assert rep["final_refit"]["misfit"] < 1e-4 * rep["start"]["misfit"]
        found = shell_table(st["best"], sym, cell, shells)
        for t, f in zip(true, found):
            assert abs(f["R"] - t["R"]) < 0.003, (steps, t, f)
            assert abs(f["sigma2"] - t["sigma2"]) < 0.15 * t["sigma2"], (steps, t, f)


def test_continuation_keeps_the_gradient_scale_and_refuses_other_runs(one_edge_run, tmp_path):
    from feffgpu.rmc import run_search
    d = one_edge_run
    first = json.loads((d / "out/rs/report.json").read_text())
    cfg = one_edge_toml(d, output=str(tmp_path / "cont"), generations=3, search_extra=f'continue_from = "{d / "out"}"', name="cont.toml")
    run_search(cfg, log=None)
    rep = json.loads((tmp_path / "cont/rs/report.json").read_text())
    assert rep["scale"] == first["scale"] and rep["scale_source"].startswith("the first run")
    assert rep["search"]["generations_done"] == 15 and rep["from_population"]["generations_done"] == 12
    hist = json.loads((tmp_path / "cont/rs/history.json").read_text())["generations"]
    assert hist[0]["generation_total"] == 13
    # another structure, another start lattice and another method are refused
    sym, pos, cell = rock_salt(2)
    alt = d / "alt"
    alt.mkdir()
    write_poscar(alt / "start.vasp", sym, pos + 0.05, cell)
    (alt / "target.chi").write_text((d / "target.chi").read_text())
    with pytest.raises(ValueError, match="start lattice"):
        run_search(one_edge_toml(alt, output=str(tmp_path / "x1"), generations=1, search_extra=f'continue_from = "{d / "out"}"'), log=None)
    write_poscar(alt / "start.vasp", np.where(sym == "Zn", "C", "Zn"), pos, cell)
    with pytest.raises(ValueError, match="different structure"):
        run_search(one_edge_toml(alt, output=str(tmp_path / "x2"), generations=1, search_extra=f'continue_from = "{d / "out"}"'), log=None)
    other = tmp_path / "other"
    (other / "rs").mkdir(parents=True)
    (other / "run.json").write_text((d / "out/run.json").read_text())
    (other / "rs/state.npz").write_bytes((d / "out/rs/state.npz").read_bytes())
    (other / "rs/report.json").write_text(json.dumps({**first, "method": "another program"}))
    with pytest.raises(ValueError, match="another method"):
        run_search(one_edge_toml(d, output=str(tmp_path / "x3"), generations=1, search_extra=f'continue_from = "{other}"', name="x3.toml"), log=None)


def test_shared_values_from_an_earlier_run_are_held(one_edge_run, tmp_path):
    from feffgpu.rmc import run_search
    d = one_edge_run
    earlier = json.loads((d / "out/run.json").read_text())["shared"]
    cfg = one_edge_toml(d, output=str(tmp_path / "held"), generations=3, amplitude=f'mode = "from_run"\nfrom_run = "{d / "out"}"', name="held.toml")
    run_search(cfg, log=None)
    run = json.loads((tmp_path / "held/run.json").read_text())
    assert run["shared"]["s02"]["C"] == pytest.approx(earlier["s02"]["C"], rel=1e-6) and run["shared"]["de0"]["C"] == pytest.approx(earlier["de0"]["C"], abs=1e-5)
    assert run["intervals"] is None and json.loads((tmp_path / "held/rs/history.json").read_text())["refits"] == []
    cfg = one_edge_toml(d, output=str(tmp_path / "fixed"), generations=2, amplitude='mode = "fixed"\ns02 = 0.85\nde0 = 2.0', name="fixed.toml")
    run_search(cfg, log=None)
    run = json.loads((tmp_path / "fixed/run.json").read_text())
    assert run["shared"]["s02"]["C"] == pytest.approx(0.85) and run["shared"]["de0"]["C"] == pytest.approx(2.0)
    cfg = one_edge_toml(d, output=str(tmp_path / "fixed_cont"), generations=2, amplitude='mode = "fixed"\ns02 = 0.85\nde0 = 2.0',
                        search_extra=f'continue_from = "{d / "out"}"', name="fixed_cont.toml")
    run_search(cfg, log=None)                                                                     # a continued run holds the declared values, not the earlier run's
    run = json.loads((tmp_path / "fixed_cont/run.json").read_text())
    assert run["shared"]["s02"]["C"] == pytest.approx(0.85) and run["shared"]["de0"]["C"] == pytest.approx(2.0)


def test_the_configured_bounds_reach_every_refit(one_edge_run, tmp_path):
    """A range of the E0 shift that excludes the planted value (+1.5 eV) stops the fit on the range's edge, in the refits of
    the search and in the final one, and the bound is reported; S0^2 stays inside its range."""
    from feffgpu.rmc import run_search
    cfg = one_edge_toml(one_edge_run, output=str(tmp_path / "bounds"), generations=4, name="bounds.toml",
                        amplitude="s02_bounds = [0.95, 1.5]\nde0_bounds = [-0.5, 0.5]\nrefit_from = 2\nrefit_every = 2")
    run_search(cfg, log=None)
    run = json.loads((tmp_path / "bounds/run.json").read_text())
    hist = json.loads((tmp_path / "bounds/rs/history.json").read_text())
    assert len(hist["refits"]) == 2
    assert all(0.95 - 1e-6 <= r["s02"]["C"] <= 1.5 + 1e-6 for r in hist["refits"]) and all(abs(r["de0"]["C"]) <= 0.5 + 1e-6 for r in hist["refits"])
    assert run["shared"]["de0"]["C"] == pytest.approx(0.5, abs=1e-5) and "C" in run["at_bound"]


def test_exchanges_and_hops_on_a_general_structure(tmp_path):
    """A rock salt whose cations are two species (sharing one scattering potential) and whose C sublattice has vacancies
    given by a file of sites: exchanges and hops run, every site holds at most one atom, the limit holds."""
    from feffgpu.rmc import run_search
    sym, pos, cell = rock_salt(2)
    np.savetxt(tmp_path / "c_sites.txt", pos[sym == "C"])
    rng = np.random.default_rng(1)
    zn = np.nonzero(sym == "Zn")[0]
    sym = sym.copy()
    sym[rng.choice(zn, len(zn) // 2, replace=False)] = "Cu"
    keep = np.ones(len(sym), bool)
    keep[rng.choice(np.nonzero(sym == "C")[0], 4, replace=False)] = False
    sym, pos = sym[keep], pos[keep]
    write_poscar(tmp_path / "start.vasp", sym, pos, cell)
    truth = pos + rng.normal(0, 0.04, pos.shape)
    k, chi = synthetic_edge_data(dict(name="C", window=WIN_ZNC, phase=str(PHASE_ZNC), symbols=sym, cell=cell, absorber="C",
                                      potentials={"Zn": 1, "Cu": 1, "C": 2}, rpath=4.3, nleg=3), truth[None])
    write_chi(tmp_path / "target.chi", k, chi)
    cfg = one_edge_toml(tmp_path, generations=6, extra="max_displacement = 0.3").read_text()
    cfg = cfg.replace("potentials = { Zn = 1, C = 2 }", "potentials = { Zn = 1, Cu = 1, C = 2 }").replace("C-Zn", "C-Zn\" = 1.8, \"C-Cu")
    cfg += '\n[moves]\nexchange = ["Zn", "Cu"]\nexchange_probability = 1.0\nfrom_generation = 2\nhops = { element = "C", sites = "c_sites.txt", probability = 1.0 }\n'
    (tmp_path / "config.toml").write_text(cfg)
    run_search(tmp_path / "config.toml", log=None)
    hist = json.loads((tmp_path / "out/rs/history.json").read_text())["generations"]
    rep = json.loads((tmp_path / "out/rs/report.json").read_text())
    assert sum(g["exchanges"] for g in hist) >= 5 * 8 and sum(g["hops"] for g in hist) >= 5 * 4
    z = np.load(tmp_path / "out/rs/state.npz")
    for so in z["site_of"].reshape(-1, len(sym)):
        assert len(set(so.tolist())) == len(so)                                                # one atom per site
    from feffgpu.rmc.constraints import nearest_sites
    import torch
    disp = z["population"].reshape(-1, len(sym), 3) - nearest_sites(torch.tensor(z["sites"], dtype=torch.float32), float(cell[0, 0]),
                                                                    torch.tensor(z["population"].reshape(-1, len(sym), 3), dtype=torch.float32),
                                                                    torch.tensor(z["site_of"].reshape(-1, len(sym)))).numpy()
    assert np.linalg.norm(disp, axis=-1).max() <= 0.3 + 1e-4
    arr = rep["arrangement"]["final_best"]
    assert arr["vacancies"] == 4 and "warren_cowley_Zn" in arr and rep["arrangement"]["population_spread"]


def test_evax_rules_without_gradient_steps_keep_the_limits(one_edge_run, tmp_path):
    """No gradient steps: the box acts by EvAX's halving, and a state with a pair below a closest distance is refused."""
    from feffgpu.rmc import run_search
    cfg = one_edge_toml(one_edge_run, output=str(tmp_path / "evax"), generations=8, name="evax.toml",
                        search_extra="step = 0.05").read_text().replace("gradient_steps = 10", "gradient_steps = 0")
    cfg = cfg.replace("closest_tolerance = 0.05", "closest_tolerance = 0.0\nmax_displacement = 0.12")
    (one_edge_run / "evax.toml").write_text(cfg)
    run_search(one_edge_run / "evax.toml", log=None)
    rep = json.loads((tmp_path / "evax/rs/report.json").read_text())
    assert rep["settings"]["box_rule"] == "halve"
    z = np.load(tmp_path / "evax/rs/state.npz")
    from feffgpu.analysis import short_pairs
    sp = short_pairs(z["population"].reshape(-1, 64, 3), z["symbols"].astype(str), float(z["cell"][0, 0]), {"C-Zn": 1.8, "C-C": 2.6, "Zn-Zn": 2.6})
    assert all(v["per_center"] == 0 for v in sp.values())
    d = z["population"].reshape(-1, 64, 3) - z["ideal"][0][None]
    assert np.linalg.norm(d - float(z["cell"][0, 0]) * np.round(d / float(z["cell"][0, 0])), axis=-1).max() <= 0.12 + 1e-4


def test_two_edges_weights_adapt_and_the_fit_improves(tmp_path):
    """Two spectra of one sample (the C edge, and a second edge with the W phase file as a stand-in): EvAX's weights adapt and
    sum to one; the misfit of both improves."""
    from feffgpu.rmc import run_search
    sym, pos, cell, truth = rock_salt_inputs(tmp_path)
    k, chi = synthetic_edge_data(dict(name="Zn", window=WIN_ZNC, phase=str(PHASE_W), symbols=sym, cell=cell, absorber="Zn", rpath=4.3, nleg=3),
                                 truth[None], s02=0.8, de0=-1.0)
    write_chi(tmp_path / "target_zn.chi", k, chi)
    cfg = one_edge_toml(tmp_path, generations=8).read_text()
    second = f"""
  [[sample.edge]]
  name = "Zn"
  absorber = "Zn"
  chi = "target_zn.chi"
  phase = "{PHASE_W}"
  window = {WIN_TOML}
  rpath = 4.3
  nleg = 3
"""
    cfg = cfg.replace("\n[amplitude]", second + "\n[amplitude]")
    (tmp_path / "config.toml").write_text(cfg)
    run_search(tmp_path / "config.toml", log=None)
    hist = json.loads((tmp_path / "out/rs/history.json").read_text())["generations"]
    rep = json.loads((tmp_path / "out/rs/report.json").read_text())
    w = hist[-1]["weights"]
    assert len(w) == 2 and abs(sum(w) - 1) < 1e-9 and abs(w[0] - 0.5) > 1e-3
    for e in ("C", "Zn"):
        assert rep["final_refit"]["ratios"][e] < 0.5 * rep["start"]["ratios"][e]


def test_analysis_commands_on_a_finished_run(one_edge_run, tmp_path, capsys):
    from feffgpu.rmc.cli import main
    out = one_edge_run / "out"
    assert main(["shells", str(out), str(out), "--shell", "C-Zn:1.6:2.8", "--shell", "C-C:2.6:3.6", "--json"]) == 0
    res = json.loads(capsys.readouterr().out)
    row = res["shells"][0]
    assert row["N"]["best"] == pytest.approx(6.0, abs=0.2) and row["N"]["states"][0]["n"] == 8 and row["N"]["runs"]["n"] == 2
    assert 2.0 < row["R"]["best"] < 2.4 and row["sigma2"]["states"][0]["sd"] >= 0
    assert main(["short-pairs", str(out), "--json"]) == 0
    sp = json.loads(capsys.readouterr().out)["rs"]
    assert set(sp["thresholds"]) == {"C-Zn", "C-C", "Zn-Zn"}
    assert main(["rdf", str(out), "--center", "C", "--json"]) == 0
    g = json.loads(capsys.readouterr().out)["rs"]
    r, gz = np.array(g["r"]), np.array(g["g"]["Zn"])
    assert gz[(r > 1.6) & (r < 2.8)].sum() * 0.01 == pytest.approx(6.0, abs=0.2)             # six Zn around a C, an input of the lattice
    assert main(["angles", str(out), "--center", "C", "--neighbors", "Zn,Zn", "--ranges", "1.6:2.8,1.6:2.8", "--json"]) == 0
    ang = json.loads(capsys.readouterr().out)["rs"]
    assert ang["angles"] == 8 * 32 * 15 and 80 < ang["mean"] < 125
    assert main(["plot", str(out), "--out", str(tmp_path / "figs"), "--json"]) == 0
    figs = json.loads(capsys.readouterr().out)
    assert len(figs) == 2 and all((tmp_path / "figs/rs" / f).exists() for f in ("fits.png", "pairs.png"))
    assert main(["signal-map", str(out), "--components", "Zn:1.5,2.8,4.3", "--out", str(tmp_path / "map"), "--json"]) == 0
    sm = json.loads(capsys.readouterr().out)["rs"]["C"]
    first = sm["components"]["Zn 1.5-2.8 A"]
    assert first["N"] == pytest.approx(6.0, abs=0.2) and "multiple scattering" in sm["shares"]
    band = "1-2"
    total = sum(v[band] for v in sm["shares"].values())
    assert total == pytest.approx(1.0, abs=1e-3)                                               # the shares of a band add up to one
    assert (tmp_path / "map" / "rs_C.png").exists()
