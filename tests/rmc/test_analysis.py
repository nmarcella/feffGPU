"""The analysis of run files (CPU only, on a run directory written here in the documented format): shell numbers with
their spread over states and runs, pairs below the closest distances, pair distributions, angles, figures, and the
rules of a continuation. The windows of every edge come from the run's own files (a comparison script of the research
code once read them from the wrong place and fell back to a default R range)."""
import json
import numpy as np
import pytest
from rmc_helpers import rock_salt
from feffgpu.rmc import outputs, analysis
from feffgpu.exafs import k_grid, r_factor


def fake_run(root, seed=0, label="rs"):
    sym, pos, cell = rock_salt(2)
    rng = np.random.default_rng(seed)
    P = 3
    pop = pos[None, None] + rng.normal(0, 0.05, (P, 1, len(pos), 3))
    d = root / label
    d.mkdir(parents=True)
    k = k_grid(24.0)
    data_c = np.sin(2 * 2.2 * k) * np.exp(-0.01 * k ** 2) / np.maximum(k, 1)
    data_zn = np.sin(2 * 2.5 * k) * np.exp(-0.01 * k ** 2) / np.maximum(k, 1)
    windows = {"C": [3.0, 12.0, 1.0, 2, 1.0, 4.0], "Zn": [3.0, 11.0, 2.0, 2, 1.2, 3.2]}           # two edges with different R windows
    model = {"C": 0.95 * data_c, "Zn": 1.05 * data_zn}
    ratios = {e: r_factor(k, data, model[e], kmin=w[0], kmax=w[1], dk=w[2], kweight=w[3], rmin=w[4], rmax=w[5])
              for e, data, w in (("C", data_c, windows["C"]), ("Zn", data_zn, windows["Zn"]))}
    np.savez_compressed(d / "spectra.npz", k=k, data_C=data_c, data_Zn=data_zn, window_C=np.array(windows["C"], float), window_Zn=np.array(windows["Zn"], float),
                        model_start_C=0.5 * data_c, model_start_Zn=0.5 * data_zn, model_refit_C=model["C"], model_refit_Zn=model["Zn"])
    misfits = np.array([0.03, 0.02, 0.025])
    np.savez_compressed(d / "state.npz", positions=pop[1], start=pos[None], population=pop, site_of=np.tile(np.arange(len(pos)), (P, 1, 1)), misfits=misfits,
                        ratios=np.tile([0.02, 0.03], (P, 1)), edges=np.array(["C", "Zn"]), cell=cell, symbols=sym, ideal=pos[None], sites=pos,
                        site_ids=np.arange(len(pos)), start_site_of=np.arange(len(pos)), best_state=1)
    report = {"method": outputs.METHOD, "edges": ["C", "Zn"], "absorbers": {"C": "C", "Zn": "Zn"}, "closest_distance": {"C-Zn": 2.0, "C-C": 2.9}, "final_refit": {"ratios": ratios},
              "scale": 1.0, "search": {}}
    (d / "report.json").write_text(json.dumps(report))
    (root / "run.json").write_text(json.dumps({"method": outputs.METHOD, "samples": [label], "shared": {"s02": {"C": 0.9}, "de0": {"C": 1.0}}}))
    return d, sym, pos, cell, pop


def test_reading_a_run(tmp_path):
    d, sym, pos, cell, pop = fake_run(tmp_path)
    assert outputs.sample_dirs(tmp_path) == [d] and outputs.sample_dirs(d) == [d]
    s = outputs.read_sample(d)
    assert s["report"]["method"] == outputs.METHOD and s["state"]["population"].shape == pop.shape and s["checkpoint"] is None
    st = analysis.load_states(d)
    assert np.allclose(st["best"], pop[1]) and st["cell"].shape == (3, 3)
    assert outputs.shared_values_of_run(tmp_path) == ({"C": 0.9}, {"C": 1.0})
    with pytest.raises(FileNotFoundError):
        outputs.sample_dirs(tmp_path / "nothing")


def test_shell_numbers_with_their_spread_over_states_and_runs(tmp_path):
    d1, sym, pos, cell, pop = fake_run(tmp_path / "seed1", seed=1)
    d2, *_ = fake_run(tmp_path / "seed2", seed=2)
    res = analysis.shell_statistics([d1, d2], [analysis.parse_shell("C-Zn:1.6:2.8")])
    row = res["shells"][0]
    assert row["N"]["best"] == pytest.approx(6.0) and row["N"]["states"][0]["n"] == 3 and row["N"]["runs"]["n"] == 2
    from feffgpu.analysis import shell_table
    per_state = [shell_table(p, sym, cell, [("C", "Zn", 1.6, 2.8)])[0]["sigma2"] for p in pop]
    assert row["sigma2"]["states"][0]["mean"] == pytest.approx(np.mean(per_state)) and row["sigma2"]["states"][0]["sd"] == pytest.approx(np.std(per_state, ddof=1))
    assert 0.003 < row["sigma2"]["best"] < 0.008 and "states" in res["method"]
    with pytest.raises(ValueError):
        analysis.parse_shell("C-Zn:1.6")


def test_short_pairs_rdf_and_angles(tmp_path):
    d, sym, pos, cell, pop = fake_run(tmp_path)
    sp = analysis.short_pair_statistics(d)
    assert set(sp["thresholds"]) == {"C-Zn", "C-C"} and sp["best"]["C-Zn"]["closest"] > 1.9
    sp = analysis.short_pair_statistics(d, {"C-Zn": 2.3})
    assert sp["population"]["C-Zn"]["per_center"] > 1                                            # most first neighbors at 2.19 A are below 2.3
    r, g, info = analysis.run_rdf(d, "C", ["Zn"], rmax=3.0)
    assert info == {"centers": 32, "configurations": 3} and g["Zn"].sum() * 0.01 == pytest.approx(6.0, abs=0.01)
    ang = analysis.run_angles(d, "C", ("Zn", "Zn"), ((1.6, 2.8), (1.6, 2.8)))
    assert ang["angles"] == 3 * 32 * 15 and abs(ang["mean"] - 108.0) < 2.0                       # octahedra: 12 right angles and 3 straight ones


def test_figures_use_the_windows_of_the_run(tmp_path):
    d, *_ = fake_run(tmp_path)
    sp = outputs.read_sample(d)["spectra"]
    comps = {c.name: c for c in analysis._comparisons(sp)}
    assert comps["Zn"].r_range == (1.2, 3.2) and comps["Zn"].dk == 2.0 and comps["C"].r_range == (1.0, 4.0)
    rep = json.loads((d / "report.json").read_text())
    for name, c in comps.items():                                                                 # the stored windows reproduce the reported R-factors
        assert c.r_factor(sp[f"model_refit_{name}"]) == pytest.approx(rep["final_refit"]["ratios"][name], rel=1e-6)
    written = analysis.plot_run(d, tmp_path / "figs")
    assert [p.name for p in written] == ["fits.png", "pairs.png"] and all(p.stat().st_size > 10000 for p in written)


def test_continuation_rules_on_files(tmp_path):
    d, sym, pos, cell, pop = fake_run(tmp_path)
    got = outputs.load_continuation(d, sym, pos[None], 2, 1)
    assert got["population"].shape == (2, 1, len(pos), 3) and list(got["order"]) == [1, 2]       # the best states first
    with pytest.raises(ValueError, match="different structure"):
        outputs.load_continuation(d, sym[::-1], pos[None], 2, 1)
    with pytest.raises(ValueError, match="start lattice"):
        outputs.load_continuation(d, sym, pos[None] + 0.01, 2, 1)
    with pytest.raises(ValueError, match="states"):
        outputs.load_continuation(d, sym, pos[None], 4, 1)
    rep = json.loads((d / "report.json").read_text())
    (d / "report.json").write_text(json.dumps({**rep, "method": "another program"}))
    with pytest.raises(ValueError, match="another method"):
        outputs.load_continuation(d, sym, pos[None], 2, 1)
