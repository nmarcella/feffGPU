"""Comparison with a measured spectrum: planted S0^2 and E0 shift recovered from a synthetic "measured" spectrum made
by the forward model, with intervals; S0^2 and the shift shared between two runs of the same edge."""
import json
import numpy as np
import pytest
from scipy.interpolate import CubicSpline
from mdexafs_testing import gpu, rock_salt_run

WINDOW = {"kmin": 3.0, "kmax": 13.0, "dk": 1.0, "kweight": 2, "rmin": 1.0, "rmax": 5.0}


def _measured(path, k, chi, s02, de0, noise, seed):
    """S0^2 chi(sqrt(k^2 - ETOK dE0)) on the model's own k points, plus Gaussian noise, written as a two-column file."""
    from feffgpu.exafs import ETOK
    q = k ** 2 - ETOK * de0
    kq = np.sqrt(np.clip(q, 0, None))
    data = np.where(q > 0, s02 * CubicSpline(k, chi)(kq), 0.0) + noise * np.random.default_rng(seed).normal(size=len(k))
    np.savetxt(path, np.column_stack([k, data]), header="synthetic measured spectrum\nk chi")


@gpu
def test_planted_s02_and_shift_are_recovered(tmp_path):
    from feffgpu.md import average_cache, load_config, run
    from feffgpu.md.compare import compare_runs
    cfg_path = rock_salt_run(tmp_path, n_frames=4, output="run", absorbers=16, compare={"chi": "measured.chi", "window": WINDOW})
    cfg = load_config(cfg_path)
    run(cfg, log=lambda m: None)
    avg = average_cache(tmp_path / "run/chi.h5")["C"]
    s02, de0 = 0.82, 2.5
    _measured(tmp_path / "measured.chi", avg["k"], avg["mean"], s02, de0, 3e-4, seed=1)
    rep = compare_runs([(cfg, cfg.output_path)], tmp_path / "run/compare", log=lambda m: None)
    sh = rep["shared"]
    assert sh["s02"]["C"] == pytest.approx(s02, rel=0.02) and sh["de0"]["C"] == pytest.approx(de0, abs=0.2)
    lo, hi = sh["s02_interval"]["C"]
    elo, ehi = sh["de0_interval"]["C"]
    z = sh["z"]                                                                       # the intervals are +- z standard deviations (90 %)
    sd_s02, sd_de0 = (hi - lo) / (2 * z), (ehi - elo) / (2 * z)
    assert abs(sh["s02"]["C"] - s02) < 3 * sd_s02 and abs(sh["de0"]["C"] - de0) < 3 * sd_de0 and sh["positive_definite"]
    assert "standard deviations" in sh["interval_method"] and not sh["at_bound"]
    assert sh["sampling"]["blocks"] >= 2 and sh["sampling"]["s02_se"]["C"] > 0 and "jackknife" in sh["sampling"]["method"]
    spec = rep["runs"][0]["spectra"][0]
    assert spec["r_factor"] < 1e-2 and spec["s02_fitted"] and spec["de0_fitted"]
    assert (tmp_path / "run/compare/compare.png").stat().st_size > 10000
    assert json.loads((tmp_path / "run/compare/compare.json").read_text())["shared"]["s02"]["C"] == pytest.approx(sh["s02"]["C"])


@gpu
def test_two_runs_share_one_s02_and_shift(tmp_path):
    from feffgpu.md import average_cache, load_config, run
    from feffgpu.md.compare import compare_runs
    runs = []
    for i, seed in enumerate((0, 5)):
        sub = tmp_path / f"r{i}"
        sub.mkdir()
        p = rock_salt_run(sub, n_frames=3, output="out", absorbers=16, seed=seed, sd=0.05 + 0.02 * i,
                          compare={"chi": "measured.chi", "window": WINDOW})
        cfg = load_config(p)
        run(cfg, log=lambda m: None)
        avg = average_cache(sub / "out/chi.h5")["C"]
        _measured(sub / "measured.chi", avg["k"], avg["mean"], 0.9, -1.5, 3e-4, seed=i)
        runs.append((cfg, cfg.output_path))
    rep = compare_runs(runs, tmp_path / "joint", log=lambda m: None)
    assert rep["shared"]["fitted_s02"] == ["C"] and rep["shared"]["s02"]["C"] == pytest.approx(0.9, rel=0.02)
    assert rep["shared"]["de0"]["C"] == pytest.approx(-1.5, abs=0.2)
    lo, hi = rep["shared"]["s02_interval"]["C"]
    assert abs(rep["shared"]["s02"]["C"] - 0.9) < 3 * (hi - lo) / (2 * rep["shared"]["z"])
    assert [r["spectra"][0]["s02"] for r in rep["runs"]] == [rep["shared"]["s02"]["C"]] * 2          # one shared value
    assert len(rep["runs"]) == 2 and all((tmp_path / "joint" / r["figure"]).is_file() for r in rep["runs"])


@gpu
def test_a_window_outside_the_data_is_refused(tmp_path):
    from feffgpu.md import average_cache, load_config, run
    from feffgpu.md.compare import compare_runs
    p = rock_salt_run(tmp_path, n_frames=2, output="run", absorbers=8, compare={"chi": "measured.chi", "window": WINDOW})
    cfg = load_config(p)
    run(cfg, log=lambda m: None)
    avg = average_cache(tmp_path / "run/chi.h5")["C"]
    k = avg["k"]
    short = k <= 10.0
    np.savetxt(tmp_path / "measured.chi", np.column_stack([k[short], avg["mean"][short]]))
    with pytest.raises(ValueError, match="reaches outside the measured range"):
        compare_runs([(cfg, cfg.output_path)], tmp_path / "c", log=lambda m: None)
