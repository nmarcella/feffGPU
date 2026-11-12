"""Thermodynamic logs (CP2K .ener, ASE MD log, CSV), block averages, drift, and the convergence of an averaged
spectrum with the number of frames."""
import math
import numpy as np
import pytest


def _ar1(n, phi, sd, seed):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0, sd)
    return x


def test_cp2k_ener_file_temperature_and_conserved_drift(tmp_path):
    from feffgpu.analysis.convergence import HARTREE_EV, read_log, thermo_summary
    n = 4000
    t = 0.5 * np.arange(n)                                                       # fs
    temp = 300 + _ar1(n, 0.9, 3.0, 1)
    cons = -100.0 + 2e-8 * t + 1e-7 * np.random.default_rng(2).normal(size=n)   # Hartree, a small linear drift
    lines = ["#     Step Nr.          Time[fs]        Kin.[a.u.]          Temp[K]            Pot.[a.u.]        Cons Qty[a.u.]        "
             "UsedTime[s]"]
    lines += [f"{i:12d} {t[i]:16.6f} {0.01:20.10f} {temp[i]:16.6f} {cons[i] - 0.01:20.10f} {cons[i]:20.10f} {0.1:12.4f}" for i in range(n)]
    (tmp_path / "run.ener").write_text("\n".join(lines) + "\n")
    log = read_log(tmp_path / "run.ener")
    assert log["format"] == "cp2k" and log["time_unit"] == "fs" and log["conserved"] == "conserved"
    s = thermo_summary(log, skip=100.0, blocks=10)
    assert s["rows"] == n - 200 and abs(s["temperature"]["mean"] - 300) < 4 * s["temperature"]["se_block"] + 0.5
    d = s["conserved_drift"]
    assert d["slope"] == pytest.approx(2e-8 * HARTREE_EV, rel=0.05) and d["unit"] == "eV"


def test_ase_log_and_csv(tmp_path):
    from feffgpu.analysis.convergence import read_log, thermo_summary
    rows = ["Time[ps]      Etot[eV]     Epot[eV]     Ekin[eV]    T[K]"]
    rows += [f"{0.002 * i:10.4f} {-10 + 1e-4 * i:12.4f} {-11 + 1e-4 * i:12.4f} {1.0:12.4f} {300 + (i % 7) - 3:8.1f}" for i in range(200)]
    (tmp_path / "md.log").write_text("\n".join(rows) + "\n")
    log = read_log(tmp_path / "md.log")
    assert log["format"] == "ase" and log["time_unit"] == "ps" and set(log["columns"]) >= {"total", "potential", "kinetic", "temperature"}
    s = thermo_summary(log)
    assert s["quantities"]["total"]["drift"]["slope"] == pytest.approx(1e-4 / 0.002, rel=1e-2) and "conserved_drift" not in s
    (tmp_path / "log.csv").write_text("time,temperature,etot\n" + "\n".join(f"{i},{300 + i % 3},{-5.0}" for i in range(50)) + "\n")
    c = read_log(tmp_path / "log.csv")
    assert c["format"] == "csv" and c["temperature"] == "temperature" and c["conserved"] == "etot"
    assert thermo_summary(c)["temperature"]["mean"] == pytest.approx(301.0, abs=0.05)
    with pytest.raises(ValueError, match="could not be guessed"):
        (tmp_path / "x.txt").write_text("1 2 3\n4 5 6\n")
        read_log(tmp_path / "x.txt")


def test_block_average_sees_correlation():
    from feffgpu.analysis.convergence import block_average
    phi, sd, n = 0.95, 1.0, 200000
    x = _ar1(n, phi, sd, 3)
    mean, se = block_average(x, blocks=20)
    var = sd ** 2 / (1 - phi ** 2)
    se_true = math.sqrt(var / n * (1 + phi) / (1 - phi))                         # standard error of the mean of an AR(1) series
    assert 0.5 < se / se_true < 1.6 and se > 3 * x.std() / math.sqrt(n)           # far above the plain standard error


def test_spectrum_convergence_with_the_number_of_frames():
    from feffgpu.analysis.convergence import spectrum_convergence
    from feffgpu.exafs import k_grid
    k = k_grid(16.0)
    signal = np.sin(2 * 2.5 * k) * np.exp(-0.01 * k ** 2) / np.maximum(k, 1.0)
    rng = np.random.default_rng(0)
    chi = signal[None] + 0.05 * rng.normal(size=(64, len(k))) / np.maximum(k, 1.0)
    c = spectrum_convergence(k, chi, points=8)
    assert c["n_frames"][-1] == 64 and c["r_factor"][-1] == pytest.approx(0.0, abs=1e-20)
    assert c["r_factor"][0] > 10 * c["r_factor"][-2] and 0 < c["split_half_r_factor"] < 0.1
    same = spectrum_convergence(k, np.repeat(signal[None], 5, 0))
    assert max(same["r_factor"]) < 1e-20 and same["split_half_r_factor"] < 1e-20
