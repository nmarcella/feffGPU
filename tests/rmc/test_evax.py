"""Reading an EvAX calculation and writing the inputs of ``rmc-exafs`` for it (``feffgpu.rmc.evax``). The EvAX files of
these tests are made here, in EvAX's formats; no file of EvAX's own examples is used. No GPU is needed."""
import json
import numpy as np
import pytest
from feffgpu.exafs import ETOK
from feffgpu.rmc import EvaxCalculation, read_parameters, scf_radius, validate_file
from feffgpu.rmc.cli import main

A = 3.165


def write_project(tmp, layout="plain", periodic=True, supercell="5 5 5", outputs=True, k_power_io=2, extra=""):
    """A made-up EvAX calculation on bcc W: start.dat, the structure file, a data file, and (optionally) output files."""
    inp = tmp / "in"
    inp.mkdir(parents=True, exist_ok=True)
    (inp / "w.p1").write_text(f"bcc W\n1.0\n {A} 0.0 0.0\n 0.0 {A} 0.0\n 0.0 0.0 {A}\n W\n 2\nDirect\n 0.0 0.0 0.0\n 0.5 0.5 0.5\n")
    k = 0.05 * np.arange(0, 361)
    chi = 0.1 * np.sin(2 * 2.74 * k) * np.exp(-0.01 * k ** 2)
    np.savetxt(inp / "w.dat", np.column_stack([chi * k ** k_power_io, k]), header="chi*k^n   k")
    sep = ":" if layout == "colon" else ""
    lines = ["# a comment", "BEGIN", f"Title{sep} test", f"File_with_structure{sep} input_files/w.p1", f"Supercell {supercell}", f"pbc{sep} {int(periodic)}",
             f"Number_of_states{sep} 8", f"Froze_in{sep} 200", "Stop_after 300", "Maximal_step_length 0.004", "Initial_displacement 0.03",
             "Maximal_displacement 0.35", "Column_for_k 2", "Column_for_chi 1", f"k_power_in_IO {k_power_io}", "S02 0.9", "dE0 2.0",
             f"File_with_experimental_EXAFS_signal{sep} input_files/w.dat", "N_legs 8", f"R_max_for_FEFF{sep} 5.6", "FEFF_criteria 1.0",
             f"Edge{sep} W", f"Edge_type{sep} L3", "Space w", f"k_min{sep} 3", f"k_max{sep} 16", "k_power 2", "R_min 1.0", "R_max 5.0", "R0_for_WT 3.0",
             "r_scf -1", extra, "END"]
    (inp / "start.dat").write_text("\n".join(lines) + "\n")
    out = None
    if outputs:
        out = tmp / "out"
        out.mkdir(exist_ok=True)
        cells = np.array([[i, j, l] for i in range(5) for j in range(5) for l in range(5)], float)
        sites = (np.array([[0, 0, 0], [0.5, 0.5, 0.5]])[None] + cells[:, None]).reshape(-1, 3) * A
        final = sites + np.random.default_rng(0).normal(0, 0.05, sites.shape)
        for name, pos in (("phantoms_eq.xyz", sites), ("phantoms.xyz", final)):
            rows = [f"W {p[0]:.9f} {p[1]:.9f} {p[2]:.9f}" for p in pos] + ["W 99.0 99.0 99.0"] * 3      # three periodic copies after the real atoms
            (out / name).write_text(f"{len(rows)}\n#First {len(pos)} lines - real atoms, remaining 3 lines - copies due to PBC.\n" + "\n".join(rows) + "\n")
        kk = 0.05 * np.arange(60, 321)
        np.savetxt(out / "ipolEXAFS.dat", np.column_stack([0.11 * np.sin(5.5 * kk) * kk ** k_power_io, kk]))
        np.savetxt(out / "EXAFS.dat", np.column_stack([0.1 * np.sin(5.5 * kk) * kk ** k_power_io, kk]))
        (out / "partialEXAFS.dat").write_text("#nleg 2 2 3\n#deg 14 12 48\n#r 2.741 4.476 4.32\n#path W*-W- W*-W- W*-W-W-\n0.05 0.1 0.1 0.1\n")
    return inp, out, sites if outputs else None, final if outputs else None


@pytest.mark.parametrize("layout", ["plain", "colon"])
def test_the_parameter_file_is_read_in_both_layouts(tmp_path, layout):
    inp, out, *_ = write_project(tmp_path, layout=layout, outputs=False, extra="'Temperature': -1")
    p = read_parameters(inp / "start.dat")
    assert p["Edge"] == ["W"] and p["Supercell"] == ["5", "5", "5"] and p["File_with_structure"] == ["input_files/w.p1"]
    assert p["Temperature"] == ["-1"] and "BEGIN" not in p and "#" not in "".join(p)
    calc = EvaxCalculation(inp / "start.dat")
    s = calc.spectra[0]
    assert (s.element, s.edge, s.name, s.data_file) == ("W", "L3", "W_L3", "w.dat")
    assert (s.k_column, s.chi_column, s.k_power_io, s.kweight, s.nlegs) == (1, 0, 2, 2, 8)
    assert (s.kmin, s.kmax, s.rmin, s.rmax, s.wavelet_r0, s.rpath, s.criteria) == (3.0, 16.0, 1.0, 5.0, 3.0, 5.6, 1.0)
    assert calc.periodic and calc.supercell == [5, 5, 5] and np.allclose(calc.lattice, np.eye(3) * A)


def test_the_radius_of_the_self_consistency_loop_follows_evax_s_rule():
    d = 2.556
    fcc = [d] * 12 + [d * 2 ** 0.5] * 6 + [d * 3 ** 0.5] * 24 + [2 * d] * 12
    assert scf_radius(fcc) == pytest.approx(d * 3 ** 0.5)          # nearest; the first at least 1 A longer; the first 0.1 d1 beyond that
    assert scf_radius(fcc, r_scf=6.0) == 6.0
    assert scf_radius([2.0, 2.1, 2.2]) == 5.0                      # no distance 1 A beyond the nearest: EvAX's fallback
    assert scf_radius([2.0, 12.0, 14.0]) == 5.0                    # outside 1 to 10 A


def test_the_data_and_evax_s_target(tmp_path):
    inp, out, *_ = write_project(tmp_path, outputs=False)
    calc = EvaxCalculation(inp / "start.dat")
    k, chi = calc.data(0)
    truth = 0.1 * np.sin(2 * 2.74 * k) * np.exp(-0.01 * k ** 2)
    assert k[0] == 0.0 and np.allclose(chi[1:], truth[1:], atol=1e-9)          # the k weight of the file is taken out
    kt, target = calc.target(0)                                               # EvAX's rule: shift on the data, then divide by S02
    assert kt[0] == pytest.approx(3.0) and kt[-1] == pytest.approx(16.0)
    kq = np.sqrt(kt ** 2 - ETOK * 2.0)
    expect = np.interp(kq, k, truth * k ** 2) / kt ** 2 / 0.9
    assert np.allclose(target, expect, rtol=1e-6, atol=1e-9)
    assert calc.fitted(0) is None and calc.final() is None and calc.path_groups(0) is None


def test_the_output_files(tmp_path):
    inp, out, sites, final = write_project(tmp_path)
    calc = EvaxCalculation(inp / "start.dat", outputs=out)
    sym, pos, cell = calc.sites()
    assert len(sym) == 250 and set(sym.tolist()) == {"W"} and np.allclose(pos, sites, atol=1e-8)       # the real atoms only, in EvAX's order
    assert np.allclose(cell, np.eye(3) * 5 * A) and np.allclose(calc.final(), final, atol=1e-8)
    k, target = calc.target(0)                                                 # EvAX's own file, k weight taken out
    assert np.allclose(target, 0.11 * np.sin(5.5 * k)) and np.allclose(calc.fitted(0)[1], 0.1 * np.sin(5.5 * k))
    assert calc.path_groups(0) == [dict(legs=2, degeneracy=14.0, half_length=2.741), dict(legs=2, degeneracy=12.0, half_length=4.476),
                                   dict(legs=3, degeneracy=48.0, half_length=4.32)]
    by_element = EvaxCalculation(inp / "start.dat", outputs={"W": out})
    assert np.allclose(by_element.final(), final, atol=1e-8)


def test_the_inputs_of_rmc_exafs(tmp_path):
    inp, out, sites, final = write_project(tmp_path)
    calc = EvaxCalculation(inp / "start.dat", outputs=out)
    res = calc.write_inputs(tmp_path / "run", target="evax", gradient_steps=20, states=4, generations=50, energy_shift=-0.1)
    run = tmp_path / "run"
    assert sorted(p.name for p in run.iterdir()) == ["W_L3.chi", "feff_W_L3.inp", "rmc.toml", "start.extxyz"]
    assert res["phase_commands"] == ["feffgpu phase feff_W_L3.inp phase_W_L3.pkl"] and not any("N_legs" in n for n in res["notes"])
    text = (run / "rmc.toml").read_text()
    for piece in ('criteria_rule = "evax"', 'objective = "wavelet"', "wavelet_r0 = 3.0", "dk = 0.0", "nleg = 8", "rpath = 5.6", "criteria = 1.0",
                  "s02 = { W_L3 = 1.0 }", "de0 = { W_L3 = -0.1000 }", "states = 4", "generations = 50", "froze_in = 200", "step = 0.004", "start_kick = 0.03",
                  "gradient_steps = 20", "max_displacement = 0.35", "[refine]", "steps = 300", "core_distance = 1.92"):
        assert piece in text, piece
    feff = (run / "feff_W_L3.inp").read_text()
    assert " EDGE      L3" in feff and " EXCHANGE  0      0   0" in feff and "       1   74    W     -1      -1       1" in feff
    radius = float([ln for ln in feff.splitlines() if ln.startswith(" SCF")][0].split()[1])
    # bcc: nearest 2.741; the first distance 1 A longer is 4.476; the first one 0.274 A beyond that is 5.249
    assert radius == pytest.approx(A * 11 ** 0.5 / 2, abs=1e-5)
    (run / "phase_W_L3.pkl").write_bytes(b"")                                   # the phase file comes from `feffgpu phase`; here only its name
    problems = [p for p in validate_file(run / "rmc.toml") if "phase" not in p]
    assert problems == []
    data = calc.write_inputs(tmp_path / "data", target="data")                 # the measured spectrum: S02 and the shift on the model
    text = (tmp_path / "data" / "rmc.toml").read_text()
    assert "s02 = { W_L3 = 0.9 }" in text and "de0 = { W_L3 = -2.0000 }" in text and "states = 8" in text and "generations = 300" in text
    assert data["config"].endswith("rmc.toml")


def test_a_supercell_that_is_too_narrow_and_a_structure_without_periodic_boundaries(tmp_path):
    inp, out, *_ = write_project(tmp_path / "narrow", supercell="3 3 3", outputs=False)
    calc = EvaxCalculation(inp / "start.dat")
    assert calc.minimum_supercell() == [4, 4, 4]
    with pytest.raises(ValueError, match=r"supercell=\[4, 4, 4\]"):
        calc.write_inputs(tmp_path / "narrow" / "run")
    calc.write_inputs(tmp_path / "narrow" / "run", supercell=[4, 4, 4])
    from ase.io import read
    atoms = read(tmp_path / "narrow" / "run" / "start.extxyz")
    assert len(atoms) == 128 and np.allclose(np.array(atoms.get_cell()), np.eye(3) * 4 * A)
    inp, out, *_ = write_project(tmp_path / "cluster", supercell="2 2 2", periodic=False, outputs=False)
    calc = EvaxCalculation(inp / "start.dat")
    sym, pos, cell = calc.sites()
    extent = pos.max(0) - pos.min(0)
    assert np.allclose(np.diag(cell), extent + 2 * (5.6 + 0.5) + 1.0) and pos.min() > 5.6      # no atom has an image within the cutoff
    res = calc.write_inputs(tmp_path / "cluster" / "run")
    assert any("no periodic boundaries" in n for n in res["notes"])


def test_the_command(tmp_path, capsys):
    inp, out, *_ = write_project(tmp_path)
    assert main(["from-evax", str(inp / "start.dat"), str(tmp_path / "run"), "--outputs", str(out), "--target", "evax", "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["spectra"] == ["W_L3"] and rep["phase_commands"] == ["feffgpu phase feff_W_L3.inp phase_W_L3.pkl"]
    assert main(["from-evax", str(inp / "start.dat"), str(tmp_path / "run2"), "--outputs", f"W={out}"]) == 0
    assert "feffgpu phase" in capsys.readouterr().out
    assert main(["from-evax", str(tmp_path / "missing.dat"), str(tmp_path / "x")]) == 2
