"""The tutorial notebooks (tutorials/): what is stored is clean, their configuration is valid, the tutorial data are
what the notebooks say, and, on a GPU with the notebook tools installed (``feffgpu[notebooks]``), the notebooks run
from top to bottom and the worked example finds the planted values again."""
import importlib.util
import json
import re
import shutil
import tomllib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "tutorials"
NOTEBOOKS = sorted(TUTORIALS.glob("*.ipynb"))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _code_cells(path):
    return ["".join(c["source"]) for c in json.loads(path.read_text())["cells"] if c["cell_type"] == "code"]


def _printed(nb):
    """Everything the code cells of an executed notebook printed."""
    return "".join(out.get("text", "") for cell in nb["cells"] if cell["cell_type"] == "code" for out in cell.get("outputs", []))


def test_there_are_notebooks():
    assert NOTEBOOKS, "tutorials/ holds no notebook"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.name for p in NOTEBOOKS])
def test_a_stored_notebook_is_clean(path):
    text = path.read_text()
    nb = json.loads(text)
    assert nb["nbformat"] == 4
    assert "/home/" not in text and "/Users/" not in text, "a machine path is stored in the notebook"
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert not any(out.get("output_type") == "error" for out in cell.get("outputs", [])), "an error is stored"
    for source in _code_cells(path):
        if source.startswith("%%"):                               # a cell that writes a file
            continue
        lines = []
        for line in source.splitlines():                          # shell and magic lines are not Python
            bare = line.lstrip()
            lines.append(line[: len(line) - len(bare)] + "pass" if bare.startswith(("!", "%")) else line)
        compile("\n".join(lines), path.name, "exec")


def test_the_configuration_of_the_worked_example_is_valid(tmp_path):
    from feffgpu.md import validate_file
    cells = [c for c in _code_cells(TUTORIALS / "01_md_exafs_iron.ipynb") if c.startswith("%%writefile")]
    assert len(cells) == 1
    body = cells[0].split("\n", 1)[1]
    config = tomllib.loads(body)
    assert config["edge"][0]["element"] == "Fe" and config["compare"]["chi"] == "Fe_synthetic_spectrum.chi"
    for name in ("Fe_bcc_300K.extxyz", "Fe_synthetic_spectrum.chi"):
        shutil.copy(TUTORIALS / "data" / name, tmp_path / name)
    (tmp_path / "config.toml").write_text(body)
    assert validate_file(tmp_path / "config.toml") == []
    # the generator of the synthetic spectrum uses the same physics as the notebook
    generator = _load(TUTORIALS / "make_fe_trajectory.py", "make_fe_trajectory")
    same = tomllib.loads(generator.EDGE_SETTINGS)["edge"][0]
    assert same == config["edge"][0]


def test_the_tutorial_trajectory_is_what_the_notebook_says():
    from ase.io import read
    frames = read(TUTORIALS / "data" / "Fe_bcc_300K.extxyz", index=":")
    assert len(frames) == 60 and all(len(f) == 128 and set(f.get_chemical_symbols()) == {"Fe"} for f in frames)
    assert np.allclose(frames[0].cell.lengths(), 4 * 2.8665)
    assert frames[1].info["time_fs"] - frames[0].info["time_fs"] == 500.0
    d = np.concatenate([f.get_all_distances(mic=True)[np.triu_indices(128, 1)] for f in frames[::10]])
    first = d[d < 2.68]
    assert len(first) / len(frames[::10]) == pytest.approx(128 * 8 / 2, rel=0.01)       # eight first neighbors
    assert first.mean() == pytest.approx(2.485, abs=0.005) and first.var() == pytest.approx(0.0051, abs=0.0006)
    k, chi = np.loadtxt(TUTORIALS / "data" / "Fe_synthetic_spectrum.chi", unpack=True)
    assert k[0] == 1.0 and k[-1] == 15.0 and np.allclose(np.diff(k), 0.05) and np.abs(chi).max() < 1.0


def test_the_generator_makes_a_trajectory_of_the_same_kind():
    generator = _load(TUTORIALS / "make_fe_trajectory.py", "make_fe_trajectory")
    positions, box, times = generator.trajectory(frames=4, seed=5, settle_fs=600.0, every_fs=100.0)
    sites, _ = generator.bcc_block()
    assert positions.shape == (4, 128, 3) and box == pytest.approx(4 * 2.8665) and list(times) == [100.0, 200.0, 300.0, 400.0]
    moved = np.sqrt(((positions - sites) ** 2).sum(-1))
    assert 0.03 < moved.mean() < 0.2                               # thermal motion, and no atom left its site
    assert moved.max() < 0.6


@pytest.mark.gpu
@pytest.mark.slow
def test_the_worked_example_runs_and_finds_the_planted_values():
    pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    tool = _load(ROOT / "tools" / "run_tutorials.py", "run_tutorials")
    printed = _printed(tool.run_notebook(TUTORIALS / "01_md_exafs_iron.ipynb", check=True))
    s02 = float(re.search(r"S0²: ([0-9.]+)", printed).group(1))
    de0 = float(re.search(r"ΔE0: ([0-9.+-]+) eV", printed).group(1))
    assert abs(s02 - 0.80) < 0.03 and abs(de0 - 2.0) < 0.25         # planted: 0.80 and +2.0 eV
    n_first = float(re.search(r"2\.00 to 2\.68 Å:  N = ([0-9.]+)", printed).group(1))
    assert abs(n_first - 8.0) < 0.1
    assert "/home/" not in printed


@pytest.mark.gpu
@pytest.mark.slow
def test_the_notebook_for_your_own_trajectory_runs_on_the_example():
    pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    tool = _load(ROOT / "tools" / "run_tutorials.py", "run_tutorials")
    printed = _printed(tool.run_notebook(TUTORIALS / "02_md_exafs_your_trajectory.ipynb", check=True))
    assert "my_run/config.toml: valid" in printed and "R-factor:" in printed and "packed: my_run_results.zip" in printed


@pytest.mark.gpu
@pytest.mark.slow
def test_the_evax_notebook_runs_when_the_example_files_are_there(monkeypatch):
    """The RMC notebook works on the files of an EvAX example, which the repository does not hold: the test runs
    only when FEFFGPU_EVAX_EXAMPLE names the folder with in/ and out/. With them, the two programs must agree on
    EvAX's misfit, on the list of scattering paths and on the spectrum of EvAX's final atoms."""
    import os
    if not os.environ.get("FEFFGPU_EVAX_EXAMPLE"):
        pytest.skip("needs the files of the EvAX example (set FEFFGPU_EVAX_EXAMPLE)")
    pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    monkeypatch.setenv("FEFFGPU_TUTORIAL_STEPS", "2")
    tool = _load(ROOT / "tools" / "run_tutorials.py", "run_tutorials")
    printed = _printed(tool.run_notebook(TUTORIALS / "03_rmc_exafs_from_evax.ipynb", check=True))
    from_files = float(re.search(r"from EvAX's own wavelet files:\s+([0-9.]+)", printed).group(1))
    ours = float(re.search(r"through this package's objective:\s+([0-9.]+)", printed).group(1))
    assert abs(ours / from_files - 1.0) < 0.03                      # the same misfit
    table = re.findall(r"^\s+(\d)\s+(\d\.\d\d)\s+(\d+)\s+(\d+)$", printed, flags=re.M)        # legs, half length, EvAX, this package
    assert len(table) >= 8 and all(int(mine) in (int(theirs), 2 * int(theirs)) for _, _, theirs, mine in table)   # the same paths
    single = float(re.search(r"single scattering:\s+([0-9.e+-]+)", printed).group(1))
    feff_count = float(re.search(r"FEFF's degeneracies:\s+([0-9.e+-]+)", printed).group(1))
    evax_count = float(re.search(r"degeneracies of EvAX's list:\s+([0-9.e+-]+)", printed).group(1))
    rest = float(re.search(r"energy shift of [0-9.+-]+ eV:\s+([0-9.e+-]+)", printed).group(1))
    assert single < 0.005 and evax_count < feff_count < 0.006 and rest < 0.0015      # the same spectrum for the same atoms
    assert "/home/" not in printed


def test_the_evax_notebook_is_kept_without_outputs():
    """Its outputs would show a measured spectrum that the repository does not hold."""
    nb = json.loads((TUTORIALS / "03_rmc_exafs_from_evax.ipynb").read_text())
    assert nb["metadata"]["feffgpu"] == {"needs_env": "FEFFGPU_EVAX_EXAMPLE", "store_outputs": False}
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")
