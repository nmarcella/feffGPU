"""The configuration of RMC-EXAFS: every example validates, problems are reported by the dotted path of the field, paths
are relative to the file, the resolved configuration can be read back, the simplest case needs nothing else."""
import json
import tomllib
from pathlib import Path
import numpy as np
import pytest
from rmc_helpers import ROOT, PHASE_ZNC, rock_salt, write_poscar, write_chi
from feffgpu.config_tools import to_toml
from feffgpu.rmc import config as C

EXAMPLES = sorted((ROOT / "examples/rmc_exafs").glob("*.toml"))


@pytest.mark.parametrize("name", sorted(C.example_configs()))
def test_every_shipped_example_matches_the_schema(name):
    text = C.example_configs()[name]
    cfg = C.RmcConfig.model_validate(tomllib.loads(text))
    assert cfg.run.output and cfg.sample
    back = C.RmcConfig.model_validate(tomllib.loads(to_toml(cfg.model_dump(mode="json"))))
    assert back.model_dump(mode="json") == cfg.model_dump(mode="json")                    # the writer of config.resolved.toml round-trips


@pytest.mark.parametrize("path", EXAMPLES, ids=[p.name for p in EXAMPLES])
def test_the_repository_examples_match_the_schema(path):
    cfg = C.RmcConfig.model_validate(tomllib.loads(path.read_text()))
    assert (cfg.search is not None) != (cfg.refine is not None)                           # each example runs one engine
    assert len(cfg.sample) == 1 and len(cfg.sample[0].edge) == 1 and cfg.moves.hops is None
    assert len(EXAMPLES) >= 3 and any(C.RmcConfig.model_validate(tomllib.loads(q.read_text())).refine is not None for q in EXAMPLES)


def test_the_simplest_case_one_sample_one_edge_no_moves(tmp_path):
    text = """
[run]
output = "out"
[[sample]]
label = "s"
structure = "s.vasp"
  [[sample.edge]]
  name = "C"
  absorber = "C"
  chi = "s.chi"
  phase = "p.pad"
  window = { kmin = 3.0, kmax = 10.0, rmin = 1.0, rmax = 3.0 }
[search]
generations = 5
"""
    cfg = C.RmcConfig.model_validate(tomllib.loads(text))
    assert cfg.moves.exchange == [] and cfg.moves.hops is None                              # nothing about exchanges, vacancies or sites is required
    assert cfg.amplitude.mode == "fixed" and cfg.amplitude.s02 == 1.0 and cfg.amplitude.de0 == 0.0     # S0^2 and the E0 shift: declared and held
    assert cfg.constraints.max_displacement == 0.4 and cfg.search.states == 32


def test_validate_file_reports_every_problem_by_its_field(tmp_path):
    f = tmp_path / "bad.toml"
    f.write_text("""
[run]
output = "out"
[[sample]]
label = "a b"
structure = "s.vasp"
  [[sample.edge]]
  name = "C"
  absorber = "C"
  chi = "c.chi"
  phase = "p.pad"
  window = { kmin = 3.0, kmax = 12.0, rmin = 1.0, rmax = 4.0 }
  [[sample.edge]]
  name = "Zn"
  absorber = "Zn"
  chi = "c.chi"
  phase = "p.pad"
  window = { kmin = 5.0, kmax = 4.0, rmin = 1.0, rmax = 4.0 }
[search]
states = 5
generations = 10
unknown_key = 1
""")
    problems = C.validate_file(f)
    joined = "\n".join(problems)
    assert any(p.startswith("sample.0.label:") for p in problems), joined
    assert any(p.startswith("sample.0.edge.1.window:") and "kmax" in p for p in problems), joined
    assert any(p.startswith("search.states:") and "even" in p for p in problems), joined
    assert any(p.startswith("search.unknown_key:") for p in problems), joined
    assert C.validate_file(tmp_path / "missing.toml")[0].endswith("does not exist")
    (tmp_path / "broken.toml").write_text("[run\noutput = ")
    assert "not valid TOML" in C.validate_file(tmp_path / "broken.toml")[0]


def test_validate_file_checks_the_input_files(tmp_path):
    sym, pos, cell = rock_salt(2)
    write_poscar(tmp_path / "s.vasp", sym, pos, cell)
    k = 0.05 * np.arange(301)
    write_chi(tmp_path / "s.chi", k, np.sin(4 * k) / (1 + k ** 2))
    base = f"""
[run]
output = "runs/x"
[[sample]]
label = "s"
structure = "s.vasp"
  [[sample.edge]]
  name = "C"
  absorber = "C"
  chi = "s.chi"
  phase = "{PHASE_ZNC}"
  potentials = {{ Zn = 1, C = 2 }}
  window = {{ kmin = 3.0, kmax = 12.0, dk = 1.0, rmin = 1.0, rmax = 3.0 }}
  rpath = 4.0
[search]
generations = 3
"""
    f = tmp_path / "ok.toml"
    f.write_text(base)
    assert C.validate_file(f) == []
    assert C.output_directory(f) == (tmp_path / "runs/x").resolve()
    bad = base.replace("kmax = 12.0", "kmax = 16.0").replace("rpath = 4.0", "rpath = 6.0").replace("Zn = 1, C = 2", "C = 2")
    bad = bad.replace('absorber = "C"', 'absorber = "Cu"')
    (tmp_path / "bad.toml").write_text(bad)
    problems = C.validate_file(tmp_path / "bad.toml")
    joined = "\n".join(problems)
    assert any(p.startswith("sample.0.edge.0.chi:") and "measured range" in p for p in problems), joined          # window beyond the data
    assert any(p.startswith("sample.0.edge.0.rpath:") for p in problems), joined                                   # cell too small for rpath
    assert any(p.startswith("sample.0.edge.0.potentials:") and "Zn" in p for p in problems), joined                # a potential missing
    assert any(p.startswith("sample.0.edge.0.absorber:") for p in problems), joined


def test_paths_are_relative_to_the_file_and_cross_field_rules(tmp_path):
    (tmp_path / "sub").mkdir()
    f = tmp_path / "sub" / "c.toml"
    f.write_text(C.example_configs()["one_edge"])
    cfg = C.load_config(f)
    assert Path(cfg.sample[0].structure) == (tmp_path / "sub" / "NiO.vasp").resolve()
    assert Path(cfg.run.output) == (tmp_path / "sub" / "runs/one_edge").resolve()
    data = tomllib.loads(C.example_configs()["one_edge"])
    data["constraints"]["box_rule"] = "halve"                                                   # EvAX's halving cannot act on gradient steps
    with pytest.raises(Exception, match="box_rule"):
        C.RmcConfig.model_validate(data)
    data = tomllib.loads(C.example_configs()["one_edge"])
    data["amplitude"]["mode"] = "from_run"
    with pytest.raises(Exception, match="from_run"):
        C.RmcConfig.model_validate(data)
    data = tomllib.loads(C.example_configs()["two_samples"])
    data["sample"][1]["edge"][0]["absorber"] = "O"                                              # a shared name with another absorber
    with pytest.raises(Exception, match="same absorber"):
        C.RmcConfig.model_validate(data)
    data = tomllib.loads(C.example_configs()["one_edge"])
    data["sample"][0]["hop_sites"] = "sites.txt"                                                # hop sites without the element that hops
    with pytest.raises(Exception, match="moves.hops"):
        C.RmcConfig.model_validate(data)
    data = tomllib.loads(C.example_configs()["rearrangements"])
    data["moves"]["exchange"] = ["Ca"]
    with pytest.raises(Exception, match="two different elements"):
        C.RmcConfig.model_validate(data)


def test_json_schema_documents_every_key():
    schema = C.json_schema()
    assert schema["title"] == "RmcConfig" and {"run", "sample", "search"} <= set(schema["properties"])
    missing = []
    for name, model in schema["$defs"].items():
        for key, prop in model.get("properties", {}).items():
            if not prop.get("description"):
                missing.append(f"{name}.{key}")
    assert not missing, missing
    json.dumps(schema)


def test_s02_and_the_e0_shift_are_held_unless_a_fit_is_asked_for(tmp_path):
    """Declared and held is the default. The refinement holds them always: the declared values, or those of an earlier
    run (a reference material); a requested fit is refused, not ignored."""
    from feffgpu.rmc.outputs import METHOD
    from feffgpu.rmc.run import _held_amplitude
    assert C.AmplitudeSettings().mode == "fixed"
    data = tomllib.loads(C.example_configs()["refine"])
    cfg = C.RmcConfig.model_validate(data)
    assert cfg.amplitude.mode == "fixed" and _held_amplitude(cfg, cfg.sample[0], print) is cfg       # the declared values, as written
    data["amplitude"]["mode"] = "fit"
    with pytest.raises(ValueError, match="holds S0\\^2 and the E0 shift and does not fit them"):
        _held_amplitude(C.RmcConfig.model_validate(data), cfg.sample[0], print)
    (tmp_path / "reference").mkdir()
    (tmp_path / "reference" / "run.json").write_text(json.dumps({"method": METHOD, "shared": {"s02": {"Cu": 0.87}, "de0": {"Cu": 2.5}}}))
    data["amplitude"] = {"mode": "from_run", "from_run": str(tmp_path / "reference")}
    lines = []
    held = _held_amplitude(C.RmcConfig.model_validate(data), cfg.sample[0], lines.append)
    assert held.amplitude.s02 == {"Cu": 0.87} and held.amplitude.de0 == {"Cu": 2.5} and "held" in lines[0]
    (tmp_path / "reference" / "run.json").write_text(json.dumps({"method": METHOD, "shared": {"s02": {"Ni": 0.9}, "de0": {"Ni": 0.0}}}))
    with pytest.raises(ValueError, match="no shared values for the spectrum 'Cu'"):
        _held_amplitude(C.RmcConfig.model_validate(data), cfg.sample[0], print)
