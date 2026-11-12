"""The configuration: every example validates, wrong keys and values give messages that name the field, input files
are checked, and the functions used by the MCP server (json_schema, example_configs, validate_file,
output_directory)."""
import tomllib
from pathlib import Path
import pytest
from mdexafs_testing import PHASE_ZNC, ROOT, write_config

EXAMPLES = sorted((ROOT / "examples/md_exafs").glob("*.toml"))


def _minimal(**over):
    data = {"system": {"trajectory": "traj.extxyz"}, "lattice": {"dynamic": True},
            "edge": [{"element": "C", "potentials": "file", "phase": str(PHASE_ZNC)}]}
    for k, v in over.items():
        data[k] = v
    return data


def test_every_example_configuration_validates():
    from feffgpu.md.config import example_configs, parse_config
    texts = dict(example_configs())
    assert {"nvt", "npt_lammps", "dynamics"} <= set(texts)
    for p in EXAMPLES:
        texts[f"examples/{p.name}"] = p.read_text()
    assert any(name.startswith("examples/") for name in texts), "examples/md_exafs has no configuration"
    for name, text in texts.items():
        cfg = parse_config(tomllib.loads(text))                                  # structure only: the files need not exist
        assert cfg.edge, name


def test_wrong_keys_and_values_name_the_field(tmp_path):
    from feffgpu.md.config import ConfigError, parse_config
    bad = _minimal(edge=[{"element": "C", "potentials": "file", "phase": "p.pad", "rpath": -2.0, "critera": 1.0}],
                   frames={"absorbers_per_frame": "some", "step": 0}, run={"devise": "cuda"})
    with pytest.raises(ConfigError) as err:
        parse_config(bad)
    problems = err.value.problems
    assert any(p.startswith("edge.0.rpath:") for p in problems)
    assert any(p.startswith("edge.0.critera:") and "unknown key" in p for p in problems)
    assert any(p.startswith("frames.absorbers_per_frame:") for p in problems)
    assert any(p.startswith("frames.step:") for p in problems)
    assert any(p.startswith("run.devise:") for p in problems)
    for data, where in ((_minimal(edge=[{"element": "Qq"}]), "edge.0.element"),
                        (_minimal(edge=[{"element": "C", "potentials": "file"}]), "edge.0"),
                        (_minimal(lattice={"a": 3.0, "b": 3.0}), "lattice"),
                        (_minimal(lattice={"dynamic": True, "a": 3.0, "b": 3.0, "c": 3.0}), "lattice"),
                        (_minimal(lattice={"pbc": [True, False, True]}), "lattice.pbc"),
                        (_minimal(average={"path_types": ["C-Zn"], "max_distance": [3.0, 4.0]}), "average"),
                        (_minimal(compare={"chi": "m.chi", "window": {"kmin": 9, "kmax": 4}}), "compare.0.window"),
                        ({"edge": [{"element": "C"}]}, "system")):
        with pytest.raises(ConfigError) as e:
            parse_config(data)
        assert any(p.startswith(where) for p in e.value.problems), (where, e.value.problems)


def test_atoms_map_both_forms_and_defaults():
    from feffgpu.md.config import parse_config
    a = parse_config(_minimal(atoms={"1": "Zn", "2": "C"}))
    b = parse_config(_minimal(atoms={"Zn": 1, "C": 2}))                         # the md_exafs 2.0 form
    assert a.atoms == b.atoms == {"1": "Zn", "2": "C"}
    cfg = parse_config(_minimal(compare={"chi": "m.chi"}))
    e = cfg.edge[0]
    assert e.name == "C" and e.potentials == "file" and e.rpath == 6.0 and e.nleg == 4 and e.phase_kmax_resolved == e.kmax + 2
    assert cfg.compare[0].edge == "C" and cfg.frames.absorbers_per_frame == "all" and cfg.run.devices == "cuda"


def test_validate_file_reports_problems_and_missing_inputs_without_raising(tmp_path):
    from feffgpu.md.config import validate_file
    p = write_config(tmp_path / "c.toml", _minimal(compare={"chi": "measured.chi"}))
    problems = validate_file(p)
    assert any(x.startswith("system.trajectory: no such file") for x in problems)
    assert any(x.startswith("compare.0.chi: no such file") for x in problems)
    (tmp_path / "traj.extxyz").write_text("")
    (tmp_path / "measured.chi").write_text("1 0\n2 0\n")
    assert validate_file(p) == []
    (tmp_path / "broken.toml").write_text("[system\ntrajectory = 'x'")
    assert validate_file(tmp_path / "broken.toml")[0].startswith("file: not valid TOML")
    assert validate_file(tmp_path / "missing.toml")[0].startswith("file: no such configuration file")
    q = write_config(tmp_path / "f.toml", _minimal(edge=[{"element": "C", "potentials": "file", "phase": "nope.pad"}]))
    assert any(x.startswith("edge.0.phase: no such file") for x in validate_file(q))
    d = write_config(tmp_path / "d.toml", _minimal(dynamics={"structure": "start.cif"}, system={"trajectory": "to_be_written.extxyz"}))
    probs = validate_file(d)
    assert any(x.startswith("dynamics.structure") for x in probs) and not any(x.startswith("system.trajectory") for x in probs)


def test_output_directory_paths_and_schema(tmp_path):
    from feffgpu.md.config import json_schema, load_config, output_directory, resolved_dict, to_toml, parse_config
    sub = tmp_path / "cfgs"
    sub.mkdir()
    p = write_config(sub / "c.toml", _minimal(system={"trajectory": "../traj.extxyz", "output": "results/run1"}))
    assert output_directory(p) == (sub / "results/run1").resolve()
    assert output_directory(write_config(sub / "d.toml", _minimal())) == (sub / "mdexafs_out").resolve()
    cfg = load_config(p)
    assert Path(cfg.system.trajectory) == (tmp_path / "traj.extxyz").resolve() and cfg.output_path == (sub / "results/run1").resolve()
    again = parse_config(tomllib.loads(to_toml(resolved_dict(cfg))))
    assert again.model_dump() == cfg.model_dump()                                # the resolved file reproduces the configuration
    schema = json_schema()
    sections = {"system", "lattice", "atoms", "frames", "edge", "average", "compare", "rdf", "convergence", "dynamics", "run"}
    assert sections <= set(schema["properties"])
    assert schema["$defs"]["EdgeConfig"]["properties"]["rpath"]["description"]


def test_every_setting_has_a_description():
    from feffgpu.md import config as C
    for model in (C.MDExafsConfig, C.SystemConfig, C.LatticeConfig, C.FramesConfig, C.EdgeConfig, C.AverageConfig, C.WindowConfig,
                  C.CompareConfig, C.PeakConfig, C.ProjectionConfig, C.AngleConfig, C.RDFConfig, C.ConvergenceConfig, C.DynamicsConfig,
                  C.RunConfig):
        for name, field in model.model_fields.items():
            assert field.description and len(field.description) > 10, f"{model.__name__}.{name}"
