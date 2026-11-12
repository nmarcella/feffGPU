"""The md-exafs command: exit codes, --json, one clear message on a user error, and the commands end to end."""
import json
import numpy as np
import pytest
from mdexafs_testing import A_W, gpu, rock_salt_run, write_config

WINDOW = {"kmin": 3.0, "kmax": 13.0, "dk": 1.0, "kweight": 2, "rmin": 1.0, "rmax": 5.0}


def test_validate_schema_example_and_user_errors(tmp_path, capsys):
    from feffgpu.md.cli import main
    p = write_config(tmp_path / "c.toml", {"system": {"trajectory": "missing.extxyz"}, "lattice": {"dynamic": True},
                                           "edge": [{"element": "C", "rpath": -1.0}]})
    assert main(["validate", str(p), "--json"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert not out["valid"] and any(m.startswith("edge.0.rpath") for m in out["problems"])
    assert main(["schema"]) == 0
    assert "edge" in json.loads(capsys.readouterr().out)["properties"]
    assert main(["example", "--json"]) == 0
    assert "nvt" in json.loads(capsys.readouterr().out)["examples"]
    assert main(["example", "nvt", "--out", str(tmp_path / "nvt.toml")]) == 0
    assert (tmp_path / "nvt.toml").read_text().startswith("# MD-EXAFS example")
    assert main(["run", str(tmp_path / "nothing.toml")]) == 2                          # one message, exit code 2
    err = capsys.readouterr().err
    assert err.startswith("md-exafs:") and "no such configuration file" in err and "Traceback" not in err
    assert main(["run", str(p), "--json"]) == 2
    assert "edge.0.rpath" in json.loads(capsys.readouterr().out)["problems"][0]
    assert main(["example", "nonexistent"]) == 2
    with pytest.raises(SystemExit):
        main(["frobnicate"])


def test_supercell_from_a_cif(tmp_path, capsys):
    from ase.build import bulk
    import ase.io
    from feffgpu.md.cli import main
    ase.io.write(tmp_path / "W.cif", bulk("W", "bcc", a=A_W, cubic=True))
    assert main(["supercell", str(tmp_path / "W.cif"), "--size", "2,2,3", "--out", str(tmp_path / "cell.extxyz"), "--json"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["atoms"] == 2 * 12 and np.allclose(np.diag(res["cell"]), [2 * A_W, 2 * A_W, 3 * A_W], atol=1e-6)
    assert len(ase.io.read(tmp_path / "cell.extxyz")) == 24
    assert main(["supercell", str(tmp_path / "W.cif"), "--size", "2,2,2", "--out", str(tmp_path / "cp2k.xyz"), "--cp2k"]) == 0
    lines = (tmp_path / "cp2k.xyz").read_text().splitlines()
    assert len(lines) == 16 and lines[0].split()[0] == "W"
    assert main(["supercell", str(tmp_path / "W.cif"), "--size", "2,2", "--out", str(tmp_path / "x.xyz")]) == 2


@gpu
def test_commands_end_to_end(tmp_path, capsys):
    from feffgpu.md.cli import main
    rdf = {"r_max": 6.0, "bin_width": 0.02, "device": "cpu", "peak": [{"center": "C", "neighbor": "Zn", "r_min": 1.8, "r_max": 2.6}],
           "projection": [{"center": "C", "neighbor": "Zn", "direction": [1, 0, 0], "angle": 15.0, "symmetry": True}],
           "angle": [{"center": "C", "neighbors": ["Zn", "Zn"], "ranges": [[1.8, 2.6], [1.8, 2.6]]}]}
    cfg = rock_salt_run(tmp_path, n_frames=4, output="out", absorbers=8, extra={"rdf": rdf, "average": {"path_types": ["C-Zn", "C-C-Zn"],
                                                                                                         "max_distance": [3.0, 4.0]}})
    assert main(["run", str(cfg), "--json"]) == 0
    s = json.loads(capsys.readouterr().out)
    assert s["frames_in_cache"] == 4 and s["average"]["edges"]["C"]["n_frames"] == 4
    out = tmp_path / "out"
    for f in ("chi.h5", "config.resolved.toml", "report.json", "average/average.json", "average/C.chi", "average/C.parts.dat",
              "average/C.png"):
        assert (out / f).is_file(), f
    assert main(["validate", str(out / "config.resolved.toml")]) == 0                   # the resolved file is a valid configuration
    capsys.readouterr()
    chi = np.loadtxt(out / "average/C.chi")
    parts = np.loadtxt(out / "average/C.parts.dat")
    assert chi.shape[1] == 4 and np.allclose(parts[:, 1] + parts[:, 2], chi[:, 1], atol=1e-7)   # the two selected parts make the average
    sharded = rock_salt_run(tmp_path, n_frames=4, output="sh", absorbers=8)
    for i in (1, 2):
        assert main(["run", str(sharded), "--shard", f"{i}/2"]) == 0
    capsys.readouterr()
    assert main(["merge", str(tmp_path / "sh"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["frames"] == 4
    assert main(["average", str(tmp_path / "sh"), "--json"]) == 0
    a = json.loads(capsys.readouterr().out)
    assert a["edges"]["C"]["n_frames"] == 4
    assert main(["plot", str(cfg), "--json"]) == 0
    assert all((tmp_path / f).is_file() for f in json.loads(capsys.readouterr().out)["figures"])
    assert main(["convergence", str(cfg), "--json"]) == 0
    c = json.loads(capsys.readouterr().out)
    assert c["spectra"]["C"]["n_frames"][-1] == 4 and (out / "convergence/convergence.png").is_file()
    assert main(["rdf", str(cfg), "--json"]) == 0
    r = json.loads(capsys.readouterr().out)
    peak = r["peaks"][0]
    assert peak["integration"]["N"] == pytest.approx(6.0, abs=0.05) and "gaussian" in peak and "skewed" in peak
    assert r["projections"][0]["shell"]["N"] == pytest.approx(6.0, abs=0.3)
    assert (out / "rdf/rdf.png").is_file() and (out / "rdf/C-Zn.dat").is_file() and r["angles"][0]["angles"] > 0
