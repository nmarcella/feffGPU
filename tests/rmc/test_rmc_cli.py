"""The command `rmc-exafs`: subcommands, exit codes, one message per user error, --json output."""
import json
import pytest
from feffgpu.rmc.cli import main, parser


def test_every_subcommand_exists():
    names = set(parser()._subparsers._group_actions[0].choices)
    assert {"run", "refine", "validate", "schema", "example", "shells", "short-pairs", "rdf", "angles", "plot", "signal-map"} <= names


def test_schema_and_examples(capsys):
    assert main(["schema"]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["title"] == "RmcConfig"
    assert main(["example", "--json"]) == 0
    names = json.loads(capsys.readouterr().out)
    assert "one_edge" in names
    assert main(["example", "one_edge"]) == 0
    assert "[[sample.edge]]" in capsys.readouterr().out


def test_user_errors_give_one_message_and_exit_code_2(tmp_path, capsys):
    assert main(["example", "no_such_example"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("rmc-exafs example:") and len(err.strip().splitlines()) == 1
    f = tmp_path / "c.toml"
    assert main(["example", "one_edge", "--out", str(f)]) == 0
    capsys.readouterr()
    assert main(["validate", str(f), "--json"]) == 2                                            # the files it names do not exist
    out = json.loads(capsys.readouterr().out)
    assert out["valid"] is False and all(p.split(":")[0].startswith("sample.0") for p in out["problems"])
    assert main(["run", str(f)]) == 2
    assert capsys.readouterr().err.startswith("rmc-exafs run:")
    assert main(["shells", str(tmp_path)]) == 2
    assert main(["shells", str(tmp_path / "none"), "--shell", "Zn-C:1:2"]) == 2
    (tmp_path / "x.toml").write_text("[run\n")
    assert main(["run", str(tmp_path / "x.toml")]) == 2
    with pytest.raises(SystemExit):
        main(["no-such-command"])
