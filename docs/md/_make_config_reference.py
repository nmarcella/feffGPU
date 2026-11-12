"""Write docs/md/configuration.md from the pydantic models of feffgpu.md.config (every key, its type, its default and
its description). Run from anywhere after changing a configuration key:

    python docs/md/_make_config_reference.py
"""
from __future__ import annotations
import sys
import types
import typing
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from pydantic import BaseModel                          # noqa: E402
from pydantic_core import PydanticUndefined            # noqa: E402
from feffgpu.md import config as C                     # noqa: E402

INTRO = """# Configuration

An MD-EXAFS run is described by one TOML file. It is validated before anything runs (`md-exafs validate
config.toml`); a key that does not exist, or a value of the wrong kind, gives a message that starts with the key's
path, for example `edge.0.rpath: Input should be greater than 0`. Paths in the file are relative to the file. The
command writes the resolved configuration, with every default filled in and absolute paths, to
`config.resolved.toml` in the output folder; running that file repeats the run.

The machine-readable form of this page is the JSON Schema printed by `md-exafs schema`. Example files:
`md-exafs example` lists them, `md-exafs example nvt --out config.toml` writes one.

This page is generated from the configuration models by `docs/md/_make_config_reference.py`; do not edit it by hand.

A complete file:

```toml
{example}
```
"""

SECTIONS = [
    ("system", C.SystemConfig, "[system]", "The trajectory and the output folder."),
    ("lattice", C.LatticeConfig, "[lattice]", None),
    ("atoms", None, "[atoms]", "Numeric atom types and their elements, for files with numeric types (LAMMPS dumps): `1 = \"Zn\"`, "
                               "`2 = \"C\"`. The form of `md_exafs` 2.0 (`Zn = 1`) is accepted too. For LAMMPS dumps the map applies to "
                               "the `type` column; for other files to the atomic numbers that ASE read (a type written as 1 is read as "
                               "H)."),
    ("frames", C.FramesConfig, "[frames]", None),
    ("edge", C.EdgeConfig, "[[edge]]", "One block per absorbing element. Several blocks give several edges in one run."),
    ("average", C.AverageConfig, "[average]", None),
    ("compare", C.CompareConfig, "[compare] or [[compare]]", "One table for one measured spectrum, or one `[[compare]]` block per "
                                                            "spectrum (for example one per edge)."),
    ("compare.window", C.WindowConfig, "window = {...} in [compare]", "An inline table: `window = { kmin = 3.0, kmax = 14.0, dk = 1.0, "
                                                                      "kweight = 2, rmin = 1.0, rmax = 5.0 }`."),
    ("rdf", C.RDFConfig, "[rdf]", None),
    ("rdf.peak", C.PeakConfig, "[[rdf.peak]]", None),
    ("rdf.projection", C.ProjectionConfig, "[[rdf.projection]]", None),
    ("rdf.angle", C.AngleConfig, "[[rdf.angle]]", None),
    ("convergence", C.ConvergenceConfig, "[convergence]", None),
    ("dynamics", C.DynamicsConfig, "[dynamics]", None),
    ("run", C.RunConfig, "[run]", None),
]


def _type(ann):
    """A short, readable name of a type annotation."""
    origin = typing.get_origin(ann)
    args = typing.get_args(ann)
    if ann is type(None):
        return "none"
    if origin in (typing.Union, types.UnionType):
        parts = [_type(a) for a in args if a is not type(None)]
        return " or ".join(parts)
    if origin is typing.Literal:
        return " / ".join(f'"{a}"' if isinstance(a, str) else str(a) for a in args)
    if origin in (list, tuple):
        return f"list of {_type(args[0])}" if args else "list"
    if origin is dict:
        return "table"
    if isinstance(ann, type) and issubclass(ann, BaseModel):
        return "table"
    return {int: "integer", float: "number", str: "text", bool: "true / false", dict: "table"}.get(ann, getattr(ann, "__name__", str(ann)))


def _toml(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, list):
        return "[" + ", ".join(_toml(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{}" if not v else "{" + ", ".join(f"{k} = {_toml(x)}" for k, x in v.items()) + "}"
    if v is None:
        return "not set"
    return repr(v)


def _default(field):
    if field.default is not PydanticUndefined:
        return _toml(field.default)
    if field.default_factory is not None:
        value = field.default_factory()
        if isinstance(value, BaseModel):
            return "the defaults below"
        return _toml(value)
    return "**required**"


def table(model):
    rows = ["| Key | Type | Default | Description |", "|---|---|---|---|"]
    for name, field in model.model_fields.items():
        desc = (field.description or "").replace("|", "\\|")
        rows.append(f"| `{name}` | {_type(field.annotation)} | {_default(field)} | {desc} |")
    return "\n".join(rows)


def main():
    example = C.example_configs()["nvt"].strip()
    out = [INTRO.format(example=example)]
    for key, model, title, note in SECTIONS:
        out.append(f"## `{title}`\n")
        text = note or (model.__doc__ or "").strip()
        if text:
            out.append(" ".join(text.split()) + "\n")
        if model is not None:
            out.append(table(model) + "\n")
    path = HERE / "configuration.md"
    path.write_text("\n".join(out))
    print(f"written {path}")


if __name__ == "__main__":
    main()
