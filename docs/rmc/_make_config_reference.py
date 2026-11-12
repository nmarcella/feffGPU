"""Regenerate the key tables of docs/rmc/configuration.md from the pydantic models of feffgpu.rmc.config.

    python docs/rmc/_make_config_reference.py

Everything between the markers "<!-- BEGIN GENERATED -->" and "<!-- END GENERATED -->" in configuration.md is
replaced; the text around them is kept."""
from __future__ import annotations
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from pydantic_core import PydanticUndefined  # noqa: E402
from feffgpu.rmc import config as C  # noqa: E402

SECTIONS = [("[run]", C.RunSettings), ("[[sample]]", C.SampleSettings), ("[[sample.edge]]", C.EdgeSettings), ("window = { ... }", C.WindowSettings),
            ("[amplitude]", C.AmplitudeSettings), ("[search]", C.SearchSettings), ("[moves]", C.MoveSettings), ("[moves.hops]", C.HopSettings),
            ("[constraints]", C.ConstraintSettings), ("[refine]", C.RefineSettings)]


def _name(ann):
    """A plain name of a type annotation: 'number', 'integer', 'text', 'true or false', 'list of ...', 'table', 'one of ...'."""
    import typing
    origin, args = typing.get_origin(ann), typing.get_args(ann)
    if origin is typing.Literal:
        return "one of " + ", ".join(f'"{a}"' for a in args)
    if origin in (typing.Union, getattr(__import__("types"), "UnionType", None)):
        names = [_name(a) for a in args if a is not type(None)]
        return " or ".join(names)
    if origin in (list, tuple):
        return "list of " + (_name(args[0]) if args else "values")
    if origin is dict:
        return "table of " + (_name(args[1]) if len(args) > 1 else "values")
    return {float: "number", int: "integer", str: "text", bool: "true or false"}.get(ann, "table")


def type_name(field):
    return _name(field.annotation)


def default(field):
    if field.is_required():
        return "required"
    if field.default_factory is not None:
        v = field.default_factory()
        return "`" + repr(v).replace("'", '"') + "`" if not hasattr(v, "model_dump") else "the defaults below"
    v = field.default
    if v is None:
        return "none"
    if v is PydanticUndefined:
        return "required"
    if isinstance(v, str):
        return f'`"{v}"`'
    if isinstance(v, bool):
        return "`true`" if v else "`false`"
    return f"`{v}`"


def table(title, model):
    lines = [f"### `{title}`", "", (model.__doc__ or "").strip().split("\n")[0], "", "| key | type | default | meaning |", "|---|---|---|---|"]
    for name, field in model.model_fields.items():
        desc = (field.description or "").replace("|", "/").replace("\n", " ")
        lines.append(f"| `{name}` | {type_name(field)} | {default(field)} | {desc} |")
    return "\n".join(lines) + "\n"


def main():
    page = HERE / "configuration.md"
    text = page.read_text()
    begin, end = "<!-- BEGIN GENERATED -->", "<!-- END GENERATED -->"
    if begin not in text or end not in text:
        raise SystemExit(f"{page}: the markers are missing")
    body = "\n".join(table(t, m) for t, m in SECTIONS)
    head, rest = text.split(begin, 1)
    _, tail = rest.split(end, 1)
    page.write_text(head + begin + "\n\n" + body + "\n" + end + tail)
    print(f"written {page}")


if __name__ == "__main__":
    main()
