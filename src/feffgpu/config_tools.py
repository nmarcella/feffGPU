"""Configuration files: the pieces that MD-EXAFS and RMC-EXAFS share.

Both tools are run from one TOML file that is validated with pydantic models. This module holds what is the same for
both: the base model (unknown keys are refused), the window of a comparison with a measured spectrum, reading TOML,
making relative paths absolute, turning a validation error into messages that start with the path of the field, and
writing a configuration back as TOML (the fully resolved configuration that every run stores next to its outputs).
"""
from __future__ import annotations
import json
import math
import tomllib
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["ConfigError", "ConfigModel", "WindowModel", "read_toml", "absolute_path", "validation_messages", "drop_none", "to_toml"]


class ConfigError(ValueError):
    """A problem in a configuration file.

    Attributes
    ----------
    problems : list of str
        One message per problem, each starting with the dotted path of the field (for example
        ``edge.0.window.kmax: ...``).
    """

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


class ConfigModel(BaseModel):
    """Base of every configuration section: a key that is not a setting of the section is an error."""
    model_config = ConfigDict(extra="forbid")


class WindowModel(ConfigModel):
    """The window of a comparison with a measured spectrum, in xraylarch's terms (``feffgpu.exafs.FitWindow``)."""
    kmin: float = Field(..., ge=0, description="Center of the low-k sill of the Hanning window (1/A).")
    kmax: float = Field(..., gt=0, description="Center of the high-k sill of the Hanning window (1/A).")
    dk: float = Field(1.0, ge=0, description="Width of each sill of the Hanning window (1/A).")
    kweight: int = Field(2, ge=0, le=4, description="Power of k that multiplies chi(k) before the Fourier transform.")
    rmin: float = Field(..., ge=0, description="Start of the R range of the R-factor (A).")
    rmax: float = Field(..., gt=0, description="End of the R range of the R-factor (A).")

    @model_validator(mode="after")
    def _ranges(self):
        if not self.kmax > self.kmin:
            raise ValueError(f"kmax ({self.kmax}) must be larger than kmin ({self.kmin})")
        if not self.rmax > self.rmin:
            raise ValueError(f"rmax ({self.rmax}) must be larger than rmin ({self.rmin})")
        return self

    def fit_window(self):
        """The ``feffgpu.exafs.FitWindow`` of these numbers."""
        from .exafs import FitWindow
        return FitWindow(kmin=self.kmin, kmax=self.kmax, dk=self.dk, kweight=self.kweight, rmin=self.rmin, rmax=self.rmax)


# -- reading ----------------------------------------------------------------------------------------------------------
def read_toml(path):
    """Read a TOML file into a dictionary.

    Raises
    ------
    ConfigError
        If the file does not exist or is not valid TOML.
    """
    p = Path(path)
    if not p.is_file():
        raise ConfigError([f"file: no such configuration file: {p}"])
    try:
        with open(p, "rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as err:
        raise ConfigError([f"file: not valid TOML: {err}"]) from None


def absolute_path(value, base, keep=()):
    """A path of a configuration made absolute: relative paths are relative to ``base`` (the folder of the TOML file).

    ``None``, an empty string and the words in ``keep`` (settings that take either a path or a keyword) are returned
    unchanged."""
    if value is None or value == "" or value in keep:
        return value
    p = Path(value).expanduser()
    return str(p if p.is_absolute() else (Path(base) / p).resolve())


_TYPE_TAGS = {"int", "float", "str", "bool", "list", "dict", "none", "tuple"}


def _is_type_tag(part):
    """pydantic adds the member of a union type to an error's location (``int``, ``literal['all']``, ``list[float]``)."""
    if not isinstance(part, str):
        return False
    return part in _TYPE_TAGS or "[" in part or part.startswith(("function-", "literal", "union"))


def validation_messages(err):
    """One message per problem of a pydantic ``ValidationError``, each starting with the dotted path of the field.

    The errors of the alternatives of one field (a field that accepts a number or a word, for example) are joined
    into one message. An unknown key and a missing required key get plain wording."""
    order, messages = [], {}
    for e in err.errors():
        loc = ".".join(str(p) for p in e.get("loc", ()) if not _is_type_tag(p)) or "file"
        msg = str(e.get("msg", "invalid")).replace("Value error, ", "")
        if e.get("type") == "extra_forbidden":
            msg = "unknown key (not a setting of this section; check the spelling)"
        elif e.get("type") == "missing":
            msg = "required setting is missing"
        if loc not in messages:
            order.append(loc)
            messages[loc] = []
        if msg not in messages[loc]:
            messages[loc].append(msg)
    return [f"{loc}: {' or '.join(messages[loc])}" for loc in order]


# -- writing ----------------------------------------------------------------------------------------------------------
def drop_none(x):
    """A copy of a nested dictionary or list without the entries whose value is None (TOML has no null)."""
    if isinstance(x, dict):
        return {k: drop_none(v) for k, v in x.items() if v is not None}
    if isinstance(x, list):
        return [drop_none(v) for v in x]
    return x


def _key(k):
    k = str(k)
    return k if k and all(ch.isalnum() or ch in "_-" for ch in k) else json.dumps(k)


def _value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{_key(k)} = {_value(x)}" for k, x in v.items() if x is not None) + " }"
    raise TypeError(f"cannot write {type(v).__name__} to TOML")


def _is_table_list(v):
    return isinstance(v, list) and len(v) > 0 and all(isinstance(x, dict) for x in v)


def _is_table(key, v, inline_keys, top):
    """A dictionary is written as a table when it is a top-level section, when it holds another table or a list of
    tables, or when it is long; a small flat dictionary inside a section (and every key in ``inline_keys``) is written
    inline, like ``window = { kmin = 3.0, ... }``."""
    if not isinstance(v, dict):
        return False
    if key in inline_keys:
        return False
    if top:
        return True
    return any(isinstance(x, dict) or _is_table_list(x) for x in v.values()) or len(v) > 6


def _emit(d, path, lines, inline_keys):
    top = path == ""
    for k, v in d.items():
        if v is None or _is_table(k, v, inline_keys, top) or _is_table_list(v):
            continue
        lines.append(f"{_key(k)} = {_value(v)}")
    for k, v in d.items():
        if _is_table(k, v, inline_keys, top):
            name = f"{path}.{_key(k)}" if path else _key(k)
            lines.append("")
            lines.append(f"[{name}]")
            _emit(v, name, lines, inline_keys)
    for k, v in d.items():
        if _is_table_list(v):
            name = f"{path}.{_key(k)}" if path else _key(k)
            for item in v:
                lines.append("")
                lines.append(f"[[{name}]]")
                _emit(item, name, lines, inline_keys)


def to_toml(data, header="", inline_keys=("window",)):
    """A configuration dictionary as TOML text.

    Sections become tables, lists of dictionaries become arrays of tables, small flat dictionaries inside a section
    are written inline. Entries whose value is None are left out. The text reads back to the same dictionary.

    Parameters
    ----------
    data : dict
        For example ``model.model_dump(mode="json")`` of a configuration.
    header : str
        Comment lines written first (without the leading ``#``).
    inline_keys : tuple of str
        Keys whose dictionary is always written inline.

    Returns
    -------
    str
    """
    lines = [f"# {h}" for h in header.splitlines()] if header else []
    _emit(data, "", lines, set(inline_keys))
    return "\n".join(lines).lstrip("\n") + "\n"
