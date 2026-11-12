"""The files of an RMC-EXAFS run: writers, readers, and the rules for continuing a run.

Layout of a run of ``rmc-exafs run`` (every key is described in ``docs/rmc/outputs.md``)::

    <output>/run.json                 the run: samples, shared S0^2 and E0 shifts with intervals, timing
    <output>/config.resolved.toml     the configuration with absolute paths and every default
    <output>/<label>/report.json      one sample: settings, records of the start and the end, the final refit,
                                      the state of the search (for a continuation)
    <output>/<label>/history.json     one record per generation and per refit
    <output>/<label>/state.npz        the final population, the best state, the start lattice, sites, cell, symbols
    <output>/<label>/spectra.npz      the data and the models of every edge on the k grid, the windows
    <output>/<label>/checkpoint.npz   the population every run.save_every generations

A continued run (``search.continue_from``) reads ``report.json`` and ``state.npz`` of every sample. It refuses a
population saved by another method, a different structure and a different start lattice, and it keeps the scale of
the gradient steps of its first run.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

__all__ = ["METHOD", "FORMAT", "write_json", "read_json", "read_run", "sample_dirs", "read_sample", "load_continuation", "shared_values_of_run"]

METHOD = "feffgpu.rmc: population search under EvAX's rules"
FORMAT = 1


def _default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


def write_json(path, obj):
    """Write ``obj`` as indented JSON (numpy numbers and arrays converted)."""
    Path(path).write_text(json.dumps(obj, indent=1, default=_default))


def read_json(path):
    return json.loads(Path(path).read_text())


def read_run(run_dir):
    """``run.json`` of a run directory."""
    p = Path(run_dir) / "run.json"
    if not p.exists():
        raise FileNotFoundError(f"{run_dir}: no run.json (not the output directory of a finished run of rmc-exafs)")
    return read_json(p)


def sample_dirs(path):
    """The sample directories of a run: ``[path]`` if ``path`` is a sample directory (it holds ``state.npz``), else the
    subdirectories of the samples of ``run.json`` (or every subdirectory with a ``report.json``)."""
    path = Path(path)
    if (path / "state.npz").exists() or (path / "checkpoint.npz").exists():
        return [path]
    if (path / "run.json").exists():
        out = [path / s for s in read_run(path)["samples"]]
        return [p for p in out if p.exists()]
    out = sorted(p.parent for p in path.glob("*/report.json"))
    if not out:
        raise FileNotFoundError(f"{path}: neither a sample directory (state.npz) nor a run directory (run.json)")
    return out


def read_sample(sample_dir):
    """Everything of one sample: ``{"report", "history", "state", "spectra", "dir"}`` (the npz files as dicts of arrays;
    a missing file gives None)."""
    d = Path(sample_dir)
    out = {"dir": d}
    out["report"] = read_json(d / "report.json") if (d / "report.json").exists() else None
    out["history"] = read_json(d / "history.json") if (d / "history.json").exists() else None
    for name in ("state", "spectra", "checkpoint"):
        f = d / f"{name}.npz"
        if f.exists():
            with np.load(f, allow_pickle=False) as z:
                out[name] = {k: z[k] for k in z.files}
        else:
            out[name] = None
    return out


def load_continuation(sample_dir, symbols, ideal, states, cells):
    """The population of an earlier run to continue, after the checks.

    Parameters
    ----------
    sample_dir : path
        ``<earlier output>/<label>``.
    symbols : array of str, shape (n,)
    ideal : array, shape (C, n, 3)
        This run's start lattice.
    states, cells : int
        The number of states and of cells per state of this run.

    Returns
    -------
    dict
        ``population`` (states, C, n, 3) and ``site_of`` (states, C, n) of the best ``states`` saved states, ``order``
        (their indices), ``misfits``, and the earlier ``report``.

    Raises
    ------
    ValueError
        Another method, another structure, another start lattice, too few states, or another number of cells.
    """
    d = Path(sample_dir)
    if not (d / "report.json").exists() or not (d / "state.npz").exists():
        raise ValueError(f"{d}: no report.json and state.npz to continue from")
    rep = read_json(d / "report.json")
    if rep.get("method") != METHOD:
        raise ValueError(f"{d}: not a run of this search (method {rep.get('method')!r}). A population of another method is not continued.")
    with np.load(d / "state.npz", allow_pickle=False) as z:
        sym = z["symbols"].astype(str)
        if list(sym) != [str(v) for v in symbols]:
            raise ValueError(f"{d}: a different structure (the elements of the atoms differ)")
        if z["ideal"].shape != np.asarray(ideal).shape or float(np.abs(np.asarray(z["ideal"], float) - np.asarray(ideal, float)).max()) > 1e-4:
            raise ValueError(f"{d}: a different start lattice (the displacement limit is measured from it)")
        pop, site_of, M = np.asarray(z["population"], float), np.asarray(z["site_of"]), np.asarray(z["misfits"], float)
    if pop.shape[0] < states or pop.shape[1] != cells:
        raise ValueError(f"{d}: {pop.shape[0]} states of {pop.shape[1]} cells saved; {states} states of {cells} cells asked")
    order = np.argsort(M, kind="stable")[:states]
    return {"population": pop[order], "site_of": site_of[order], "order": order, "misfits": M[order], "report": rep}


def shared_values_of_run(run_dir):
    """The shared S0^2 and E0 shifts of the final refit of an earlier run: ``({name: s02}, {name: de0})``."""
    run = read_run(run_dir)
    if run.get("method") != METHOD:
        raise ValueError(f"{run_dir}: not a run of this search (method {run.get('method')!r})")
    sh = run["shared"]
    return {k: float(v) for k, v in sh["s02"].items()}, {k: float(v) for k, v in sh["de0"].items()}
