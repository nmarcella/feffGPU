"""Analysis of RMC-EXAFS runs: the states of a run, statistics over the states and over runs, figures.

The generic tools of pair analysis are in ``feffgpu.analysis`` (shared with MD-EXAFS); this module applies them to
the files of a run (``feffgpu.rmc.outputs``):

- ``shell_statistics``: N, R, the MSRD (sigma2) and the third cumulant of stated shells, for the best state, with the
  spread over the states of a run and, for several runs (seeds), over the runs;
- ``short_pair_statistics``: pairs closer than the chemistry allows;
- ``run_rdf``, ``run_angles``: partial pair distributions and bond angles of the states;
- ``plot_run``: the fits of every edge and the pair distributions of a run;
- ``signal_map``: which distances make which features of the final fit.

What the spreads are. The states of one run descend from each other by selection, so their spread is a lower bound
of the uncertainty of a shell number; independent runs (other seeds) give a fairer spread. Neither includes what was
held fixed: the number of atoms of each element (so the total number of neighbors of an element), the start
lattice, the windows and the forward model.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from ..analysis import shell_table, short_pairs, partial_rdfs, bond_angles, angle_histogram
from .outputs import read_sample

__all__ = ["load_states", "parse_shell", "parse_pair_value", "shell_statistics", "short_pair_statistics", "run_rdf", "run_angles", "plot_run",
           "signal_map"]

QUANTITIES = ("N", "R", "sigma2", "c3")


def load_states(sample_dir):
    """The configurations of a sample directory: ``{"symbols", "cell", "population" (P, C, n, 3), "best" (C, n, 3),
    "ideal" (C, n, 3), "misfits" (P,), "report", "dir"}`` (from ``state.npz``, or ``checkpoint.npz`` of a run that
    has not finished)."""
    s = read_sample(sample_dir)
    z = s["state"] if s["state"] is not None else s["checkpoint"]
    if z is None:
        raise FileNotFoundError(f"{sample_dir}: no state.npz or checkpoint.npz")
    cell = np.asarray(z["cell"], float)
    pop = np.asarray(z["population"], float)
    misfits = np.asarray(z["misfits"], float)
    best = np.asarray(z["positions"], float) if "positions" in z else pop[int(np.argmin(misfits))]
    return {"symbols": np.asarray(z["symbols"]).astype(str), "cell": None if np.isnan(cell).any() else cell, "population": pop, "best": best,
            "ideal": np.asarray(z["ideal"], float), "misfits": misfits, "report": s["report"], "dir": Path(sample_dir)}


def parse_shell(text):
    """``"A-B:r_min:r_max"`` -> (center A, neighbor B, r_min, r_max)."""
    try:
        pair, lo, hi = text.split(":")
        a, b = pair.split("-")
        return a.strip(), b.strip(), float(lo), float(hi)
    except ValueError as err:
        raise ValueError(f"a shell is written A-B:r_min:r_max, for example Zn-C:1.8:2.6 (got {text!r})") from err


def parse_pair_value(text):
    """``"A-B:value"`` -> ("A-B", value)."""
    try:
        pair, v = text.split(":")
        a, b = pair.split("-")
        return f"{a.strip()}-{b.strip()}", float(v)
    except ValueError as err:
        raise ValueError(f"write A-B:value, for example Zn-C:1.9 (got {text!r})") from err


def _stats(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"mean": float("nan"), "sd": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0, "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}


def shell_statistics(paths, shells):
    """Shell numbers of one or several runs of the same sample.

    Parameters
    ----------
    paths : list of str or Path
        Sample directories (one per run, for example one per seed).
    shells : list of tuple
        Each ``(center, neighbor, r_min, r_max)``: the two elements and the distance range (A).

    Returns
    -------
    dict
        ``{"shells": [row, ...], "runs": [dirs], "method": text}``; every row has ``center``, ``neighbor``,
        ``r_min``, ``r_max`` and for each of N, R, sigma2 (the MSRD, A^2) and c3 (A^3): ``best`` (the best state of
        the first run), ``states`` (mean and sd over the states of each run), ``runs`` (mean and sd over the runs of
        their best states; with one run, None).
    """
    runs = [load_states(p) for p in paths]
    rows = []
    for center, neighbor, lo, hi in shells:
        row = {"center": center, "neighbor": neighbor, "r_min": float(lo), "r_max": float(hi)}
        per_run_best, per_run_states = [], []
        for d in runs:
            states = [shell_table(P, d["symbols"], d["cell"], [(center, neighbor, lo, hi)])[0] for P in d["population"]]
            best = shell_table(d["best"], d["symbols"], d["cell"], [(center, neighbor, lo, hi)])[0]
            per_run_states.append(states)
            per_run_best.append(best)
        for q in QUANTITIES:
            row[q] = {"best": per_run_best[0][q],
                      "states": [_stats([s[q] for s in states]) for states in per_run_states],
                      "runs": _stats([b[q] for b in per_run_best]) if len(runs) > 1 else None}
        rows.append(row)
    return {"shells": rows, "runs": [str(d["dir"]) for d in runs],
            "method": "pair histograms of the positions integrated over the stated ranges (no fit); 'states': spread over the states of a run "
                      "(they share ancestors: a lower bound); 'runs': spread of the best states over independent runs"}


def short_pair_statistics(path, thresholds=None):
    """Pairs below a distance: ``{"best": ..., "population": ...}`` from ``feffgpu.analysis.short_pairs``.

    ``thresholds`` ``{"A-B": distance}``; default the closest distances of the run (its report)."""
    d = load_states(path)
    if thresholds is None:
        thresholds = (d["report"] or {}).get("closest_distance") or {}
    if not thresholds:
        raise ValueError(f"{path}: give thresholds (the run had no closest distances)")
    pop = d["population"].reshape(-1, *d["population"].shape[2:])
    return {"thresholds": dict(thresholds), "best": short_pairs(d["best"], d["symbols"], d["cell"], thresholds),
            "population": short_pairs(pop, d["symbols"], d["cell"], thresholds)}


def run_rdf(path, center, elements=None, rmax=6.0, dr=0.01, which="population"):
    """Partial pair distributions of the center element with each element, over the states (or the best state).

    Returns ``r`` (bin centers, A), ``{element: pairs per center per A}`` and the numbers of centers."""
    d = load_states(path)
    sym = d["symbols"]
    pos = d["best"] if which == "best" else d["population"].reshape(-1, *d["population"].shape[2:])
    centers = np.nonzero(sym == center)[0]
    if len(centers) == 0:
        raise ValueError(f"{path}: no {center} atoms")
    elements = elements or sorted(set(sym.tolist()))
    r, g, dist = partial_rdfs(pos, d["cell"], centers, sym, elements, rmax=rmax, dr=dr)
    return r, g, {"centers": int(len(centers)), "configurations": int(np.asarray(pos).reshape(-1, len(sym), 3).shape[0])}


def run_angles(path, center, neighbors, ranges, which="population", bin_width=1.0):
    """Angles B-A-C at the center element A between neighbors B and C in stated distance ranges.

    Returns the histogram (bin centers, counts), the number of angles and their mean and standard deviation (deg)."""
    d = load_states(path)
    pos = d["best"] if which == "best" else d["population"].reshape(-1, *d["population"].shape[2:])
    ang = bond_angles(pos, d["symbols"], d["cell"], center, tuple(neighbors), tuple(tuple(r) for r in ranges))
    x, h = angle_histogram(ang, bin_width)
    return {"angles": int(ang.size), "mean": float(ang.mean()) if ang.size else float("nan"), "sd": float(ang.std()) if ang.size else float("nan"),
            "bin_centers": x.tolist(), "counts": h.tolist()}


def _comparisons(sp, device="cpu"):
    """SpectrumComparison objects of the edges of a spectra.npz, with S0^2 1 and no shift (the stored models are scaled)."""
    from ..exafs import FitWindow, SpectrumComparison
    out = []
    for key in sorted(k for k in sp if k.startswith("window_")):
        name = key[len("window_"):]
        kmin, kmax, dk, kw, rmin, rmax = (float(v) for v in sp[key])
        out.append(SpectrumComparison(name, sp["k"], sp[f"data_{name}"], FitWindow(kmin=kmin, kmax=kmax, dk=dk, kweight=int(kw), rmin=rmin, rmax=rmax),
                                      device=device))
    return out


def plot_run(path, out_dir=None, rmax_pairs=6.0):
    """Figures of a sample directory: ``fits.png`` (every edge: data, start, final; the windows are the run's own) and
    ``pairs.png`` (partial pair distributions of every absorber, the start lattice filled, the final states as lines).
    Returns the paths written."""
    from ..exafs import plot_comparison
    from ..analysis import plot_pair_distributions
    path = Path(path)
    out_dir = Path(out_dir) if out_dir else path
    out_dir.mkdir(parents=True, exist_ok=True)
    s = read_sample(path)
    written = []
    if s["spectra"] is not None:
        sp = s["spectra"]
        comps = _comparisons(sp)
        models = [{c.name: sp[f"model_start_{c.name}"] for c in comps}, {c.name: sp[f"model_refit_{c.name}"] for c in comps}]
        f = out_dir / "fits.png"
        plot_comparison(comps, models, labels=["start", "final"], out=f, title=f"{path.name}: fits")
        written.append(f)
    d = load_states(path)
    sym = d["symbols"]
    absorbers = [str(a) for a in ((d["report"] or {}).get("absorbers") or {}).values()]
    centers = [a for a in dict.fromkeys(absorbers) if (sym == a).any()] or sorted(set(sym.tolist()))
    curves = {}
    pop = d["population"].reshape(-1, *d["population"].shape[2:])
    for c in centers:
        idx = np.nonzero(sym == c)[0]
        for el in sorted(set(sym.tolist())):
            r, g0, _ = partial_rdfs(d["ideal"], d["cell"], idx, sym, [el], rmax=rmax_pairs, dr=0.02)
            _, g1, _ = partial_rdfs(pop, d["cell"], idx, sym, [el], rmax=rmax_pairs, dr=0.02)
            curves[f"{c}-{el}"] = {"start lattice": g0[el], "final states": g1[el]}
    f = out_dir / "pairs.png"
    plot_pair_distributions(r, curves, out=f, title=f"{path.name}: pair distributions")
    written.append(f)
    return written


def _resolved_config(sample_dir):
    root = Path(sample_dir).parent
    f = root / "config.resolved.toml"
    if not f.exists():
        raise FileNotFoundError(f"{root}: no config.resolved.toml (the signal map rebuilds the forward model from it)")
    from .config import load_config
    return load_config(f)


def signal_map(path, components=None, bands=((1.0, 2.0), (2.0, 3.0), (3.0, 4.0)), out=None, device="cuda"):
    """Which distances make which features of the final fit of a sample.

    The forward model is rebuilt from the run's ``config.resolved.toml`` (the structure, chi and phase files must
    still exist), the best state's single scattering is split into distance components, and the multiple scattering
    is the rest of the stored calculated spectrum.

    Parameters
    ----------
    path : path
        A sample directory.
    components : dict, optional
        ``{element: [b0, b1, ...]}`` (A): the components of each neighbor element between consecutive boundaries;
        default one component per element from the start of the pair tables to the cutoff.
    bands : sequence of (float, float)
        R bands of chi(R) for the shares (A).
    out : path, optional
        A figure per edge, ``<out>_<edge>.png``.
    device : str

    Returns
    -------
    dict
        Per edge: the components (N, mean distance) and the shares of every part per band.
    """
    from ..analysis.signal_map import component_spectra, band_shares, plot_signal_map
    from .run import build_sample
    path = Path(path)
    cfg = _resolved_config(path)
    s = read_sample(path)
    sset = next((x for x in cfg.sample if x.label == path.name), None)
    if sset is None:
        raise ValueError(f"{path}: the sample {path.name!r} is not in the run's configuration")
    sample = build_sample(sset, cfg, device=device)
    d = load_states(path)
    sp, rep = s["spectra"], s["report"]
    result = {}
    for e in sample.edges:
        e.set_s02(float(rep["final_refit"]["s02"][e.name]))
        e.set_de0(float(rep["final_refit"]["de0"][e.name]) - e.e0)
        comps = []
        for el in e.elements:
            bs = (components or {}).get(el, [e.r0, e.rmax])
            comps += [(el, lo, hi) for lo, hi in zip(bs[:-1], bs[1:])]
        parts, info = component_spectra(e, d["best"], comps)
        total = np.asarray(sp[f"calc_refit_{e.name}"], float)
        parts["multiple scattering"] = total - sum(parts.values())
        shares = band_shares(e, parts, total, list(bands))
        result[e.name] = {"components": {k: {kk: v[kk] for kk in ("element", "r_min", "r_max", "N", "R")} for k, v in info.items()}, "shares": shares}
        if out is not None:
            plot_signal_map(e, parts, info, total, data=e.data, out=f"{out}_{e.name}.png", title=f"{path.name}, {e.name} edge")
    return result
