"""Convergence of a simulation: its thermodynamic log, and an averaged spectrum against the number of configurations.

Thermodynamic logs (``read_log``): a CP2K ``.ener`` file, the log of ASE's ``MDLogger``, or a CSV file with a header
row. ``thermo_summary`` gives, for every quantity, the mean, the standard deviation, the block-averaged standard error
and the drift (a straight-line fit against time), and for the temperature the mean and spread. The drift of a
conserved quantity (the total energy of NVE dynamics, CP2K's "Cons Qty" of a thermostatted run) measures the
integration error; the total energy of Langevin dynamics is not conserved and its drift means nothing.

Spectra (``spectrum_convergence``): the running weighted mean of the configurations' spectra after n
configurations, compared with the mean of all of them (the R-factor in a given window, and the relative difference
of k-weighted chi(k)), and the R-factor between the means of the first and the second half (the size of the
sampling noise that remains). These take plain arrays: a trajectory's frames or a reverse Monte Carlo model's states.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np

__all__ = ["read_log", "thermo_summary", "block_average", "linear_drift", "spectrum_convergence", "HARTREE_EV"]

HARTREE_EV = 27.211386245988                 # eV per Hartree


def _numeric_rows(lines, sep=None):
    rows = []
    for ln in lines:
        t = ln.strip()
        if not t or t[0] in "#%;!":
            continue
        parts = t.split(sep) if sep else t.split()
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            continue
    if not rows:
        return np.zeros((0, 0))
    width = min(len(r) for r in rows)
    return np.array([r[:width] for r in rows], float)


def _guess_format(lines):
    head = "\n".join(lines[:5])
    if "Cons Qty" in head or "Step Nr." in head or "Time[fs]" in head:
        return "cp2k"
    if "Time[ps]" in head:
        return "ase"
    first = next((ln for ln in lines if ln.strip()), "")
    if "," in first:
        return "csv"
    raise ValueError("the format of the log could not be guessed: give it (\"cp2k\", \"ase\" or \"csv\")")


def read_log(path, format="auto"):
    """Read a thermodynamic log.

    Parameters
    ----------
    path : str or Path
    format : {"auto", "cp2k", "ase", "csv"}
        ``"cp2k"``: a CP2K ``.ener`` file (step, time in fs, kinetic energy, temperature, potential energy, conserved
        quantity in Hartree, used time); ``"ase"``: the log of ``ase.md.MDLogger`` (time in ps, energies in eV,
        temperature in K, one header line); ``"csv"``: comma-separated values with a header row (units as written).

    Returns
    -------
    dict
        ``format``; ``time`` (array) and ``time_unit`` (``"fs"``, ``"ps"`` or the CSV column name); ``columns``
        (``{name: array}``, energies of CP2K and ASE logs in eV); ``units`` (``{name: unit}``); ``temperature`` (the
        name of the temperature column or None); ``conserved`` (the name of the column whose drift is meaningful,
        or None); ``total`` (the total-energy column of an ASE log, or None).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no such log file: {p}")
    lines = p.read_text().splitlines()
    fmt = _guess_format(lines) if format == "auto" else format
    if fmt == "cp2k":
        a = _numeric_rows(lines)
        if a.shape[0] == 0 or a.shape[1] < 6:
            raise ValueError(f"{p}: no CP2K energy rows (step, time, kinetic, temperature, potential, conserved) found")
        cols = {"step": a[:, 0], "kinetic": a[:, 2] * HARTREE_EV, "temperature": a[:, 3], "potential": a[:, 4] * HARTREE_EV,
                "conserved": a[:, 5] * HARTREE_EV}
        units = {"step": "", "kinetic": "eV", "temperature": "K", "potential": "eV", "conserved": "eV"}
        return {"format": "cp2k", "time": a[:, 1], "time_unit": "fs", "columns": cols, "units": units, "temperature": "temperature",
                "conserved": "conserved", "total": None}
    if fmt == "ase":
        hdr_i = next((i for i, ln in enumerate(lines) if "Time[ps]" in ln), None)
        if hdr_i is None:
            raise ValueError(f"{p}: no ASE log header (Time[ps] ...) found")
        names = lines[hdr_i].split()
        a = _numeric_rows(lines[hdr_i + 1:])
        if a.shape[0] == 0:
            raise ValueError(f"{p}: the ASE log has no rows")
        cols, units = {}, {}
        for j, nm in enumerate(names[1:a.shape[1]], start=1):
            base = nm.split("[")[0]
            unit = nm.split("[")[1].rstrip("]") if "[" in nm else ""
            key = {"Etot": "total", "Epot": "potential", "Ekin": "kinetic", "T": "temperature", "Etot/N": "total_per_atom",
                   "Epot/N": "potential_per_atom", "Ekin/N": "kinetic_per_atom"}.get(base, base)
            cols[key], units[key] = a[:, j], unit
        temp = "temperature" if "temperature" in cols else None
        total = "total" if "total" in cols else ("total_per_atom" if "total_per_atom" in cols else None)
        return {"format": "ase", "time": a[:, 0], "time_unit": "ps", "columns": cols, "units": units, "temperature": temp,
                "conserved": None, "total": total}
    if fmt == "csv":
        hdr = next((ln for ln in lines if ln.strip() and ln.strip()[0] not in "#%;!"), None)
        if hdr is None:
            raise ValueError(f"{p}: empty CSV file")
        names = [h.strip() for h in hdr.split(",")]
        a = _numeric_rows(lines[lines.index(hdr) + 1:], sep=",")
        if a.shape[0] == 0:
            raise ValueError(f"{p}: the CSV file has no numeric rows")
        cols = {nm: a[:, j] for j, nm in enumerate(names[:a.shape[1]])}
        low = {nm.lower(): nm for nm in cols}
        tname = next((low[k] for k in ("time", "time_fs", "time_ps", "t_fs", "t_ps", "step") if k in low), names[0])
        temp = next((low[k] for k in ("temperature", "temp", "t_k", "temperature_k") if k in low), None)
        cons = next((low[k] for k in ("conserved", "cons_qty", "etot", "total", "total_energy") if k in low), None)
        return {"format": "csv", "time": cols[tname], "time_unit": tname, "columns": {k: v for k, v in cols.items() if k != tname},
                "units": {k: "" for k in cols if k != tname}, "temperature": temp, "conserved": cons, "total": None}
    raise ValueError(f"unknown log format {fmt!r} (use \"cp2k\", \"ase\" or \"csv\")")


def block_average(x, blocks=10):
    """Mean and block-averaged standard error of a series.

    The series is cut into ``blocks`` contiguous blocks; the standard error is the standard deviation of the block
    means divided by the square root of the number of blocks. For a correlated series (consecutive steps of a
    dynamics) this is larger, and more honest, than the plain standard error, provided the blocks are longer than
    the correlation time.

    Returns
    -------
    mean : float
    se : float
        NaN when there are fewer than two blocks with data.
    """
    x = np.asarray(x, float)
    if x.size == 0:
        return math.nan, math.nan
    b = int(max(1, min(blocks, x.size)))
    means = np.array([v.mean() for v in np.array_split(x, b) if v.size])
    if len(means) < 2:
        return float(x.mean()), math.nan
    return float(x.mean()), float(means.std(ddof=1) / math.sqrt(len(means)))


def linear_drift(t, y):
    """Slope and intercept of a straight-line fit of ``y`` against ``t``, and the change over the whole time span."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    if len(t) < 2 or np.ptp(t) == 0:
        return {"slope": math.nan, "intercept": float(y[0]) if len(y) else math.nan, "change": math.nan}
    slope, intercept = np.polyfit(t, y, 1)
    return {"slope": float(slope), "intercept": float(intercept), "change": float(slope * (t[-1] - t[0]))}


def thermo_summary(log, skip=0.0, blocks=10):
    """Statistics of a thermodynamic log (from ``read_log``).

    Parameters
    ----------
    log : dict
    skip : float
        Leave out the rows with ``time < time[0] + skip`` (equilibration), in the log's time unit.
    blocks : int
        Number of blocks of the block-averaged standard errors.

    Returns
    -------
    dict
        ``rows`` (used), ``time_span`` and ``time_unit``; ``quantities``: per column ``mean``, ``std``, ``se_block``,
        ``unit``, ``drift`` (``slope`` per time unit, ``change`` over the span); ``temperature`` (mean, std, se) and
        ``conserved_drift`` (``slope``, ``change``, ``relative_change`` = change / abs(mean)) when the log has them.
    """
    t = np.asarray(log["time"], float)
    keep = t >= t[0] + float(skip) if len(t) else np.zeros(0, bool)
    if not keep.any():
        raise ValueError(f"skip = {skip} leaves no rows of the log")
    tk = t[keep]
    out = {"rows": int(keep.sum()), "time_span": float(tk[-1] - tk[0]), "time_unit": log["time_unit"], "format": log["format"],
           "quantities": {}}
    for name, y in log["columns"].items():
        if name == "step":
            continue
        yk = np.asarray(y, float)[keep]
        mean, se = block_average(yk, blocks)
        out["quantities"][name] = {"mean": mean, "std": float(yk.std()), "se_block": se, "unit": log["units"].get(name, ""),
                                   "drift": linear_drift(tk, yk)}
    if log.get("temperature"):
        q = out["quantities"][log["temperature"]]
        out["temperature"] = {"mean": q["mean"], "std": q["std"], "se_block": q["se_block"]}
    if log.get("conserved"):
        q = out["quantities"][log["conserved"]]
        d = q["drift"]
        out["conserved_drift"] = {"slope": d["slope"], "change": d["change"], "unit": q["unit"],
                                  "relative_change": d["change"] / abs(q["mean"]) if q["mean"] else math.nan}
    return out


def _kweighted_rel(k, a, b, kweight, kmin, kmax):
    sel = (k >= kmin) & (k <= kmax)
    w = k[sel] ** kweight
    num = np.sqrt(np.sum((w * (a[sel] - b[sel])) ** 2))
    den = np.sqrt(np.sum((w * b[sel]) ** 2))
    return float(num / den) if den > 0 else math.nan


def spectrum_convergence(k, chi, weights=None, window=None, points=12):
    """How the mean spectrum of the first n configurations approaches the mean of all of them.

    Parameters
    ----------
    k : array, shape (NK,)
        Uniform k grid from zero (1/A).
    chi : array, shape (F, NK)
        Spectra of the configurations, in their order (frames in time order).
    weights : array, shape (F,), optional
        Weight of each configuration (for example its number of absorbers); default equal.
    window : feffgpu.exafs.FitWindow or dict, optional
        k window, k weight and R range of the R-factor; default kmin 3, kmax the last k minus 1, dk 1, k weight 2,
        R 1 to 6 A.
    points : int
        Number of values of n (spaced evenly on a logarithmic scale from 1 to F).

    Returns
    -------
    dict
        ``n_frames`` (list), ``r_factor`` (the R-factor of the running mean against the final mean, list),
        ``relative_difference`` (the k-weighted relative difference in the k window, list), ``split_half_r_factor``
        (the R-factor between the means of the first and the second half, NaN for fewer than two configurations),
        ``window`` (dict).
    """
    from ..exafs.transform import r_factor
    k = np.asarray(k, float)
    chi = np.asarray(chi, float)
    if chi.ndim != 2 or chi.shape[1] != len(k):
        raise ValueError("chi must be (number of configurations, len(k))")
    F = chi.shape[0]
    if F == 0:
        raise ValueError("no configuration")
    w = np.ones(F) if weights is None else np.asarray(weights, float)
    if window is None:
        win = {"kmin": 3.0, "kmax": float(k[-1]) - 1.0, "dk": 1.0, "kweight": 2, "rmin": 1.0, "rmax": 6.0}
    elif isinstance(window, dict):
        win = dict(window)
    else:
        win = window.as_dict()
    cum = np.cumsum(w[:, None] * chi, 0) / np.cumsum(w)[:, None]
    final = cum[-1]
    ns = sorted({int(round(v)) for v in np.geomspace(1, F, max(2, int(points)))} | {F})
    rf, rel = [], []
    for n in ns:
        m = cum[n - 1]
        rf.append(r_factor(k, final, m, kmin=win["kmin"], kmax=win["kmax"], rmin=win["rmin"], rmax=win["rmax"], kweight=win["kweight"],
                           dk=win["dk"]) if np.any(final) else math.nan)
        rel.append(_kweighted_rel(k, m, final, win["kweight"], win["kmin"], win["kmax"]))
    half = math.nan
    if F >= 2:
        h = F // 2
        a = (w[:h, None] * chi[:h]).sum(0) / w[:h].sum()
        b = (w[h:, None] * chi[h:]).sum(0) / w[h:].sum()
        half = r_factor(k, b, a, kmin=win["kmin"], kmax=win["kmax"], rmin=win["rmin"], rmax=win["rmax"], kweight=win["kweight"],
                        dk=win["dk"])
    return {"n_frames": ns, "r_factor": [float(v) for v in rf], "relative_difference": rel, "split_half_r_factor": float(half),
            "window": win}
