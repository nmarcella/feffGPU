"""Averages of the cached spectra, computed from the cache alone (no new forward calculation).

For one edge and a selection of frames (``[average] start, stop, step``) and of path types (``[average] path_types``,
``max_distance``):

- the mean spectrum over frames, each frame weighted by its number of absorbers;
- the spread between frames (the weighted standard deviation of the frames' spectra);
- the standard error of the mean from blocks of consecutive frames (frames of a trajectory are correlated; the plain
  standard error would be too small), and the plain one for comparison;
- the convergence of the mean against the number of frames (``feffgpu.analysis.convergence.spectrum_convergence``).

A path type without a distance limit is read from the stored parts by type. A single-scattering type with a limit is
recomputed from the stored distance histogram and the stored two-leg tables (exact up to the 0.002 A table step); a
multiple-scattering type with a limit adds the stored effective-length bins below it, so the limit must be a multiple
of the edge's ``ms_bin_width`` (or at least ``rpath``).
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
from .cache import edge_names, open_cache, read_rows
from .paths import path_type_name

__all__ = ["frame_rows", "resolve_selection", "selected_spectra", "weighted_statistics", "edge_average", "average_cache", "write_average"]


def _read(ds, rows):
    """Rows of an h5py dataset in any order (h5py wants increasing indices)."""
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) == 0:
        return np.zeros((0,) + ds.shape[1:], dtype=ds.dtype)
    o = np.argsort(rows, kind="stable")
    out = np.empty((len(rows),) + ds.shape[1:], dtype=ds.dtype)
    out[o] = ds[rows[o]]
    return out


def frame_rows(g, n_rows, start=None, stop=None, step=1):
    """The rows of an edge group to average, in frame order, and their frame indices.

    Parameters
    ----------
    g : h5py.Group
    n_rows : int
        Completed rows of the file.
    start, stop : int, optional
        Range of trajectory frame indices (``start <= index < stop``).
    step : int
        Every step-th frame of those.

    Returns
    -------
    rows : numpy.ndarray of int
    frames : numpy.ndarray of int
    """
    order, fi = read_rows(g, n_rows)
    keep = np.ones(len(fi), bool)
    if start is not None:
        keep &= fi >= int(start)
    if stop is not None:
        keep &= fi < int(stop)
    rows, frames = order[keep][:: int(step)], fi[keep][:: int(step)]
    if len(rows) == 0:
        held = f"{fi.min()} to {fi.max()}" if len(fi) else "none"
        raise ValueError(f"no cached frame in the range start {start}, stop {stop} (the cache holds frames {held})")
    return rows, frames


def resolve_selection(g, path_types=None, max_distance=None):
    """The path types of a selection in canonical spelling, each with its distance limit.

    Only the types whose first element is this edge's absorbing element are used, so one ``[average]`` list can name
    the types of several edges; an edge with none of its types in the list keeps its total spectrum.

    Returns
    -------
    list of dict
        ``name``, ``kind`` (``"ss"`` or ``"ms"``), ``index`` (in the edge's single- or multiple-scattering types),
        ``max_distance`` (A or None), ``applied`` (a sentence on what was applied).
    """
    names = json.loads(g.attrs["path_types"])
    ss, ms = json.loads(g.attrs["ss_types"]), json.loads(g.attrs["ms_types"])
    rpath = float(g.attrs["rpath"])
    edges = g["ms_bin_edges"][:]
    if not path_types:
        return []
    if isinstance(max_distance, (list, tuple)):
        limits = [float(v) for v in max_distance]
    else:
        limits = [None if max_distance is None else float(max_distance)] * len(path_types)
    element = str(g.attrs["element"])
    pairs = [(n, d) for n, d in zip(path_types, limits) if str(n).split("-")[0].strip() == element]
    out = []
    for name, d in pairs:
        parts = [p.strip() for p in str(name).split("-")]
        canon = path_type_name(parts[0], parts[1:]) if len(parts) >= 2 else name
        if canon not in names:
            raise ValueError(f"path type {name!r} is not a type of edge {g.name.split('/')[-1]}; its types are: {', '.join(names)}")
        kind = "ss" if canon in ss else "ms"
        idx = ss.index(canon) if kind == "ss" else ms.index(canon)
        if d is not None and d >= rpath - 1e-9:
            d = None
        applied = "all distances up to rpath"
        if d is not None and kind == "ss":
            applied = f"pairs up to {d:g} A"
        elif d is not None:
            j = int(np.argmin(np.abs(edges - d)))
            if abs(edges[j] - d) > 1e-6:
                width = float(edges[1] - edges[0]) if len(edges) > 1 else rpath
                raise ValueError(f"max_distance {d:g} A of the multiple-scattering type {canon} is not a boundary of the stored length "
                                 f"bins (multiples of {width:g} A up to {rpath:g} A); use a boundary, or run again with a smaller "
                                 "ms_bin_width")
            applied = f"paths with effective length below {d:g} A"
        out.append({"name": canon, "kind": kind, "index": idx, "max_distance": d, "applied": applied})
    return out


def selected_spectra(g, rows, items, chunk=256):
    """Per-frame spectra of the selected parts and of their sum.

    Parameters
    ----------
    g : h5py.Group
    rows : array of int
    items : list of dict
        From ``resolve_selection``; empty for the total spectrum.
    chunk : int
        Frames read at a time.

    Returns
    -------
    total : numpy.ndarray, shape (F, NK)
    parts : dict
        ``{name: (F, NK)}`` of the selected types (empty for the total).
    """
    rows = np.asarray(rows, dtype=np.int64)
    if not items:
        return _read(g["chi"], rows).astype(np.float64), {}
    NK = g["k"].shape[0]
    r_table = g["r_table"][:]
    edges = g["ms_bin_edges"][:]
    tables = g["ss_tables"][:].astype(np.float64) if any(it["kind"] == "ss" and it["max_distance"] is not None for it in items) else None
    n_ss = len(json.loads(g.attrs["ss_types"]))
    parts = {it["name"]: np.zeros((len(rows), NK)) for it in items}
    for s in range(0, len(rows), chunk):
        rr = rows[s:s + chunk]
        types = _read(g["chi_types"], rr).astype(np.float64)
        need_hist = any(it["kind"] == "ss" and it["max_distance"] is not None for it in items)
        need_bins = any(it["kind"] == "ms" and it["max_distance"] is not None for it in items)
        hist = _read(g["ss_histogram"], rr).astype(np.float64) if need_hist else None
        groups = _read(g["phase_group"], rr) if need_hist else None
        bins = _read(g["ms_binned"], rr).astype(np.float64) if need_bins else None
        for it in items:
            out = parts[it["name"]]
            if it["max_distance"] is None:
                out[s:s + len(rr)] = types[:, it["index"] if it["kind"] == "ss" else n_ss + it["index"]]
            elif it["kind"] == "ss":
                cols = r_table <= it["max_distance"] + 1e-9
                p = it["index"]
                for gi in np.unique(groups):
                    m = groups == gi
                    out[s:s + len(rr)][m] = hist[m][:, p][:, cols] @ tables[gi, p][cols]
            else:
                nb = int(np.sum(edges[1:] <= it["max_distance"] + 1e-9))
                out[s:s + len(rr)] = bins[:, it["index"], :nb].sum(1)
    total = sum(parts.values())
    return total, parts


def weighted_statistics(chi, weights, blocks=10):
    """Weighted mean, spread between frames, and standard errors of a stack of spectra (F, NK) in frame order.

    Returns
    -------
    dict
        ``mean``, ``spread`` (weighted standard deviation between frames), ``stderr`` (from ``blocks`` blocks of
        consecutive frames: the standard deviation of the block means over the square root of the number of blocks;
        NaN with fewer than four frames), ``stderr_plain`` (spread over the square root of the effective number of
        frames), ``blocks`` (used), ``effective_frames``.
    """
    x = np.asarray(chi, float)
    w = np.asarray(weights, float)
    W = w.sum()
    mean = (w[:, None] * x).sum(0) / W
    spread = np.sqrt((w[:, None] * (x - mean) ** 2).sum(0) / W)
    neff = W ** 2 / (w ** 2).sum()
    B = int(min(blocks, len(x) // 2))
    if B >= 2:
        bm = []
        for idx in np.array_split(np.arange(len(x)), B):
            bm.append((w[idx, None] * x[idx]).sum(0) / w[idx].sum())
        bm = np.array(bm)
        stderr = bm.std(0, ddof=1) / math.sqrt(B)
    else:
        B, stderr = 0, np.full(x.shape[1], math.nan)
    return {"mean": mean, "spread": spread, "stderr": stderr, "stderr_plain": spread / math.sqrt(neff), "blocks": B,
            "effective_frames": float(neff)}


def edge_average(f, name, average_cfg=None, window=None):
    """The averaged spectrum of one edge of an open cache.

    Parameters
    ----------
    f : h5py.File
    name : str
        Edge name.
    average_cfg : feffgpu.md.config.AverageConfig, optional
        Frames and path types; default every frame and every path.
    window : dict or FitWindow, optional
        Window of the convergence R-factor.

    Returns
    -------
    dict
        ``edge``, ``k``, ``mean``, ``spread``, ``stderr``, ``stderr_plain``, ``parts`` (``{name: mean (NK,)}``; with no
        selection, every type plus ``"single scattering"`` and ``"multiple scattering"``), ``selection`` (the applied
        types and limits), ``frames`` (indices), ``n_frames``, ``absorbers`` (total), ``blocks``, ``convergence``,
        ``chi_frames`` (the per-frame spectra of the selection, F x NK, for further analysis) and ``weights`` (the
        number of absorbers of each frame).
    """
    from ..analysis.convergence import spectrum_convergence
    g = f["edges"][name]
    n_rows = int(f.attrs["n_rows"])
    a = average_cfg
    rows, frames = frame_rows(g, n_rows, getattr(a, "start", None), getattr(a, "stop", None), getattr(a, "step", 1) or 1)
    items = resolve_selection(g, getattr(a, "path_types", None), getattr(a, "max_distance", None))
    total, parts = selected_spectra(g, rows, items)
    w = _read(g["n_absorbers"], rows).astype(float)
    stats = weighted_statistics(total, w, getattr(a, "blocks", 10) or 10)
    k = g["k"][:]
    if items:
        part_means = {nm: (w[:, None] * v).sum(0) / w.sum() for nm, v in parts.items()}
    else:
        names = json.loads(g.attrs["path_types"])
        part_means = {}
        for s in range(0, len(rows), 256):
            t = _read(g["chi_types"], rows[s:s + 256]).astype(np.float64)
            ws = w[s:s + 256]
            for j, nm in enumerate(names):
                part_means[nm] = part_means.get(nm, 0) + (ws[:, None] * t[:, j]).sum(0)
        part_means = {nm: v / w.sum() for nm, v in part_means.items()}
        part_means["single scattering"] = (w[:, None] * _read(g["chi_ss"], rows)).sum(0) / w.sum()
        part_means["multiple scattering"] = (w[:, None] * _read(g["chi_ms"], rows)).sum(0) / w.sum()
    conv = spectrum_convergence(k, total, w, window=window)
    note = " (no type of [average] path_types belongs to this edge)" if getattr(a, "path_types", None) else ""
    selection = ([{"type": it["name"], "applied": it["applied"]} for it in items]
                 or [{"type": "every path", "applied": "total spectrum" + note}])
    return {"edge": name, "element": str(g.attrs["element"]), "absorption_edge": str(g.attrs["edge"]), "k": k, "mean": stats["mean"],
            "spread": stats["spread"], "stderr": stats["stderr"], "stderr_plain": stats["stderr_plain"], "blocks": stats["blocks"],
            "effective_frames": stats["effective_frames"], "parts": part_means, "selection": selection, "frames": frames,
            "n_frames": int(len(frames)), "absorbers": int(w.sum()), "convergence": conv, "chi_frames": total, "weights": w}


def average_cache(cache_path, average_cfg=None, windows=None):
    """``edge_average`` for every edge named in ``average_cfg.edges`` (default all) of a cache file.

    Parameters
    ----------
    cache_path : str or Path
    average_cfg : AverageConfig, optional
    windows : dict, optional
        ``{edge name: window}`` for the convergence R-factors.

    Returns
    -------
    dict
        ``{edge name: edge_average result}``.
    """
    with open_cache(cache_path) as f:
        names = edge_names(f)
        use = list(getattr(average_cfg, "edges", None) or names)
        bad = [n for n in use if n not in names]
        if bad:
            raise ValueError(f"edges {bad} are not in {cache_path} (it holds {names})")
        return {n: edge_average(f, n, average_cfg, (windows or {}).get(n)) for n in use}


def _fmt_cols(cols):
    return "\n".join(" ".join(f"{v: .8e}" for v in row) for row in np.column_stack(cols))


def write_average(folder, results, system_name="", cache_path=""):
    """Write the averaged spectra: ``<edge>.chi`` (k, mean, spread, stderr), ``<edge>.parts.dat`` (k and the parts) and
    ``average.json`` (the summary) into ``folder``.

    Returns
    -------
    dict
        The summary written to ``average.json``.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    summary = {"system": system_name, "cache": str(cache_path), "edges": {}}
    for name, r in results.items():
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
        fr = r["frames"]
        sel = "; ".join(f"{s['type']} ({s['applied']})" for s in r["selection"])
        head = [f"MD-EXAFS averaged spectrum: system {system_name}, edge {name} ({r['element']} {r['absorption_edge']} edge)",
                f"frames: {r['n_frames']} (trajectory frames {int(fr[0])} to {int(fr[-1])}), {r['absorbers']} absorbers in all; each frame "
                "weighted by its number of absorbers",
                f"paths: {sel}",
                "chi: FEFF's convention (S0^2 = 1, no Debye-Waller factor; the disorder is in the frames)",
                f"spread: standard deviation between frames; stderr: standard error of the mean from {r['blocks']} blocks of "
                "consecutive frames",
                "k(1/A) chi spread stderr"]
        table = _fmt_cols([r["k"], r["mean"], r["spread"], r["stderr"]])
        (folder / f"{safe}.chi").write_text("".join(f"# {h}\n" for h in head) + table + "\n")
        pnames = list(r["parts"])
        (folder / f"{safe}.parts.dat").write_text(f"# MD-EXAFS parts of the averaged spectrum: system {system_name}, edge {name}\n"
                                                  f"# columns: k(1/A) | {' | '.join(pnames)}\n"
                                                  + _fmt_cols([r["k"]] + [r["parts"][p] for p in pnames]) + "\n")
        c = r["convergence"]
        summary["edges"][name] = {"element": r["element"], "edge": r["absorption_edge"], "n_frames": r["n_frames"],
                                  "first_frame": int(fr[0]), "last_frame": int(fr[-1]), "absorbers": r["absorbers"],
                                  "selection": r["selection"], "blocks": r["blocks"], "effective_frames": r["effective_frames"],
                                  "files": {"chi": f"{safe}.chi", "parts": f"{safe}.parts.dat"}, "parts": pnames,
                                  "convergence": {"n_frames": c["n_frames"], "r_factor_to_final": c["r_factor"],
                                                  "relative_difference_to_final": c["relative_difference"],
                                                  "split_half_r_factor": c["split_half_r_factor"], "window": c["window"]},
                                  "stderr_method": f"standard deviation of the means of {r['blocks']} blocks of consecutive frames "
                                                   "over the square root of the number of blocks (frames of a trajectory are "
                                                   "correlated)"}
    (folder / "average.json").write_text(json.dumps(summary, indent=1))
    return summary
