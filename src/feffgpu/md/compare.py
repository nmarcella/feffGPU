"""Comparison of averaged MD-EXAFS spectra with measured ones.

The averaged spectrum of an edge (``[average]``: frames and path types, from the cache) is compared with the measured
spectrum of a ``[compare]`` block through ``feffgpu.exafs.SpectrumComparison``. S0^2 and the E0 shift act on the
model (xraylarch's convention): ``model(k) = S0^2 chi(sqrt(k^2 - ETOK dE0))``. They are fitted with
``feffgpu.exafs.fit_shared``, one value per edge name, shared between all the runs compared together (several
trajectories or temperatures of one material), and their intervals come from ``feffgpu.exafs.shared_intervals``.

The rule of this project: the ensemble of frames is scored against the measured spectrum as it is. No Debye-Waller
factors are fitted; the only free numbers are S0^2 and the shift of each edge.

Two uncertainties are reported. The intervals of ``shared_intervals`` come from the curvature of the misfit with the
averaged spectrum held fixed. The sampling error of the finite ensemble is not in them; it is estimated separately by
a jackknife over blocks of consecutive frames: the fit is repeated with each block left out (in every run at once),
and the spread of the results gives a standard error of S0^2 and of the shift.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from ..exafs import (FitWindow, SpectrumComparison, check_window_inside_data, fit_shared, plot_comparison, read_chi, shared_intervals,
                     to_grid)
from ..parallel import primary_device
from .average import average_cache

__all__ = ["build_comparisons", "compare_runs"]


def build_comparisons(cfg, output, device="cpu"):
    """The comparisons of one run: ``[(SpectrumComparison, averaged chi (NK,), compare settings)]`` for every
    ``[compare]`` block, with the averaged spectra read from the run's cache.

    Parameters
    ----------
    cfg : MDExafsConfig
    output : Path
        The run's output folder (holds ``chi.h5``).
    device : str
    """
    if not cfg.compare:
        raise ValueError(f"{cfg.system.name}: the configuration has no [compare] block")
    windows = {c.edge: FitWindow(**c.window.model_dump()) for c in cfg.compare}
    results = average_cache(Path(output) / "chi.h5", cfg.average, windows)
    out = []
    for c in cfg.compare:
        if c.edge not in results:
            raise ValueError(f"[compare] edge {c.edge!r} is not among the averaged edges {list(results)} ([average] edges)")
        r = results[c.edge]
        k = r["k"]
        w = FitWindow(**c.window.model_dump())
        if k[-1] < w.kmax + w.dk / 2 - 1e-9:
            raise ValueError(f"[compare] {c.edge}: the window reaches {w.kmax + w.dk / 2:.2f} 1/A (sill included) but the spectrum was "
                             f"calculated to kmax {k[-1]:.2f} 1/A; raise [[edge]] kmax or lower the window's kmax")
        kd, cd = read_chi(c.chi, c.k_column, c.chi_column)
        check_window_inside_data(kd, w.kmin, w.kmax, w.dk, name=Path(c.chi).name)
        cmp = SpectrumComparison(c.edge, k, to_grid(k, kd, cd), w, s02=c.s02, e0=0.0, fit_s02=c.fit_s02, fit_e0=c.fit_e0, device=device)
        cmp.set_de0(c.e0)
        out.append((cmp, r["mean"], c))
    return out, results


def _jackknife(meta, blocks, bounds, device):
    """Standard errors of the shared S0^2 and shifts from refits with each block of consecutive frames left out."""
    n_frames = min(len(res[c.edge]["frames"]) for _l, _cfg, _o, comps, res in meta for _cmp, _chi, c in comps)
    B = int(min(blocks, n_frames))
    if B < 2:
        return None
    s02, de0 = {}, {}
    for b in range(B):
        groups = []
        for _label, _cfg, _out, comps, res in meta:
            group = []
            for cmp, _chi, c in comps:
                frames, w = res[c.edge]["chi_frames"], res[c.edge]["weights"]
                keep = np.ones(len(w), bool)
                keep[np.array_split(np.arange(len(w)), B)[b]] = False
                chi_b = (w[keep, None] * frames[keep]).sum(0) / w[keep].sum()
                cb = SpectrumComparison(c.edge, cmp.k, cmp.data, cmp.fit_window, s02=c.s02, e0=0.0, fit_s02=c.fit_s02, fit_e0=c.fit_e0,
                                        device=device)
                cb.set_de0(c.e0)
                group.append((cb, torch.as_tensor(chi_b, device=device, dtype=torch.float32), 1.0))
            groups.append(group)
        f = fit_shared(groups, s02_bounds=bounds[0], de0_bounds=bounds[1])
        for n, v in f["s02"].items():
            s02.setdefault(n, []).append(v)
        for n, v in f["de0"].items():
            de0.setdefault(n, []).append(v)

    def se(vals):
        v = np.asarray(vals, float)
        return float(np.sqrt((len(v) - 1) / len(v) * ((v - v.mean()) ** 2).sum()))
    return {"blocks": B, "s02_se": {n: se(v) for n, v in s02.items()}, "de0_se": {n: se(v) for n, v in de0.items()},
            "method": f"jackknife over {B} blocks of consecutive frames: the fit repeated with each block left out (in every run at once); "
                      "the standard error of the finite ensemble's sampling, which the curvature intervals do not include"}


def compare_runs(runs, out_folder, device=None, log=None):
    """Compare one or several runs with their measured spectra, S0^2 and the E0 shift shared by edge name.

    Parameters
    ----------
    runs : list of tuple
        ``(MDExafsConfig, Path)``: each run's configuration and output folder.
    out_folder : str or Path
        Where ``compare.json`` and the figures are written.
    device : str, optional
        Torch device of the fit (default the first run's device setting).
    log : callable, optional

    Returns
    -------
    dict
        The content of ``compare.json``: per run and spectrum the R-factor, S0^2, the shift, the number of independent
        points; the shared values with their intervals and how the intervals were obtained.
    """
    log = log or (lambda s: print(s, flush=True))
    device = device or primary_device(runs[0][0].run.devices, quiet=True)
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    groups, meta = [], []
    lo_s, hi_s, lo_e, hi_e = [], [], [], []
    labels = []
    blocks = []
    for cfg, output in runs:
        comps, results = build_comparisons(cfg, output, device)
        label = cfg.system.name
        if label in labels:
            label = f"{label} ({Path(output).name})"
        labels.append(label)
        groups.append([(cmp, torch.as_tensor(np.asarray(chi), device=device, dtype=torch.float32), 1.0) for cmp, chi, _c in comps])
        meta.append((label, cfg, output, comps, results))
        blocks.append(cfg.average.blocks)
        for _cmp, _chi, c in comps:
            lo_s.append(c.s02_bounds[0]); hi_s.append(c.s02_bounds[1]); lo_e.append(c.e0_bounds[0]); hi_e.append(c.e0_bounds[1])
    bounds = ((min(lo_s), max(hi_s)), (min(lo_e), max(hi_e)))
    fit = fit_shared(groups, s02_bounds=bounds[0], de0_bounds=bounds[1])
    iv = shared_intervals(groups)
    sampling = _jackknife(meta, min(blocks), bounds, device)
    shared = {"s02": fit["s02"], "de0": fit["de0"], "fitted_s02": fit["s02_names"], "fitted_de0": fit["de0_names"],
              "at_bound": fit["at_bound"],
              "s02_interval": {n: [iv["s02_lo"][n], iv["s02_hi"][n]] for n in iv["s02_lo"]},
              "de0_interval": {n: [iv["de0_lo"][n], iv["de0_hi"][n]] for n in iv["de0_lo"]},
              "s02_sd_log": iv["s02_sd_log"], "de0_sd": iv["de0_sd"], "correlation_de0_s02": iv.get("de0_s02_correlation", {}),
              "positive_definite": iv["positive_definite"], "z": iv["z"], "interval_method": iv["method"], "objective": fit["residual"],
              "sampling": sampling}
    report = {"command": "compare", "rule": "each averaged spectrum (an ensemble of frames) is scored against the measured spectrum; no "
                                            "Debye-Waller factor is fitted; S0^2 and the E0 shift act on the model, one value per edge "
                                            "name", "runs": [], "shared": shared}
    if fit["at_bound"]:
        log(f"warning: {', '.join(fit['at_bound'])} ended on a bound of the allowed range: that value is not a fit")
    for n in fit["s02"]:
        extra = f"; sampling standard error {sampling['s02_se'][n]:.3g} (jackknife over {sampling['blocks']} blocks)" if sampling else ""
        log(f"{n}: S0^2 {fit['s02'][n]:.4f}, {iv['z']:g}-sigma interval {iv['s02_lo'][n]:.4f} to {iv['s02_hi'][n]:.4f} (structure held "
            f"fixed){extra}")
    for n in fit["de0"]:
        extra = f"; sampling standard error {sampling['de0_se'][n]:.3g} eV" if sampling else ""
        log(f"{n}: dE0 {fit['de0'][n]:.3f} eV, interval {iv['de0_lo'][n]:.3f} to {iv['de0_hi'][n]:.3f} eV{extra}")
    for (label, cfg, output, comps, _res), ratios in zip(meta, fit["ratios"]):
        rows = []
        for (cmp, chi, c), rf in zip(comps, ratios):
            rows.append({"edge": c.edge, "label": c.label or Path(c.chi).name, "chi_file": str(c.chi), "window": c.window.model_dump(),
                         "r_factor": float(rf), "s02": cmp.s02, "s02_fitted": cmp.fit_s02, "de0": cmp.de0, "de0_fitted": cmp.fit_e0,
                         "independent_points": float(cmp.n_ind), "description": cmp.describe()})
            log(f"{label}: {cmp.describe()}; R-factor {rf:.5f}")
        fig_name = "compare.png" if len(meta) == 1 else f"compare_{''.join(ch if ch.isalnum() else '_' for ch in label)}.png"
        plot_comparison([cmp for cmp, _chi, _c in comps], [{cmp.name: chi for cmp, chi, _c in comps}], labels=[f"MD average ({label})"],
                        out=out_folder / fig_name, title=f"{label}: averaged spectrum against the measured one")
        report["runs"].append({"run": label, "output": str(output), "spectra": rows, "figure": fig_name})
    (out_folder / "compare.json").write_text(json.dumps(report, indent=1))
    return report
