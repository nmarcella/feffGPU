"""Structural and thermodynamic analysis of a trajectory: the drivers of ``md-exafs rdf`` and ``md-exafs convergence``.

``rdf_analysis`` streams the frames of ``[rdf]`` (default the frames of ``[frames]``) through
``feffgpu.analysis.rdf.RDFAccumulator`` and ``AngleAccumulator``, then characterizes each ``[[rdf.peak]]`` with
``feffgpu.analysis.peaks.analyze_peak``: the integration numbers (exact, from the distances) next to the Gaussian and
skewed fits. ``convergence_analysis`` reads the thermodynamic log (``feffgpu.analysis.convergence``) and, when the
cache exists, the convergence of every edge's averaged spectrum against the number of frames.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
from ..analysis.convergence import read_log, thermo_summary
from ..analysis.peaks import analyze_peak
from ..analysis.rdf import AngleAccumulator, RDFAccumulator
from ..parallel import primary_device
from .run import make_reader

__all__ = ["rdf_analysis", "convergence_analysis"]


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items() if not isinstance(v, np.ndarray) or v.ndim == 0}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def rdf_analysis(cfg, log=None, device=None):
    """Partial RDFs, peaks, projections and bond angles of a configuration's trajectory.

    Writes into ``<output>/rdf/``: ``rdf.json`` (the numbers), ``<center>-<neighbor>.dat`` (r, g, pairs per center per
    A, running coordination), ``projection_<i>.dat``, ``angles_<i>.dat``, ``rdf.png``, ``peaks.png`` and
    ``angles.png``.

    Parameters
    ----------
    cfg : MDExafsConfig
    log : callable, optional
    device : str, optional
        Torch device of the histograms (default ``[rdf] device`` or the first of ``[run] devices``).

    Returns
    -------
    dict
        The content of ``rdf.json``.
    """
    from .plots import plot_angles, plot_peaks, plot_rdf
    log = log or (lambda s: print(s, flush=True))
    rc = cfg.rdf
    if rc is None:
        raise ValueError("the configuration has no [rdf] section")
    dev = device or rc.device or primary_device(cfg.run.devices, quiet=True)
    reader = make_reader(cfg)
    start = rc.start if rc.start is not None else cfg.frames.start
    stop = rc.stop if rc.stop is not None else cfg.frames.stop
    step = rc.step if rc.step is not None else cfg.frames.step
    sel = reader.selection(start, stop, step)
    first = reader.read_frame(sel[0])
    shells = [(p.center, p.neighbor, p.r_min, p.r_max) for p in rc.peak]
    projections = [{"center": p.center, "neighbor": p.neighbor, "direction": p.direction, "angle": p.angle, "symmetry": p.symmetry,
                    "r_min": p.r_min, "r_max": p.r_max} for p in rc.projection]
    acc = RDFAccumulator(first.symbols, pairs=[tuple(p) for p in rc.pairs] or None, r_max=rc.r_max, bin_width=rc.bin_width, shells=shells,
                         projections=projections, device=dev)
    ang = AngleAccumulator(first.symbols, [{"center": a.center, "neighbors": a.neighbors, "ranges": a.ranges, "bin_width": a.bin_width}
                                           for a in rc.angle]) if rc.angle else None
    t0 = time.time()
    log(f"RDF of {len(sel)} frames (r_max {rc.r_max} A, bins {rc.bin_width} A, device {dev})")
    for i, fr in enumerate(reader.iter_frames(sel)):
        acc.add(fr.positions, fr.cell)
        if ang is not None:
            ang.add(fr.positions, fr.cell)
        if (i + 1) % max(1, len(sel) // 10) == 0:
            log(f"  {i + 1}/{len(sel)} frames ({time.time() - t0:.1f} s)")
    res = acc.result()
    out = Path(cfg.output_path) / "rdf"
    out.mkdir(parents=True, exist_ok=True)
    r = res["r"]
    for label, p in res["pairs"].items():
        g = p["g"] if p["g"] is not None else np.full_like(r, np.nan)
        np.savetxt(out / f"{label}.dat", np.column_stack([r, g, p["pairs_per_center_per_A"], p["coordination"]]),
                   header=f"{label}: {res['frames']} frames, {p['centers']} centers, {p['neighbors']} neighbors\n"
                          "r(A) g(r) pairs_per_center_per_A running_coordination", fmt="%.6e")
    peaks = []
    for pk, sh in zip(rc.peak, res["shells"]):
        dens = res["pairs"][f"{pk.center}-{pk.neighbor}"]["pairs_per_center_per_A"]
        an = analyze_peak(r, dens, pk.r_min, pk.r_max, pk.fit, exact=sh)
        an["label"] = f"{pk.center}-{pk.neighbor} {pk.r_min:g}-{pk.r_max:g} A"
        an["center"], an["neighbor"] = pk.center, pk.neighbor
        peaks.append(an)
        integ = an["integration"]
        log(f"  {an['label']}: N {integ['N']:.3f}, R {integ['R']:.4f} A, MSRD {integ['sigma2']:.5f} A^2, C3 {integ['c3']:.2e} A^3 "
            "(integration)")
    projs = []
    for i, p in enumerate(res["projections"]):
        g = p["g"] if p["g"] is not None else np.full_like(r, np.nan)
        np.savetxt(out / f"projection_{i}.dat", np.column_stack([r, g, p["pairs_per_center_per_A"], p["coordination"]]),
                   header=f"{p['label']}: within {p['angle']} degrees; solid-angle fraction {p['solid_angle_fraction']:.5f}\n"
                          "r(A) g(r) pairs_per_center_per_A running_coordination", fmt="%.6e")
        projs.append({"label": p["label"], "center": p["center"], "neighbor": p["neighbor"], "angle": p["angle"], "symmetry": p["symmetry"],
                      "directions": np.asarray(p["directions"]).tolist(), "solid_angle_fraction": p["solid_angle_fraction"],
                      "shell": p["shell"], "file": f"projection_{i}.dat"})
    angles = []
    if ang is not None:
        for i, a in enumerate(ang.result()):
            np.savetxt(out / f"angles_{i}.dat", np.column_stack([a["angle"], a["counts"], a["per_center"]]),
                       header=f"{a['label']} angles: {a['angles']} in all\nangle(degrees) counts per_center", fmt="%.6e")
            angles.append({"label": a["label"], "ranges": a["ranges"], "angles": a["angles"], "mean": a["mean"], "file": f"angles_{i}.dat"})
        plot_angles(ang.result(), out=out / "angles.png", title=cfg.system.name)
    marks = [{"center": p.center, "neighbor": p.neighbor, "r_min": p.r_min, "r_max": p.r_max} for p in rc.peak]
    plot_rdf(res, out=out / "rdf.png", peaks=marks, title=f"{cfg.system.name}: {res['frames']} frames")
    if peaks:
        densities = [res["pairs"][f"{p['center']}-{p['neighbor']}"]["pairs_per_center_per_A"] for p in peaks]
        plot_peaks(r, densities, peaks, out=out / "peaks.png", title=cfg.system.name)
    pairs = {label: {"centers": p["centers"], "neighbors": p["neighbors"], "file": f"{label}.dat"} for label, p in res["pairs"].items()}
    report = {"command": "rdf", "system": cfg.system.name, "frames": res["frames"], "first_frame": sel[0], "last_frame": sel[-1],
              "r_max": res["r_max"], "bin_width": res["bin_width"], "periodic": bool(acc.periodic), "pairs": pairs,
              "peaks": [_jsonable({k: v for k, v in pk.items()}) for pk in peaks], "projections": projs, "angles": angles,
              "numbers_note": "integration: from the distances themselves (preferred, nothing fitted); gaussian / skewed: least-squares "
                              "fits of the pair density with standard errors from the fit covariance",
              "seconds": round(time.time() - t0, 2)}
    for pk in report["peaks"]:
        for form in ("gaussian", "skewed"):
            if form in pk and isinstance(pk[form], dict):
                pk[form].pop("r_fit", None)
                pk[form].pop("fitted", None)
    (out / "rdf.json").write_text(json.dumps(report, indent=1, default=float))
    return report


def convergence_analysis(cfg=None, log_file=None, log_format="auto", skip=None, blocks=None, cache=None, out_folder=None, log=None):
    """The thermodynamic log and the convergence of the averaged spectra.

    Parameters
    ----------
    cfg : MDExafsConfig, optional
        Gives the log (``[convergence]`` or ``[dynamics]``), the cache (``<output>/chi.h5``) and the output folder.
    log_file : str or Path, optional
        A log file (overrides the configuration).
    log_format : str
    skip : float, optional
        Equilibration to leave out, in the log's time unit (default ``[convergence] skip`` or 0).
    blocks : int, optional
    cache : str or Path, optional
        A cache file (overrides ``<output>/chi.h5``).
    out_folder : str or Path, optional
        Where ``convergence.json`` and ``convergence.png`` go (default ``<output>/convergence``, or the log's folder).
    log : callable, optional

    Returns
    -------
    dict
        The content of ``convergence.json``.
    """
    from .average import average_cache
    from .plots import plot_convergence
    say = log or (lambda s: print(s, flush=True))
    cc = getattr(cfg, "convergence", None)
    dyn_log = cfg.dynamics.log if cfg is not None and cfg.dynamics is not None else None
    lf = log_file or (cc.log if cc is not None and cc.log else None) or dyn_log
    fmt = log_format if log_format != "auto" or cc is None else cc.format
    skip = skip if skip is not None else (cc.skip if cc is not None else 0.0)
    blocks = blocks if blocks is not None else (cc.blocks if cc is not None else 10)
    report = {"command": "convergence"}
    lg, thermo = None, None
    if lf:
        lg = read_log(lf, fmt)
        thermo = thermo_summary(lg, skip=skip, blocks=blocks)
        report["log"] = {"file": str(lf), **thermo}
        if "temperature" in thermo:
            t = thermo["temperature"]
            say(f"temperature {t['mean']:.1f} K (standard deviation {t['std']:.1f} K, block standard error {t['se_block']:.2f} K) over "
                f"{thermo['time_span']:.4g} {thermo['time_unit']}")
        if "conserved_drift" in thermo:
            d = thermo["conserved_drift"]
            say(f"conserved quantity drift: {d['change']:.4g} {d['unit']} over the span (relative {d['relative_change']:.2e})")
    spectra = None
    cpath = Path(cache) if cache else (Path(cfg.output_path) / "chi.h5" if cfg is not None else None)
    if cpath is not None and cpath.is_file():
        windows = {c.edge: c.window.model_dump() for c in cfg.compare} if cfg is not None else {}
        res = average_cache(cpath, cfg.average if cfg is not None else None, windows)
        spectra = {name: r["convergence"] for name, r in res.items()}
        report["spectra"] = {name: {"n_frames": c["n_frames"], "r_factor_to_final": c["r_factor"],
                                    "relative_difference_to_final": c["relative_difference"],
                                    "split_half_r_factor": c["split_half_r_factor"], "window": c["window"]} for name, c in spectra.items()}
        for name, c in spectra.items():
            say(f"[{name}] R-factor between the averages of the first and second half of the frames: {c['split_half_r_factor']:.3g}")
    if lg is None and spectra is None:
        raise ValueError("nothing to analyze: give a log file ([convergence] log, or --log) or run md-exafs run first so that chi.h5 "
                         "exists")
    folder = Path(out_folder) if out_folder else (Path(cfg.output_path) / "convergence" if cfg is not None else Path(lf).parent)
    folder.mkdir(parents=True, exist_ok=True)
    plot_convergence(spectrum=spectra, thermo=thermo, log=lg, out=folder / "convergence.png",
                     title=(cfg.system.name if cfg is not None else Path(lf).name))
    report["figure"] = "convergence.png"
    (folder / "convergence.json").write_text(json.dumps(_jsonable(report), indent=1, default=float))
    return report
