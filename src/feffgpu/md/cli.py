"""The ``md-exafs`` command (also ``feffgpu md``).

::

    md-exafs run config.toml [--shard 3/16] [--potentials-only]   # trajectory -> cache of per-frame spectra (then average, compare)
    md-exafs merge OUTPUT                                         # shard files -> chi.h5
    md-exafs average CONFIG_OR_OUTPUT                             # averages from the cache ([average])
    md-exafs compare RUN [RUN ...]                                # against measured spectra, S0^2 and E0 shared by edge name
    md-exafs rdf config.toml                                      # radial distributions, peaks, projections, angles ([rdf])
    md-exafs convergence CONFIG_OR_OUTPUT_OR_LOG                  # thermodynamic log, convergence of the averages
    md-exafs plot CONFIG_OR_OUTPUT                                # figures of the averages and their convergence
    md-exafs validate config.toml                                 # check a configuration without running it
    md-exafs schema                                               # the JSON Schema of the configuration
    md-exafs example [NAME] [--out FILE]                          # the example configurations
    md-exafs supercell structure.cif --size 3,3,3 --out cell.xyz  # a supercell for a dynamics program (ASE)
    md-exafs dynamics config.toml                                 # the optional runner ([dynamics], ASE calculator)

Every command that prints a result accepts ``--json``. A user error ends the command with one message and exit code 2
(``validate`` exits with 1 when the file has problems).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

__all__ = ["main"]


class _UserError(Exception):
    pass


def _printer(as_json):
    stream = sys.stderr if as_json else sys.stdout
    return lambda s: print(s, file=stream, flush=True)


def _emit(obj, as_json, text=None):
    if as_json:
        print(json.dumps(obj, indent=1, default=float), flush=True)
    elif text:
        print(text, flush=True)


def _cmd_run(a):
    from .config import load_config
    from .run import run
    log = _printer(a.json)
    cfg = load_config(a.config)
    summary = run(cfg, shard=a.shard or None, log=log, potentials_only=a.potentials_only)
    if not a.potentials_only and not a.shard and not a.no_analysis:
        summary["average"] = _average(cfg, cfg.output_path, log)
        if cfg.compare:
            from .compare import compare_runs
            summary["compare"] = compare_runs([(cfg, cfg.output_path)], cfg.output_path / "compare", log=log)
        (cfg.output_path / "report.json").write_text(json.dumps(summary, indent=1, default=float))
    _emit(summary, a.json, f"output: {summary['output']}")
    return 0


def _average(cfg, output, log, cache=None, folder=None):
    from ..exafs import FitWindow
    from .average import average_cache, write_average
    from .plots import plot_average
    windows = {c.edge: FitWindow(**c.window.model_dump()) for c in cfg.compare}
    cache = Path(cache) if cache else Path(output) / "chi.h5"
    res = average_cache(cache, cfg.average, windows)
    folder = Path(folder) if folder else Path(output) / "average"
    summary = write_average(folder, res, cfg.system.name, cache)
    for name, r in res.items():
        w = windows.get(name)
        safe = summary["edges"][name]["files"]["chi"][:-4]
        plot_average(r, out=folder / f"{safe}.png", window=w.as_dict() if w else None,
                     title=f"{cfg.system.name}: {name}, {r['n_frames']} frames")
        summary["edges"][name]["files"]["figure"] = f"{safe}.png"
        chi_file = folder / summary["edges"][name]["files"]["chi"]
        log(f"[{name}] average of {r['n_frames']} frames written to {chi_file}; R-factor between the halves "
            f"{r['convergence']['split_half_r_factor']:.3g}")
    (folder / "average.json").write_text(json.dumps(summary, indent=1))
    return summary


def _cmd_merge(a):
    from .cache import merge_shards
    from .config import output_directory
    log = _printer(a.json)
    folder = output_directory(a.output) if Path(a.output).suffix.lower() == ".toml" else Path(a.output)
    res = merge_shards(folder, force=a.force, log=log)
    _emit(res, a.json, f"run `md-exafs average {a.output}` for the averages")
    return 0


def _cmd_average(a):
    from .run import load_run
    log = _printer(a.json)
    cfg, out = load_run(a.path)
    folder = Path(a.out) if a.out else Path(out) / "average"
    summary = _average(cfg, out, log, cache=a.cache, folder=folder)
    _emit(summary, a.json, f"averages in {folder}")
    return 0


def _cmd_compare(a):
    from .compare import compare_runs
    from .run import load_run
    log = _printer(a.json)
    runs = [load_run(p) for p in a.runs]
    folder = Path(a.out) if a.out else (runs[0][1] / ("compare" if len(runs) == 1 else "compare_joint"))
    rep = compare_runs(runs, folder, device=a.device, log=log)
    _emit(rep, a.json, f"comparison in {folder}")
    return 0


def _cmd_rdf(a):
    from .analyze import rdf_analysis
    from .run import load_run
    log = _printer(a.json)
    cfg, _out = load_run(a.config)
    rep = rdf_analysis(cfg, log=log, device=a.device)
    _emit(rep, a.json, f"RDFs in {Path(cfg.output_path) / 'rdf'}")
    return 0


def _cmd_convergence(a):
    from .analyze import convergence_analysis
    from .run import load_run
    log = _printer(a.json)
    p = Path(a.path)
    opts = dict(log_format=a.format, skip=a.skip, blocks=a.blocks, cache=a.cache, out_folder=a.out, log=log)
    if p.is_dir() or p.suffix.lower() == ".toml":
        cfg, _out = load_run(p)
        rep = convergence_analysis(cfg, log_file=a.log, **opts)
    else:
        rep = convergence_analysis(None, log_file=p, **opts)
    _emit(rep, a.json, "convergence written")
    return 0


def _cmd_plot(a):
    from ..exafs import FitWindow
    from .average import average_cache
    from .plots import plot_average, plot_convergence
    from .run import load_run
    cfg, out = load_run(a.path)
    folder = Path(a.out) if a.out else Path(out) / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    windows = {c.edge: FitWindow(**c.window.model_dump()) for c in cfg.compare}
    res = average_cache(Path(out) / "chi.h5", cfg.average, windows)
    files = []
    for name, r in res.items():
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
        w = windows.get(name)
        plot_average(r, out=folder / f"average_{safe}.png", window=w.as_dict() if w else None, title=f"{cfg.system.name}: {name}")
        files.append(str(folder / f"average_{safe}.png"))
    plot_convergence(spectrum={n: r["convergence"] for n, r in res.items()}, out=folder / "convergence.png", title=cfg.system.name)
    files.append(str(folder / "convergence.png"))
    _emit({"command": "plot", "figures": files}, a.json, "\n".join(files))
    return 0


def _cmd_validate(a):
    from .config import validate_file
    problems = validate_file(a.config)
    if a.json:
        print(json.dumps({"path": str(a.config), "valid": not problems, "problems": problems}, indent=1))
    elif problems:
        print("\n".join(problems))
    else:
        print(f"{a.config}: valid")
    return 1 if problems else 0


def _cmd_schema(a):
    from .config import json_schema
    print(json.dumps(json_schema(), indent=1))
    return 0


def _cmd_example(a):
    from .config import example_configs
    ex = example_configs()
    if not a.name:
        _emit({"examples": sorted(ex)}, a.json, "\n".join(sorted(ex)))
        return 0
    if a.name not in ex:
        raise _UserError(f"no example {a.name!r}; available: {', '.join(sorted(ex))}")
    if a.out:
        Path(a.out).write_text(ex[a.name])
        _emit({"example": a.name, "written": a.out}, a.json, f"written {a.out}")
    else:
        sys.stdout.write(ex[a.name])
    return 0


def _cmd_supercell(a):
    import numpy as np
    import ase.io
    from ase.build import make_supercell
    try:
        size = [int(v) for v in a.size.split(",")]
    except ValueError as err:
        raise _UserError(f"--size is three positive integers like 3,3,3 (got {a.size!r})") from err
    if len(size) != 3 or any(v < 1 for v in size):
        raise _UserError(f"--size is three positive integers like 3,3,3 (got {a.size!r})")
    if not Path(a.input).is_file():
        raise FileNotFoundError(f"no such structure file: {a.input}")
    atoms = make_supercell(ase.io.read(a.input), np.diag(size))
    if a.cp2k:
        with open(a.out, "w") as f:
            for s, p in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
                f.write(f"{s}  {p[0]:24.16E}  {p[1]:24.16E}  {p[2]:24.16E}\n")
    else:
        ase.io.write(a.out, atoms, format=a.format or None)
    cell = atoms.cell.array
    res = {"command": "supercell", "input": a.input, "output": a.out, "size": size, "atoms": len(atoms), "cell": cell.tolist(),
           "format": "cp2k (element x y z, no header)" if a.cp2k else (a.format or "from the file name")}
    _emit(res, a.json, f"{len(atoms)} atoms written to {a.out}; cell lengths {', '.join(f'{v:.6f}' for v in atoms.cell.lengths())} A")
    return 0


def _cmd_dynamics(a):
    from .config import load_config
    from .dynamics import run_dynamics
    cfg = load_config(a.config)
    res = run_dynamics(cfg, log=_printer(a.json), device=a.device)
    _emit(res, a.json, f"trajectory: {res['trajectory']}")
    return 0


def _parser():
    ap = argparse.ArgumentParser(prog="md-exafs", description="MD-EXAFS: the EXAFS spectrum of a molecular dynamics trajectory, averaged "
                                 "over frames and absorbing atoms, computed with feffgpu's GPU forward model.")
    sub = ap.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def cmd(name, help_, func):
        p = sub.add_parser(name, help=help_, description=help_)
        p.set_defaults(func=func)
        return p

    p = cmd("run", "compute the spectrum of every selected frame and write the cache (then the averages and the comparison)", _cmd_run)
    p.add_argument("config", help="the TOML configuration")
    p.add_argument("--shard", default="", help="compute only this part of the frames, INDEX/COUNT (for example 3/16)")
    p.add_argument("--potentials-only", action="store_true", help="compute the phase files of every edge and stop (before a job array)")
    p.add_argument("--no-analysis", action="store_true", help="do not write the averages and the comparison after a whole run")
    p.add_argument("--json", action="store_true", help="print the summary as JSON (progress lines go to standard error)")
    p = cmd("merge", "combine the shard files of an output folder into chi.h5", _cmd_merge)
    p.add_argument("output", help="the output folder of the run, or its configuration file")
    p.add_argument("--force", action="store_true", help="overwrite a chi.h5 that is not itself a merged file")
    p.add_argument("--json", action="store_true")
    p = cmd("average", "averaged spectra, spread, standard errors and convergence from the cache ([average])", _cmd_average)
    p.add_argument("path", help="the TOML configuration or the run's output folder")
    p.add_argument("--cache", default=None, help="a cache file to use instead of OUTPUT/chi.h5")
    p.add_argument("--out", default=None, help="the folder of the averages (default OUTPUT/average)")
    p.add_argument("--json", action="store_true")
    p = cmd("compare", "compare averaged spectra with measured ones; S0^2 and the E0 shift are shared by edge name between the runs",
            _cmd_compare)
    p.add_argument("runs", nargs="+", help="configurations or output folders of the runs")
    p.add_argument("--out", default=None, help="the folder of the comparison (default OUTPUT/compare of the first run)")
    p.add_argument("--device", default=None, help="torch device of the fit (default the first run's device)")
    p.add_argument("--json", action="store_true")
    p = cmd("rdf", "radial distribution functions, peaks, projections and bond angles of the trajectory ([rdf])", _cmd_rdf)
    p.add_argument("config", help="the TOML configuration")
    p.add_argument("--device", default=None, help="torch device of the histograms (cpu works)")
    p.add_argument("--json", action="store_true")
    p = cmd("convergence", "statistics of the thermodynamic log and convergence of the averaged spectra", _cmd_convergence)
    p.add_argument("path", help="a configuration, an output folder, or a log file (CP2K .ener, ASE log, CSV)")
    p.add_argument("--log", default=None, help="a log file (overrides the configuration)")
    p.add_argument("--format", default="auto", choices=["auto", "cp2k", "ase", "csv"])
    p.add_argument("--skip", type=float, default=None, help="equilibration to leave out, in the log's time unit")
    p.add_argument("--blocks", type=int, default=None, help="number of blocks of the block averages")
    p.add_argument("--cache", default=None, help="a cache file (default OUTPUT/chi.h5)")
    p.add_argument("--out", default=None, help="the folder of the results")
    p.add_argument("--json", action="store_true")
    p = cmd("plot", "figures of the averaged spectra and their convergence", _cmd_plot)
    p.add_argument("path", help="the TOML configuration or the run's output folder")
    p.add_argument("--out", default=None, help="the folder of the figures (default OUTPUT/figures)")
    p.add_argument("--json", action="store_true")
    p = cmd("validate", "check a configuration without running anything", _cmd_validate)
    p.add_argument("config")
    p.add_argument("--json", action="store_true")
    cmd("schema", "print the JSON Schema of the configuration", _cmd_schema)
    p = cmd("example", "list the example configurations, or print or write one", _cmd_example)
    p.add_argument("name", nargs="?", default="")
    p.add_argument("--out", default=None, help="write the example to this file")
    p.add_argument("--json", action="store_true")
    p = cmd("supercell", "build a supercell of a structure file (CIF, POSCAR, ...) with ASE, for a dynamics program", _cmd_supercell)
    p.add_argument("input", help="the structure file")
    p.add_argument("--size", required=True, help="repetitions along a, b, c, for example 3,3,3")
    p.add_argument("--out", required=True, help="output file")
    p.add_argument("--format", default=None, help="ASE's output format (default from the file name)")
    p.add_argument("--cp2k", action="store_true", help="write CP2K's plain coordinate format (element x y z, no header)")
    p.add_argument("--json", action="store_true")
    p = cmd("dynamics", "run the optional dynamics of [dynamics] with an ASE calculator and write the trajectory", _cmd_dynamics)
    p.add_argument("config")
    p.add_argument("--device", default=None)
    p.add_argument("--json", action="store_true")
    return ap


def main(argv=None):
    """Entry point of ``md-exafs``; returns the exit code."""
    from .config import ConfigError
    ap = _parser()
    args = ap.parse_args(list(sys.argv[1:] if argv is None else argv))
    as_json = bool(getattr(args, "json", False))
    try:
        return int(args.func(args) or 0)
    except ConfigError as err:
        msg = "the configuration has problems:\n  " + "\n  ".join(err.problems)
        problems = err.problems
    except (_UserError, ValueError, FileNotFoundError, ImportError, KeyError, RuntimeError) as err:
        msg = err.args[0] if isinstance(err, KeyError) and err.args else str(err)
        problems = [msg]
    if as_json:
        print(json.dumps({"error": msg, "problems": problems}, indent=1), flush=True)
    print(f"md-exafs: {msg}", file=sys.stderr, flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())
