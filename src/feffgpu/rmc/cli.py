"""The command ``rmc-exafs``: reverse Monte Carlo fitting of EXAFS spectra, one TOML file per run.

    rmc-exafs run config.toml              the population search under EvAX's rules
    rmc-exafs refine config.toml           the gradient refinement (one sample, one edge, one element)
    rmc-exafs validate config.toml         check a configuration without running (no GPU)
    rmc-exafs schema                       the JSON Schema of the configuration
    rmc-exafs example [NAME]               the example configurations that ship with the package
    rmc-exafs shells RUN [RUN ...] --shell Zn-C:1.8:2.6     shell numbers with their spread over states and runs
    rmc-exafs short-pairs RUN              pairs closer than the closest distances
    rmc-exafs rdf RUN --center Zn          partial pair distributions
    rmc-exafs angles RUN --center Zn --neighbors C,C --ranges 1.8:2.6,1.8:2.6
    rmc-exafs plot RUN                     figures of the fits and the pair distributions
    rmc-exafs signal-map RUN               which distances make which features of the fit

RUN is the output directory of a run (all samples) or one sample's subdirectory. Every command that prints a result
accepts ``--json``. A user error ends with one message and the exit code 2.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

__all__ = ["main"]


class UserError(Exception):
    """A problem of the input that the user can fix."""


def _print_json(obj):
    from .outputs import _default
    print(json.dumps(obj, indent=1, default=_default))


def _samples(run, label=None):
    from .outputs import sample_dirs
    dirs = sample_dirs(run)
    if label:
        dirs = [d for d in dirs if d.name == label]
        if not dirs:
            raise UserError(f"{run}: no sample {label!r}")
    return dirs


def _log(args):
    if getattr(args, "json", False):
        return lambda text: print(text, file=sys.stderr, flush=True)
    return lambda text: print(text, flush=True)


def cmd_run(args):
    from .run import run_search
    summary = run_search(args.config, log=_log(args))
    if args.json:
        _print_json(summary)
    return 0


def cmd_refine(args):
    from .run import run_refine
    summary = run_refine(args.config, log=_log(args))
    if args.json:
        _print_json({k: v for k, v in summary.items() if k != "history"})
    return 0


def cmd_validate(args):
    from .config import validate_file, output_directory
    problems = validate_file(args.config)
    if args.json:
        out = {"valid": not problems, "problems": problems}
        if not problems:
            out["output"] = str(output_directory(args.config))
        _print_json(out)
    elif problems:
        print(f"{args.config}: {len(problems)} problem(s)")
        for p in problems:
            print(f"  {p}")
    else:
        print(f"{args.config}: valid; the run writes to {output_directory(args.config)}")
    return 0 if not problems else 2


def cmd_schema(args):
    from .config import json_schema
    _print_json(json_schema())
    return 0


def cmd_example(args):
    from .config import example_configs
    ex = example_configs()
    if not args.name:
        if args.json:
            _print_json(sorted(ex))
        else:
            for name in sorted(ex):
                first = next((ln.lstrip("# ").strip() for ln in ex[name].splitlines() if ln.startswith("#")), "")
                print(f"{name}: {first}")
        return 0
    if args.name not in ex:
        raise UserError(f"no example {args.name!r}; the examples are {sorted(ex)}")
    if args.out:
        Path(args.out).write_text(ex[args.name])
        print(f"written {args.out}")
    else:
        print(ex[args.name], end="")
    return 0


def cmd_shells(args):
    from .analysis import shell_statistics, parse_shell
    if not args.shell:
        raise UserError("give at least one --shell A-B:r_min:r_max")
    shells = [parse_shell(s) for s in args.shell]
    paths = []
    for run in args.runs:
        paths += _samples(run, args.sample)
    labels = sorted({p.name for p in paths})
    if len(labels) > 1 and not args.sample:
        raise UserError(f"the runs hold several samples {labels}: choose one with --sample")
    res = shell_statistics(paths, shells)
    if args.json:
        _print_json(res)
        return 0
    print(f"{len(paths)} run(s) of sample {labels[0]}; {res['method']}")
    for row in res["shells"]:
        print(f"{row['center']}-{row['neighbor']} {row['r_min']:g}-{row['r_max']:g} A:")
        for q, unit in (("N", ""), ("R", " A"), ("sigma2", " A^2"), ("c3", " A^3")):
            v = row[q]
            st = v["states"][0]
            line = f"   {q:7s} best {v['best']:.5g}{unit}; states {st['mean']:.5g} +- {st['sd']:.2g}"
            if v["runs"]:
                line += f"; runs {v['runs']['mean']:.5g} +- {v['runs']['sd']:.2g} ({v['runs']['n']} runs)"
            print(line)
    return 0


def cmd_short_pairs(args):
    from .analysis import short_pair_statistics, parse_pair_value
    th = dict(parse_pair_value(t) for t in args.threshold) if args.threshold else None
    out = {d.name: short_pair_statistics(d, th) for d in _samples(args.run, args.sample)}
    if args.json:
        _print_json(out)
        return 0
    for label, res in out.items():
        print(f"{label}:")
        for which in ("best", "population"):
            for key, v in res[which].items():
                print(f"   {which:10s} {key} below {v['threshold']:g} A: {v['per_center']:.3f} per center, {100 * v['share_of_centers']:.1f}% of the centers; "
                      f"closest {v['closest']:.3f} A")
    return 0


def cmd_rdf(args):
    import numpy as np
    from .analysis import run_rdf
    out = {}
    for d in _samples(args.run, args.sample):
        r, g, info = run_rdf(d, args.center, args.elements.split(",") if args.elements else None, rmax=args.rmax, dr=args.dr, which=args.which)
        out[d.name] = {"r": r, "g": g, **info}
        if args.out:
            np.savez_compressed(Path(args.out).with_suffix("").as_posix() + f"_{d.name}.npz", r=r, **{f"g_{k}": v for k, v in g.items()})
    if args.json:
        _print_json({k: {"r": v["r"].tolist(), "g": {e: x.tolist() for e, x in v["g"].items()}, "centers": v["centers"],
                         "configurations": v["configurations"]} for k, v in out.items()})
        return 0
    for label, v in out.items():
        print(f"{label}: {args.center} centers {v['centers']}, {v['configurations']} configurations; pairs per center below {args.rmax} A: "
              + ", ".join(f"{e} {float(x.sum() * args.dr):.2f}" for e, x in v["g"].items()) + (f"; written {args.out}" if args.out else ""))
    return 0


def cmd_angles(args):
    from .analysis import run_angles
    nb = [v.strip() for v in args.neighbors.split(",")]
    rg = [[float(x) for x in r.split(":")] for r in args.ranges.split(",")]
    if len(nb) != 2 or len(rg) != 2 or any(len(r) != 2 for r in rg):
        raise UserError("--neighbors B,C and --ranges lo:hi,lo:hi (A) take two entries each")
    out = {d.name: run_angles(d, args.center, nb, rg, which=args.which, bin_width=args.bin) for d in _samples(args.run, args.sample)}
    if args.json:
        _print_json(out)
        return 0
    for label, v in out.items():
        print(f"{label}: {nb[0]}-{args.center}-{nb[1]}: {v['angles']} angles, mean {v['mean']:.2f} deg, standard deviation {v['sd']:.2f} deg")
    return 0


def cmd_plot(args):
    from .analysis import plot_run
    written = []
    for d in _samples(args.run, args.sample):
        written += plot_run(d, Path(args.out) / d.name if args.out else None)
    if args.json:
        _print_json([str(p) for p in written])
    else:
        print("written " + ", ".join(str(p) for p in written))
    return 0


def cmd_signal_map(args):
    from .analysis import signal_map
    from ..parallel import primary_device
    comps = None
    if args.components:
        comps = {}
        for item in args.components:
            el, bs = item.split(":")
            comps[el.strip()] = [float(v) for v in bs.split(",")]
    bands = [tuple(float(x) for x in b.split(":")) for b in args.band] if args.band else [(1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
    out = {}
    for d in _samples(args.run, args.sample):
        out[d.name] = signal_map(d, comps, bands, out=(Path(args.out) / d.name).as_posix() if args.out else None, device=primary_device(args.devices, quiet=True))
    if args.json:
        _print_json(out)
        return 0
    for label, res in out.items():
        for edge, v in res.items():
            print(f"{label}, {edge} edge: share of every part in the R bands (Re<part, model> / <model, model>)")
            for name, row in v["shares"].items():
                comp = v["components"].get(name)
                head = f"N {comp['N']:.2f}, R {comp['R']:.3f} A" if comp else "the rest"
                print(f"   {name:24s} ({head}): " + ", ".join(f"{b} A {s:+.3f}" for b, s in row.items() if b != "peak_R"))
    return 0


def parser():
    ap = argparse.ArgumentParser(prog="rmc-exafs", description="Reverse Monte Carlo fitting of EXAFS spectra on the GPU (feffgpu.rmc). "
                                 "One TOML file per run; see `rmc-exafs example` and the documentation.")
    sub = ap.add_subparsers(dest="command", required=True)

    def add(name, func, help_, json_=True):
        p = sub.add_parser(name, help=help_, description=help_)
        p.set_defaults(func=func)
        if json_:
            p.add_argument("--json", action="store_true", help="print the result as JSON (progress lines go to standard error)")
        return p
    p = add("run", cmd_run, "run the population search of a configuration")
    p.add_argument("config", help="the TOML configuration")
    p = add("refine", cmd_refine, "run the gradient refinement of a configuration ([refine] section)")
    p.add_argument("config")
    p = add("validate", cmd_validate, "check a configuration and the files it names, without running (no GPU)")
    p.add_argument("config")
    add("schema", cmd_schema, "print the JSON Schema of the configuration", json_=False)
    p = add("example", cmd_example, "list the example configurations, or print (or write) one")
    p.add_argument("name", nargs="?", default="")
    p.add_argument("--out", default="", help="write the example to this file")
    p = add("shells", cmd_shells, "shell numbers (N, R, MSRD, third cumulant) from stated distance ranges, with their spread over the states and over runs")
    p.add_argument("runs", nargs="+", help="run directories or sample directories (several: one per seed)")
    p.add_argument("--shell", action="append", default=[], help="A-B:r_min:r_max (A); repeat for several shells")
    p.add_argument("--sample", default="", help="the sample (label) of a run with several")
    p = add("short-pairs", cmd_short_pairs, "pairs at distances below the closest distances (the run's, or --threshold)")
    p.add_argument("run")
    p.add_argument("--threshold", action="append", default=[], help="A-B:distance (A); repeat; default the run's closest distances")
    p.add_argument("--sample", default="")
    p = add("rdf", cmd_rdf, "partial pair distributions of a center element")
    p.add_argument("run")
    p.add_argument("--center", required=True, help="the center element")
    p.add_argument("--elements", default="", help="neighbor elements, comma-separated (default all)")
    p.add_argument("--rmax", type=float, default=6.0)
    p.add_argument("--dr", type=float, default=0.01)
    p.add_argument("--which", choices=["population", "best"], default="population")
    p.add_argument("--out", default="", help="write the distributions to <out>_<sample>.npz")
    p.add_argument("--sample", default="")
    p = add("angles", cmd_angles, "bond angles B-A-C at a center element")
    p.add_argument("run")
    p.add_argument("--center", required=True)
    p.add_argument("--neighbors", required=True, help="B,C")
    p.add_argument("--ranges", required=True, help="the A-B and A-C distance ranges, lo:hi,lo:hi (A)")
    p.add_argument("--which", choices=["population", "best"], default="population")
    p.add_argument("--bin", type=float, default=1.0, help="bin width (degrees)")
    p.add_argument("--sample", default="")
    p = add("plot", cmd_plot, "figures of a run: the fits and the pair distributions")
    p.add_argument("run")
    p.add_argument("--out", default="", help="directory of the figures (default the sample directories)")
    p.add_argument("--sample", default="")
    p = add("signal-map", cmd_signal_map, "which distances make which features of the final fit (needs the run's input files)")
    p.add_argument("run")
    p.add_argument("--components", action="append", default=[], help="El:b0,b1,... (A): the components of a neighbor element; repeat")
    p.add_argument("--band", action="append", default=[], help="lo:hi (A): an R band of chi(R); repeat (default 1-2, 2-3, 3-4)")
    p.add_argument("--out", default="", help="directory of the figures")
    p.add_argument("--devices", default="cuda")
    p.add_argument("--sample", default="")
    return ap


def main(argv=None):
    """Run ``rmc-exafs`` with the arguments ``argv`` (default ``sys.argv[1:]``). Returns the exit code."""
    ap = parser()
    args = ap.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except UserError as err:
        print(f"rmc-exafs {args.command}: {err}", file=sys.stderr)
        return 2
    except (ValueError, FileNotFoundError, KeyError) as err:
        from pydantic import ValidationError
        if isinstance(err, ValidationError):
            from ..config_tools import validation_messages
            msg = "; ".join(validation_messages(err))
        else:
            msg = str(err).strip("'\"")
        print(f"rmc-exafs {args.command}: {msg}", file=sys.stderr)
        return 2
    except Exception as err:
        import tomllib
        if isinstance(err, tomllib.TOMLDecodeError):
            print(f"rmc-exafs {args.command}: not valid TOML: {err}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    sys.exit(main())
