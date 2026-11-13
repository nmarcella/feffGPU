#!/usr/bin/env python3
"""Validation of paths of five to eight legs against FEFF itself.

Two made-up cases whose path cutoff is several bond lengths, so that long paths exist, with NLEG 8:

  L000  the rock-salt environment of the stored case A000 (C K edge, Zn and C neighbors) with its cutoff reduced to
        6.6 A: paths of up to six legs.
  L001  a compact carbon cluster (a fragment of the diamond lattice, 17 atoms, slightly disordered), C K edge, cutoff
        6.2 A: paths of up to eight legs.

For each, FEFF's six modules run on the feff.inp (a feff85exafs build, FEFF_HOME; fortran/README.md) and the GPU
chain runs on the same file. chi is compared on chi.dat's own grid (the R-factor of the analysis window), and the
degeneracy-weighted number of paths for every number of legs. With --store the cases are written to
data/validation/<name> (feff.inp and FEFF's chi.dat, paths.dat, pot.pad, phase.pad), where the regression tests
(tests/test_validation_cases.py) pick them up.

usage: FEFF_HOME=/path/to/feff85exafs python tools/validate_legs.py [--store] [--out DIR]
"""
from pathlib import Path
import argparse, json, shutil, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from validate_random import run_feff, compare        # noqa: E402


def rock_salt_input():
    """The stored case A000 with NLEG 8 and a cutoff of 6.6 A."""
    lines = (ROOT / "data/validation/A000/feff.inp").read_text().splitlines()
    out = []
    for line in lines:
        word = line.split()[0] if line.split() else ""
        if word == "NLEG":
            line = "NLEG 8"
        elif word == "RPATH":
            line = "RPATH 6.6"
        elif word == "TITLE":
            line = "TITLE made-up environment L000: Zn-C rock salt of case A000, K edge of C, paths of up to eight legs"
        out.append(line)
    return "\n".join(out) + "\n", 6.6


def carbon_cluster_input(seed=7):
    """A fragment of the diamond lattice around one atom (4 neighbors at 1.545 A, 12 at 2.52 A), every atom displaced
    at random by 0.03 A per coordinate."""
    a = 3.567
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    cells = np.array([[i, j, k] for i in range(-2, 3) for j in range(-2, 3) for k in range(-2, 3)])
    sites = np.concatenate([(cells[:, None] + basis[None]).reshape(-1, 3), (cells[:, None] + basis[None] + 0.25).reshape(-1, 3)]) * a
    d = np.linalg.norm(sites, axis=1)
    keep = np.argsort(d)[:17]
    pos = sites[keep] + np.random.default_rng(seed).normal(0, 0.03, (17, 3))
    pos -= pos[0]
    lines = ["TITLE made-up environment L001: compact carbon cluster, K edge of C, paths of up to eight legs", "CONTROL 1 1 1 1 1 1", "PRINT 0 0 0 0 0 0",
             "EDGE K", "S02 1.0", "RPATH 6.2", "EXAFS 20.0", "AFOLP 1.15", "CRITERIA 0.0 2.0", "NLEG 8", "", "POTENTIALS", "     0    6   C", "     1    6   C",
             "", "ATOMS"]
    lines += [f"  {p[0]:11.6f} {p[1]:11.6f} {p[2]:11.6f}  {0 if i == 0 else 1}" for i, p in enumerate(pos)]
    return "\n".join(lines + ["END", ""]), 6.2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "validation_runs" / "legs"))
    ap.add_argument("--store", action="store_true", help="write the cases to data/validation")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    from feffgpu.chain import run
    summary = {}
    for name, (text, rpath) in (("L000", rock_salt_input()), ("L001", carbon_cluster_input())):
        d = Path(args.out) / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "feff.inp").write_text(text)
        _, times = run_feff((d, 0))
        if not isinstance(times, dict):
            print(f"{name}: FEFF failed: {times}")
            continue
        out = run(d / "feff.inp", device=args.device)
        res = compare(d, {"rpath": rpath}, out)
        summary[name] = res
        (d / "comparison.json").write_text(json.dumps(res, indent=1))
        print(f"{name}: R-factor of the analysis window {res['R_window']:.2e}; Fermi level difference {res['xmu_diff_ha']:.1e} hartree; FEFF {sum(times.values()):.0f} s, "
              f"here {sum(res['timing_gpu'].values()):.0f} s")
        print("      paths by legs [entries, summed degeneracy]: FEFF " + str({int(k): v for k, v in sorted(res["paths_feff"].items())}))
        print("                                                  here " + str({int(k): v for k, v in sorted(res["paths_gpu"].items()) if v[0]}))
        if args.store:
            target = ROOT / "data/validation" / name
            target.mkdir(parents=True, exist_ok=True)
            for f in ("feff.inp", "chi.dat", "paths.dat", "pot.pad", "phase.pad"):
                shutil.copy(d / f, target / f)
    (Path(args.out) / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
