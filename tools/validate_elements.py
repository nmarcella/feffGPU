#!/usr/bin/env python3
"""Validation of feffgpu against FEFF 8.5L element by element and edge by edge.

For every element Z (default 1 to 99) and each of the edges K, L3 and M5 that the element has, one FEFF input is
written and run through FEFF 8.5L (its six modules, a build at FEFF_HOME; fortran/README.md) and through feffgpu; the
two chi(k) are compared. Every case uses the same recipe, so that only the element and the edge change:

  structure    an fcc cluster of the element itself; nearest-neighbor distance d = 2 x covalent radius (at least 1.4 A);
               every atom but the absorber displaced by a Gaussian of 0.01 d per coordinate (seeded by Z), so that
               the cluster has no exact symmetry; all atoms within 2.55 d of the absorber (87 atoms on the ideal lattice)
  cards        EDGE <edge>, S02 1, SCF 1.55 d (self-consistent potentials on the first two shells), RPATH 2.05 d
               (four shells), EXAFS 20, AFOLP 1.15, CRITERIA 0 2.0, NLEG 4

Where FEFF 8.5L's own self-consistency loop stops for an element (it does for a few), run that element again with
--no-scf: the comparison is then made with overlapped potentials, and the table says so.

Output: validation_runs/<tag>/<Z><edge>/ (inputs, FEFF's outputs, the two spectra), summary.json. `--collect` writes the
small files that the repository keeps (data/validation/elements.npz and elements.json) for
tools/plot_validation_elements.py.

    FEFF_HOME=/path/to/feff85exafs/local_install python tools/validate_elements.py            # the whole sweep
    python tools/validate_elements.py --zmin 26 --zmax 30 --edges K,L3                        # a part of it
    python tools/validate_elements.py --collect                                              # write data/validation/elements.*
"""
from pathlib import Path
import argparse, json, os, shutil, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_random import STRUCTURES, available_holes, compare, run_feff, write_input      # noqa: E402

HOLE = {"K": 1, "L3": 4, "M5": 9}
DISORDER = 0.01                                              # Gaussian displacement per coordinate, in units of d


def element_case(z, edge, scf=True):
    """The FEFF input of one element and edge: dict(name, title, cards, potentials, atoms, meta). scf=False leaves the
    SCF card out (overlapped potentials without self-consistency)."""
    from ase.data import covalent_radii
    from feffgpu.io import SYMBOLS
    sym = SYMBOLS[z]
    d = max(2.0 * float(covalent_radii[z]), 1.4)
    a = d / STRUCTURES["fcc"]["nn"]
    rcl = 2.55 * d
    m = int(np.ceil(rcl / a)) + 1
    cells = np.stack(np.meshgrid(*[np.arange(-m, m + 1)] * 3, indexing="ij"), -1).reshape(-1, 3)
    pos = np.concatenate([(cells + np.array(frac)) * a for frac, _slot in STRUCTURES["fcc"]["basis"]])
    r = np.linalg.norm(pos, axis=1)
    pos = pos[r <= rcl + 1e-9]
    pos = pos[np.argsort(np.linalg.norm(pos, axis=1), kind="stable")]
    rng = np.random.default_rng(1000 + z)                    # the same structure for every edge of an element
    pos[1:] += rng.normal(0.0, DISORDER * d, pos[1:].shape)
    atoms = [(float(x), float(y), float(zc), 0 if i == 0 else 1) for i, (x, y, zc) in enumerate(pos)]
    rfms, rpath = round(1.55 * d, 3), round(2.05 * d, 3)
    cards = ["CONTROL 1 1 1 1 1 1", "PRINT 0 0 0 0 0 0", f"EDGE {edge}", "S02 1.0"] + ([f"SCF {rfms} 0 30 .2 1"] if scf else []) + [
        f"RPATH {rpath}", "EXAFS 20.0", "AFOLP 1.15", "CRITERIA 0.0 2.0", "NLEG 4"]
    meta = dict(family="E", species=[z], structure="fcc", absorber_Z=z, absorber=sym, edge=edge, ihole=HOLE[edge], n_atoms=len(atoms),
                cluster_radius=round(rcl, 3), lattice_constant=round(a, 4), nearest=round(d, 4), scf=rfms if scf else None, rpath=rpath, kmax=20.0, pwcrit=2.0,
                afolp=True, lmax_override=None)
    return dict(name=f"{z:03d}_{sym}_{edge}", title=f"{sym} fcc, {edge} edge, {len(atoms)} atoms (element sweep)", cards=cards,
                potentials=[(0, z, sym), (1, z, sym)], atoms=atoms, meta=meta)


def all_cases(out_dir):
    out = []
    for f in sorted(Path(out_dir).glob("*/case.json")):
        out.append(json.loads(f.read_text()))
    return out


def collect(out_dir):
    """Write data/validation/elements.npz and elements.json from the case folders of a finished sweep."""
    cases = all_cases(out_dir)
    arrays, rows = {}, []
    for c in cases:
        m, res = c["meta"], c.get("result")
        row = dict(case=c["name"], Z=m["absorber_Z"], absorber=m["absorber"], edge=m["edge"], atoms=m["n_atoms"], nearest=m["nearest"],
                   rpath=m["rpath"], scf_radius=m["scf"], self_consistent=m["scf"] is not None)
        if res is None:
            status = str(c.get("status", "not run"))
            row.update(status="FEFF 8.5L failed" if status.startswith("FEFF failed") else ("feffgpu failed" if status.startswith("GPU") else status),
                       detail=status.splitlines()[0][:160])
        else:
            z = np.load(Path(out_dir) / c["name"] / "chi_compare.npz")
            arrays[f"{c['name']}_k"], arrays[f"{c['name']}_feff"], arrays[f"{c['name']}_feffgpu"] = z["k"].astype(np.float32), z["chi_feff"], z["chi_gpu"]
            deg = lambda v: float(v[1]) if isinstance(v, (list, tuple)) else float(v)
            pf = {str(k): deg(v) for k, v in res["paths_feff"].items()}
            pg = {str(k): deg(v) for k, v in res["paths_gpu"].items()}
            row.update(status="compared", r_factor_r_space=res["R_window"], r_factor_k_space=res["R_k"], max_relative_difference=res["max_rel"],
                       fermi_level_difference_hartree=res["xmu_diff_ha"], paths_feff=pf, paths_feffgpu=pg,
                       same_path_counts=all(abs(pf.get(k, 0.0) - pg.get(k, 0.0)) < 0.5 for k in set(pf) | set(pg)),
                       partial_wave_limits_equal=res["lmax_gpu"] == res["lmax_feff"], seconds_feff=round(sum(c.get("timing_feff", {}).values()), 1),
                       seconds_feffgpu=round(sum(res["timing_gpu"].values()), 1))
        rows.append(row)
    out = ROOT / "data/validation"
    np.savez_compressed(out / "elements.npz", **arrays)
    doc = ("FEFF 8.5L against feffgpu, element by element and edge by edge (tools/validate_elements.py): for every element and each of its "
           "K, L3 and M5 edges the same recipe (an fcc cluster of the element, self-consistent potentials, paths to four legs). "
           "elements.npz holds, per compared case, k (1/A), chi of FEFF 8.5L and chi of feffgpu. r_factor_r_space: k 3 to 19, R 1 A to the "
           "path cutoff, k^2 weight; r_factor_k_space: k >= 2, k^2 weight.")
    (out / "elements.json").write_text(json.dumps({"description": doc, "cases": rows}, indent=1) + "\n")
    done = [r for r in rows if r["status"] == "compared"]
    rr = np.array([r["r_factor_r_space"] for r in done])
    print(f"collected {len(rows)} cases, {len(done)} compared" + (f": median R {np.median(rr):.2e}, worst {rr.max():.2e}, "
          f"{int((rr < 2e-5).sum())} below 2e-5" if len(done) else ""))
    for r in rows:
        if r["status"] != "compared":
            print(f"   {r['case']}: {r['status']}: {r.get('detail', '')[:110]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="elements")
    ap.add_argument("--zmin", type=int, default=1)
    ap.add_argument("--zmax", type=int, default=99)
    ap.add_argument("--edges", default="K,L3,M5")
    ap.add_argument("--workers", type=int, default=8, help="FEFF processes in parallel")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true", help="skip the cases that already have a result or a recorded failure")
    ap.add_argument("--no-scf", action="store_true", help="leave the SCF card out (for the elements on which FEFF 8.5L's own self-consistency "
                    "loop stops); the cases of the chosen elements are run again")
    ap.add_argument("--reuse-feff", action="store_true", help="where a case folder holds FEFF 8.5L's outputs for the same input, keep them and "
                    "run only feffgpu again")
    ap.add_argument("--collect", action="store_true", help="only write data/validation/elements.* from the finished case folders")
    args = ap.parse_args()
    out_dir = ROOT / "validation_runs" / args.tag
    if args.collect:
        return collect(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for z in range(args.zmin, args.zmax + 1):
        holes = set(available_holes(z))
        for edge in args.edges.split(","):
            if HOLE[edge] in holes:
                cases.append(element_case(z, edge, scf=not args.no_scf))
    todo = []
    for c in cases:
        d = out_dir / c["name"]
        c["dir"] = str(d)
        old = d / "case.json"
        if args.resume and old.exists():
            prev = json.loads(old.read_text())
            if prev.get("result") or prev.get("status"):
                c["skip"] = True
                continue
        if args.reuse_feff and old.exists() and (d / "chi.dat").exists() and (d / "feff.inp").exists():
            prev = json.loads(old.read_text())
            write_input(c, d / "feff.inp.new")
            same = (d / "feff.inp.new").read_text() == (d / "feff.inp").read_text()
            (d / "feff.inp.new").unlink()
            if same and "timing_feff" in prev:
                c["timing_feff"] = prev["timing_feff"]
                todo.append(c)
                continue
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        write_input(c, d / "feff.inp")
        todo.append(c)
    need_feff = [c for c in todo if "timing_feff" not in c]
    print(f"{len(cases)} cases ({len(todo)} to run, FEFF 8.5L for {len(need_feff)} of them): elements {args.zmin} to {args.zmax}, "
          f"edges {args.edges}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        res = dict(ex.map(run_feff, [(c["dir"], None) for c in need_feff]))
    for c in need_feff:
        r = res[c["name"]]
        if isinstance(r, str):
            c["status"] = f"FEFF failed: {r}"
            print(f"{c['name']}: FEFF 8.5L failed: {r.splitlines()[0][:160]}", flush=True)
            (Path(c["dir"]) / "case.json").write_text(json.dumps(c, indent=1, default=float))
        else:
            c["timing_feff"] = r
    print(f"FEFF 8.5L done for {sum(1 for c in need_feff if 'timing_feff' in c)} of {len(need_feff)} cases in {time.time() - t0:.0f} s", flush=True)
    from feffgpu.chain import run
    n_done = 0
    for c in todo:
        if "timing_feff" not in c:
            continue
        d = Path(c["dir"])
        os.environ.pop("FEFF_LMAX", None)
        t1 = time.time()
        c.pop("result", None)
        c.pop("status", None)
        try:
            out = run(d / "feff.inp", device=args.device)
            c["result"] = compare(d, c["meta"], out)
            r = c["result"]
            n_done += 1
            print(f"{c['name']}: R {r['R_window']:.1e} (k space {r['R_k']:.1e}), Fermi level difference {r['xmu_diff_ha']:.1e} Ha, paths FEFF "
                  f"{r['paths_feff']} feffgpu {r['paths_gpu']}, FEFF {sum(c['timing_feff'].values()):.0f} s, feffgpu {time.time() - t1:.0f} s "
                  f"[{n_done} compared, {time.time() - t0:.0f} s]", flush=True)
        except Exception as e:
            c["status"] = f"GPU chain failed: {e!r}"
            (d / "gpu_error.txt").write_text(traceback.format_exc())
            print(f"{c['name']}: feffgpu failed: {e!r}"[:300], flush=True)
        (d / "case.json").write_text(json.dumps(c, indent=1, default=float))
    (out_dir / "summary.json").write_text(json.dumps(all_cases(out_dir), indent=1, default=float))
    print(f"finished in {time.time() - t0:.0f} s; summary in {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
