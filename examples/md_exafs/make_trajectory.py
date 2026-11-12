"""Make the inputs of the MD-EXAFS example: a small synthetic trajectory and a synthetic "measured" spectrum.

The system is a rock-salt block of Zn and C (3 x 3 x 3 conventional cells, 216 atoms) whose phase file ships with
the package (data/validation/A000/phase.pad: the K edge of C, potentials 0 C, 1 Zn, 2 C). It is a test system for the
code, not a real material.

- ``trajectory.extxyz``: frames of independent Gaussian displacements of every atom (standard deviation 0.06 A per
  component) around the ideal sites, with the cell in every frame. A stand-in for a molecular dynamics trajectory.
- ``measured.chi``: the averaged spectrum of an independent set of such frames, computed with the package's forward
  model, then multiplied by S0^2 = 0.85, shifted by dE0 = +1.5 eV, and given a little noise. The comparison step of
  the example should find these two numbers again.

The files go to ``--out`` (default: this folder), together with a copy of the phase file and of ``config.toml``, so
that the folder is self-contained. A GPU is used when available:

    python make_trajectory.py --out ~/mdexafs_example
    cd ~/mdexafs_example
    md-exafs run config.toml
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent.parent / "data/validation/A000/phase.pad"   # in a checkout of the repository
A = 4.3765                                  # the lattice constant of the A000 validation input (A)


def rock_salt(L=3, a=A):
    """Sites, element symbols and box length of an L x L x L rock-salt block."""
    sites, sym = [], []
    basis = (((0, 0, 0), "C"), ((0.5, 0, 0), "Zn"), ((0, 0.5, 0), "Zn"), ((0, 0, 0.5), "Zn"),
             ((0.5, 0.5, 0), "C"), ((0.5, 0, 0.5), "C"), ((0, 0.5, 0.5), "C"), ((0.5, 0.5, 0.5), "Zn"))
    for i in range(L):
        for j in range(L):
            for k in range(L):
                for d, s in basis:
                    sites.append(((i + d[0]) * a, (j + d[1]) * a, (k + d[2]) * a))
                    sym.append(s)
    return np.array(sites), np.array(sym), L * a


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=20, help="frames of the trajectory")
    ap.add_argument("--sd", type=float, default=0.06, help="standard deviation of the displacements per component (A)")
    ap.add_argument("--s02", type=float, default=0.85, help="S0^2 planted in the measured spectrum")
    ap.add_argument("--de0", type=float, default=1.5, help="E0 shift planted in the measured spectrum (eV)")
    ap.add_argument("--noise", type=float, default=2e-4, help="standard deviation of the noise added to chi(k)")
    ap.add_argument("--device", default=None, help="torch device (default cuda when available)")
    ap.add_argument("--out", default=str(HERE), help="folder of the generated files (default this folder)")
    ap.add_argument("--phase", default=str(PHASE), help="the phase file data/validation/A000/phase.pad of the repository")
    args = ap.parse_args()
    import torch
    from scipy.interpolate import CubicSpline
    from feffgpu.exafs import ETOK, k_grid
    from feffgpu.io.trajectory import write_extxyz_frame
    from feffgpu.md.spectrum import EdgeSpectra
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not Path(args.phase).is_file():
        raise SystemExit(f"phase file not found: {args.phase} (give --phase, the file data/validation/A000/phase.pad of the repository)")
    if Path(args.phase).resolve() != (out / "phase.pad").resolve():
        shutil.copyfile(args.phase, out / "phase.pad")
    if out != HERE:
        shutil.copyfile(HERE / "config.toml", out / "config.toml")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sites, sym, box = rock_salt()
    cell = np.eye(3) * box
    rng = np.random.default_rng(0)
    with open(out / "trajectory.extxyz", "w") as f:
        for i in range(args.frames):
            write_extxyz_frame(f, sym, sites + rng.normal(0.0, args.sd, sites.shape), cell, info=f"frame={i}")
    n_zn, n_c = int((sym == "Zn").sum()), int((sym == "C").sum())
    print(f"wrote trajectory.extxyz: {args.frames} frames of {len(sites)} atoms (Zn {n_zn}, C {n_c})")
    # the "measured" spectrum: an independent set of frames through the forward model, then S0^2, the shift and noise
    k = k_grid(16.0)
    calc = EdgeSpectra(str(out / "phase.pad"), "C", {"Zn": 1, "C": 2}, ["C", "Zn"], k, rpath=5.5, nleg=4, criteria=2.0, device=device)
    other = np.random.default_rng(12345)
    absorbers = np.nonzero(sym == "C")[0]
    chi = np.mean([calc.spectrum(sites + other.normal(0.0, args.sd, sites.shape), cell, absorbers, sym).chi for _ in range(16)], axis=0)
    q = k ** 2 - ETOK * args.de0
    data = np.where(q > 0, args.s02 * CubicSpline(k, chi)(np.sqrt(np.clip(q, 0, None))), 0.0)
    data = data + args.noise * other.normal(size=len(k))
    keep = k >= 1.0
    np.savetxt(out / "measured.chi", np.column_stack([k[keep], data[keep]]), fmt="%.6f %.8e",
               header=f"synthetic measured spectrum of the MD-EXAFS example: S0^2 {args.s02}, dE0 {args.de0} eV, noise {args.noise}\nk chi")
    print(f"wrote measured.chi (planted S0^2 {args.s02}, dE0 {args.de0} eV), phase.pad and config.toml in {out}")


if __name__ == "__main__":
    main()
