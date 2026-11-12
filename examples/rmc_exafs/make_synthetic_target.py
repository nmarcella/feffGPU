"""Make the inputs of the RMC-EXAFS example: a start structure and a synthetic "measured" spectrum with a known answer.

The structure is a 2 x 2 x 2 block of a Zn-C rock salt (64 atoms; the compound of the FEFF validation case
data/validation/A000, whose phase file is used: potentials 0 C absorber, 1 Zn, 2 C). The "measured" spectrum is the
C K edge of a disordered copy of it, made with the package's own forward model: every atom displaced at random
(0.05 A per coordinate), S0^2 0.90, E0 shift +1.5 eV, single scattering of every pair and multiple scattering to
three legs. The run then starts from the ideal block and has to find the disorder.

    python make_synthetic_target.py          # writes start.vasp, target.chi and truth.json next to this script
    rmc-exafs validate config.toml
    rmc-exafs run config.toml

truth.json holds the planted S0^2 and E0 shift and the shell numbers of the disordered block, for comparison with the
result (rmc-exafs shells). Nothing else is needed; no data file is committed.
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PHASE = ROOT / "data/validation/A000/phase.pad"
A, L, SD, S02, DE0 = 4.3765, 2, 0.05, 0.90, 1.5
SHELLS = [("C", "Zn", 1.6, 2.8), ("C", "C", 2.6, 3.6), ("C", "Zn", 3.2, 4.3)]


def rock_salt():
    sites, sym = [], []
    for i in range(L):
        for j in range(L):
            for l in range(L):
                for d, s in (((0, 0, 0), "C"), ((0.5, 0, 0), "Zn"), ((0, 0.5, 0), "Zn"), ((0, 0, 0.5), "Zn"),
                             ((0.5, 0.5, 0), "C"), ((0.5, 0, 0.5), "C"), ((0, 0.5, 0.5), "C"), ((0.5, 0.5, 0.5), "Zn")):
                    sites.append(((i + d[0]) * A, (j + d[1]) * A, (l + d[2]) * A))
                    sym.append(s)
    return np.array(sym), np.array(sites), np.eye(3) * L * A


def write_poscar(path, sym, pos, cell):
    species = list(dict.fromkeys(sym.tolist()))
    lines = ["Zn-C rock salt, 2 x 2 x 2 cells (ideal start)", "1.0"] + [" ".join(f"{v:.8f}" for v in row) for row in cell]
    lines += [" ".join(species), " ".join(str(int((sym == s).sum())) for s in species), "Cartesian"]
    for s in species:
        lines += [" ".join(f"{v:.8f}" for v in p) for p in pos[sym == s]]
    path.write_text("\n".join(lines) + "\n")


def main():
    import torch
    from feffgpu.exafs import FitWindow, k_grid
    from feffgpu.exafs.edge import EdgeModel
    from feffgpu.analysis import shell_table
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sym, pos, cell = rock_salt()
    write_poscar(HERE / "start.vasp", sym, pos, cell)
    truth = pos + np.random.default_rng(0).normal(0.0, SD, pos.shape)
    k = k_grid(24.0)
    edge = EdgeModel("C", k, np.zeros(len(k)), FitWindow(kmin=3.0, kmax=12.0, dk=1.0, kweight=2, rmin=1.0, rmax=4.0), str(PHASE), sym, cell, "C",
                     potentials={"Zn": 1, "C": 2}, rpath=4.3, nleg=3, criteria=2.5, device=device)
    with torch.no_grad():
        chi = edge.model(edge.spectra(truth[None]), s02=S02, de0=DE0)[0].cpu().numpy()
    with open(HERE / "target.chi", "w") as f:
        f.write("# synthetic C K-edge chi(k) of a disordered Zn-C rock salt (make_synthetic_target.py)\n# k (1/A)   chi\n")
        for a, b in zip(k, chi):
            f.write(f"{a:8.3f} {b: .10e}\n")
    rows = shell_table(truth, sym, cell, SHELLS)
    (HERE / "truth.json").write_text(json.dumps({"s02": S02, "de0": DE0, "displacement_sd": SD,
                                                 "shells": [{k_: r[k_] for k_ in ("center", "neighbor", "r_min", "r_max", "N", "R", "sigma2", "c3")} for r in rows]},
                                                indent=1))
    print("written start.vasp, target.chi, truth.json; the planted shells:")
    for r in rows:
        print(f"  {r['center']}-{r['neighbor']} {r['r_min']}-{r['r_max']} A: N {r['N']:.2f}, R {r['R']:.4f} A, sigma2 {r['sigma2']:.5f} A^2")


if __name__ == "__main__":
    main()
