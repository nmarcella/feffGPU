"""Make the input files of the MD-EXAFS tutorials: a short synthetic trajectory of bcc iron, and a synthetic
spectrum to compare it with.

Nothing here is measured, and the forces are not a model of iron. The atoms of a 4 x 4 x 4 block of bcc cells (128
atoms, lattice constant 2.8665 A) are joined to their eight first and six second neighbors by Morse springs, each
at rest at its distance in the ideal lattice. The block is moved with Langevin dynamics at 300 K (time step 2 fs),
and after 2 ps one frame is kept every 0.5 ps. The spring strength is chosen so that the distance between first
neighbors spreads by about 0.07 A. The result has the features of a real trajectory that matter for the tutorials:
neighbors move together, the distribution of the bond length is slightly asymmetric, and a finite number of frames
samples the thermal motion only approximately.

- ``data/Fe_bcc_300K.extxyz``: 60 frames, each with its cell (extended xyz, read by ASE, OVITO and most programs).
- ``data/Fe_synthetic_spectrum.chi``: the averaged spectrum of a second, independent and ten times longer trajectory
  of the same model (600 frames), computed with ``md-exafs``, then multiplied by S0^2 = 0.80, shifted by
  dE0 = +2.0 eV and given noise of standard deviation 3e-4. It stands in for a measured spectrum; the comparison
  step of the tutorial should find the two planted numbers again, within the sampling error of the 60 frames.

    python tutorials/make_fe_trajectory.py                 # writes both files into tutorials/data (GPU: about 4 min)
    python tutorials/make_fe_trajectory.py --no-spectrum   # the trajectory only (a few seconds, no GPU)
"""
from __future__ import annotations
import argparse
import tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
A_FE = 2.8665                 # lattice constant of the block (A)
MASS = 55.845                 # atomic mass of iron (u)
KB = 8.617333e-5              # Boltzmann's constant (eV/K)
TIME_UNIT = 10.18051          # fs: the time unit of the system eV, A, u
MORSE_DEPTH = 0.55            # eV: depth of every spring (sets how far the atoms move)
MORSE_ALPHA = 1.4             # 1/A: inverse width of every spring (sets the asymmetry)

# The settings of the calculation, the same as in the tutorial notebook (01_md_exafs_iron.ipynb).
EDGE_SETTINGS = """
[[edge]]
name = "Fe"
element = "Fe"
edge = "K"
potentials = "reference"
cluster_radius = 6.0
scf_radius = 4.0
rpath = 5.2
nleg = 4
criteria = 2.0
kmax = 16.0
"""


def bcc_block(n=4, a=A_FE):
    """Ideal sites (n_atoms, 3) and the box length of an n x n x n block of bcc cells."""
    cells = np.array([(i, j, k) for i in range(n) for j in range(n) for k in range(n)], float)
    sites = np.concatenate([cells, cells + 0.5]) * a
    return sites, n * a


def springs(sites, box, a=A_FE):
    """The pairs of first and second neighbors of the ideal lattice: (i, j, rest length)."""
    d = sites[None] - sites[:, None]
    d -= box * np.round(d / box)
    r = np.sqrt((d ** 2).sum(-1))
    i, j = np.nonzero(np.triu((r > 0.1) & (r < 1.1 * a), 1))
    return i, j, r[i, j]


def forces(pos, box, i, j, rest, depth=MORSE_DEPTH, alpha=MORSE_ALPHA):
    """Forces (eV/A) of the Morse springs, with the minimum-image convention."""
    d = pos[j] - pos[i]
    d -= box * np.round(d / box)
    r = np.sqrt((d ** 2).sum(-1))
    e = np.exp(-alpha * (r - rest))
    pull = (2.0 * depth * alpha * (1.0 - e) * e / r)[:, None] * d      # on atom i, toward j when stretched
    f = np.zeros_like(pos)
    np.add.at(f, i, pull)
    np.add.at(f, j, -pull)
    return f


def trajectory(frames=60, temperature=300.0, seed=0, n=4, every_fs=500.0, settle_fs=2000.0, dt_fs=2.0, friction_per_ps=5.0):
    """Langevin dynamics of the block. Returns the positions of the kept frames (frames, n_atoms, 3), the box length
    and the time of each frame (fs)."""
    rng = np.random.default_rng(seed)
    sites, box = bcc_block(n)
    i, j, rest = springs(sites, box)
    dt = dt_fs / TIME_UNIT
    kt = KB * temperature
    damp = np.exp(-friction_per_ps * 1e-3 * dt_fs)
    kick = np.sqrt(kt / MASS * (1.0 - damp ** 2))
    pos = sites.copy()
    vel = rng.normal(0.0, np.sqrt(kt / MASS), pos.shape)
    f = forces(pos, box, i, j, rest)
    keep_every, settle = int(round(every_fs / dt_fs)), int(round(settle_fs / dt_fs))
    kept, times = [], []
    for step in range(1, settle + frames * keep_every + 1):
        vel += 0.5 * dt * f / MASS
        pos += 0.5 * dt * vel
        vel = damp * vel + kick * rng.normal(size=vel.shape)          # the thermostat
        pos += 0.5 * dt * vel
        f = forces(pos, box, i, j, rest)
        vel += 0.5 * dt * f / MASS
        if step > settle and (step - settle) % keep_every == 0:
            kept.append(pos - (pos - sites).mean(0))                  # without the drift of the whole block
            times.append((step - settle) * dt_fs)
    return np.array(kept), box, np.array(times)


def write_extxyz(path, positions, box, times):
    """Write the frames as extended xyz with ASE (every frame carries its cell)."""
    from ase import Atoms
    from ase.io import write
    images = []
    for pos, t in zip(positions, times):
        atoms = Atoms("Fe" * pos.shape[0], positions=pos, cell=[box, box, box], pbc=True)
        atoms.info["time_fs"] = float(t)
        images.append(atoms)
    write(path, images, format="extxyz")


def synthetic_spectrum(path, frames=600, seed=1, s02=0.80, de0=2.0, noise=3e-4, device=None):
    """The averaged spectrum of an independent trajectory, with S0^2, an energy shift and noise planted."""
    import torch
    from scipy.interpolate import CubicSpline
    from feffgpu.exafs import ETOK
    from feffgpu.md import average_cache, load_config, run
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    positions, box, times = trajectory(frames=frames, seed=seed)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_extxyz(tmp / "other.extxyz", positions, box, times)
        (tmp / "config.toml").write_text(
            '[system]\nname = "independent trajectory"\ntrajectory = "other.extxyz"\noutput = "out"\n'
            '[lattice]\ndynamic = true\n[frames]\nabsorbers_per_frame = 32\nseed = 0\n'
            + EDGE_SETTINGS + f'[average]\nblocks = 10\n[run]\ndevices = "{device}"\n')
        cfg = load_config(tmp / "config.toml")
        run(cfg)
        average = average_cache(cfg.output_path / "chi.h5", cfg.average)["Fe"]
        k, chi = np.asarray(average["k"], float), np.asarray(average["mean"], float)
    q = k ** 2 - ETOK * de0                                           # the model at k' = sqrt(k^2 - 0.2625 dE0)
    data = np.where(q > 0, s02 * CubicSpline(k, chi)(np.sqrt(np.clip(q, 0, None))), 0.0)
    data = data + noise * np.random.default_rng(seed + 1000).normal(size=len(k))
    keep = (k >= 1.0) & (k <= 15.0)
    np.savetxt(path, np.column_stack([k[keep], data[keep]]), fmt="%.6f %.8e",
               header="synthetic spectrum of the MD-EXAFS tutorial, not a measurement: Fe K edge of an independent,\n"
                      f"longer trajectory of the same spring model, with S0^2 {s02}, dE0 {de0} eV and noise {noise} planted\nk chi")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "data"), help="folder of the two files (default tutorials/data)")
    ap.add_argument("--frames", type=int, default=60, help="frames of the trajectory")
    ap.add_argument("--temperature", type=float, default=300.0, help="temperature of the thermostat (K)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-spectrum", action="store_true", help="write the trajectory only")
    ap.add_argument("--device", default=None, help="torch device of the spectrum (default cuda when available)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    positions, box, times = trajectory(frames=args.frames, temperature=args.temperature, seed=args.seed)
    write_extxyz(out / "Fe_bcc_300K.extxyz", positions, box, times)
    print(f"wrote Fe_bcc_300K.extxyz: {len(positions)} frames of {positions.shape[1]} atoms, box {box:.4f} A")
    if not args.no_spectrum:
        synthetic_spectrum(out / "Fe_synthetic_spectrum.chi", device=args.device)
        print("wrote Fe_synthetic_spectrum.chi (planted S0^2 0.80, dE0 +2.0 eV)")


if __name__ == "__main__":
    main()
