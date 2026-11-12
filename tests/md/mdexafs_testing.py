"""Helpers of the MD-EXAFS tests: structures and trajectories generated in code, configuration files written in the
test's folder, and the two phase files of the repository (bcc W, K edge; Zn-C rock salt, K edge of C)."""
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
PHASE_W = ROOT / "tests/data/W/phase.pad"                 # bcc W, K edge: potentials 0 W (absorber), 1 W
PHASE_ZNC = ROOT / "data/validation/A000/phase.pad"       # Zn-C rock salt, K edge of C: potentials 0 C, 1 Zn, 2 C
A_W = 3.1652                                              # the lattice constant of the W phase file's input (A)
A_ZNC = 4.3765                                            # the rock-salt lattice constant of the A000 input (A)
GPU = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")]


def gpu(f):
    """Mark a test as needing the GPU."""
    for m in GPU:
        f = m(f)
    return f


def bcc_sites(L, a=A_W):
    """Sites (2 L^3, 3) of an L x L x L bcc block and its cubic box length."""
    from feffgpu.io import bcc_block
    return bcc_block(L, a)


def rock_salt(L=3, a=A_ZNC):
    """Sites, symbols and box length of an L x L x L rock-salt block of Zn and C."""
    sites, sym = [], []
    for i in range(L):
        for j in range(L):
            for l in range(L):
                for d, s in (((0, 0, 0), "C"), ((0.5, 0, 0), "Zn"), ((0, 0.5, 0), "Zn"), ((0, 0, 0.5), "Zn"),
                             ((0.5, 0.5, 0), "C"), ((0.5, 0, 0.5), "C"), ((0, 0.5, 0.5), "C"), ((0.5, 0.5, 0.5), "Zn")):
                    sites.append(((i + d[0]) * a, (j + d[1]) * a, (l + d[2]) * a))
                    sym.append(s)
    return np.array(sites), np.array(sym), L * a


def gaussian_frames(sites, n_frames, sd, seed=0):
    """Frames of independent isotropic Gaussian displacements (standard deviation ``sd`` per component, A)."""
    rng = np.random.default_rng(seed)
    return [sites + rng.normal(0.0, sd, sites.shape) for _ in range(n_frames)]


def write_extxyz(path, symbols, frames, cell=None):
    """Frames as an extended xyz file (with the cell when given)."""
    from feffgpu.io.trajectory import write_extxyz_frame
    with open(path, "w") as f:
        for i, p in enumerate(frames):
            c = cell[i] if (cell is not None and np.asarray(cell).ndim == 3) else cell
            write_extxyz_frame(f, symbols, p, c, info=f"frame={i}")
    return Path(path)


def write_lammps(path, types, frames, box):
    """Frames as a LAMMPS text dump with numeric types."""
    from feffgpu.io.trajectory import write_lammps_dump_frame
    with open(path, "w") as f:
        for i, p in enumerate(frames):
            write_lammps_dump_frame(f, types, p, box, timestep=10 * i)
    return Path(path)


def write_config(path, data):
    """A configuration dictionary written as TOML."""
    from feffgpu.md.config import to_toml
    Path(path).write_text(to_toml(data))
    return Path(path)


def rock_salt_run(tmp_path, n_frames=6, output="out", absorbers=8, nleg=3, rpath=5.0, kmax=16.0, extra=None, seed=0, sd=0.06,
                  compare=None):
    """A Zn-C rock-salt trajectory (extended xyz, cell in every frame) and its configuration with the A000 phase file
    (absorber C). Returns the configuration path."""
    sites, sym, box = rock_salt(3)
    write_extxyz(tmp_path / "traj.extxyz", sym, gaussian_frames(sites, n_frames, sd, seed), np.eye(3) * box)
    data = {"system": {"name": "rock salt test", "trajectory": "traj.extxyz", "output": output},
            "lattice": {"dynamic": True},
            "frames": {"absorbers_per_frame": absorbers, "seed": 3},
            "edge": [{"element": "C", "potentials": "file", "phase": str(PHASE_ZNC), "rpath": rpath, "nleg": nleg, "criteria": 2.0,
                      "kmax": kmax}],
            "run": {"frames_per_block": 2}}
    if compare:
        data["compare"] = compare
    for k, v in (extra or {}).items():
        data.setdefault(k, {}).update(v) if isinstance(v, dict) else data.__setitem__(k, v)
    return write_config(tmp_path / f"{output}.toml", data)
