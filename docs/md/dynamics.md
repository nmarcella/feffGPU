# Making a trajectory: the optional dynamics runner

MD-EXAFS works on trajectories made elsewhere (LAMMPS, CP2K, VASP, ASE, ...). For users who do not have one,
`md-exafs dynamics` makes a trajectory with NVT Langevin dynamics in ASE and any ASE calculator, in particular a MACE
foundation model. It is a convenience: small, generic, with a fixed temperature and a fixed cell. Production dynamics
with more control (NPT, other thermostats, long runs) belong in a dedicated program, whose trajectories MD-EXAFS
reads directly.

## Calculators

| `calculator` | What it is | Needs |
|---|---|---|
| `"mace_mp"` | a MACE foundation model (`mace.calculators.mace_mp`); `model` names it (`"small"`, `"medium"`, `"large"` or a file) | `pip install 'feffgpu[mace]'`; MACE downloads the model the first time |
| `"emt"` | ASE's effective-medium potential, for a few fcc metals (Al, Ni, Cu, Pd, Ag, Pt, Au); for tests and demonstrations | nothing |
| `"module:function"` | any function that returns an ASE calculator, called with `calculator_options` | the module |

The interatomic potential decides how good the trajectory is. This package does not validate the forces of any
potential; check that the structure, the cell and the vibrations of your system are reasonable (the RDF tools and the
thermodynamic log help) before you compare its spectrum with a measurement.

## Settings

```toml
[system]
trajectory = "trajectory.extxyz"   # written by md-exafs dynamics, then read by md-exafs run

[lattice]
dynamic = true                     # the frames carry their cell

[dynamics]
structure = "start.cif"            # any file that ASE reads, with a periodic cell
supercell = [3, 3, 3]
calculator = "mace_mp"
model = "medium"
temperature = 300.0                # K
timestep = 2.0                     # fs
friction = 0.01                    # 1/fs
equilibration = 1000               # steps before frames are written
steps = 10000                      # production steps
save_every = 50                    # a frame every 50 steps
log_every = 10                     # a line of the log every 10 steps
seed = 0
log = "md.log"

[run]
devices = "cuda"                   # the device of the calculator (MACE)
```

Every key is in [](configuration.md).

## What it does

1. Reads the starting structure and repeats it `supercell` times.
2. Draws velocities from the Maxwell-Boltzmann distribution at `temperature` (seeded), removes the motion of the
   center of mass and keeps the center of mass fixed.
3. Runs `equilibration + steps` steps of ASE's Langevin integrator at `temperature` with `friction`.
4. During production, writes a frame every `save_every` steps to `[system] trajectory` (extended xyz with the cell, the
   step, the time and the instantaneous temperature), and the thermodynamic log of ASE's `MDLogger` every `log_every`
   steps to `[dynamics] log`.
5. Writes `dynamics.json` into the output folder ([](outputs.md)).

Then:

```bash
md-exafs run config.toml
md-exafs convergence config.toml    # the temperature and energies of the log, and the convergence of the spectra
```

## Testing

The test suite runs the runner with ASE's EMT calculator on a 32-atom copper block for 40 steps on the CPU, and
checks that the trajectory streams through the reader and that the log is read by the convergence tools. The MACE
path is not exercised by the tests, because it downloads model files.
