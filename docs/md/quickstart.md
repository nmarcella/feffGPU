# MD-EXAFS in 10 minutes

MD-EXAFS computes the EXAFS spectrum of a molecular dynamics trajectory. For every frame it takes the mean, over
the absorbing atoms, of FEFF's chi(k) (S0² = 1 and no Debye-Waller factor: the disorder is in the positions), and
then averages over the frames. Each frame's spectrum is computed on the GPU with this package's forward model:
single scattering from every pair within the cutoff (exact pair tables), multiple scattering from FEFF's path finder
and path stage for every absorber. The spectra of all frames, with their parts by path type, are kept in one HDF5
file, so that averages, selections of paths and comparisons with a measured spectrum need no new calculation.

This page runs the example of the repository from start to finish: a generated trajectory, its averaged spectrum,
the comparison with a synthetic "measured" spectrum, and the radial distributions.

## What you need

- `feffgpu` installed with a CUDA build of PyTorch ([](../installation.md)). The example also runs on a CPU, slowly.
- A checkout of the repository: the example uses `examples/md_exafs/` and the phase file
  `data/validation/A000/phase.pad`. It does not need the free-atom table, because the phase file is given.

The example system is a rock-salt block of Zn and C atoms (216 atoms) whose phase file ships with the repository: the
K edge of C, with potentials 0 = the C absorber, 1 = Zn, 2 = C. It is a test system for the code, not a material
of interest.

## 1. Make a trajectory and a "measured" spectrum

```bash
python examples/md_exafs/make_trajectory.py --out ~/mdexafs_example
```

This writes into `~/mdexafs_example/` (any folder outside the checkout will do):

- `trajectory.extxyz`: 20 frames in which every atom is displaced from its ideal site by a Gaussian random vector
  (0.06 Å per component), each frame with its cell. It stands in for a molecular dynamics trajectory;
- `measured.chi`: the averaged spectrum of 16 other frames of the same kind, computed with the forward model, then
  multiplied by S0² = 0.85, shifted by ΔE0 = +1.5 eV and given a little noise;
- copies of `config.toml` and of the phase file.

## 2. Check the configuration

```bash
cd ~/mdexafs_example
md-exafs validate config.toml
```

The essential parts of `config.toml`:

```toml
[system]
trajectory = "trajectory.extxyz"
output = "mdexafs_out"

[lattice]
dynamic = true                      # every frame carries its cell

[frames]
absorbers_per_frame = "all"

[[edge]]
element = "C"
potentials = "file"
phase = "phase.pad"
rpath = 5.5                         # path cutoff (Å)
nleg = 4                            # single, double and triple scattering
kmax = 16.0

[compare]
chi = "measured.chi"
window = { kmin = 3.0, kmax = 13.0, dk = 1.0, kweight = 2, rmin = 1.0, rmax = 5.2 }
```

Every key is described in [](configuration.md). A key that does not exist or a wrong value is reported with its
path (`edge.0.rpath: ...`) before anything runs.

## 3. Run

```bash
md-exafs run config.toml
```

The command reads the frames one by one, computes the spectrum of every frame, writes the cache
`mdexafs_out/chi.h5` block by block, and then writes the averages and the comparison. On the 12 GB GPU used for
development the 20 frames took about 25 s (about 1.2 s per frame with all 108 absorbers). A run that is stopped
continues where it stopped when it is started again.

## 4. Look at the results

`mdexafs_out/average/` holds the averaged spectrum:

- `C.chi`: k, the mean chi(k), the spread between frames and the standard error of the mean (from blocks of
  consecutive frames);
- `C.parts.dat`: the parts by path type (`C-Zn`, `C-C`, `C-C-Zn`, ...) and the single- and multiple-scattering parts;
- `C.png`: the averaged spectrum and its largest parts in k and in R;
- `average.json`: the numbers, including how the mean converges with the number of frames.

`mdexafs_out/compare/` holds the comparison with `measured.chi`:

- `compare.json`: the R-factor, the fitted S0² and E0 shift, their intervals with the structure held fixed, and
  their sampling standard errors from a jackknife over blocks of frames;
- `compare.png`: the measured and the calculated spectrum in k and in R.

The fitted S0² and shift should come out close to the planted 0.85 and 1.5 eV, within a few sampling standard errors:
the "measured" spectrum and the run are two different finite sets of frames. In the run made for this page they were
0.853 ± 0.007 and 1.51 ± 0.03 eV (sampling standard errors).

## 5. Select path types without a new calculation

Add a selection to `[average]` and average again:

```toml
[average]
path_types = ["C-Zn", "C-C", "C-C-Zn"]
max_distance = [2.5, 3.5, 4.5]      # Å, one limit per type
```

```bash
md-exafs average config.toml
```

Only the cache is read. Single-scattering parts are recomputed from the stored pair-distance histograms, so their
limits can be any distance; multiple-scattering parts are stored in bins of effective path length, so their limits
must lie on a bin boundary ([](path_types.md)).

## 6. Radial distributions

```bash
md-exafs rdf config.toml
```

`mdexafs_out/rdf/` then holds the partial radial distribution functions, the numbers of each `[[rdf.peak]]` (the
coordination number, mean distance, mean-square relative displacement and third cumulant by integration, next to
Gaussian and skewed fits), a projected RDF and a bond-angle distribution ([](analysis.md)).

## 7. Your own trajectory

```bash
md-exafs example nvt --out config.toml      # a commented template
```

- Point `[system] trajectory` at your file: extended xyz, CP2K xyz, LAMMPS text dumps, ASE `.traj`, VASP XDATCAR
  and the other formats that ASE reads ([](trajectories.md)).
- Give the cell in `[lattice]` (or `dynamic = true` when the frames carry it) and, for numeric atom types, the map
  in `[atoms]`.
- For the potentials, the default `potentials = "reference"` computes one phase file per edge from a cluster of
  your trajectory; it uses the free-atom table that ships with the package. See [](potentials.md).
- Split long runs over several GPUs or nodes with shards ([](shards.md)).
- No trajectory yet? `md-exafs dynamics` makes one with a MACE potential or any ASE calculator ([](dynamics.md)).

## Where MD-EXAFS comes from

This tool continues the program MD-EXAFS (`md_exafs` 2.0, <https://github.com/Joint-Photon-Sciences-Institute/MD-EXAFS>,
MIT license), which wrote one FEFF input per frame and absorbing atom and ran FEFF on each. The new version computes
the spectra with the GPU forward model of this package, needs neither OVITO nor a FEFF executable, and keeps the
configuration sections and the analysis tools of the original ([](migration.md)). When you publish results, cite
FEFF ([](../citing.md)) and the original MD-EXAFS repository.
