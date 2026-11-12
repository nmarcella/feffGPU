# MD-EXAFS example

A synthetic trajectory from start to finish: its averaged spectrum, the comparison with a synthetic "measured"
spectrum, and the radial distributions. Nothing in this folder is measured data; the script generates every input.

The system is a rock-salt block of Zn and C (3 x 3 x 3 conventional cells, 216 atoms) whose phase file ships with the
repository (`data/validation/A000/phase.pad`: the K edge of C; potentials 0 = the C absorber, 1 = Zn, 2 = C). It is a
test system for the code, not a material of interest.

## Run it

From the root of the repository, with `feffgpu` installed:

```bash
python examples/md_exafs/make_trajectory.py --out ~/mdexafs_example     # about 20 s on a GPU
cd ~/mdexafs_example
md-exafs validate config.toml
md-exafs run config.toml                                    # about 30 s on a GPU: cache, averages, comparison
md-exafs rdf config.toml                                    # radial distributions (CPU)
md-exafs convergence config.toml                            # convergence of the averaged spectrum
```

The times are those of the 12 GB GPU used for development; yours will differ.

## What the script writes

- `trajectory.extxyz`: 20 frames in which every atom is displaced from its ideal site by a Gaussian random vector
  (0.06 Å per component), each frame with its cell. It stands in for a molecular dynamics trajectory.
- `measured.chi`: the averaged spectrum of 16 other frames of the same kind, computed with the package's forward model,
  multiplied by S0² = 0.85, shifted by ΔE0 = +1.5 eV, with noise of standard deviation 2e-4 added.
- `phase.pad` and `config.toml`: copies, so that the folder is self-contained.

## What to look at

- `mdexafs_out/average/C.chi`, `C.parts.dat`, `C.png`: the averaged spectrum, its parts by path type, and a figure.
- `mdexafs_out/compare/compare.json` and `compare.png`: S0² and ΔE0 should come out close to the planted 0.85 and
  1.5 eV. They are not exactly equal: the run and the "measured" spectrum are two different finite sets of frames. The
  `sampling` entry gives the standard error of that kind of difference (a jackknife over blocks of frames).
- `mdexafs_out/rdf/rdf.json`: the C-Zn first shell has N = 6 neighbors at about a/2 = 2.19 Å by construction, and its
  mean-square relative displacement is close to 2 x 0.06² = 0.0072 Å² (independent displacements of the two atoms).
- Change `[average] path_types` in `config.toml` and run `md-exafs average config.toml`: the parts are recomputed from
  the cache, without a new calculation.

Every key of `config.toml` is explained in the documentation page "Configuration" (`docs/md/configuration.md`), and
the steps above in "MD-EXAFS in 10 minutes" (`docs/md/quickstart.md`).
