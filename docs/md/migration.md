# Coming from `md_exafs` 2.0

MD-EXAFS in this package continues the program `md_exafs` 2.0
(<https://github.com/Joint-Photon-Sciences-Institute/MD-EXAFS>). The workflow is shorter: the original wrote one FEFF
input per frame and absorbing atom with OVITO, ran FEFF on each (locally or with a SLURM script), and averaged the
`chi.dat` files (or a SQLite database built from FEFF's path files). Here one command computes every frame's spectrum
on the GPU into one HDF5 file, and the averages, selections of paths and comparisons are computed from it. OVITO, the
FEFF input tree, the FEFF executable, the launcher and the database are gone.

## Commands

| `md_exafs` 2.0 | Now |
|---|---|
| `md-exafs-process config.toml` + `md-exafs-feff-local` or `sbatch run_slurm.sh` | `md-exafs run config.toml` (`--shard i/n` for several GPUs or nodes, then `md-exafs merge`) |
| `md-exafs-average --config averaging.toml` | `md-exafs average config.toml` (runs after `md-exafs run` too) |
| `md-exafs-average --build-database ...`, `--use-database` | not needed: the cache `chi.h5` is always written |
| `md-exafs-md --check_convergence --input x.ener --output plot.png` | `md-exafs convergence x.ener` |
| `md-exafs-md --rdf rdf.toml` | `md-exafs rdf config.toml` (the `[rdf]` section of the run's configuration) |
| `md-exafs-md --get_lattice_vectors` | not needed: `[lattice] dynamic = true` reads the cell of every frame |
| `md-exafs-md-input-gen --cp2k --input s.cif --size 3,3,3 --output s.xyz` | `md-exafs supercell s.cif --size 3,3,3 --out s.xyz --cp2k` (without `--cp2k`: any format ASE writes) |
| `plot_chi_from_database.py` | `md-exafs plot config.toml` |

## Keys of the processing file

| `md_exafs` 2.0 | Now | Notes |
|---|---|---|
| `[system] name` | `[system] name` | |
| `[system] trajectory_file` | `[system] trajectory` | any format ASE reads; `reader = "mdanalysis"` for binary formats |
| `[system] output_directory` | `[system] output` | holds the cache and the results instead of FEFF inputs |
| `[lattice] a`, `b`, `c` (vectors) | `[lattice] a`, `b`, `c` | vectors are still accepted; lengths and angles, or `matrix`, too |
| `[lattice] pbc` | `[lattice] pbc` | all true (periodic) or all false (clusters) |
| `[lattice] dynamic` | `[lattice] dynamic` | |
| `[atoms] U = 1` (element = type) | `[atoms] 1 = "Zn"` (type = element) | the old form is still accepted |
| `[processing] start_frame` | `[frames] start` | |
| `[processing] end_frame` | `[frames] stop` | excluded, as before; may be left out for the end of the file |
| `[processing] frame_step` | `[frames] step` | |
| `[processing] atoms_per_frame` | `[frames] absorbers_per_frame` | now also `"all"`; a subset is drawn from `[frames] seed` and the frame index, so it is reproducible |
| `[processing] cutoff_radius` | `[[edge]] rpath` | the path cutoff; the radius of the cluster that defines the potentials is `[[edge]] cluster_radius` |
| `[feff] absorbing_element` | `[[edge]] element` | one `[[edge]]` block per absorbing element; several edges in one run |
| `[feff] header` (FEFF cards) | `[[edge]] edge`, `nleg`, `criteria`, `rpath`, `scf_radius`, `cards` | other cards of the reference cluster go into `cards` |
| `[feff] feff_executable` | — | no FEFF executable |

## Keys of the averaging file

| `md_exafs` 2.0 | Now | Notes |
|---|---|---|
| `[averaging] input_directory` | — | the cache is in `[system] output` |
| `[averaging] frame_range` | `[average] start`, `stop` | plus `step` |
| `[averaging] output_file` | — | files in `<output>/average/` ([](outputs.md)) |
| `[averaging] paths` (path numbers) | — | paths are not numbered: select path types |
| `[averaging.multipath] paths` | `[average] path_types` | the two spellings of a path and its time reverse name one type ([](path_types.md)) |
| `[averaging.multipath] max_distance` | `[average] max_distance` | one number or one per type; multiple-scattering limits on the bins of `ms_bin_width` |
| `[averaging.multipath] num_processes` | — | |
| `[averaging.database]` | — | the cache `chi.h5` |

## Keys of the RDF file

| `md_exafs` 2.0 | Now | Notes |
|---|---|---|
| `[input] trajectory_file` | `[system] trajectory` | the RDF uses the run's configuration |
| `[output] directory`, `report_file`, `plot_file` | — | fixed names in `<output>/rdf/` |
| `[analysis] frame_start`, `frame_end`, `frame_step` | `[rdf] start`, `stop`, `step` | default: the frames of `[frames]` |
| `[rdf] cutoff_radius` | `[rdf] r_max` | |
| `[rdf] num_bins` | `[rdf] bin_width` | `bin_width = cutoff_radius / num_bins` |
| `[[peaks]] center`, `neighbor`, `r_min`, `r_max` | `[[rdf.peak]]` with the same keys | |
| `[[peaks]] fit_type` | `[[rdf.peak]] fit` | `"gaussian"`, `"skewed"`, `"both"`, `"none"` |
| `[[peaks]] constrain_gaussian_mean` | — | not provided |
| `[projections] angle_threshold` | `[[rdf.projection]] angle` | per projection |
| `[projections] use_symmetry` | `[[rdf.projection]] symmetry` | per projection |
| `[[projections.bonds]] center`, `neighbor`, `direction` | `[[rdf.projection]]` with the same keys | |

## What changed in the results

- **The spectra.** The original ran FEFF on a cluster around each sampled absorber. Here single scattering comes from
  every pair within `rpath` through exact pair tables, and multiple scattering from FEFF's path finder and path stage
  for every absorber, on the GPU; the GPU code is validated against FEFF 8.5L ([](../validation/index.md)). Paths
  have at most four legs, where the original's FEFF runs could use up to eight ([](path_types.md)).
- **The potentials.** The original computed potentials for every cluster (with self-consistency when its header asked
  for it). By default the new version computes them once per edge from a reference cluster; `per_frame` recomputes
  them at intervals. How much this changes a spectrum depends on the material and has not been measured for this
  package ([](potentials.md)).
- **Path types.** A path and its time reverse name one type (`Zn-C-Zn` and `Zn-Zn-C` are the same paths); the original
  kept FEFF's spelling, which varies from cluster to cluster.
- **Peak fits.** The Gaussian is fitted to the pair density (pairs per center per Å), like the skewed form, instead of
  to g(r); the fitted N is the area inside the range, with standard errors. The plain integration numbers are now the
  exact moments of the distances.
- **Projected RDFs** are normalized by the solid angle of their cones; the original's curves are smaller by that
  fraction.
- **Not carried over:** the X-ray weighted total RDF of `--rdf-save`, the fixed-mean Gaussian fit, the selection of
  paths by FEFF path number, and the tar archive of the FEFF inputs.
