# Outputs

Everything a run writes goes into the output folder `[system] output` (default `mdexafs_out`, relative to the
configuration file):

```text
mdexafs_out/
  config.resolved.toml        the configuration with every default filled in and absolute paths
  chi.h5                      the cache: the spectrum of every frame and its parts (a whole run, or merged shards)
  chi.shard-3-of-16.h5        the cache of shard 3 of 16 (before `md-exafs merge`)
  report.json                 the summary of the run (and, after a whole run, of its averages and comparison)
  report.shard-3-of-16.json   the summary of one shard
  report.potentials.json      the summary of `md-exafs run --potentials-only`
  potentials/<edge>/          phase files computed by the run (potentials = "reference" or "per_frame")
  average/                    averaged spectra (md-exafs average, and after md-exafs run)
  compare/                    the comparison with measured spectra (md-exafs compare, and after md-exafs run)
  rdf/                        radial distributions (md-exafs rdf)
  convergence/                thermodynamic log and convergence (md-exafs convergence)
  figures/                    figures of md-exafs plot
  dynamics.json               the summary of md-exafs dynamics
```

Units throughout: k in 1/Å, distances in Å, energies in eV, chi(k) without unit. Spectra follow FEFF's convention:
S0² = 1 and no Debye-Waller factor.

## The cache `chi.h5`

One HDF5 file per run (per shard before the merge). Rows are appended in the order in which frames are computed; the
column `frame_index` says which frame a row holds. Every reader sorts the rows by `frame_index`, so a merged file and a
single run give the same averages. Floating-point arrays are 32-bit (compressed with gzip); the k grid and the
distance grids are 64-bit.

### Attributes of the file

| Attribute | Type | Meaning |
|---|---|---|
| `format` | text | `"feffgpu md-exafs cache"` |
| `format_version` | integer | 1 |
| `settings` | text (JSON) | the settings that determine every frame's spectrum: trajectory and its size in bytes, reader, format, topology, `[atoms]`, `[lattice]`, the frame selection and absorber settings, and the physics settings of every edge. A run continues, and shards merge, only with equal settings |
| `settings_key` | text | a 16-character hash of `settings` |
| `config` | text (TOML) | the resolved configuration of the run |
| `shard`, `shard_index`, `shard_count` | text, integer, integer | the shard (`"3/16"`), or `"1/1"` for a whole run; a merged file has `"merged from 16 shards"`, 1 and 1 |
| `merged` | true / false | whether the file was made by `md-exafs merge` |
| `n_rows` | integer | rows that belong to completed blocks of frames; rows beyond it (an interrupted block) are dropped when the run continues |
| `edge_names` | text (JSON) | the edge names, in the order of the configuration |
| `feffgpu_version`, `torch_version`, `numpy_version`, `h5py_version`, `python_version` | text | the versions that wrote the file |

### Datasets of the file

| Dataset | Shape | Meaning |
|---|---|---|
| `selection` | (S,) int64 | positions in the trajectory of every selected frame of the run (all shards together) |

### The group `edges/<name>` of every edge

Attributes:

| Attribute | Meaning |
|---|---|
| `element`, `edge` | the absorbing element and the absorption edge |
| `potentials` | `"file"`, `"reference"` or `"per_frame"` |
| `rpath`, `nleg`, `criteria`, `kmax` | the path cutoff (Å), the largest number of legs, the importance filter (percent), the largest k (1/Å) |
| `pair_rmin`, `pair_dr` | start and step of the single-scattering distance table (Å) |
| `ms_bin_width` | width of the effective-length bins of the multiple-scattering parts (Å; 0 for one bin) |
| `ipot_of` | JSON: `{element: scatterer potential index}` |
| `absorbers_per_frame`, `seed` | the absorber setting (`"all"` or a number, as text) and the seed of the subsets |
| `replicate` | JSON: the repetition of the cell for the calculation |
| `path_types` | JSON list: every path type, single scattering first (the order of `chi_types`) |
| `ss_types`, `ms_types` | JSON lists: the single- and multiple-scattering types (`P` and `M` of them) |
| `ss_elements` | JSON list: the neighbor element of each single-scattering type |

Datasets that describe the grids and the phase files:

| Dataset | Shape | Meaning |
|---|---|---|
| `k` | (NK,) float64 | the k grid, 0 to `kmax` in steps of 0.05 1/Å |
| `r_table` | (NR,) float64 | the distance grid of the single-scattering tables, `pair_rmin` to `rpath` in steps of `pair_dr` |
| `ms_bin_edges` | (NB + 1,) float64 | the boundaries of the effective-length bins (Å) |
| `phase_keys` | (G,) text | the phase files used: `"file"`, `"reference"` or `"frame_<index>"` |
| `ss_tables` | (G, P, NR, NK) float32 | for every phase file, the two-leg chi(k) of a single pair at every distance of `r_table`, for each single-scattering type |

Datasets with one row per frame (F rows):

| Dataset | Shape | Meaning |
|---|---|---|
| `frame_index` | (F,) int64 | position of the frame in the trajectory |
| `phase_group` | (F,) int32 | which entry of `phase_keys` the frame used |
| `n_absorbers` | (F,) int32 | absorbing atoms averaged in the frame (the weight of the frame in averages) |
| `n_paths` | (F, max(2, nleg - 2)) int64 | multiple-scattering paths with three, four, ... `nleg` legs, over all absorbers of the frame |
| `n_short_pairs` | (F,) int64 | absorber-neighbor pairs closer than `pair_rmin` (placed on the first point of the table; should be 0) |
| `seconds` | (F,) float32 | computing time of the frame (all edges) |
| `chi` | (F, NK) float32 | the total chi(k) of the frame: the mean over its absorbers |
| `chi_ss`, `chi_ms` | (F, NK) float32 | its single- and multiple-scattering parts |
| `chi_types` | (F, T, NK) float32 | its parts by path type, in the order of `path_types`; they sum to `chi` |
| `ss_histogram` | (F, P, NR) float32 | the absorber-neighbor distance histogram per single-scattering type, per absorber, with the linear-interpolation weights of the table |
| `ms_binned` | (F, M, NB, NK) float32 | the multiple-scattering parts by type and effective-length bin; summed over the bins they give the multiple-scattering columns of `chi_types` |

The single-scattering part of type `p` in a distance range is recomputed from the cache as

```python
cols = r_table <= r_limit
chi_p = ss_histogram[f, p, cols] @ ss_tables[phase_group[f], p, cols]
```

which is what `md-exafs average` does for a limit in `[average] max_distance`.

Reading the cache in Python:

```python
import h5py, numpy as np
with h5py.File("mdexafs_out/chi.h5") as f:
    g = f["edges/Zn"]
    order = np.argsort(g["frame_index"][:])
    k, chi = g["k"][:], g["chi"][:][order]
    mean = (g["n_absorbers"][:][order, None] * chi).sum(0) / g["n_absorbers"][:].sum()
```

## `report.json`

Written by `md-exafs run`: `command`, `system`, `output`, `cache` (the file written), `shard`, `frames_selected`,
`frames_in_shard`, `frames_computed_now`, `frames_in_cache`, `seconds`, `seconds_per_frame`, `edges` (per edge:
`element`, `edge`, `potentials`, `path_types`, `absorber_atoms`, `absorbers_per_frame`), `mean_paths_per_frame`
(`"3"`, `"4"`), `short_pairs`, `device`. After a whole run (no shard) it also holds `average` (the content of
`average/average.json`) and, with a `[compare]` block, `compare` (the content of `compare/compare.json`). A shard
writes `report.shard-<i>-of-<n>.json` with the first set of keys. `md-exafs run --potentials-only` writes
`report.potentials.json` (`command`, `output`, `phase_files` per edge, `seconds`).

## `potentials/<edge>/`

For `potentials = "reference"` the files `reference.feff.inp` (the FEFF input of the cluster), `reference.pkl` (the
phase file, a pickled `feffgpu.PhaseData`) and `reference.json`; for `"per_frame"` the same with `frame_<index>`.
The JSON file holds `mode`, `key`, `frame`, `atom` (the absorbing atom at the center), `cluster_atoms`,
`cluster_radius`, `ipot_of`, `potentials` (labels), `Z`, `seconds` and `settings` (what the file was made from; a
change makes the run compute it again).

## `average/`

- `<edge>.chi`: comment lines (system, frames, selection, conventions), then the columns k, chi (the mean over frames
  weighted by the number of absorbers), spread (the weighted standard deviation between frames) and stderr (the
  standard error of the mean from blocks of consecutive frames).
- `<edge>.parts.dat`: k and the mean of every part: with a selection, the selected types; without, every type and the
  `single scattering` and `multiple scattering` parts. The header names the columns.
- `<edge>.png`: the averaged spectrum and its largest parts, k-weighted chi(k) with bands and |chi(R)|.
- `average.json`: `system`, `cache`, and per edge `element`, `edge`, `n_frames`, `first_frame`, `last_frame`,
  `absorbers` (total over the frames), `selection` (each type and what was applied), `blocks`, `effective_frames`,
  `files`, `parts`, `convergence` (`n_frames`, `r_factor_to_final`, `relative_difference_to_final`,
  `split_half_r_factor`, `window`) and `stderr_method`.

## `compare/`

- `compare.json`: `command`; `rule`; `runs` (per run: `run`, `output`, `figure`, and per spectrum `edge`, `label`,
  `chi_file`, `window`, `r_factor`, `s02`, `s02_fitted`, `de0` (eV), `de0_fitted`, `independent_points`,
  `description`); `shared`: `s02` and `de0` (`{edge name: value}`), `fitted_s02`, `fitted_de0`, `at_bound` (names whose
  value ended on a bound: not a fit), `s02_interval` and `de0_interval` (`[low, high]`, plus or minus `z` standard
  deviations with the structure held fixed), `s02_sd_log` (standard deviation of ln S0²), `de0_sd`,
  `correlation_de0_s02`, `positive_definite`, `z` (1.645), `interval_method`, `objective`, and `sampling` (`blocks`,
  `s02_se`, `de0_se`, `method`: the jackknife over blocks of frames).
- `compare.png` (one run) or `compare_<run>.png` (several runs): data in black, the model in blue, k-weighted chi(k) in
  the window and |chi(R)|.

## `rdf/`

- `<A>-<B>.dat`: r (bin centers), g(r) (NaN for clusters), pairs per center per Å, running coordination number.
- `projection_<i>.dat`: the same for each `[[rdf.projection]]`.
- `angles_<i>.dat`: angle (bin centers, degrees), counts, angles per center atom per frame.
- `rdf.png`, `peaks.png`, `angles.png`.
- `rdf.json`: `command`, `system`, `frames`, `first_frame`, `last_frame`, `r_max`, `bin_width`, `periodic`, `pairs`
  (`centers`, `neighbors`, `file` per pair), `peaks` (per peak: `label`, `center`, `neighbor`, `r_min`, `r_max`,
  `integration` with `N`, `R`, `sigma2`, `c3`, `source`; `gaussian` and `skewed` with `N`, `R`, `sigma2`, `c3`, their
  standard errors `N_se`, `R_se`, `sigma2_se`, `c3_se`, `parameters`, `parameters_se`, `r_squared`, `method`, or
  `error`), `projections` (`label`, `center`, `neighbor`, `angle`, `symmetry`, `directions`,
  `solid_angle_fraction`, `shell` with `N`, `R`, `sigma2`, `c3`, `skew`, `n_pairs`, and `file`), `angles` (`label`,
  `ranges`, `angles`, `mean`, `file`), `numbers_note`, `seconds`.

## `convergence/`

- `convergence.json`: `command`; `log` (when a log was read): `file`, `format`, `rows`, `time_span`, `time_unit`,
  `quantities` (per column `mean`, `std`, `se_block`, `unit`, `drift` with `slope`, `intercept`, `change`),
  `temperature` (`mean`, `std`, `se_block`) and `conserved_drift` (`slope`, `change`, `unit`, `relative_change`) when
  the log has them; `spectra` (when the cache exists): per edge `n_frames`, `r_factor_to_final`,
  `relative_difference_to_final`, `split_half_r_factor`, `window`; `figure`.
- `convergence.png`.

## `dynamics.json`

Written by `md-exafs dynamics`: `command`, `trajectory`, `log`, `frames`, `atoms`, `steps`, `temperature_target`,
`temperature_mean` (K, over the production steps), `calculator`, `seconds`.
