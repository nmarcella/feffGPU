# Configuration

An MD-EXAFS run is described by one TOML file. It is validated before anything runs (`md-exafs validate
config.toml`); a key that does not exist, or a value of the wrong kind, gives a message that starts with the key's
path, for example `edge.0.rpath: Input should be greater than 0`. Paths in the file are relative to the file. The
command writes the resolved configuration, with every default filled in and absolute paths, to
`config.resolved.toml` in the output folder; running that file repeats the run.

The machine-readable form of this page is the JSON Schema printed by `md-exafs schema`. Example files:
`md-exafs example` lists them, `md-exafs example nvt --out config.toml` writes one.

This page is generated from the configuration models by `docs/md/_make_config_reference.py`; do not edit it by hand.

A complete file:

```toml
# MD-EXAFS example: an NVT trajectory in extended xyz format, one absorbing element, potentials computed from a
# reference cluster, the averaged spectrum compared with a measured one, and the radial distributions.
#
# Paths are relative to this file. Run with:
#   md-exafs validate nvt.toml
#   md-exafs run nvt.toml          # cache, averages and comparison in mdexafs_out/
#   md-exafs rdf nvt.toml          # radial distributions in mdexafs_out/rdf/
# Every key is described in the documentation page "Configuration" (docs/md/configuration.md).

[system]
name = "example_nvt"                 # a label for outputs and figures
trajectory = "trajectory.extxyz"     # any format that ASE reads frame by frame
reader = "ase"                       # "ase" or "mdanalysis" (binary formats, extra feffgpu[mda])
output = "mdexafs_out"               # output folder

[lattice]
# A static cell for every frame (NVT). Give lengths and angles, three vectors, or a matrix:
a = 13.1295
b = 13.1295
c = 13.1295
alpha = 90.0
beta = 90.0
gamma = 90.0
# a = [13.1295, 0.0, 0.0]            # (or the lattice vectors)
# matrix = [[13.1295, 0, 0], [0, 13.1295, 0], [0, 0, 13.1295]]
replicate = [1, 1, 1]                # repeat the cell when it is narrower than twice rpath

[frames]
start = 0                            # first frame of the file
step = 1                             # every frame
# stop = 1000                        # end of the range (excluded); default: the end of the file
absorbers_per_frame = "all"          # or a number: a random subset per frame, from the seed and the frame index
seed = 0

[[edge]]
name = "Zn"                          # S0^2 and the E0 shift are shared between runs by this name
element = "Zn"
edge = "K"
potentials = "reference"             # computed once from a cluster of one frame (with the free-atom table of the package)
cluster_radius = 7.0                 # radius of the reference cluster (A)
scf_radius = 0.0                     # 0: no self-consistency (FEFF's default); for example 4.0 for a self-consistent potential
rpath = 6.0                          # path cutoff (A)
nleg = 4                             # largest number of legs (2, 3 or 4)
criteria = 2.0                       # importance filter of multiple-scattering paths (percent)
kmax = 18.0                          # the spectrum is calculated on 0 to kmax in steps of 0.05 1/A

[average]
# Default: every frame and every path. A selection of path types, with optional distance limits:
# path_types = ["Zn-C", "Zn-Zn", "Zn-C-C"]
# max_distance = [3.0, 4.5, 5.0]
blocks = 10                          # blocks of consecutive frames for the standard error

[compare]
chi = "measured_Zn.chi"              # k and chi columns (Athena and xraylarch files are read as they are)
window = { kmin = 3.0, kmax = 14.0, dk = 1.0, kweight = 2, rmin = 1.0, rmax = 5.0 }
fit_s02 = true                       # S0^2 and the E0 shift are fitted on the model
fit_e0 = true

[rdf]
r_max = 8.0                          # largest distance (A)
bin_width = 0.01                     # bin width (A)

[[rdf.peak]]
center = "Zn"
neighbor = "C"
r_min = 1.6
r_max = 2.8
fit = "both"                         # integration always; fits: "gaussian", "skewed", "both" or "none"

[[rdf.projection]]
center = "Zn"
neighbor = "Zn"
direction = [1, 1, 0]                # bonds within `angle` degrees of this direction (or its opposite)
angle = 20.0
symmetry = true                      # also every cubic equivalent of the direction

[[rdf.angle]]
center = "Zn"
neighbors = ["C", "C"]
ranges = [[1.6, 2.8], [1.6, 2.8]]

[run]
devices = "cuda"
```

## `[system]`

The trajectory and the output folder.

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | text | "md_exafs" | Label of the system, written in output headers and figure titles. |
| `trajectory` | text | **required** | The trajectory file: any format that ASE reads frame by frame (extended xyz, CP2K xyz, LAMMPS text dump, ASE .traj, VASP XDATCAR, ...), or a binary format with reader = "mdanalysis". Relative to the configuration file. |
| `format` | text | not set | ASE's name of the file format, for example "extxyz", "lammps-dump-text", "vasp-xdatcar" or "traj". Guessed from the file name and content when not given. With reader = "mdanalysis": MDAnalysis's format name (optional). |
| `reader` | "ase" / "mdanalysis" | "ase" | Library that reads the trajectory: "ase" (default) or "mdanalysis" (optional extra feffgpu[mda], for DCD, XTC, TRR, NetCDF and other binary formats). |
| `topology` | text | not set | Topology file for reader = "mdanalysis" (for example a PDB or LAMMPS data file that names the atoms). Relative to the configuration file. |
| `output` | text | "mdexafs_out" | Output folder of the run (the cache chi.h5, phase files, averages, figures). Relative to the configuration file. |

## `[lattice]`

The periodic cell. Give a static cell (lengths and angles, three vectors, or a matrix), or read the cell of every frame (dynamic), or declare clusters (pbc all false).

| Key | Type | Default | Description |
|---|---|---|---|
| `dynamic` | true / false | not set | true: read the cell stored in every frame (NPT runs; extended xyz Lattice, LAMMPS box, XDATCAR header). false: use the static cell given here. Not given: static when a, b, c or matrix are given, otherwise the cell of every frame. |
| `a` | number or list of number | not set | First cell length (A) or first lattice vector [x, y, z] (A). |
| `b` | number or list of number | not set | Second cell length (A) or second lattice vector (A). |
| `c` | number or list of number | not set | Third cell length (A) or third lattice vector (A). |
| `alpha` | number | 90.0 | Angle between b and c (degrees), when a, b and c are lengths. |
| `beta` | number | 90.0 | Angle between a and c (degrees), when a, b and c are lengths. |
| `gamma` | number | 90.0 | Angle between a and b (degrees), when a, b and c are lengths. |
| `matrix` | list of list of number | not set | The three lattice vectors as rows of a 3 x 3 matrix (A); an alternative to a, b and c. |
| `pbc` | list of true / false | [true, true, true] | Periodic boundaries along a, b, c: all true (a periodic system) or all false (clusters, no cell). Mixed periodicity is not supported. |
| `replicate` | list of integer | [1, 1, 1] | Repeat the cell this many times along a, b, c for the EXAFS calculation, needed when the cell is narrower than twice rpath. The absorbers are taken from the original cell only. |

## `[atoms]`

Numeric atom types and their elements, for files with numeric types (LAMMPS dumps): `1 = "Zn"`, `2 = "C"`. The form of `md_exafs` 2.0 (`Zn = 1`) is accepted too. For LAMMPS dumps the map applies to the `type` column; for other files to the atomic numbers that ASE read (a type written as 1 is read as H).

## `[frames]`

Which frames of the trajectory are used and how many absorbers per frame.

| Key | Type | Default | Description |
|---|---|---|---|
| `start` | integer | 0 | First frame (0 is the first frame of the file). |
| `stop` | integer | not set | End of the frame range (excluded), as in Python's range; not given: the end of the file. |
| `step` | integer | 1 | Use every step-th frame. |
| `absorbers_per_frame` | integer or "all" | "all" | "all": every atom of the absorbing element; a number: a random subset of that size per frame, drawn from the seed and the frame index only. |
| `seed` | integer | 0 | Seed of the random absorber subsets. |

## `[[edge]]`

One block per absorbing element. Several blocks give several edges in one run.

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | text | not set | Name of the edge in outputs and in [average] and [compare]; S0^2 and the E0 shift are shared between runs by this name. Default: the element. |
| `element` | text | **required** | Absorbing element (symbol). |
| `edge` | "K" / "L1" / "L2" / "L3" / "M1" / "M2" / "M3" / "M4" / "M5" | "K" | Absorption edge. |
| `potentials` | "reference" / "per_frame" / "file" | "reference" | Where the potentials and phase shifts come from: "reference" (computed once from a reference cluster of one frame), "per_frame" (computed again every potentials_every frames) or "file" (the phase file given in phase). |
| `phase` | text | not set | Phase file for potentials = "file": a phase.pad from FEFF or a pickle from `feffgpu phase`. Relative to the configuration file. |
| `ipots` | table | {} | Optional {element = potential index} for a phase file whose potentials are not found by atomic number (potentials = "file"). |
| `reference_frame` | integer | not set | Frame whose cluster defines the potentials ("reference"). Default: the first selected frame. |
| `reference_atom` | integer | not set | Index of the absorbing atom at the center of the reference cluster. Default: the first atom of the element. |
| `cluster_radius` | number | 7.0 | Radius of the reference cluster (A); every element must have an atom in it besides the absorber. |
| `scf_radius` | number | 0.0 | Radius of FEFF's self-consistency loop for the potentials (A); 0 for no self-consistency (FEFF's default). Self-consistent potentials take tens of seconds per cluster. |
| `phase_kmax` | number | not set | Largest k of the phase grid (1/A). Default: kmax + 2. |
| `cards` | list of text | [] | Extra FEFF cards for the reference cluster's input, written as given (for example "EXCHANGE 0 0.0 0.0"); only cards that feffgpu reads are accepted. |
| `potentials_every` | integer | 10 | For potentials = "per_frame": new potentials every this many selected frames. |
| `rpath` | number | 6.0 | Path cutoff (A): half the total path length, FEFF's RPATH. Single scattering includes every pair up to it. |
| `nleg` | 2 / 3 / 4 | 4 | Largest number of legs: 2 (single scattering only), 3 or 4. The path finder of this package stops at four legs. |
| `criteria` | number | 2.0 | FEFF's plane-wave importance filter for multiple-scattering paths (percent of the shortest single-scattering path's importance); 0 keeps every path. |
| `kmax` | number | 20.0 | Largest k of the calculated spectrum (1/A); the grid is 0 to kmax in steps of 0.05. |
| `pair_rmin` | number | 1.0 | Start of the single-scattering distance table (A). Pairs closer than this are counted and reported; lower it for systems with shorter bonds. |
| `pair_dr` | number | 0.002 | Step of the single-scattering distance table (A). |
| `ms_bin_width` | number | 0.5 | Width of the effective-length bins in which the multiple-scattering parts are stored (A); distance limits of multiple-scattering path types act on these bins. 0: one bin. |

## `[average]`

The averaged spectrum: which edges, frames and path types. Computed from the cache (md-exafs average).

| Key | Type | Default | Description |
|---|---|---|---|
| `edges` | list of text | [] | Edges to average (names); default all. |
| `path_types` | list of text | [] | Path types to include, for example ["Zn-C", "Zn-C-C"]; default every path (the total spectrum). Each edge uses the types that start with its absorbing element; an edge with none of them keeps its total spectrum. |
| `max_distance` | number or list of number | not set | Largest effective path length (A) of each path type in path_types (one number for all, or one per type). Single scattering: any value; multiple scattering: a multiple of the edge's ms_bin_width. |
| `start` | integer | not set | First frame of the average (trajectory index); default the first in the cache. |
| `stop` | integer | not set | End of the frames of the average (excluded); default after the last in the cache. |
| `step` | integer | 1 | Use every step-th frame of the cache's frames between start and stop. |
| `blocks` | integer | 10 | Number of blocks for the block-averaged standard error and the convergence curve. |

## `[compare] or [[compare]]`

One table for one measured spectrum, or one `[[compare]]` block per spectrum (for example one per edge).

| Key | Type | Default | Description |
|---|---|---|---|
| `edge` | text | not set | Name of the edge this spectrum belongs to; may be left out when the run has one edge. |
| `chi` | text | **required** | Measured chi(k): a text file with k (1/A) and chi columns (Athena and xraylarch .chi files are read as they are). Relative to the configuration file. |
| `k_column` | integer | 0 | Zero-based column of k. |
| `chi_column` | integer | 1 | Zero-based column of chi. |
| `window` | table | the defaults below | The k window, k weight and R range of the comparison. |
| `fit_s02` | true / false | true | Fit S0^2 (on the model). |
| `fit_e0` | true / false | true | Fit the E0 shift (on the model). |
| `s02` | number | 1.0 | S0^2: the start value when fitted, else the held value. |
| `e0` | number | 0.0 | E0 shift (eV): the start value when fitted, else the held value. |
| `s02_bounds` | list of number | [0.2, 5.0] | Allowed range of S0^2 (a safety range, not a prior). |
| `e0_bounds` | list of number | [-15.0, 15.0] | Allowed range of the E0 shift (eV). |
| `label` | text | not set | Label of the spectrum in tables and figures; default the file name. |

## `window = {...} in [compare]`

An inline table: `window = { kmin = 3.0, kmax = 14.0, dk = 1.0, kweight = 2, rmin = 1.0, rmax = 5.0 }`.

| Key | Type | Default | Description |
|---|---|---|---|
| `kmin` | number | 3.0 | Center of the low-k sill of the Hanning window (1/A). |
| `kmax` | number | 14.0 | Center of the high-k sill of the Hanning window (1/A). |
| `dk` | number | 1.0 | Width of each sill of the Hanning window (1/A). |
| `kweight` | integer | 2 | Power of k that multiplies chi(k) before the Fourier transform. |
| `rmin` | number | 1.0 | Start of the R range of the R-factor (A). |
| `rmax` | number | 6.0 | End of the R range of the R-factor (A). |

## `[rdf]`

Radial distribution functions of the trajectory (md-exafs rdf).

| Key | Type | Default | Description |
|---|---|---|---|
| `r_max` | number | 8.0 | Largest distance of the RDFs (A). Any value: periodic images are included when it exceeds half the cell width. |
| `bin_width` | number | 0.01 | Bin width (A). |
| `pairs` | list of list of text | [] | Center-neighbor element pairs, for example [["Zn", "C"]]; default every pair of the system's elements. |
| `start` | integer | not set | First frame; default [frames] start. |
| `stop` | integer | not set | End of the frame range (excluded); default [frames] stop. |
| `step` | integer | not set | Frame step; default [frames] step. |
| `device` | text | not set | Torch device of the histogram accumulation; default the first of [run] devices ("cpu" works too). |
| `peak` | list of table | [] | Peaks to characterize ([[rdf.peak]] blocks). |
| `projection` | list of table | [] | RDFs along crystal directions ([[rdf.projection]] blocks). |
| `angle` | list of table | [] | Bond-angle distributions ([[rdf.angle]] blocks). |

## `[[rdf.peak]]`

A peak of a partial RDF to characterize.

| Key | Type | Default | Description |
|---|---|---|---|
| `center` | text | **required** | Element of the center atoms. |
| `neighbor` | text | **required** | Element of the neighbor atoms. |
| `r_min` | number | **required** | Lower end of the peak's distance range (A). |
| `r_max` | number | **required** | Upper end of the peak's distance range (A). |
| `fit` | "gaussian" / "skewed" / "both" / "none" | "both" | Fitted forms besides the plain integration: "gaussian", "skewed" (skew-normal), "both" or "none". |

## `[[rdf.projection]]`

An RDF of the bonds along a crystal direction.

| Key | Type | Default | Description |
|---|---|---|---|
| `center` | text | **required** | Element of the center atoms. |
| `neighbor` | text | **required** | Element of the neighbor atoms. |
| `direction` | list of number | **required** | Direction [x, y, z] in Cartesian coordinates (for a cubic cell, the crystal direction [h, k, l]). |
| `angle` | number | 30.0 | A bond counts when its angle to the direction (or to its opposite) is at most this (degrees). |
| `symmetry` | true / false | false | Also count bonds along every cubic-symmetry equivalent of the direction. |
| `r_min` | number | not set | Lower end of the distance range of the reported numbers (A); default 0. |
| `r_max` | number | not set | Upper end of the distance range of the reported numbers (A); default the RDF's r_max. |

## `[[rdf.angle]]`

A bond-angle distribution B-A-C.

| Key | Type | Default | Description |
|---|---|---|---|
| `center` | text | **required** | Element of the center atom A. |
| `neighbors` | list of text | **required** | Elements of the two neighbors [B, C] (they may be the same). |
| `ranges` | list of list of number | **required** | Distance ranges [[r_min, r_max], [r_min, r_max]] (A) of A-B and A-C that define a neighbor. |
| `bin_width` | number | 1.0 | Bin width of the angle histogram (degrees). |

## `[convergence]`

The thermodynamic log of the dynamics (md-exafs convergence).

| Key | Type | Default | Description |
|---|---|---|---|
| `log` | text | not set | Log file: a CP2K .ener file, an ASE molecular-dynamics log, or a CSV file with a header row. Relative to the configuration file. Default: the log written by [dynamics]. |
| `format` | "auto" / "cp2k" / "ase" / "csv" | "auto" | Format of the log; "auto" guesses it from the content. |
| `skip` | number | 0.0 | Leading part of the log to leave out of the statistics (equilibration), in the log's time unit (fs for CP2K, ps for ASE logs, the time column of a CSV). |
| `blocks` | integer | 10 | Number of blocks for block averages. |

## `[dynamics]`

The optional runner: NVT Langevin dynamics with an ASE calculator, written to [system] trajectory.

| Key | Type | Default | Description |
|---|---|---|---|
| `structure` | text | **required** | Starting structure: any file that ase.io reads (CIF, POSCAR, extended xyz, ...). Relative to the configuration file. |
| `supercell` | list of integer | [1, 1, 1] | Repeat the starting structure this many times along a, b, c. |
| `calculator` | text | "mace_mp" | "mace_mp" (a MACE foundation model, extra feffgpu[mace]), "emt" (ASE's effective-medium potential for a few metals, for tests), or "module:function" returning any ASE calculator. |
| `model` | text | "medium" | Model name or file for calculator = "mace_mp". |
| `calculator_options` | table | {} | Extra keyword arguments for the calculator's constructor. |
| `temperature` | number | 300.0 | Thermostat temperature (K). |
| `timestep` | number | 2.0 | Time step (fs). |
| `friction` | number | 0.01 | Langevin friction (1/fs). |
| `steps` | integer | 10000 | Production steps (after equilibration). |
| `equilibration` | integer | 1000 | Steps before frames are written. |
| `save_every` | integer | 10 | Write a frame every this many steps. |
| `log_every` | integer | 10 | Write a line of the thermodynamic log every this many steps. |
| `seed` | integer | 0 | Seed of the initial velocities and of the thermostat. |
| `log` | text | "md.log" | ASE molecular-dynamics log written by the runner. Relative to the configuration file. |

## `[run]`

Devices and batch sizes.

| Key | Type | Default | Description |
|---|---|---|---|
| `devices` | text or integer or list of integer | "cuda" | Devices: "cuda", "cuda:1", "cpu", "all", a count or a list of indices. This version runs on the first; split a run over several GPUs with shards. |
| `absorber_batch` | integer | 0 | Absorbers per batch of the distance histogram; 0 chooses it from the free GPU memory. |
| `path_chunk` | integer | 0 | Paths per batch of the path stage; 0 chooses it from the free GPU memory. |
| `ms_group` | integer | 64 | Absorbers whose multiple-scattering paths are enumerated together. |
| `frames_per_block` | integer | 8 | Frames computed between two writes to the cache. |
