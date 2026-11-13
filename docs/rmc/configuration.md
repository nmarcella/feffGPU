# Configuration

Every run of `rmc-exafs` is described by one TOML file. The file is validated before anything runs; a problem is
reported with the dotted path of the key (for example `sample.0.edge.1.window.kmax: kmax (2.0) must be larger
than kmin (3.0)`). Keys that the models below do not know are refused, so a typing error never passes silently.

```bash
rmc-exafs validate config.toml     # the keys, then the files they name (structure, chi, phase), without a GPU
rmc-exafs schema                   # the JSON Schema of the file
rmc-exafs example                  # the example configurations that ship with the package
rmc-exafs example one_edge --out config.toml
```

## Rules

- **Relative paths are relative to the TOML file**, not to the directory you run from.
- A run writes the **resolved configuration** to `config.resolved.toml` in its output directory: every path
  absolute, every default filled in. That file can be run again as it is.
- **Sections.** `[run]` and at least one `[[sample]]` with at least one `[[sample.edge]]` are required. `[search]`
  is required by `rmc-exafs run`, `[refine]` by `rmc-exafs refine`. `[amplitude]`, `[moves]` and `[constraints]`
  can be left out (their defaults apply). The simplest run is one sample with one edge and no `[moves]` section.
- **Spectrum names.** Spectra with the same `name` in several samples share S0² and the E0 shift, so they must have
  the same absorber. Within one sample the names must differ.
- **Keys that depend on others.** A sample's `hop_sites` needs a `[moves.hops]` section (the element that hops);
  gradient steps need `box_rule = "project"` or `"auto"`; `[amplitude].from_run` belongs to the modes `"fit"`
  and `"from_run"`.
- **EvAX's keywords.** The keys of `[search]` and `[constraints]` that have an EvAX counterpart name it in brackets
  in their description. Where the default differs from EvAX's rule, [](evax_differences.md) lists it with the
  switch that restores EvAX's rule.

## A complete file

```toml
[run]
output = "runs/my_fit"
seed = 1
devices = "cuda"

[[sample]]
label = "sample_a"
structure = "start.vasp"
supercell = [1, 1, 1]

  [[sample.edge]]
  name = "Zn"
  absorber = "Zn"
  chi = "sample_a_Zn.chi"
  phase = "phase_Zn.pkl"
  potentials = { Zn = 1, O = 2 }
  window = { kmin = 3.0, kmax = 13.0, dk = 2.0, kweight = 2, rmin = 1.0, rmax = 3.6 }
  rpath = 5.0
  nleg = 3

[amplitude]
mode = "fit"
s02_bounds = [0.5, 1.5]
de0_bounds = [-8.0, 8.0]
refit_from = 101
refit_every = 10

[search]
states = 16
generations = 600
froze_in = 500
gradient_steps = 50

[moves]
exchange = ["Zn", "Cd"]
from_generation = 101
hops = { element = "O", sites = "o_sites.txt", probability = 0.5 }

[constraints]
max_displacement = 0.4
closest_distance = { "Zn-O" = 1.8, "O-O" = 2.6 }
```

## Every key

The tables are generated from the models in `feffgpu/rmc/config.py` by `docs/rmc/_make_config_reference.py`.

<!-- BEGIN GENERATED -->

### `[run]`

[run]: where the outputs go, the seed and the devices.

| key | type | default | meaning |
|---|---|---|---|
| `output` | text | required | Directory of the outputs. Every sample writes into a subdirectory named by its label; run.json and config.resolved.toml go into the directory itself. |
| `seed` | integer | `1` | Seed of the random numbers. Every sample draws from its own generator, seeded with seed + 11, in a fixed order: the same configuration and seed give the same run. |
| `devices` | text or integer or list of integer | `"cuda"` | The device(s): 'cuda', 'cuda:1', 'cpu', 'all', a number of GPUs, or a list of GPU indices (feffgpu.parallel.resolve_devices). This version runs on the first of them. |
| `save_every` | integer | `25` | Generations between checkpoints (checkpoint.npz and history.json of every sample). |

### `[[sample]]`

[[sample]]: one sample: a start structure and its measured spectra.

| key | type | default | meaning |
|---|---|---|---|
| `label` | text | required | Name of the sample (letters, digits, '.', '_', '-'): its output subdirectory. |
| `structure` | text | required | The start structure (POSCAR, CIF, extended xyz, or any file ASE reads), with its periodic cell. The position of every atom is its lattice site. |
| `format` | text | none | ASE's format name of the structure file, if its name does not tell. |
| `supercell` | list of integer | `[1, 1, 1]` | Replication of the structure along its three cell vectors. |
| `hop_sites` | text | none | The sites of the hop sublattice of this sample ('fluorite' or a file); default moves.hops.sites. |
| `edge` | list of table | required | The measured spectra of this sample, one [[sample.edge]] block each. |

### `[[sample.edge]]`

[[sample.edge]]: one measured spectrum of a sample and its forward model.

| key | type | default | meaning |
|---|---|---|---|
| `name` | text | required | Label of the spectrum. Spectra with the same name in several samples (the same edge of the same compound) share S0^2 and the E0 shift. |
| `absorber` | text | required | The absorbing element (its atoms are the absorbers). |
| `chi` | text | required | The measured chi(k): a text file with k (1/A) and chi in columns; comment lines are skipped, so Athena's and xraylarch's .chi files are read as they are. Never divided by S0^2. |
| `chi_columns` | list of integer | `[0, 1]` | Zero-based columns of k and chi in the chi file. |
| `phase` | text | required | The phase file of this edge: FEFF 8.5L's phase.pad, or a pickle written by `feffgpu phase` (.pkl). Its potential 0 is the absorber. |
| `potentials` | table of integer | `{}` | {element: scattering potential index} in the phase file (1 or more), as in the POTENTIALS block of the feff.inp it was made from. May be left out when the phase file has a single scattering potential. |
| `window` | table | required | The k window, k weight and R window: { kmin, kmax, dk, kweight, rmin, rmax }. |
| `rpath` | number | `6.0` | Pair and path cutoff (A): FEFF's RPATH, EvAX's R_max_for_FEFF. The periodic cell must be at least twice as wide. |
| `nleg` | integer | `4` | Largest number of legs of the scattering paths: 2 (single scattering only) up to 8 [N_legs; EvAX's default is 8]. |
| `criteria` | number | `2.5` | Importance filter of the multiple-scattering paths, in percent; what it measures is set by criteria_rule (FEFF's CRITERIA, EvAX's FEFF_criteria). |
| `criteria_rule` | one of "feff", "evax" | `"feff"` | 'feff': every path of each structure as it is, judged alone by FEFF's plane-wave importance relative to the shortest single-scattering path (what FEFF's path finder does on a structure without symmetry); the paths are found again every paths_every generations. 'evax': EvAX's rule: the paths are chosen once on the start lattice; paths of like geometry form groups as in EvAX's path finder (see group_tolerance), and a group is kept when the size of its signal on the lattice is more than criteria percent of the size of the lattice's whole spectrum. Not with exchanges or hops. |
| `group_tolerance` | number | `0.5` | With criteria_rule = 'evax': two paths belong to one group when their atoms, one by one, lie within this distance (A) of each other in EvAX's path coordinates [Pathfinder_robustness]. |
| `multiple_scattering` | true or false | `true` | False: single scattering only. |
| `objective` | one of "r_factor", "wavelet", "cauchy_wavelet" | `"r_factor"` | The misfit of this spectrum: xraylarch's R-factor on chi(R), EvAX's wavelet residual, or xraylarch's Cauchy wavelet residual. |
| `wavelet_r0` | number | none | EvAX's R0_for_WT (A) of the 'wavelet' objective; default the middle of the R window. |
| `pair_min` | number | `1.5` | Shortest pair distance that can enter single scattering (A): the start of the pair tables. |
| `pair_windows` | table of list of number | `{}` | Optional {element: [r0, r1, ...]} (A): the pairs with that element are counted from the first to the last boundary (default pair_min to rpath). Inner boundaries split the sum without changing the spectrum. |
| `ignore` | list of text | `[]` | Elements the forward model of this spectrum does not see (they neither absorb nor scatter). An atom that every spectrum ignores does not move. |

### `window = { ... }`

The window of a comparison with a measured spectrum, in xraylarch's terms (``feffgpu.exafs.FitWindow``).

| key | type | default | meaning |
|---|---|---|---|
| `kmin` | number | required | Center of the low-k sill of the Hanning window (1/A). |
| `kmax` | number | required | Center of the high-k sill of the Hanning window (1/A). |
| `dk` | number | `1.0` | Width of each sill of the Hanning window (1/A). |
| `kweight` | integer | `2` | Power of k that multiplies chi(k) before the Fourier transform. |
| `rmin` | number | required | Start of the R range of the R-factor (A). |
| `rmax` | number | required | End of the R range of the R-factor (A). |

### `[amplitude]`

[amplitude]: S0^2 and the E0 shift, one value per spectrum name, shared by the samples.

| key | type | default | meaning |
|---|---|---|---|
| `mode` | one of "fit", "fixed", "from_run" | `"fit"` | 'fit': refitted jointly over all samples on the best states (feffgpu.exafs.fit_shared); 'fixed': held at s02 and de0 (EvAX's rule); 'from_run': held at the shared values of the run named by from_run. |
| `s02` | number or table of number | `1.0` | S0^2: the start value ('fit') or the held value ('fixed'); one number for every spectrum or {name: value}. |
| `de0` | number or table of number | `0.0` | The E0 shift (eV): the start or held value; one number or {name: value}. The pair tables and paths are evaluated at it; a fit changes it by interpolation. |
| `s02_bounds` | list of number | `[0.5, 1.5]` | Allowed range of S0^2 in the fits: a safety range, not a prior. A value on a bound is reported. |
| `de0_bounds` | list of number | `[-8.0, 8.0]` | Allowed change of the E0 shift in the fits (eV), relative to the start value. |
| `refit_every` | integer | `10` | Generations between refits during the search ('fit'). |
| `refit_from` | integer | `101` | First generation of a refit during the search ('fit'); 0: no refit during the search. A final refit on the best states is always made in mode 'fit'. |
| `fit_iterations` | integer | `60` | L-BFGS iterations of a refit. |
| `from_run` | text | none | A finished run of this command (its output directory). Mode 'from_run': its shared S0^2 and E0 shifts are held. Mode 'fit': they are the start values. |
| `noise_floor` | number | `0.0` | Least misfit taken as the noise level in the intervals of S0^2 and the E0 shift (the misfit at the optimum is used when it is larger). |

### `[search]`

[search]: the population search under EvAX's rules (EvAX's keyword in brackets).

| key | type | default | meaning |
|---|---|---|---|
| `states` | integer | `32` | Number of states of the population [Number_of_states]; even (EvAX breeds in pairs). |
| `generations` | integer | required | Number of generations of this run [Stop_after]. |
| `cells_per_state` | integer | `1` | Copies of the cell in one state, whose spectra are averaged (EvAX: 1). |
| `step` | number | `0.005` | The random move: a uniform step within +/- step on every coordinate of every atom (A) [Maximal_step_length]. |
| `acceptance` | number | `0.8` | Probability of accepting a worse move of mean size at the start, cooled linearly to zero at froze_in [Acceptance_rate]; 0 or less: worse moves are never accepted. |
| `froze_in` | integer | `1500` | Generation count (over all continued runs) at which worse moves stop being accepted [Froze_in]. |
| `means_from` | integer | `0` | Generation count at which the running mean of the size of the misfit changes (the acceptance) starts again from zero moves; 0: never (EvAX's rule). |
| `weights_means_from` | integer | `0` | Generation count at which the running mean of the spectrum weights starts again; 0: never (EvAX's rule). |
| `adapt_weights` | true or false | `true` | Adapt the weights of the spectra of a sample during the run [adjust_spectra_weights]; false: equal weights. |
| `gradient_steps` | integer | `0` | FIRE steps on the gradient of the misfit after every random move (an addition to EvAX; 0: EvAX's rule). |
| `gradient_force` | number | `1.0` | Largest force (eV/A) of the misfit on an atom of the start structure: sets the size of the gradient steps. A continued run keeps the scale of its first run. |
| `start_kick` | number | `0.01` | Random displacement that makes the start states differ (A) [Initial_displacement]. |
| `start_distribution` | one of "gaussian", "uniform" | `"uniform"` | 'uniform': every coordinate uniform in [-start_kick, start_kick] / sqrt(2) (EvAX's start); 'gaussian': normal with standard deviation start_kick. |
| `paths_every` | integer | `10` | Generations between new searches of every state's multiple-scattering paths (between them the paths follow the atoms). |
| `generation_offset` | integer | none | Generations done before this run (default: 0, or those of continue_from). |
| `continue_from` | text | none | A finished run of this command (its output directory): its population, S0^2, E0 shifts, spectrum weights, running means and generation count continue here. |

### `[moves]`

[moves]: rearrangements in addition to EvAX's random move.

| key | type | default | meaning |
|---|---|---|---|
| `exchange` | list of text | `[]` | Two elements whose atoms exchange positions and sites; empty: no exchanges. A sample without both elements makes none. |
| `exchange_probability` | number | `0.5` | Probability that a state proposes exchanges in a generation. |
| `exchange_count` | integer | `1` | Exchanges per cell of a state that proposes them. |
| `hops` | table | none | Hops onto vacant sites (see [moves.hops]); absent: no hops. |
| `from_generation` | integer | `1` | First generation of this run with exchanges and hops. |

### `[moves.hops]`

[moves.hops]: hops of the atoms of one element onto vacant sites of their sublattice.

| key | type | default | meaning |
|---|---|---|---|
| `element` | text | required | The element that hops. |
| `sites` | text | required | All sites of its sublattice, occupied or vacant: 'fluorite' (the built-in generator: the anion sites of a fluorite structure) or a file (three columns x y z in A, or a structure file whose atoms are the sites). |
| `radius` | number | none | Two sites are neighbors within this distance (A); default 1.1 times the nearest site distance (the built-in generator: the six nearest sites). |
| `source` | one of "neighbors", "any" | `"neighbors"` | Where the hopping atom comes from: a neighboring site of the vacant site, or any occupied site. |
| `probability` | number | `0.5` | Probability that a state proposes hops in a generation. |
| `count` | integer | `1` | Hops per cell of a state that proposes them. |
| `jitter` | number | `0.05` | Normal displacement of a hopped atom from its new site, per coordinate (A). |

### `[constraints]`

[constraints]: the displacement limit and the closest distances.

| key | type | default | meaning |
|---|---|---|---|
| `max_displacement` | number | `0.4` | Every atom stays within this distance (A) of its site of the fixed start lattice [Potential_type box, Maximal_displacement]; 0: no limit. |
| `box_rule` | one of "auto", "project", "halve" | `"auto"` | How the limit acts: 'halve' (EvAX: a move that leaves the limit is halved, at most 100 times, then refused), 'project' (an atom outside is put back on the limit; needed with gradient steps), 'auto' (halve without gradient steps, project with them). |
| `closest_distance` | table of number | `{}` | {"A-B": distance}: the shortest distance (A) between the elements A and B that exists in the chemistry [Potential_type hardsphere]. Pairs not listed have no limit. |
| `closest_tolerance` | number | `0.0` | A configuration is refused when a pair is closer than its limit by more than this (A); bred children always, trial states when reject_trials. |
| `reject_trials` | true or false | `true` | Refuse a trial state with a pair below its limit (EvAX's rule); false: only bred children are tested and the springs of the gradient steps keep pairs near the limits. |
| `spring` | number | `50.0` | Stiffness of the springs below the closest distances during the gradient steps (eV/A^2). |

### `[refine]`

[refine]: the gradient refinement (`rmc-exafs refine`) of one sample with one spectrum of one element.

| key | type | default | meaning |
|---|---|---|---|
| `copies` | integer | `4` | Copies of the cell refined together; their spectra are averaged. |
| `steps` | integer | required | Steps of gradient descent. |
| `step_size` | number | `0.003` | Root-mean-square move of the atoms per step (A), decaying on a cosine to step_size x final_fraction. |
| `final_fraction` | number | `0.1` | The step at the end, as a fraction of step_size. |
| `optimizer` | one of "sgd", "adam" | `"sgd"` | 'sgd': steepest descent with a normalized step and an accepted-step rule; 'adam': Adam with the step as its learning rate. |
| `ms_absorbers` | integer | `128` | Absorbers per step whose exact multiple-scattering gradient enters the running average. |
| `ms_memory` | number | `1.0` | Memory of the running average of the multiple-scattering gradient, in cycles through all absorbers. |
| `value_every` | integer | `10` | Steps between full passes of every absorber's multiple-scattering spectrum. |
| `refresh_every` | integer | `250` | Steps between new searches of the path lists. |
| `core_distance` | number | required | Hard core (A): a harmonic penalty acts on every pair closer than this. |
| `core_stiffness` | number | `20.0` | Stiffness of the hard core (eV/A^2). |
| `start_kick` | number | `0.0` | Normal displacement of every atom of every copy at the start (A). |
| `log_every` | integer | `50` | Steps between progress lines. |

<!-- END GENERATED -->

## Phase files and potentials

The phase file of an edge holds the scattering potentials of FEFF: potential 0 is the absorber with its core hole,
potentials 1, 2, ... are the scatterers, as in the `POTENTIALS` block of the `feff.inp` the file was made from
(`feffgpu phase feff.inp phase.pkl`, or FEFF 8.5L's own `phase.pad`). `potentials = { El = index }` maps every
element of the structure to its scatterer potential. The absorbing atoms also scatter for the other absorbers, with
the potential given for their element.

## The amplitude modes

| `[amplitude].mode` | S0² and the E0 shift during the search | at the end |
|---|---|---|
| `"fit"` (default) | start at `s02`, `de0` (or at the values of `from_run`, or of the run that `search.continue_from` continues); refitted jointly over the samples on their best states every `refit_every` generations from `refit_from` on (`refit_from = 0`: never during the search) | refitted once more, with intervals |
| `"fixed"` | held at `s02`, `de0` (EvAX's rule), also in a continued run | held |
| `"from_run"` | held at the shared values of the run named by `from_run` (a scan of an input then keeps one set of shared values) | held |

The pair tables and paths are evaluated at the start value of the E0 shift; a fit changes the shift by
interpolation of the calculated spectrum (exact to about 1e-4 of the signal for a few eV).
