# Differences from EvAX

The population search follows EvAX (J. Timoshenko and A. Kuzmin; [](../citing.md)) as closely as possible, and every
difference is listed here: what EvAX does, what this package does, why, and the switch that restores EvAX's rule
where there is one. The rules of EvAX were read from the source of EvAX 6.16 (function and keyword names in
brackets). EvAX itself is not contained in this package and was not run to write it.

**Running EvAX's rules.** With the defaults below and one sample, the search is EvAX's except for the rows marked
"no switch": `gradient_steps = 0`, no `[moves]` section, `box_rule = "auto"` (halving), `reject_trials = true` with
`closest_tolerance = 0`, `means_from = 0`, `weights_means_from = 0`, `adapt_weights = true`,
`start_distribution = "uniform"`, `cells_per_state = 1`, and `[amplitude] mode = "fixed"` (EvAX holds S0² and the E0
shift); `objective = "wavelet"` gives EvAX's default fit space.

## The model of the spectrum

| # | Topic | EvAX 6.16 | This package | Why | Switch |
|---|---|---|---|---|---|
| 1 | Thermal disorder | None: the positions carry all disorder. | The same. | | (none needed) |
| 2 | S0² and the E0 shift | Declared for each spectrum and held (`S02`, `dE0`). | Refitted on the best state every `refit_every` generations from `refit_from` on and at the end, one value per spectrum name shared by all samples, within declared ranges; the values act on the model (xraylarch's `feffit` convention), not on the data. | They are rarely known to the precision the structure needs; a joint fit over samples uses every spectrum of an edge. They trade with the disorder ([](uncertainty.md)). | `[amplitude] mode = "fixed"` (or `"from_run"`) |
| 3 | Several samples | One structure model (or a weighted mixture) against its own spectra; several edges of one sample are native (`Number_of_spectra`). | Several samples in one run, each with its own atoms and spectra; only S0² and the E0 shift connect them. | One shared S0² and E0 shift per edge, measured on several samples. | one `[[sample]]` block |
| 4 | Spectrum weights | Adapted (`adjust_spectra_weights`): `w = (1/a) / sum(1/a)`, `a` the running mean of the change of the spectrum's residual. | The same rule, with these choices: the residual is the R-factor, not EvAX's square-root form (row 5); all moves of a generation enter the mean before the weights are used (EvAX updates after each state); a move includes the gradient steps and the rearrangements; samples keep equal weights among themselves; the same weights enter the gradient steps and the refit; moves refused by a constraint do not count. | The generation is batched on the GPU; the rest follows from rows 3, 5, 8 and 9. | `adapt_weights = false` gives equal fixed weights; `weights_means_from` restarts the mean (default 0: never, EvAX's rule) |
| 5 | Fit space and residual | Wavelet space by default (`Space w`), residual the square root of the summed squared difference over the summed squared data; EvAX's own k window. | R space by default with xraylarch's R-factor (the ratio, not its square root) on the complex chi(R) and xraylarch's Hanning window. EvAX's window with `k_min`, `k_max`, taper `w` is exactly the Hanning window with `kmin + w/2`, `kmax - w/2`, `dk = w` (`feffgpu.exafs.evax_window`). | The conventions of xraylarch, Athena and Artemis (the R-factor that users quote). | `objective = "wavelet"` (EvAX's transform; the edges combine as a weighted sum of ratios, not of square roots) |
| 6 | Multiple scattering | Computed for every state from that state's geometry, through clusters of similar paths with an allowed error (`Clustering_precision`, `Expansion_accuracy`), renewed every 2 generations (`Update_basis_every_...`). The paths are chosen once on the start structure: paths of like geometry form groups (`Pathfinder_robustness`), and a group is kept when the size of its signal there is more than `FEFF_criteria` percent of the size of the structure's whole spectrum. | Computed for every state from its own atoms through its own path lists; every path of a list is computed exactly. By default (`criteria_rule = "feff"`) the paths are chosen by FEFF's path finder on each state as it is, every path judged alone, and the choice is renewed every `paths_every` generations (at once for the absorbers near an atom that changed site). With `criteria_rule = "evax"` they are chosen once on the start structure by EvAX's rule, with EvAX's groups (below). Single scattering is always computed for every pair within the cutoff. During gradient steps the multiple scattering of the state before the move is held. | Exact paths on the GPU instead of clusters with an error allowance; FEFF's rule also works for structures without a lattice. | `criteria_rule = "evax"` (EvAX's choice of paths); `paths_every = 1` renews FEFF's choice every generation (no switch for the exact signals, nor for the held signal during gradient steps) |
| 7 | Forward-model code | The FEFF executable on clustered paths; potentials computed once. | The GPU rewrite of FEFF 8.5L in this package: every pair exact, every path of the lists exact; potentials computed once (the phase file). Validated against FEFF 8.5L ([](../validation/index.md)). | Speed and exactness. | no switch |

## The moves and the search

| # | Topic | EvAX 6.16 | This package | Why | Switch |
|---|---|---|---|---|---|
| 8 | Gradient steps | None: random moves only. | Optional FIRE steps on the gradient of the misfit after the random move of every state (`gradient_steps`). | The exact forward model is differentiable; the search converges in far fewer generations. | `gradient_steps = 0` (default) |
| 9 | Rearrangements | None: an atom keeps its start site. | Optional exchanges of two elements over their sites and hops onto vacant sites of a site list (`[moves]`). | Arrangements of atoms on sites that a random move cannot reach. | no `[moves]` section (default) |
| 10 | Random move | Every coordinate of every atom within +/- `Maximal_step_length` (`make_move`, `move_all`). | The same (`step`). | | exact |
| 11 | Crossover | In half the parent pairs, each atom's displacement exchanged between the parents with probability 0.5 (`crossover`). | The same draw made per lattice site; a site is exchanged when both parents hold one atom of the same element on it; each child keeps its own parent's arrangement. With equal arrangements this is EvAX's rule atom by atom (tested). | With rearrangements one atom can sit on different sites in two parents. | exact without `[moves]` |
| 12 | Selection and acceptance | Two-way tournaments (`crossoverT`); the slots receive the selected parents; a bred child that violates a constraint is replaced by its unbred parent; every state accepts or rejects its own move, a worse one when `exp(logp x change / mean change) >= rho`, cooled to zero at `Froze_in` (`adjust_acceptance`). | The same, with these choices: the generation count runs over continued runs; the mean change takes all moves of a generation first; the change is that of the misfit after the gradient steps; moves refused by a constraint stay out of the mean (EvAX does not compute the spectrum of a move refused for its energy). | The generation is batched; rows 8 and 9. | `means_from` restarts the mean of the acceptance at a generation (default 0: never, EvAX's rule) |
| 13 | Population | 32 states of one cell (`Number_of_states`). | The same defaults (`states = 32`, `cells_per_state = 1`). | | exact |
| 14 | Schedule | One run (`Stop_after`). | A run can continue an earlier one (`continue_from`): population, S0², E0 shifts, weights, means and the generation count. | Long searches in stages. | a single run |
| 15 | Start | The input structure plus a uniform displacement in `[-d, d] / sqrt(2)` per coordinate (`Initial_displacement`). | The same by default (`start_distribution = "uniform"`, `start_kick` = d). | | exact; `"gaussian"` is the alternative |
| 16 | Random numbers | EvAX's generator. | One torch generator per sample, seeded with `seed + 11`, drawn in a fixed order. | | no switch |

## The constraints

| # | Topic | EvAX 6.16 | This package | Why | Switch |
|---|---|---|---|---|---|
| 17 | Center of the box | Each atom's own input position, fixed in space (`potential_box`). | The atom's site of the fixed start lattice, which is its start position; after an exchange or a hop, its new site. The lattice is never refitted: a shift of all atoms together counts as a displacement of every atom. | Without rearrangements this is EvAX's rule; the new-site rule exists because of row 9. | exact without `[moves]` |
| 18 | How the box acts | A move that puts an atom outside is halved, up to 100 times, then rejected. | Without gradient steps the same (`box_rule = "auto"` or `"halve"`; the halving acts on the random step after any rearrangement). With gradient steps a projection: an atom outside is put back on the limit along its displacement, after the random move and after every gradient step. | A relaxation of many steps cannot be halved as one move. Atoms collect on the limit with a projection; the records give the share at the limit. | `box_rule = "halve"` (only without gradient steps) |
| 19 | Closest distances | Hard: a move with a pair below the minimum of its element pair is rejected (`Potential_type hardsphere`). | The same by default (`reject_trials = true`, `closest_tolerance = 0`). During gradient steps the limits also act as springs (`spring`, eV/Å²), which give a force at the wall; with springs, pairs can sit slightly below a limit, so `closest_tolerance` sets how far below a configuration may go before it is refused. | The gradient steps need a force at the wall. | `reject_trials = false` refuses only bred children (springs alone keep pairs near the limits) |

## Translating an EvAX parameter file

The keys of [](configuration.md) name EvAX's keyword in brackets. The keywords of a typical EvAX parameter file map
as follows; the tutorial notebook `tutorials/03_rmc_exafs_from_evax.ipynb` does it for a finished EvAX calculation
and compares the results.

| EvAX | Here | Note |
|---|---|---|
| `File_with_structure`, `Supercell` | `structure`, `format`, `supercell` | EvAX's `.p1` files are VASP structure files with extra columns (`format = "vasp"`). |
| `File_with_experimental_EXAFS_signal`, `Column_for_k`, `Column_for_chi` | `chi`, `chi_columns` | Columns count from zero here. |
| `Edge`, `Edge_type` | `absorber`; the `EDGE` card of the input of the phase file | |
| `R_max_for_FEFF`, `N_legs` | `rpath`, `nleg` | `nleg` is at most 8, EvAX's default. |
| `FEFF_criteria` | `criteria` with `criteria_rule = "evax"` | The same number means something else under FEFF's rule: see below. |
| `r_scf`, `n_scf`, `ca`, `n_mix` | the `SCF` card of the input of the phase file | A negative `r_scf` is not "off": see below. |
| `S02`, `dE0` | `[amplitude] mode = "fixed"`, `s02`, `de0 = -dE0` | The sign changes: see below. |
| `Space w`, `k_min`, `k_max`, `k_power`, `R_min`, `R_max`, `R0_for_WT` | `objective = "wavelet"`, `wavelet_r0`, `window = { kmin, kmax, dk = 0, kweight, rmin, rmax }` | In wavelet space EvAX uses the range from `k_min` to `k_max` without a taper: `dk = 0`. |
| `Number_of_states`, `Froze_in`, `Stop_after`, `Maximal_step_length`, `Acceptance_rate`, `Initial_displacement`, `Update_basis_every_...` | `[search] states`, `froze_in`, `generations`, `step`, `acceptance`, `start_kick`, `paths_every` | EvAX runs until it is stopped with `Stop_after -1`; here the number of generations is stated. |
| `Potential_type box`, `Maximal_displacement` | `[constraints] max_displacement` | |
| `Pathfinder_robustness` | `group_tolerance` (with `criteria_rule = "evax"`) | The distance within which paths belong to one group. |
| `Maximal_number_of_clusters`, `Clustering_precision`, `Expansion_accuracy` | none | Every path of every atom is computed exactly (row 6). |

Five points need care:

- **`FEFF_criteria` is not FEFF's path filter.** EvAX chooses its paths once, on the start structure, in groups.
  It walks every path as a chain of atoms in one direction of travel, so a path and its reverse are two chains. Two
  chains belong to one group when they meet atoms again in the same pattern, have the same elements in the same
  order, and each atom of one lies within `Pathfinder_robustness` (default 0.5 Å) of the matching atom of the other,
  in a frame set by the atoms before it. The importance of a group is the size of (its chains per absorber) times
  (the signal of its first chain), relative to the sum of that over all groups; the size is the root of the sum over
  k up to 20 Å⁻¹ of `(k^k_power chi)^2`. Groups above `FEFF_criteria` percent are kept. `criteria_rule = "evax"` does
  the same (`group_tolerance` is `Pathfinder_robustness`), and the log and `refine.json` list the groups in the form
  of EvAX's own list, the head of its `partialEXAFS.dat`. Three consequences:
  - In low symmetry and in particles a group holds paths of different geometry (bcc: the two nearest shells, 0.42 Å
    apart, are one group of fourteen). A group is kept or dropped as a whole, so far more paths pass the same
    threshold than with groups of identical geometry.
  - Paths whose first two scatterers lie on a line with the absorber count in one direction only. EvAX's frame for
    the third atom is not defined there, the chains split into small groups below the threshold, and only the
    opposite direction of the same paths is kept. This package reproduces that, so such a path enters with half of
    FEFF's degeneracy.
  - Groups within a few percent of the threshold can differ (the order in which paths of equal length are met is not
    fixed). With the same `nleg` as EvAX's `N_legs`, on nine spectra of EvAX's library (an fcc metal, four oxides,
    two metal particles, oxide particles) the lists agree exactly or to a few groups at the threshold, out of 18 to
    80, the groups of five and six legs included.

  Under the default, `criteria_rule = "feff"`, the same number is applied to every single path of the disordered
  structure by FEFF's plane-wave estimate, relative to one nearest-neighbor path. That keeps fewer and other paths,
  and the spectrum of a given structure then differs between the two programs by an R-factor of several 0.001.

- **EvAX's own spectrum is not exact FEFF on its atoms.** It computes clusters of similar paths with an allowed
  error (`Clustering_precision`, 0.03 to 0.1 in its library). With the same atoms and the same paths, the FEFF
  program EvAX runs and this package agree to an R-factor between 1e-7 and 7e-5 (four calculations of the library);
  EvAX's `EXAFS.dat` differs from both by 0.0006 to 0.007 (seven spectra), most of it in the multiple scattering.
  EvAX's reported residual is measured with its own spectrum, so the same atoms give a somewhat larger misfit here.

- **A negative `r_scf` means that EvAX chooses the radius** of the self-consistency loop of the potentials. It takes
  the nearest-neighbor distance d1, then the first neighbor distance at least 1 Å longer than d1, then the first
  one at least 0.1 d1 longer than that. This last distance is the radius (for fcc copper 4.43 Å, the third shell),
  enlarged if needed until every element has an atom inside. EvAX always runs FEFF with an `SCF` card. A phase file
  without self-consistency has its energy origin elsewhere (about 2 eV for copper), so the energy shift of the EvAX
  run no longer fits.
- **EvAX applies S0² and the energy shift to the data**: it divides the measured spectrum by S0² and reads it at
  `k' = sqrt(k² - 0.2625 dE0)`. This package applies both to the calculated spectrum (row 2), so the same alignment
  is `de0 = -dE0`. One difference remains: EvAX multiplies the data by `k^k_power` before the shift, so its target
  carries the factor `(k'/k)^k_power` relative to shifted data. `feffgpu.exafs.shifted_target` gives the shifted
  data without this factor.
- **The energy shift belongs to the potentials it was found with.** With other potentials (another radius of the
  self-consistency loop, another cluster) it is another number; `[amplitude] mode = "fit"` finds it. EvAX runs the
  FEFF 8.5L program; this package reproduces the open-source edition of FEFF 8.5L. For the same input the two give
  the same scattering (to 2e-8 in a first-shell spectrum of copper once the energy origins are aligned) but not
  quite the same Fermi level: for fcc copper with EvAX's settings it lies 0.13 eV lower here, so the same alignment
  is `de0 = -dE0 - 0.13`.

`rmc-exafs from-evax` does this translation: it reads the parameter file, the structure and the spectra (and, with
`--outputs`, EvAX's output files) and writes the start structure, the spectra, the FEFF inputs of the potentials and
`rmc.toml`:

```bash
rmc-exafs from-evax start.dat my_run            # the measured spectra; S02 and dE0 go on the model
cd my_run && feffgpu phase feff_Cu_K.inp phase_Cu_K.pkl
rmc-exafs refine rmc.toml                       # the gradient refinement ([refine]); minutes
rmc-exafs run rmc.toml                          # or EvAX's own search ([search]); as long as EvAX takes
```

`rmc.toml` holds both sections. The search is written with EvAX's numbers (states, generations, step, start). The
refinement is written with 300 steps from the lattice with normal displacements of 0.05 Å per coordinate, a
start of thermal size: EvAX's `Initial_displacement` suits a long random search (0.01 Å in some of its published
calculations) and is a poor start for a few hundred gradient steps.

With `--outputs <EvAX's output directory> --target evax` the spectra written are EvAX's own targets and the start
keeps EvAX's order of atoms, for a comparison with its result atom by atom. `--gradient-steps`, `--states` and
`--generations` set the search; `--supercell a,b,c` asks for another repetition of the unit cell when EvAX's is
narrower than twice the cutoff (EvAX works with explicit periodic copies and has no such limit). In Python:
`feffgpu.rmc.EvaxCalculation` reads the same files (`sites()`, `final()`, `target(i)`, `fitted(i)`,
`path_groups(i)`).

A literal translation of a typical EvAX run (tens of states, thousands of generations, no gradient steps) costs
what it costs in EvAX: every state's multiple scattering is computed in every generation. `gradient_steps` and, for
structures of one element, `rmc-exafs refine` ([](refinement.md)) need far fewer evaluations.

## What is not EvAX's and has no counterpart there

- S0² and the E0 shift of several samples fitted jointly, with intervals from the curvature of the misfit
  ([](uncertainty.md)).
- The arrangement statistics of the records (vacancies, Warren-Cowley parameters of an exchange pair).
- Several spectra combine as `sum w_e R_e` (ratios); EvAX combines its square-root residuals.
- EvAX reports the square root of the ratio as its residual; this package reports the ratio (the R-factor).
