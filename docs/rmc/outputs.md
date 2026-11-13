# Output files

A run of `rmc-exafs run` writes into `run.output`:

```
<output>/run.json                  the run: samples, shared S0^2 and E0 shifts with intervals
<output>/config.resolved.toml      the configuration with absolute paths and every default (can be run again)
<output>/<label>/report.json       one sample: settings, records, final refit, state of the search
<output>/<label>/history.json      one record per generation and per refit
<output>/<label>/state.npz         the final population, the best state, the start lattice, the sites
<output>/<label>/spectra.npz       data and models of every edge on the k grid, the windows
<output>/<label>/checkpoint.npz    the population every save_every generations
```

Units: distances and positions in Å, S0² without unit, E0 shifts in eV, k in 1/Å. "Misfit" is the weighted sum of
the edges' objectives (`M = sum_e w_e R_e`); "ratio" is the R-factor of one edge (or its wavelet residual). Arrays:
P states, C cells per state, n atoms, NK points of the k grid, m sites, n_e edges.

The functions `feffgpu.rmc.read_run(dir)` and `feffgpu.rmc.read_sample(dir)` read these files.

## `run.json`

| key | meaning |
|---|---|
| `method` | `"feffgpu.rmc: population search under EvAX's rules"`: a continuation refuses files of another method |
| `format` | version of this layout (1) |
| `samples` | the sample labels, in the order of the configuration |
| `amplitude_mode` | `"fit"`, `"fixed"` or `"from_run"` |
| `shared` | `{"s02": {name: value}, "de0": {name: value}}`: the final S0² and total E0 shift of every spectrum name |
| `at_bound` | spectrum names whose fitted S0² or E0 shift sits on a bound (a value on a bound is not a fit) |
| `intervals` | mode `"fit"`: the intervals of S0² and the E0 shifts ([](uncertainty.md)): `positive_definite`, `W` (likelihood weight per sample), `z` (1.645), `s02_sd_log`, `s02_lo`, `s02_hi`, `de0_sd`, `de0_lo`, `de0_hi` (each `{name: value}`), `de0_s02_correlation`, `method`; otherwise null |
| `intervals_method` | one sentence: how the intervals were obtained |
| `misfits` | `{label: {"best": misfit, "ratios": {edge: R}}}` of the best state of every sample after the final refit |
| `generations_done` | the generation count over all continued runs |
| `seconds` | wall time of the run |

## `<label>/report.json`

| key | meaning |
|---|---|
| `method`, `format` | as in `run.json` |
| `label` | the sample |
| `structure` | one line: atoms, cell, sites |
| `species` | the element names (sorted; the order of species codes) |
| `edges` | the spectrum names of the sample, in order |
| `absorbers` | `{edge: absorbing element}` |
| `edge_descriptions` | one line per edge: window, S0², E0 shift, absorbers, objective, paths, potentials, pair types |
| `settings` | `search`, `amplitude`, `moves`, `constraints` (every key, resolved), `box_rule` (the rule in force: `"halve"` or `"project"`), `seed` |
| `closest_distance` | `{"A-B": distance}` of the limits in force for this structure |
| `max_displacement` | the displacement limit (0: none) |
| `at_limit_final` | share of the atoms of the best state at the displacement limit (null without a limit) |
| `scale` | energy of a unit misfit in the gradient steps (eV) |
| `scale_source` | `"this run's start structure"` or `"the first run (kept)"` |
| `start` | state 0 at the start: `misfit`, `ratios` `{edge: R}`, `s02`, `de0` (`{edge: value}`) |
| `start_kind` | `"start structure"` or `"continued"` |
| `from_population` | for a continued run: `path`, `states` (indices of the states taken), `misfits`, `generations_done`; else null |
| `best_state` | index of the best state of the final population |
| `final_held` | the best state at the end with the S0² and E0 shifts of the search (before the final refit): `misfit`, `ratios`, `s02`, `de0` |
| `final_refit` | the same after the final refit (equal to `final_held` when S0² and the E0 shifts are held) |
| `population_misfits` | misfit of every state (P) |
| `population_ratios` | ratio of every edge of every state (P x n_e) |
| `arrangement` | `start`, `final_best` (statistics of the arrangement of atoms on sites, below), `population` (one per state), `population_spread` (`{key: {mean, sd, min, max}}` over the states) |
| `intervals` | as in `run.json` |
| `search` | the state of the search that a continuation reads: `weights` (`weights`, `average_delta`, `num`), `acceptance` (`acceptance`, `froze_in`, `average_delta`, `num`), `generation_offset`, `generations_done`, `accepted_worse`, `children_rejected`, `refused` |
| `seconds` | wall time when the file was written |

**Arrangement statistics** (averages over the cells of a state): `atoms_off_start_site`, `sites_changed` (occupied
sites whose element differs from the start); with a hop sublattice `vacancies`, `vacancies_moved` (vacant sites that
were occupied at the start), `vacant_neighbors_of_a_vacancy` and `vacant_neighbors_random` (the value for a random
arrangement); with an exchange pair A, B: `fraction_A`, `unlike_neighbors_of_A`, `warren_cowley_A` (0 random, above 0
like neighbors preferred; nearest sites of the exchange sublattice), the same for B, and with a hop sublattice
`A_around_vacancies` (the share of A among the exchange sites nearest to a vacant site).

## `<label>/history.json`

`{"generations": [record, ...], "refits": [record, ...], "search": {...}}` (`search` as in `report.json`).

A generation record:

| key | meaning |
|---|---|
| `generation`, `generation_total` | the generation of this run and the count over all continued runs (EvAX's `t`) |
| `target_acceptance` | probability of accepting a worse move of mean size at this generation |
| `average_delta` | running mean of the size of the misfit changes (the mean change of the acceptance rule) |
| `rearrange` | whether exchanges and hops were allowed |
| `best`, `mean`, `worst`, `best_state` | misfits of the population after the generation and the index of the best state |
| `ratios_best` | `{edge: R}` of the best state |
| `misfits` | misfit of every state |
| `weights` | the spectrum weights |
| `selection` | `bred_pairs`, `children`, `children_rejected` (replaced by their unbred parent), `distinct_parents`, `atoms_exchanged_per_child`, `mean_after` and `best_after` (misfits after the selection) |
| `accepted`, `accepted_worse`, `proposed_worse`, `refused` | moves accepted, worse moves accepted, worse moves proposed, trial states refused by a constraint |
| `accepted_worse_plain`, `accepted_worse_rearranged` | accepted worse moves without and with a rearrangement |
| `gradient_steps` | FIRE steps taken |
| `exchanges`, `exchange_states`, `accepted_exchange_states` | exchanges proposed, states that proposed them, of which accepted |
| `hops`, `hops_skipped`, `hop_states`, `accepted_hop_states` | the same for hops (a hop is skipped when no vacant site has an occupied neighbor) |
| `absorbers_refound` | absorbers whose path lists were found again around moved atoms |
| `diversity` | mean root-mean-square distance between two states (Å) |
| `rms_from_sites_best`, `max_from_sites_best`, `rms_from_sites_best_per_element` | displacements of the best state from its sites (Å) |
| `pairs_below_closest_best`, `deepest_below_closest_best` | pairs of the best state below a closest distance, and the depth of the deepest (Å) |
| `at_limit_best` | share of the best state's atoms at the displacement limit (null without a limit) |
| `arrangement_best` | arrangement statistics of the best state (when rearrangements run, and at the last generation) |
| `seconds_parts`, `seconds` | time per part (`paths_all`, `evaluate_all`, `selection`, `gradient_steps`, `paths_moved`, `evaluate`) and in all |

A refit record: `generation`, `best_before` and `best_after` (misfit of the best state), `ratios`, `s02`, `de0`
(`{edge: value}`), `weights`, `at_bound`.

## `<label>/state.npz`

| key | shape | meaning |
|---|---|---|
| `positions` | (C, n, 3) | the best state |
| `start` | (C, n, 3) | state 0 at the start (after the start displacement) |
| `population` | (P, C, n, 3) | every state at the end |
| `site_of` | (P, C, n) | the site (index into `sites`) of every atom of every state |
| `misfits` | (P,) | misfit of every state |
| `ratios` | (P, n_e) | ratio of every edge of every state |
| `edges` | (n_e,) | spectrum names |
| `cell` | (3, 3) | lattice vectors as rows (NaN for a cluster) |
| `symbols` | (n,) | element of every atom |
| `ideal` | (C, n, 3) | the start lattice (the start structure) |
| `sites` | (m, 3) | positions of the sites |
| `site_ids` | (m,) | identifier of every site (the crossover draws one number per identifier) |
| `start_site_of` | (n,) | the site of every atom in the start structure |
| `best_state` | () | index of the best state |

## `<label>/spectra.npz`

| key | shape | meaning |
|---|---|---|
| `k` | (NK,) | the k grid (1/Å) |
| `data_<edge>` | (NK,) | the measured chi(k) on the grid |
| `window_<edge>` | (6,) | `kmin, kmax, dk, kweight, rmin, rmax` of the edge |
| `model_start_<edge>` | (NK,) | state 0 at the start, with the start S0² and E0 shift |
| `model_final_<edge>` | (NK,) | the best state at the end with the S0² and E0 shift of the search |
| `model_population_<edge>` | (P, NK) | every state at the end, same S0² and E0 shift |
| `model_refit_<edge>` | (NK,) | the best state after the final refit |
| `calc_refit_<edge>` | (NK,) | the calculated spectrum of the best state before S0² and the shift (S0² = 1, at the start shift) |

The models are what the data are compared with: S0² times the calculated spectrum at the shifted k. Every figure and
every R-factor recomputed from this file uses the stored window of its edge.

## `<label>/checkpoint.npz`

| key | meaning |
|---|---|
| `population`, `site_of`, `misfits` | the population at the checkpoint, as in `state.npz` |
| `generation` | the generation of this run |
| `cell`, `symbols`, `ideal`, `edges` | as in `state.npz` |
| `s02`, `de0` | (n_e,) the current values |
| `search` | the state of the search as JSON text |

## The gradient refinement (`rmc-exafs refine`)

`<output>/<label>/refine.json`: `method`, `label`, `edge`, `copies`, `atoms`, `misfit_start`, `misfit_final`,
`settings` (the `[refine]` keys), `s02`, `de0`, `steps_info` (`halvings`, `rejected_steps`, `final_step_size`),
`criteria_rule`, `path_types` (with `criteria_rule = "evax"`: every type of multiple-scattering path from a quarter
of the threshold on, with `legs`, `half_length` in Å, `degeneracy` per atom, `directions`, `importance` in percent,
`potentials` and `kept`), `timing` (seconds per part), `history` (one record per log: `step`, `misfit`, `core`,
`rms_move`, `max_move`, `step_size`, `halvings`, `seconds`), `seconds`.

`<output>/<label>/refine.npz`: `start` and `positions` (copies, n, 3), `sites` (n, 3: the start lattice), `k`,
`target` (the data divided by S0²), `model_start` and `model_final` (the calculated spectra at the E0 shift), `cell`,
`symbols`, `window`.
