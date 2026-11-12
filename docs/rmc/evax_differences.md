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
| 6 | Multiple scattering | Computed for every state from that state's geometry, through a basis of representative paths renewed every 2 generations (`Update_basis_every_...`). | Computed for every state from its own atoms through its own path lists. The lists hold atom indices, so between two searches the signal follows the atoms exactly; the choice of paths (the importance filter) is renewed every `paths_every` generations, and at once for the absorbers near an atom that changed site. During gradient steps the multiple scattering of the state before the move is held. | Exact paths on the GPU instead of a clustered basis. | `paths_every = 1` renews the choice of paths every generation (no switch for the held signal during gradient steps) |
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

## What is not EvAX's and has no counterpart there

- S0² and the E0 shift of several samples fitted jointly, with intervals from the curvature of the misfit
  ([](uncertainty.md)).
- The arrangement statistics of the records (vacancies, Warren-Cowley parameters of an exchange pair).
- Several spectra combine as `sum w_e R_e` (ratios); EvAX combines its square-root residuals.
- EvAX reports the square root of the ratio as its residual; this package reports the ratio (the R-factor).
