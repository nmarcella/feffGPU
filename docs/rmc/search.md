# The search, one generation at a time

`rmc-exafs run` refines a **population** of model structures (states) against the measured spectra by the rules of
EvAX, the reverse Monte Carlo program of J. Timoshenko and A. Kuzmin (Comput. Phys. Commun. **183**, 1237 (2012);
see [](../citing.md)). The rules were read from EvAX 6.16's source; where this package does something else, the
difference is listed in [](evax_differences.md) with the switch that restores EvAX's rule.

## The objects

- A **state** is one structure: `cells_per_state` copies of the model cell (EvAX: one) whose spectra are averaged.
- Its **spectrum** for every edge is calculated from its own atoms: single scattering of every absorber-neighbor
  pair within `rpath`, exact (the pair histogram on a 0.002 Å grid times the two-leg signal of the path stage), plus
  multiple scattering (paths of three and four legs) from the state's **own path lists** (below). S0² = 1 and no
  Debye-Waller factor: all disorder is in the positions.
- Its **misfit** is `M = sum_e w_e R_e` over the edges of the sample, `R_e` the R-factor of edge `e` (or its wavelet
  residual) with the current S0² and E0 shift on the model, and `w_e` the spectrum weights (EvAX's
  `adjust_spectra_weights`; one edge has the weight 1).
- **Sites.** The site of an atom is its position in the start structure. Exchanges and hops (if configured) move atoms
  to other sites; every state carries the site of every atom.
- The **random numbers** of a sample come from one generator seeded with `seed + 11`, drawn in a fixed order, so the
  same configuration and seed give the same run (up to the last digits of additions on the GPU).

## The start

1. Every state is the start structure plus a random displacement of every atom: uniform in
   `[-start_kick, start_kick] / sqrt(2)` per coordinate (EvAX's `Initial_displacement`; or normal with standard
   deviation `start_kick`). Atoms that every edge ignores do not move.
2. With a displacement limit, every atom is put within it (projected).
3. The paths of every absorber of every state are found (FEFF's path finder with the importance filter `criteria`;
   FEFF is cited in [](../citing.md)), and every state is evaluated from its own atoms. With
   `criteria_rule = "evax"` the paths are instead chosen once, on the start lattice, by EvAX's rule
   ([](evax_differences.md)); every state then has the same list, and nothing is found again later.
4. The scale of the gradient steps (if any) is set so that the largest force of the misfit on an atom of state 0 is
   `gradient_force` (eV/Å). A continued run keeps the scale of its first run.

## One generation

For every sample, in this order:

**0. Every `paths_every` generations** the path lists of every state are found again and every state is evaluated
again. **In mode `fit`, every `refit_every` generations from `refit_from` on,** S0² and the E0 shift of every spectrum
name are refitted jointly over all samples, each sample on its best state with that state's own multiple scattering
(`feffgpu.exafs.fit_shared`, within `s02_bounds` and `de0_bounds`), and every state is rescored.

**1. Selection and crossover** (EvAX's `selectionT`, `crossover`, `end_ea`). The states are replaced pair by pair:
the first parent of pair `iv` is the better of two random states (the first pick on a tie), the second parent the
better of two more, and the two slots receive copies of the two parents. With probability 0.5, and two different
parents, the pair breeds: for every site, with probability 0.5 (one draw for both children), child 1 takes the
displacement that parent 2 has on that site and child 2 the one parent 1 has. A site is exchanged only if both parents
hold one atom of the same element there; each child keeps the arrangement of its own parent. Without exchanges and
hops this is EvAX's rule atom by atom. A child with a pair closer than a closest distance by more than
`closest_tolerance`, or an atom outside the displacement limit, is replaced by its unbred parent (the copy already in
the slot). All picks read the population before the selection; no state is protected. A bred child keeps its
parent's path lists and is evaluated from its own atoms.

**2. The move of every state.**
- From `from_generation` on, if configured: with probability `exchange_probability` a state proposes
  `exchange_count` exchanges per cell (an atom of each of the two elements exchange positions and sites), and with
  probability `hops.probability` it proposes `hops.count` hops per cell (an atom moves from a site next to a random
  vacant site onto it, plus a normal displacement of `hops.jitter`).
- EvAX's random move: every coordinate of every atom moves by a uniform step within `+/- step` (EvAX's
  `Maximal_step_length`, `Move_type move_all`).
- If `gradient_steps > 0`: that many FIRE steps on the gradient of the misfit, with the springs of the closest
  distances and the displacement limit applied as a projection after every step. During these steps the multiple
  scattering of the state before the move is held (its gradient is that of single scattering). Where multiple
  scattering is strong (metals, linear chains of atoms) the steps then stop improving the fit; use the gradient
  refinement there ([](refinement.md)).
- The displacement limit: with gradient steps, a projection; without them (`box_rule = "auto"`), EvAX's rule: a state
  whose move takes an atom outside has its random step halved, at most 100 times, and is refused if that does not
  help.

**3. Evaluation and acceptance.** The path lists of a state are found again for the absorbers within `rpath + 0.5 Å`
of the old or the new position of an atom that changed site; the trial's multiple scattering is computed from its own
atoms. A trial with a pair below a closest distance by more than `closest_tolerance` is refused when
`reject_trials` (EvAX's hard sphere). Then:
- the **spectrum weights** take the changes `|R_e(trial) - R_e(state)|` of this generation's moves into their running
  mean and become `w_e = (1 / a_e) / sum_f (1 / a_f)` (EvAX's `adjust_spectra_weights`);
- every state **accepts or rejects its own trial** (EvAX's `mutation` with `adjust_acceptance`): a move that does not
  raise the misfit is accepted; a worse move when `exp(logp x dM / <|dM|>) >= rho`, with
  `logp = log(acceptance x (1 - t / froze_in))`, `t` the generation count over all continued runs (1 at the first),
  `<|dM|>` the running mean of the size of the proposed changes and `rho` a uniform random number. A worse move of mean
  size is accepted with the probability `acceptance x (1 - t / froze_in)`; from `t = froze_in` on only moves that do
  not raise the misfit are accepted. Refused moves count in neither running mean.

**4. Records.** One record per generation goes to `history.json` ([](outputs.md)); every `save_every` generations
the population is written to `checkpoint.npz`.

## The end

All path lists are found again, every state is evaluated, the best state of every sample is taken; in mode `fit` S0²
and the E0 shifts are refitted once more on the best states, with intervals ([](uncertainty.md)). The population, the
best state, the spectra and the reports are written ([](outputs.md)).

## Continuing a run

`search.continue_from` names a finished run of this command. Its best `states` states, S0² and E0 shifts, spectrum
weights, running means and generation count continue (the cooling goes on from where it stopped). The S0² and E0
shifts of the earlier run are the start values of the fit in mode `fit`; mode `fixed` holds the declared `s02` and
`de0` instead, and mode `from_run` the values of the run it names. A run is refused if
it was made by another method, holds another structure (other elements of the atoms) or another start lattice (the
displacement limit is measured from it). The scale of the gradient steps of the first run is kept: on a fitted
structure the gradient is much smaller, and a scale computed there would make every gradient step far too strong.
