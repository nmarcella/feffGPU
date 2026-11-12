# Constraints

A reverse Monte Carlo model has as many free coordinates as it has atoms times three; the spectra determine far
fewer numbers (the number of independent points of a window, `2 (kmax - kmin)(rmax - rmin)/π + 2`). Without limits
the model develops pairs at distances that no real material has. Two limits are available, both from EvAX.

## The displacement limit (EvAX's box)

```toml
[constraints]
max_displacement = 0.4      # A; 0: no limit
box_rule = "auto"           # "auto", "halve" or "project"
```

Every atom stays within `max_displacement` of its **site of the fixed start lattice**. The site of an atom is its
position in the start structure; after an exchange or a hop it is the atom's new site. The lattice is taken once
from the start structure and never refitted, so a shift of all atoms together is a displacement of every atom, and
no translation can make an atom change site (the limit always measures from the same sites, with the minimum image
in the periodic cell).

How the limit acts:

- **Halving** (EvAX's rule, `box_rule = "halve"`, and `"auto"` without gradient steps): a state whose random step
  takes an atom outside has the step halved towards where it started, at most 100 times; a state still outside is
  refused for that generation.
- **Projection** (`"project"`, and `"auto"` with gradient steps): an atom outside is put back on the limit along its
  displacement, after the random move and after every gradient step. A relaxation of many steps cannot be halved as
  one move, so the gradient steps need the projection. With a projection, atoms collect on the limit; the records
  give the share of atoms at the limit (`at_limit_best` in `history.json`, `at_limit_final` in `report.json`). A large
  share means the spectra push atoms further than the limit allows: the limit then sets their displacement, not the
  data.

The start states are projected into the limit in either case. A bred child with an atom outside the limit is
replaced by its unbred parent (it cannot happen without rearrangements: the displacements of a child come from
states inside the limit).

## Closest distances

```toml
[constraints]
closest_distance = { "Zn-O" = 1.8, "O-O" = 2.5, "Zn-Zn" = 2.8 }
closest_tolerance = 0.0
reject_trials = true
spring = 50.0
```

For a pair of elements, the shortest distance that exists in the chemistry. They come from chemistry (bond lengths,
ionic radii, known structures of related compounds), never from the data being fitted. Pairs not listed have no
limit.

- A **trial state** with a pair closer than its limit by more than `closest_tolerance` is refused when
  `reject_trials` is true (EvAX's hard sphere, `Potential_type hardsphere`).
- A **bred child** with such a pair is always replaced by its unbred parent.
- During **gradient steps** every pair closer than its limit feels a harmonic spring of stiffness `spring` (eV/Å²),
  energy `0.5 spring (d_min - r)²`. The spring gives a force at the wall; it lets pairs sit slightly below the limit,
  so with gradient steps a small `closest_tolerance` (for example 0.05 Å) avoids refusing most trials.

The records count the pairs below a limit in the best state (`pairs_below_closest_best`, with the depth of the
deepest); `rmc-exafs short-pairs RUN` counts them in the whole population.

## What the limits do to the result

The limits are inputs of the model, like the start lattice and the number of atoms. Report them with every result:

- with a displacement limit smaller than half the distance between neighboring sites, every atom keeps its
  neighbors' sites: the coordination number of a site is fixed by the start lattice, not measured
  ([](limits.md));
- a closest distance removes the short tail of a pair distribution by construction;
- results that change when a limit changes are set by the limit. Compare runs that differ in one limit only, with the
  same windows, start and seeds.
