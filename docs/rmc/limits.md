# What the model cannot tell you

A reverse Monte Carlo fit always ends with a structure that fits. These are the questions it cannot answer by itself.

## The number of neighbors is an input of the cell

The model holds a fixed number of atoms of every element in a fixed cell. In a model built on a lattice (a crystal
structure, a supercell of it) the **mean number of neighbors** of an element is then fixed by the cell, not measured:

- With a displacement limit smaller than half the distance between neighboring sites, no atom leaves its site, so
  every site keeps the neighbors of the start lattice. The coordination number of every shell is that of the start
  structure, exactly (the quickstart's shells come out as 6, 12 and 8 in every run).
- Exchanges of two elements over their sites change which element sits next to which (the partial coordination
  numbers, the ordering), never the total number of neighbors of a site.
- A hop of an atom onto a vacant site moves neighbors from one place to another; the total number of pairs between the
  hop sublattice and its neighbors stays the number of atoms on it times the neighbors of a site.
- Without a displacement limit, atoms can leave their shells, and the number of pairs inside a stated range can
  change; the total over all distances is still the number of atoms.

A coordination number from a lattice-based model is therefore never a result. Vary the input (the number of atoms of
an element, the vacancies, the cell) and compare the fits, or use a model without a lattice.

## A model without limits contains impossible pairs

Without closest distances the search places some pairs at distances that no real material has, because a pair
anywhere in the R window can carry signal. The tools report them (`rmc-exafs short-pairs`); with closest distances from
chemistry they are refused ([](constraints.md)). A closest distance in turn removes the short tail of a distribution
by construction.

## The spectra fix pair distances better than they fix which atom moved

The EXAFS of an absorber is a sum over its pairs. The search reproduces the distribution of distances, not the
displacement of a particular atom: many arrangements of displacements give the same distributions. Statements about
single atoms, about which atom of a pair is displaced, or about correlations that the spectra do not see (for example
between atoms farther apart than the cutoff) are not supported by the fit. Compare such statements over the states and
over independent runs ([](uncertainty.md)); a statement that changes between runs is not determined.

The tails of a distribution are determined worse than its center: a few pairs far from the center of a shell cost
little misfit. A variance (MSRD) is sensitive to them; compare it with a width from the interquartile range of the
pair distances.

## S0² and the disorder trade

A larger S0² with broader distributions makes nearly the same spectrum. With S0² fitted, the widths of the shells are
uncertain by this trade; the interval of S0² from the final refit does not include it ([](uncertainty.md)).

## Paths up to four legs, within the cutoff

The forward model includes single scattering for every pair within `rpath` and multiple scattering of paths with three
and four legs, after FEFF's importance filter, within the same cutoff. Paths with more legs and paths longer than the
cutoff are left out, as is the signal of atoms farther than `rpath`. Structure beyond the R window of the fit is not
constrained at all. The calculated spectrum has no Debye-Waller factor: all disorder is in the positions, so the
static and the thermal parts of the disorder are not separated.

## One structure is one answer

The best state is one structure that fits. The population of one run is not an ensemble of independent structures.
Report the spread over independent runs.
