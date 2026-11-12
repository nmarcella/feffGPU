# What EXAFS does and does not determine

This page is a short guide to reading results of MD-EXAFS and RMC-EXAFS with the right expectations.

## What the spectrum contains

An EXAFS spectrum of one absorption edge is an average over all absorbing atoms of that element in the sample. It
depends on the distribution of distances from the absorber to its neighbors, element by element, out to about 5 to
8 Å, and, through multiple scattering, on some angles between neighbors.

## What it determines well

- **Pair distances.** The mean distance of a well-separated shell is determined to about 0.01 to 0.02 Å.
- **The width of a distance distribution** (the MSRD), when the shell is resolved and S0² is known.
- **Asymmetry** of a first-shell distribution, when the k range is long.
- **Near-collinear arrangements**, through multiple scattering, which is strong when three atoms are almost in a
  line.

## What it determines poorly

- **Coordination numbers.** The amplitude of a shell is the product of S0², the coordination number and a damping
  that depends on the width. These are strongly correlated. A coordination number from EXAFS alone carries an
  uncertainty of the order of 10 to 20 %, more if the width is free and the k range is short.
- **Which atom is displaced.** The spectrum fixes the distribution of pair distances much better than the
  positions that produce it. Many arrangements of atoms give the same distributions. A reverse Monte Carlo model
  is one such arrangement, not the arrangement.
- **Neighbors of similar atomic number**, which scatter almost alike.
- **Two shells closer than the resolution**, about π / (2 Δk), which is 0.13 Å for a k range of 12 1/Å.
- **Anything beyond the path cutoff**, and long-range order in general.
- **Light neighbors next to heavy ones**, whose signal is small.

## Consequences for the tools

- A model with a fixed number of atoms on lattice sites has its total number of neighbors built in. That number is
  an input of the model, not a result of the fit. Compare models with different inputs instead of reading a
  coordination number off one model.
- A reverse Monte Carlo fit with no limit on closest distances will contain some pairs at distances that do not
  exist in the chemistry. They are a diagnostic of how much freedom the fit took. The tools count them
  (`feffgpu.analysis.short_pairs`).
- Report shell numbers with their spread over independent runs and over the states of a run. A single run gives a
  single arrangement.
- A good R-factor shows that a structure is consistent with the spectrum. It does not show that it is the only one.
- The number of independent points of a window, 2 Δk ΔR / π + 2, bounds how many parameters the spectrum can fix.
  A model with thousands of coordinates is far beyond that bound; it is constrained by its start structure, by the
  limits on displacements and distances, and by the spectra together.
