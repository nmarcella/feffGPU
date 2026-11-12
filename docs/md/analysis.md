# Analysis of trajectories

`md-exafs rdf` and `md-exafs convergence` analyze the trajectory itself: radial distribution functions, the numbers
of their peaks, projections on crystal directions, bond angles, and the convergence of the dynamics and of the
averaged spectrum. The functions behind them are in the shared package `feffgpu.analysis` (`rdf`, `peaks`,
`convergence`, `pairs`); they take plain arrays, so RMC-EXAFS uses them too.

## Radial distribution functions

```toml
[rdf]
r_max = 8.0              # Å
bin_width = 0.01         # Å
pairs = [["Zn", "C"], ["Zn", "Zn"]]     # default: every pair of the system's elements
device = "cpu"           # or "cuda"
```

The frames are those of `[frames]` unless `[rdf]` gives its own `start`, `stop` and `step`. They are read one at a
time and the distances histogrammed with torch on the chosen device. For every center-neighbor pair A-B:

- **pairs per center per Å**, n(r): the number of B neighbors of an A atom per Å of distance. Its integral over a
  range is a coordination number;
- **g(r)**: n(r) divided by the ideal-gas expectation `rho_B 4 pi r^2 dr`, computed with the exact volume of each
  spherical shell and with `rho_B = (N_B - delta_AB) / V` taken frame by frame (so a changing cell is handled), so that
  g(r) tends to 1 at large distances. Clusters have no g(r);
- **the running coordination number**: the integral of n(r) from zero.

`r_max` may exceed half the cell: periodic images are counted as far as needed (an atom's own images count as its
neighbors). The tests check the coordination numbers and distances of the first five shells of an ideal bcc lattice
in a 6.3 Å cell with `r_max` 5.9 Å.

## The numbers of a peak

```toml
[[rdf.peak]]
center = "Zn"
neighbor = "C"
r_min = 1.8
r_max = 3.0
fit = "both"             # "gaussian", "skewed", "both" or "none"
```

Two kinds of numbers are reported side by side; say which one you quote.

**Integration (the preferred, fit-free numbers).** N, the number of B neighbors per A atom with
`r_min <= r < r_max`; R, their mean distance; sigma2, the variance of their distances (the mean-square relative
displacement, MSRD); C3, the third cumulant. They are computed exactly from the distances themselves (power sums
accumulated frame by frame), not from the histogram, and equal what `feffgpu.analysis.shell_table` gives for the same
frames (checked in the tests). Nothing is fitted and nothing is assumed about the shape of the peak.

**Fits.** As in `md_exafs` 2.0, a Gaussian and a skewed Gaussian (skew-normal) are fitted to the peak:

- Gaussian: `n(r) = N0 exp(-(r - R)^2 / (2 s^2)) / (s sqrt(2 pi))`;
- skew-normal: `n(r) = N0 (2 / w) phi((r - xi) / w) Phi(a (r - xi) / w)`, whose mean, variance and third cumulant
  are analytic.

Both are fitted to n(r) itself, not to g(r), so that the moments of the fitted form are the mean distance and the
variance of the pairs (the argument of Sukhomlinov and Müser, which `md_exafs` 2.0 cites for its skewed fit). The
reported N is the area of the fitted form inside the range; R, sigma2 and C3 are its moments. The standard errors
come from the covariance of the least-squares fit (scaled by the scatter of the residuals) propagated to N, R,
sigma2 and C3. They say how well the form describes the histogram; they do not include the correlation between the
frames of a trajectory, and they are not the uncertainty of the structure. Use the fits to describe the shape of a
peak or to separate overlapping peaks, and the integration for the numbers themselves.

## Projected RDFs

```toml
[[rdf.projection]]
center = "Zn"
neighbor = "Zn"
direction = [1, 1, 0]    # Cartesian; for a cubic cell the crystal direction [h k l]
angle = 20.0             # degrees
symmetry = true          # every cubic equivalent of the direction too
r_min = 2.5              # the range of the reported numbers (default 0 to r_max)
r_max = 4.5
```

Only the bonds whose direction lies within `angle` of the direction or of its opposite are counted (with `symmetry`,
of any of the cubic equivalents: permutations and sign changes of the components). The results are given per center
atom per Å, as above, and as g(r) normalized by the fraction of all directions that the cones cover (computed
numerically, overlaps of the cones counted once). The numbers of the range (N, R, sigma2, C3) come from the
distances, as for peaks. `md_exafs` 2.0 normalized projected RDFs like full ones; divide its curves by the
solid-angle fraction to compare.

## Bond angles

```toml
[[rdf.angle]]
center = "Zn"
neighbors = ["C", "C"]
ranges = [[1.8, 3.0], [1.8, 3.0]]     # the Zn-C and Zn-C distances that define a neighbor
bin_width = 1.0                        # degrees
```

The angles B-A-C between pairs of neighbors of every A atom, from `feffgpu.analysis.bond_angles` frame by frame:
plain counts from 0 to 180 degrees (no division by sin(angle)), and the same per center atom and frame. With B and C
of the same element and range, each unordered pair of neighbors counts once.

## Convergence of the dynamics

```bash
md-exafs convergence md.log            # a log file directly, or a configuration with [convergence]
```

```toml
[convergence]
log = "md.log"           # CP2K .ener, ASE MD log, or CSV with a header row
format = "auto"
skip = 2.0               # equilibration to leave out, in the log's time unit (fs for CP2K, ps for ASE)
blocks = 10
```

For every quantity of the log: the mean, the standard deviation, the block-averaged standard error (the standard
deviation of the means of `blocks` consecutive blocks over the square root of their number, larger and more honest
than the plain standard error for correlated steps) and the drift (a straight-line fit against time). CP2K energies
are converted from Hartree to eV. The drift of a conserved quantity (the total energy of NVE dynamics, CP2K's
"Cons Qty" of a thermostatted run) measures the integration error; the total energy of Langevin dynamics is not
conserved, and its drift means nothing.

## Convergence of the averaged spectrum

From the cache, for every edge (`md-exafs convergence config.toml`, and in `average.json`):

- the R-factor of the average of the first n frames against the average of all frames, for n from 1 to all frames,
  in the window of the edge's `[compare]` block (or k 3 to kmax - 1, R 1 to 6 Å);
- the R-factor between the averages of the first and the second half of the frames: the size of the sampling noise
  that remains. Compare it with the R-factor of the fit to the measured spectrum: when it is not much smaller, more
  frames (or more absorbers per frame) are needed before the comparison means much;
- the standard error of the mean spectrum from blocks of consecutive frames (`stderr` in `average/<edge>.chi`).

These take plain arrays (`feffgpu.analysis.convergence.spectrum_convergence(k, chi_frames, weights, window)`), so they
apply to the states of a reverse Monte Carlo model too.
