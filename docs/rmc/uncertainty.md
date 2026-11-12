# Uncertainty

A reverse Monte Carlo fit gives one structure (the best state) out of many that fit the spectra about equally well.
Every number taken from it needs an uncertainty and a sentence on how it was obtained. Three spreads are available;
none of them includes what was held fixed.

## 1. Independent runs (seeds)

The fairest spread: run the same configuration with several `[run] seed` values (and, better, different start
displacements), each to its own output directory, then

```bash
rmc-exafs shells runs/seed1 runs/seed2 runs/seed3 runs/seed4 --shell Zn-O:1.8:2.4
```

`runs` in the output (`"runs": {"mean", "sd", "n"}`) is the spread of the best states over the runs. Report the mean
and the standard deviation over the runs, and the number of runs.

## 2. The states of one run

`states` in the output of `rmc-exafs shells` (`{"mean", "sd", "min", "max", "n"}` per run) is the spread over the
states of the final population. The states descend from each other by selection and crossover, so they are not
independent: this spread is a **lower bound**. It is useful to see whether a number is determined at all (a large
spread over states means the spectra do not fix it); it is not an error bar.

## 3. S0² and the E0 shift

In mode `fit` the final refit gives intervals (`run.json`, `intervals`): the curvature of the summed log-likelihood
at the optimum, with the likelihood weight `W_g = N_ind,g / (2 M_g)` of every sample (`N_ind` the number of
independent points of its windows, `M_g` its misfit at the optimum or `noise_floor` if larger), inverted to standard
deviations of log S0² and of the E0 shift; the intervals are +/- 1.645 standard deviations (90 %), with the correlation
of the two.

**These intervals hold the structure fixed.** They say how well S0² and the E0 shift are determined by the spectra for
that structure, not how well they are determined by the fit. In a reverse Monte Carlo model the structure can absorb a
change of S0² into a change of the disorder: a larger S0² with broader pair distributions makes nearly the same
spectrum (the quickstart shows it on a synthetic spectrum: S0² 0.995 with intervals 0.992-0.998, the planted value
0.90). To include that trade:

- hold S0² (and the E0 shift) at values from a reference compound of known structure measured with the samples, in
  the same window ([`mode = "fixed"`](configuration.md) or `"from_run"` from a fit of the reference), and say so;
- or **profile** S0²: run with S0² held at several values (mode `fixed`) and compare the misfits of the best states;
  the range of S0² whose misfit is within the noise of the data is the honest interval, and the structure changes
  along it.

## 4. Pairs at impossible distances

Without closest distances, or with soft ones, a model contains pairs at distances that do not exist in the chemistry.
They carry signal that real pairs would carry, so they bias the numbers of the real shells. Count them:

```bash
rmc-exafs short-pairs runs/my_fit                               # below the run's closest distances
rmc-exafs short-pairs runs/my_fit --threshold Zn-O:1.8 --threshold O-O:2.4
```

Report the share of centers with such a pair. A shell number that changes when the impossible pairs are added to it
or removed from it (integrate the shell over a range that starts below the closest distance, and over one that does
not) is uncertain by that change.

## 5. What was held fixed

Report these with every result; none of them is in any of the spreads above:

- the **number of atoms of every element** in the cell, so the mean number of neighbors of an element in a
  lattice-based model ([](limits.md));
- the **start lattice** and the **displacement limit**;
- the **closest distances**;
- the **windows** (k range, k weight, R range) and the **forward model** (the phase file, the cutoff, the number of
  legs, the importance filter);
- S0² and the E0 shift, when held.

## How a shell number is computed

`rmc-exafs shells` integrates the pair distribution of the positions over the stated range of distances: N is the
number of pairs per center atom in the range, R their mean distance, sigma2 their variance (the mean-square relative
displacement, MSRD) and c3 their third cumulant. Nothing is fitted. A range that cuts a peak changes all four; state
the ranges with the numbers.
