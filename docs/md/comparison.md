# Comparison with a measured spectrum

The averaged spectrum of a trajectory is a prediction with no free structural parameter: the coordination numbers,
distances, their spread and its asymmetry all come from the frames. MD-EXAFS scores this ensemble against the measured
spectrum as it is. No Debye-Waller factor and no coordination number is fitted. The only free numbers are S0², the
amplitude factor, and ΔE0, the shift of the energy origin, one of each per edge, as in the other tools of this package
([](../concepts/conventions.md)).

## Settings

```toml
[compare]
edge = "Zn"                      # may be left out when the run has one edge
chi = "measured_Zn.chi"          # two columns, k (1/Å) and chi; Athena and xraylarch .chi files are read as they are
window = { kmin = 3.0, kmax = 14.0, dk = 1.0, kweight = 2, rmin = 1.0, rmax = 5.0 }
fit_s02 = true
fit_e0 = true
s02 = 1.0                        # the start value, or the held value with fit_s02 = false
e0 = 0.0                         # eV: the start value, or the held value with fit_e0 = false
s02_bounds = [0.2, 5.0]          # a safety range, not a prior
e0_bounds = [-15.0, 15.0]
```

Several measured spectra (one per edge) are written as `[[compare]]` blocks. The averaged spectrum is the one defined by
`[average]`: every frame and every path by default, or a selection of frames and path types.

## What is fitted

Following xraylarch's `feffit`, S0² and the shift act on the model, never on the data:

```text
model(k) = S0^2 chi_avg(k'),   k' = sqrt(k^2 - 0.2625 dE0)        (k in 1/Å, dE0 in eV)
```

The shift is applied by cubic interpolation on the k grid. The misfit is the R-factor of the Fourier transform of the
k-weighted, Hanning-windowed spectra in the R window, `sum |chi_R(model) - chi_R(data)|^2 / sum |chi_R(data)|^2`. The
k window (with its sills) must lie inside the measured range and inside the calculated range (`kmax` of the edge);
otherwise the comparison stops with a message.

The fit is `feffgpu.exafs.fit_shared`, the same function that RMC-EXAFS uses.

## Several runs share S0² and the shift

```bash
md-exafs compare run_300K/config.toml run_500K/config.toml run_700K/config.toml
```

Each argument is a configuration file or an output folder. S0² and the shift belong to the edge, not to the sample:
spectra with the same edge name share one S0² and one shift, fitted jointly to all of them. Comparing, for example,
trajectories at several temperatures of one material this way gives each trajectory its R-factor under common values
of S0² and ΔE0. The results go to `compare_joint/` in the first run's output folder (or `--out`).

## Uncertainties

Every fitted number is reported with two uncertainties, which answer different questions:

- **The interval with the structure held fixed** (`s02_interval`, `de0_interval`): from the curvature of the summed
  log-likelihood at the optimum (`feffgpu.exafs.shared_intervals`), taking the misfit at the optimum as the noise level
  of the data (`W = N_independent / (2 x misfit)`), plus or minus 1.645 standard deviations (90 %). It says how well the
  measured spectrum determines S0² and ΔE0 for this averaged spectrum.
- **The sampling standard error** (`sampling` in `compare.json`): a jackknife over the blocks of consecutive frames of
  `[average] blocks`. The fit is repeated with each block left out (in every run at once), and the spread of the
  results gives the standard error. It says how much S0² and ΔE0 would move with another finite set of frames from
  the same dynamics. The interval above does not include it.

The R-factor between the averages of the first and second half of the frames (in `average.json`) tells whether the
averaged spectrum has converged well enough for the comparison to mean much.

In the example of the repository ([](quickstart.md)), whose "measured" spectrum is the forward model's average of 16
other frames with a planted S0² of 0.85 and ΔE0 of 1.5 eV, the fit gave S0² = 0.853 (interval with the structure fixed
0.850 to 0.857, sampling standard error 0.007) and ΔE0 = 1.51 eV (interval 1.47 to 1.55 eV, sampling standard error
0.03 eV). The tests check that planted values are recovered within three standard deviations of the reported
intervals, alone and with two runs sharing one S0² and one shift.

## Output

`compare/compare.json` holds, per run and spectrum, the R-factor, S0², ΔE0 and the number of independent points of the
window, and the shared values with both uncertainties and one sentence each on how they were obtained
([](outputs.md)). `compare/compare.png` shows the measured spectrum in black and the model in blue, in k (k-weighted,
inside the window) and in R. A value that ends on a bound of `s02_bounds` or `e0_bounds` is listed in `at_bound` and
is not a fit.
