# Conventions

All three tools share one set of conventions, implemented once in `feffgpu.exafs`. They are those of xraylarch
(and so of Athena and Artemis), so that a number from this package means the same as the number of that name there.

## The k grid

Spectra live on a uniform k grid that starts at zero with a step of 0.05 1/Å (`feffgpu.exafs.k_grid`). Measured
spectra are interpolated onto it (`feffgpu.exafs.to_grid`).

## The k window

The window is a Hanning window given by three numbers, as in xraylarch's `ftwindow`:

- `kmin`, `kmax`: the centers of the two sills;
- `dk`: the width of each sill.

The window is 1 between `kmin + dk/2` and `kmax - dk/2`, 0 outside `kmin - dk/2` to `kmax + dk/2`, and 0.5 at
`kmin` and `kmax`. The measured range must contain the window with its sills.

EvAX defines its window differently: it is 1 strictly inside `(kmin + w, kmax - w)` with tapers of width `w` that
reach zero at `kmin` and `kmax`. EvAX's window with `kmin`, `kmax`, `w` equals the Hanning window with
`kmin + w/2`, `kmax - w/2`, `dk = w` (`feffgpu.exafs.evax_window`; a test checks the equality).

## The Fourier transform

chi(R) is the fast Fourier transform of `k**kweight * chi(k) * window`, zero-padded to 2048 points, with
xraylarch's `xftf` normalization. The R grid has a step of π / (2048 × 0.05) ≈ 0.0307 Å. R is the distance axis of
the transform, labeled "R (Å)" in every figure.

## The misfit: the R-factor

The misfit between a model and a measured spectrum is xraylarch's R-factor on the R window `rmin` to `rmax`:

$$
R = \frac{\sum_{R} \left| \chi_\mathrm{model}(R) - \chi_\mathrm{data}(R) \right|^2}{\sum_{R} \left| \chi_\mathrm{data}(R) \right|^2}
$$

The sum runs over the real and imaginary parts. The six numbers `kmin`, `kmax`, `dk`, `kweight`, `rmin`, `rmax`
are one object, `feffgpu.exafs.FitWindow`, and appear as `window = { ... }` in the configuration files.

Other objectives exist as options where they are useful for comparisons with other programs (EvAX's wavelet
residual in RMC-EXAFS). The R-factor is the default everywhere.

## S0² and the E0 shift belong to the model

FEFF's chi(k) has the amplitude factor S0² = 1 and its own energy origin. A measurement has S0² below 1 and an
energy origin E0 that was chosen during data reduction. Following xraylarch's `feffit`, both corrections are applied
to the model, never to the data:

$$
\chi_\mathrm{model}(k) = S_0^2 \; \chi_\mathrm{calc}\!\left(\sqrt{k^2 - 0.2625\,\Delta E_0}\right)
$$

with ΔE0 in eV and k in 1/Å. There is one S0² and one ΔE0 per absorption edge. When several samples, temperatures
or trajectories of the same compound are compared in one fit, they share these two numbers per edge
(`feffgpu.exafs.fit_shared`). When an input is scanned (for example a constraint of an RMC model), the same values
are used for every value of the input, so that the scan compares structures and not amplitudes.

S0² and the coordination number multiply each other in the amplitude of a shell, so an error in one is an error in
the other. The tools never fix a sample's coordination number as a result: in a model with a fixed number of atoms
on lattice sites the total number of neighbors is an input of the model, and the documentation of RMC-EXAFS says so
where it matters.

## No Debye-Waller factor

FEFF's chi(k) for one configuration has no Debye-Waller factor, and none is added. Disorder comes from the atom
positions: from the frames of a trajectory or the states of a model.

## Uncertainty

Every fitted number is reported with an uncertainty and with a sentence that says how it was obtained. For S0² and
ΔE0 (`feffgpu.exafs.shared_intervals`): the noise of the measurement is not known, so the misfit at the optimum is
taken as the noise level, which gives each sample the weight W = N_ind / (2 M), where N_ind is the number of
independent points of its windows (2 Δk ΔR / π + 2) and M its misfit. The standard deviations follow from the
curvature of the summed weighted misfit at the optimum, and the intervals given are ±1.645 standard deviations
(90 %). These intervals hold the structure fixed; the uncertainty of the structure is estimated separately (by
repeated runs with different random seeds in RMC-EXAFS, by the spread between frames in MD-EXAFS).

## Units

Distances in Å, k in 1/Å, energies in eV, MSRD (mean-square relative displacement, the variance of a pair
distance) in Å².
