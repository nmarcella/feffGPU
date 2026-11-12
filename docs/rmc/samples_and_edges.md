# Samples and edges

## One sample, one edge

The simplest run: one structure, one measured spectrum, no rearrangements. For example an oxide whose cation edge was
measured:

```toml
[[sample]]
label = "nio"
structure = "NiO.vasp"
supercell = [3, 3, 3]

  [[sample.edge]]
  name = "Ni"
  absorber = "Ni"
  chi = "NiO_Ni.chi"
  phase = "phase_Ni.pkl"
  potentials = { Ni = 1, O = 2 }
  window = { kmin = 3.0, kmax = 13.0, dk = 2.0, kweight = 2, rmin = 1.2, rmax = 4.5 }
  rpath = 5.0
```

The misfit is the R-factor of that spectrum (its weight is 1). In `[amplitude] mode = "fit"`, S0² and the E0 shift
are fitted on that spectrum alone; `"fixed"` holds them at given values, `"from_run"` at the values of an earlier
run. The oxygen atoms move like the nickel atoms: every atom is free within the limits, and the oxygen positions are
determined only through the Ni-O pairs that the Ni edge sees.

## Several edges of one sample

Every `[[sample.edge]]` block of a sample is one spectrum of the same structure: another absorbing element, or the same
element measured in another window. Each edge has its own phase file (its potential 0 is its absorber), potentials,
window, objective and path settings. The misfit of a state is `M = sum_e w_e R_e`, with EvAX's adapted weights
(`adjust_spectra_weights`): `w_e = (1/a_e) / sum_f (1/a_f)`, `a_e` the running mean over the proposed moves of
`|R_e(trial) - R_e(state)|`. A spectrum whose misfit changes less weighs more, so every spectrum moves the total
by a similar amount. The weights start equal; `adapt_weights = false` keeps them equal.

## Several samples

Every `[[sample]]` block is a sample with its own structure, atoms, population and spectra. Samples are searched side
by side; their states never interact. Only **S0² and the E0 shift** connect them: spectra with the same `name` (the
same edge of the same element, measured on several samples) have one S0² and one E0 shift, fitted jointly on the best
state of every sample (`feffgpu.exafs.fit_shared`):

```
minimize  sum over samples g of  (1 / n_samples) x sum over spectra s of g of  w_s R_s(S0^2, dE0)
```

Each sample's search uses its own spectra only. The edges of one name must have the same absorber; their windows may
differ. A shared value is the best compromise for all samples, so a sample whose spectrum would want another value
pays a higher misfit: compare the misfits of the samples in `run.json`.

Typical uses:

- a series of samples of one compound (temperatures, compositions, treatments) measured at the same edge;
- a reference compound of known structure measured with the samples: its spectrum then pins S0² and the E0 shift that
  the other samples share. A reference whose atoms are free can trade S0² against its own disorder like any sample;
  hold its structure close to the known one with a small displacement limit.

## Shared values from an earlier run

A scan of an input (a displacement limit, a closest distance, a filling of sites) should keep the same S0² and E0
shifts for every value of the input, so that the input alone changes:

```toml
[amplitude]
mode = "from_run"
from_run = "runs/reference_fit"     # a finished run of rmc-exafs: its shared values are held
```

In mode `"fit"`, `from_run` gives start values instead.

## The k grid

Every spectrum is put on the model's uniform grid from 0 to 24 1/Å in steps of 0.05 1/Å by linear interpolation; the
window with its sills must lie inside the measured range (`rmc-exafs validate` checks it).
