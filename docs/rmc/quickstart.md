# RMC-EXAFS in 15 minutes

Reverse Monte Carlo (RMC) fitting moves the atoms of a model structure until its calculated spectra fit the measured
ones. This page runs the simplest case, **one sample and one measured edge**, on a synthetic spectrum whose answer is
known, and reads the result. It needs an NVIDIA GPU and the package installed ([](../installation.md)); the
free-atom table is not needed (the phase file is given).

## 1. The inputs (1 minute)

The example builds everything in code: a 2 x 2 x 2 block of a Zn-C rock salt (64 atoms), a copy with every atom
displaced at random by 0.05 Å per coordinate, and the C K-edge spectrum of that copy made by the package's own
forward model with S0² 0.90 and an E0 shift of +1.5 eV. That spectrum plays the measured one.

```bash
cd examples/rmc_exafs
python make_synthetic_target.py
```

It writes `start.vasp` (the ideal block: the start structure), `target.chi` (two columns, k in 1/Å and chi) and
`truth.json` (the planted values and the shells of the disordered copy), and prints the planted shells:

```
  C-Zn 1.6-2.8 A: N 6.00, R 2.1903 A, sigma2 0.00542 A^2
  C-C 2.6-3.6 A: N 12.00, R 3.0964 A, sigma2 0.00578 A^2
  C-Zn 3.2-4.3 A: N 8.00, R 3.7914 A, sigma2 0.00480 A^2
```

## 2. The configuration (3 minutes)

One TOML file describes the run ([](configuration.md) lists every key). `config.toml` of the example:

```toml
[run]
output = "runs/example"
seed = 1

[[sample]]
label = "rocksalt"
structure = "start.vasp"

  [[sample.edge]]
  name = "C"
  absorber = "C"
  chi = "target.chi"
  phase = "../../data/validation/A000/phase.pad"   # potential 0 is the C absorber
  potentials = { Zn = 1, C = 2 }
  window = { kmin = 3.0, kmax = 12.0, dk = 1.0, kweight = 2, rmin = 1.0, rmax = 4.0 }
  rpath = 4.3
  nleg = 3

[amplitude]
mode = "fit"
refit_from = 21
refit_every = 10

[search]
states = 16
generations = 200
froze_in = 150
gradient_steps = 20

[constraints]
max_displacement = 0.4
closest_distance = { "C-Zn" = 1.8, "C-C" = 2.6, "Zn-Zn" = 2.6 }
closest_tolerance = 0.05
```

- `[[sample]]` and `[[sample.edge]]`: the start structure and the measured spectrum, its phase file (which scattering
  potential each element uses), the window of the comparison (xraylarch's conventions: Hanning sills centered on
  `kmin` and `kmax`, the R-factor between `rmin` and `rmax`) and the cutoff of the pairs and paths.
- `[amplitude]`: S0² and the E0 shift are refitted on the best state from generation 21 on.
- `[search]`: 16 states, 200 generations under EvAX's rules, with 20 gradient steps after every random move.
- `[constraints]`: every atom stays within 0.4 Å of its start position; no pair comes closer than chemistry allows.

There is no `[moves]` section: no exchanges, no hops. Check the file without running anything:

```bash
rmc-exafs validate config.toml
```

## 3. The run (4 minutes)

```bash
rmc-exafs run config.toml
```

One line per generation reports the best and mean misfit, the moves accepted and the state of the limits:

```
[rocksalt] generation    1: best 0.01253 (R 0.0125), mean 0.02023; 5 pairs bred (0 children refused); 16 of 16 moves accepted ...
[rocksalt] generation   50: best 0.00043 (R 0.0004), ...
[rocksalt] generation  200: best 0.00006 (R 0.0001), ...
```

The misfit is the R-factor of the C edge: about 2.3 at the ideal start (no disorder: the calculated spectrum is far
too strong), 0.01 after one generation, 6e-5 at the end. The outputs are in `runs/example/` ([](outputs.md)).

## 4. The result (5 minutes)

Shell numbers by integration of the pair distribution of the best state, with their spread over the 16 states:

```bash
rmc-exafs shells runs/example --shell C-Zn:1.6:2.8 --shell C-C:2.6:3.6 --shell C-Zn:3.2:4.3
rmc-exafs plot runs/example        # runs/example/rocksalt/fits.png and pairs.png
```

The fit is excellent, but S0² ends near 1.0 and the E0 shift near +1.0 eV, instead of 0.90 and +1.5 eV, and the first
C-Zn shell has an MSRD of about 0.011 Å² instead of 0.0054 Å², with a long tail. The search found another structure
that makes nearly the same spectrum: more disorder with a larger S0². Now run the same search with S0² and the E0 shift
held at the planted values, as they would be held from a reference compound:

```bash
rmc-exafs run config_held.toml
rmc-exafs shells runs/example_held --shell C-Zn:1.6:2.8 --shell C-C:2.6:3.6 --shell C-Zn:3.2:4.3
```

The misfit reaches about 6e-6 and the shells are the planted ones: first C-Zn shell R 2.190 Å and MSRD 0.0054 Å², C-C
0.0058 Å², second C-Zn 0.0047 Å².

Three lessons of this example hold for every RMC fit:

1. **S0² trades with the disorder.** A misfit of 6e-5 against 6e-6 separates the two answers only because this
   spectrum has no noise. With measured data both are fits. The interval that `run.json` gives for S0² is computed
   with the structure held fixed and does not include this trade. See [](uncertainty.md).
2. **The numbers of neighbors are inputs.** 6, 12 and 8 come out exactly, in both runs: with a displacement limit every
   atom keeps its site, so the coordination numbers are those of the start lattice. See [](limits.md).
3. **Report what was held fixed**: the start lattice, the limits, the windows, S0² and the E0 shift if held.

## Next

- Several samples and edges in one fit, shared S0² and E0 shifts: [](samples_and_edges.md).
- Exchanges of two elements and hops onto vacant sites: [](configuration.md) (`[moves]`).
- One generation step by step: [](search.md); the differences from EvAX: [](evax_differences.md).
- The analysis commands: [](analysis.md). The Python API: [](api.md).
