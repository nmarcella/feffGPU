# RMC-EXAFS in 15 minutes

Reverse Monte Carlo (RMC) fitting moves the atoms of a model structure until its calculated spectra fit the measured
ones. `rmc-exafs` has two engines:

- **The gradient refinement** (`rmc-exafs refine`), the default: one structure moves downhill on the misfit of the
  exact spectra. Any elements, several edges, minutes on one GPU ([](refinement.md)).
- **The population search** (`rmc-exafs run`): EvAX's evolutionary algorithm, for exchanges and hops of atoms between
  sites and for several samples that share S0² and the energy shift ([](search.md)).

This page runs the simplest case, **one sample and one measured edge**, on a synthetic spectrum whose answer is
known: first with the refinement, then with the search. It needs an NVIDIA GPU and the package installed
([](../installation.md)); the free-atom table is not needed (the phase file is given).

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

## 2. The configuration (2 minutes)

One TOML file describes the run ([](configuration.md) lists every key). `config_refine.toml` of the example:

```toml
[run]
output = "runs/example_refine"
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
mode = "fixed"
s02 = 0.90
de0 = 1.5

[constraints]
max_displacement = 0.4
closest_distance = { "C-Zn" = 1.8, "C-C" = 2.6, "Zn-Zn" = 2.6 }

[refine]
copies = 1
steps = 200
ms_absorbers = 32
core_distance = 1.5
start_kick = 0.01
log_every = 20
```

- `[[sample]]` and `[[sample.edge]]`: the start structure and the measured spectrum, its phase file (which scattering
  potential each element uses), the window of the comparison (xraylarch's conventions: Hanning sills centered on
  `kmin` and `kmax`, the R-factor between `rmin` and `rmax`) and the cutoff of the pairs and paths.
- `[amplitude]`: the refinement holds S0² and the E0 shift, here at the planted values, as they would be held from a
  reference compound.
- `[constraints]`: every atom stays within 0.4 Å of its start position; no pair comes closer than chemistry allows.
- `[refine]`: 200 steps from the ideal block with a small random displacement.

Check the file without running anything:

```bash
rmc-exafs validate config_refine.toml
```

## 3. The run (1 minute)

```bash
rmc-exafs refine config_refine.toml
```

One line per 20 steps reports the misfit and how far the atoms have moved:

```
[rocksalt] start misfit C 1.45455; displacement limit 0.4 A
   step     0: misfit 1.45455; rms move 0.0000 A (max 0.000), step 0.00300 A (0 halvings so far), 1 s
   step    20: misfit 0.00935; rms move 0.0316 A (max 0.086), ...
   step    40: misfit 0.00075; rms move 0.0351 A (max 0.110), ...
   step   200: misfit 0.00046; rms move 0.0354 A (max 0.112), ...
[rocksalt] misfit C 1.45455 -> 0.00014; written runs/example_refine/rocksalt/refine.json, refine.npz
```

The misfit is the R-factor of the C edge: 1.5 at the start (almost no disorder: the calculated spectrum is far too
strong), below 0.001 after 40 steps. The whole run takes about a minute. (The last line is computed with the
multiple-scattering paths found again on the final atoms, which is why it differs from the last step's.) The outputs
are in `runs/example_refine/` ([](outputs.md)).

## 4. The result (3 minutes)

Shell numbers by integration of the pair distribution of the final structure:

```bash
rmc-exafs shells runs/example_refine --shell C-Zn:1.6:2.8 --shell C-C:2.6:3.6 --shell C-Zn:3.2:4.3
rmc-exafs plot runs/example_refine        # runs/example_refine/rocksalt/fits.png and pairs.png
```

| Shell | Planted R (Å), MSRD (Å²) | Refined R (Å), MSRD (Å²) |
|---|---|---|
| C-Zn 1.6-2.8 Å | 2.1903, 0.0054 | 2.1897, 0.0055 |
| C-C 2.6-3.6 Å | 3.0964, 0.0058 | 3.0961, 0.0055 |
| C-Zn 3.2-4.3 Å | 3.7914, 0.0048 | 3.7911, 0.0046 |

The distances come back to better than 0.001 Å and the widths to a few percent. A refinement holds one structure,
so the commands report no spread over states; repeat it with other values of `seed` and `start_kick` and give the
runs together to `rmc-exafs shells` to see how much the result depends on the start ([](uncertainty.md)).

## 5. The population search (8 minutes)

The same problem with EvAX's evolutionary algorithm, and with S0² and the E0 shift fitted. `config.toml` has the same
`[[sample]]` block and, in place of `[refine]`:

```toml
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

- `[amplitude]`: S0² and the E0 shift are refitted on the best state from generation 21 on.
- `[search]`: 16 states, 200 generations under EvAX's rules, with 20 gradient steps after every random move. (Those
  gradient steps hold the multiple scattering fixed; in this weakly scattering toy that is enough. See
  [](refinement.md) for where it is not.)

```bash
rmc-exafs run config.toml
rmc-exafs shells runs/example --shell C-Zn:1.6:2.8 --shell C-C:2.6:3.6 --shell C-Zn:3.2:4.3
```

One line per generation reports the best and mean misfit, the moves accepted and the state of the limits. The
misfit falls from about 2.3 to 6e-5 in 200 generations (about 4 minutes).

The fit is excellent, but S0² ends near 1.0 and the E0 shift near +1.0 eV, instead of 0.90 and +1.5 eV, and the first
C-Zn shell has an MSRD of about 0.011 Å² instead of 0.0054 Å², with a long tail. The search found another structure
that makes nearly the same spectrum: more disorder with a larger S0². With S0² and the E0 shift held at the planted
values (`rmc-exafs run config_held.toml`) the misfit reaches about 6e-6 and the shells are the planted ones, as in
the refinement above.

Three lessons of this example hold for every RMC fit:

1. **S0² trades with the disorder.** A misfit of 6e-5 against 6e-6 separates the two answers only because this
   spectrum has no noise. With measured data both are fits. Hold S0² at a value from a reference compound (the
   refinement always holds it), or compare fits with S0² held at several values. The interval that `run.json` gives
   for a fitted S0² is computed with the structure held fixed and does not include this trade. See
   [](uncertainty.md).
2. **The numbers of neighbors are inputs.** 6, 12 and 8 come out exactly, in every run: with a displacement limit
   every atom keeps its site, so the coordination numbers are those of the start lattice. See [](limits.md).
3. **Report what was held fixed**: the start lattice, the limits, the windows, S0² and the E0 shift if held.

## Next

- The gradient refinement in detail, with several elements and edges: [](refinement.md).
- How it compares with EvAX on EvAX's own published calculations: [](commissioning.md).
- Taking over an EvAX calculation (`rmc-exafs from-evax`) and the differences from EvAX: [](evax_differences.md).
- Several samples and edges in one search, shared S0² and E0 shifts: [](samples_and_edges.md).
- Exchanges of two elements and hops onto vacant sites: [](configuration.md) (`[moves]`); one generation of the
  search step by step: [](search.md).
- The analysis commands: [](analysis.md). The Python API: [](api.md).
