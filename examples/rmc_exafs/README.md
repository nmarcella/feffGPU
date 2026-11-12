# RMC-EXAFS example: one sample, one edge, a known answer

A synthetic test from start to finish. The script builds a 2 x 2 x 2 block of a Zn-C rock salt (64 atoms), displaces
every atom at random by 0.05 Å per coordinate, and makes the C K-edge spectrum of that disordered block with the
package's own forward model, with S0² 0.90 and an E0 shift of +1.5 eV. That spectrum is the "measured" one. The search
starts from the ideal block and has to find the disorder. The answer is known: `truth.json`.

The phase file is FEFF 8.5L's own, from the validation case `data/validation/A000` (potentials 0 C absorber, 1 Zn,
2 C). Nothing else is needed and nothing is committed: the script writes its inputs next to itself.

## Commands

```bash
cd examples/rmc_exafs
python make_synthetic_target.py              # start.vasp, target.chi, truth.json (prints the planted shells)
rmc-exafs validate config.toml
rmc-exafs run config.toml                    # S0^2 and the E0 shift fitted; about 4 minutes on one GPU
rmc-exafs run config_held.toml               # the same with S0^2 and the E0 shift held at 0.90 and +1.5 eV
rmc-exafs shells runs/example --shell C-Zn:1.6:2.8 --shell C-C:2.6:3.6 --shell C-Zn:3.2:4.3
rmc-exafs shells runs/example_held --shell C-Zn:1.6:2.8 --shell C-C:2.6:3.6 --shell C-Zn:3.2:4.3
rmc-exafs plot runs/example                  # fits.png and pairs.png in runs/example/rocksalt
rmc-exafs signal-map runs/example_held --components Zn:1.5,2.8,4.3 --out runs/example_held/map
```

## What to expect

The numbers below are from one run on one GPU; another GPU gives the same run up to the last digits of its additions,
so expect the same numbers to two or three significant figures.

- The misfit (the R-factor of the C edge) falls from about 2.3 at the ideal start (the ideal block has no disorder,
  so its spectrum is far too strong) to about 0.01 after the first generation and below 1e-3 within 50 generations.
- **S0² and the E0 shift held** (`config_held.toml`): the misfit reaches about 6e-6 and the shells of the best state
  are the planted ones: first C-Zn shell R 2.190 Å, MSRD 0.0054 Å² (planted 2.190 Å, 0.0054 Å²); C-C shell 0.0058 Å²
  (0.0058); second C-Zn shell 0.0047 Å² (0.0048).
- **S0² and the E0 shift fitted** (`config.toml`): the misfit reaches about 6e-5, an excellent fit, yet S0² ends near
  1.0 and the E0 shift near +1.0 eV instead of 0.90 and +1.5 eV, and the shells are broader than planted (first C-Zn
  MSRD about 0.011 Å², with a long tail). The search found another structure that makes nearly the same spectrum: a
  higher S0² with more disorder. The interval of S0² in `run.json` is narrow because it is computed with the structure
  held fixed; it does not include this trade. With real data, whose noise is far above these misfits, the two answers
  cannot be told apart by the misfit. Hold S0² at a value from a reference compound, or compare runs with S0² held at
  several values (docs/rmc/uncertainty.md).
- The numbers of neighbors (6, 12, 8) are those of the lattice in every run: with a displacement limit no atom leaves
  its site, so they are inputs of the model, not results (docs/rmc/limits.md).

`runs/example/rocksalt/report.json` holds the records of the run; `history.json` one record per generation;
`state.npz` the final population (docs/rmc/outputs.md).
