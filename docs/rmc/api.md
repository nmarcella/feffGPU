# The Python API by task

Everything the command does is available from Python. The objects take a `device` (default `"cuda"`).

## Run a configuration

```python
from feffgpu.rmc import run_search, run_refine, validate_file, load_config

problems = validate_file("config.toml")            # [] if valid; else "field.path: message" per problem
summary = run_search("config.toml")                # the content of run.json
cfg = load_config("config.toml")                   # a validated RmcConfig with absolute paths
```

## Build the pieces yourself

```python
import numpy as np
from feffgpu.exafs import FitWindow, k_grid, read_chi, to_grid
from feffgpu import load_phase
from feffgpu.exafs import EdgeModel
from feffgpu.rmc import ModelCell, RmcSample, EvaxSearch, SearchSettings, AmplitudeSettings, ConstraintSettings

model = ModelCell.from_file("start.vasp", supercell=(2, 2, 2))           # every atom's start position is its site
k = k_grid(24.0)
kd, cd = read_chi("data.chi")
edge = EdgeModel("Zn", k, to_grid(k, kd, cd), FitWindow(kmin=3, kmax=13, dk=2, kweight=2, rmin=1, rmax=3.6),
                 load_phase("phase_Zn.pkl"), model.symbols, model.cell, "Zn", potentials={"Zn": 1, "O": 2}, rpath=5.0, nleg=3)
search = EvaxSearch([RmcSample("zno", model, [edge])], SearchSettings(states=16, generations=300, gradient_steps=20),
                    amplitude=AmplitudeSettings(refit_from=51), constraints=ConstraintSettings(max_displacement=0.4),
                    seed=1, output="runs/zno")
summary = search.run()
```

A search built this way writes every output file except `config.resolved.toml` (pass `resolved=` a configuration
dictionary to write it); `rmc-exafs signal-map` rebuilds the forward model from that file, so it needs a run of
`run_search` or of the command.

## The forward model of a structure

`feffgpu.exafs.edge.EdgeModel` (shared with MD-EXAFS) calculates the spectrum of configurations of a structure and compares
it with a measured spectrum:

```python
chi = edge.spectra(positions)                      # (S, NK): exact single scattering + multiple scattering of fresh paths
hist = edge.static_histograms(positions)           # (S, T, nr): pair histograms on the tables' grid (differentiable)
chi_ss = edge.spectrum(hist)                       # their single-scattering spectrum
R = edge.residual(chi)                             # R-factor of each, with the edge's S0^2 and E0 shift on the model
```

S0² and the E0 shift of several spectra, shared by name: `feffgpu.exafs.fit_shared` and `shared_intervals`.

## Sites, moves and constraints

| task | function |
|---|---|
| a site list for hops from a file | `feffgpu.rmc.read_site_file`, `sublattice_sites(symbols, positions, cell, element, sites)` |
| the anion sites of a fluorite structure | `feffgpu.rmc.fluorite_sites(symbols, positions, cell, anion)` |
| the displacement limit | `feffgpu.rmc.Box(site_positions, max_displacement, cell)`: `project`, `displacement`, `at_limit` |
| closest distances | `feffgpu.rmc.ClosestDistances(symbols, {"A-B": d}, cell, spring)`: `energy`, `worst`, `below` |
| moves | `start_population`, `uniform_step`, `exchange_atoms`, `hop_atoms` |
| gradient steps | `fire_relax(objective, positions, steps, project=...)` with `GradientObjective` |
| EvAX's rules | `evax_selection`, `site_crossover`, `EdgeWeights`, `Annealing` (`feffgpu.rmc.rules`) |
| per-state multiple scattering | `StatePaths(edge)`: `find`, `refind`, `near`, `signal` |
| arrangement statistics | `ArrangementStatistics(model, exchange=(A, B))` |

## Analysis

| task | function |
|---|---|
| read a run | `feffgpu.rmc.read_run`, `read_sample`, `feffgpu.rmc.analysis.load_states` |
| shell numbers with spreads | `feffgpu.rmc.analysis.shell_statistics` |
| impossible pairs | `feffgpu.rmc.analysis.short_pair_statistics`, `feffgpu.analysis.short_pairs` |
| pair distributions, angles | `feffgpu.rmc.analysis.run_rdf`, `run_angles`; `feffgpu.analysis.partial_rdfs`, `bond_angles` |
| figures | `feffgpu.rmc.analysis.plot_run`, `feffgpu.exafs.plot_comparison`, `feffgpu.analysis.plot_pair_distributions` |
| decomposition of a fit | `feffgpu.rmc.analysis.signal_map`; `feffgpu.analysis.signal_map` |

## The gradient refinement

`feffgpu.rmc.GpuRmc` ([](refinement.md)).

## For programs and agents

`feffgpu.rmc.config`: `json_schema()` (the JSON Schema), `example_configs()` (`{name: TOML text}`),
`validate_file(path)` (problems by dotted path, never raises for a problem of the file), `output_directory(path)`.
`feffgpu.rmc.cli.main(argv)` runs a command and returns its exit code.
