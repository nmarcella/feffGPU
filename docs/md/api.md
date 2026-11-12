# Python API

Everything the `md-exafs` command does is available from Python. Every function that computes on a device takes
`device` (`"cuda"`, `"cuda:1"`, `"cpu"`).

## A whole run from a configuration

```python
from feffgpu.md import load_config, run, merge_shards, average_cache, write_average

cfg = load_config("config.toml")                 # validated; relative paths made absolute
summary = run(cfg)                               # per-frame spectra into <output>/chi.h5; returns report.json's content
run(cfg, shard="2/4")                            # one shard of the frames
merge_shards(cfg.output_path)                    # shard files -> chi.h5
results = average_cache(cfg.output_path / "chi.h5", cfg.average)   # {edge name: averages}
write_average(cfg.output_path / "average", results, cfg.system.name)
```

`load_config` raises `feffgpu.md.config.ConfigError`, whose `problems` holds one message per problem, each starting
with the field's path. `validate_file(path)` returns the same list without raising (and also reports input files that
do not exist); `json_schema()`, `example_configs()` and `output_directory(path)` give the schema, the shipped examples
and the output folder of a configuration file.

## Reading a trajectory

```python
import numpy as np
from feffgpu.io.trajectory import TrajectoryReader

reader = TrajectoryReader("dump.lammpstrj", type_map={1: "Zn", 2: "C"}, dynamic=True)
for frame in reader.iter_frames(reader.selection(start=100, stop=None, step=5)):
    frame.index, frame.symbols, frame.positions, frame.cell        # position in the file, (n,), (n, 3) Å, (3, 3) or None
```

A static cell is `TrajectoryReader(path, cell=np.eye(3) * 13.13)`; clusters are `periodic=False`;
`feffgpu.io.trajectory.static_cell(a, b, c, alpha, beta, gamma)` builds a cell from lengths and angles. Frames are
read one at a time.

## The spectrum of one frame

```python
from feffgpu.exafs import k_grid
from feffgpu.md import EdgeSpectra, choose_absorbers

calc = EdgeSpectra("phase.pad", "Zn", ipot_of={"Zn": 1, "C": 2}, elements=["C", "Zn"], k=k_grid(18.0),
                   rpath=6.0, nleg=4, criteria=2.0, device="cuda")
absorbers = choose_absorbers(np.nonzero(frame.symbols == "Zn")[0], 64, seed=0, frame_index=frame.index, element="Zn")
s = calc.spectrum(frame.positions, frame.cell, absorbers, frame.symbols)
s.chi, s.chi_ss, s.chi_ms          # (NK,): the mean over the absorbers, and its single- and multiple-scattering parts
s.chi_types                         # (T, NK): parts by path type, in the order of calc.types.names
s.ss_histogram, s.ms_binned         # what the cache stores for selections by distance
```

`calc.types` (`feffgpu.md.paths.PathTypes`) names the path types and turns either spelling into the canonical one
(`calc.types.normalize("Zn-Zn-C")`). `calc.tables` and `calc.r_table` are the single-scattering tables and their
distance grid.

## Phase files

```python
from feffgpu import load_phase
from feffgpu.md.potentials import potentials_of_phase, cut_cluster, reference_input

phase = load_phase("phase.pad")                                  # or a pickle from `feffgpu phase`
ipot_of = potentials_of_phase(phase, "Zn", ["C", "Zn"])           # {element: scatterer potential}
rel, sym, dist = cut_cluster(frame.positions, frame.symbols, frame.cell, center=17, radius=7.0)
text, ipot_of = reference_input(rel, sym, "Zn", edge="K", scf_radius=4.0, phase_kmax=20.0)
```

The text is a FEFF input; `feffgpu.cli.build_phase(path, xkmax, device)` computes its potentials and phase shifts.
`feffgpu.md.potentials.PhaseProvider` does all of this for a run, including the caching in the output folder.

## Averages from the cache

```python
from feffgpu.md import edge_average
from feffgpu.md.cache import open_cache
from feffgpu.md.config import AverageConfig

with open_cache("mdexafs_out/chi.h5") as f:
    r = edge_average(f, "Zn", AverageConfig(path_types=["Zn-C"], max_distance=2.6, start=200))
r["k"], r["mean"], r["spread"], r["stderr"]       # the average, the spread between frames, the block standard error
r["parts"], r["convergence"], r["chi_frames"]      # selected parts, convergence with the number of frames, per-frame spectra
```

## Comparison with measured spectra

```python
from feffgpu.md.compare import compare_runs
from feffgpu.md.run import load_run

report = compare_runs([load_run("run_a/config.toml"), load_run("run_b/mdexafs_out")], "joint_comparison")
report["shared"]["s02"], report["shared"]["s02_interval"], report["shared"]["sampling"]
```

The pieces are the shared ones: `feffgpu.exafs.SpectrumComparison`, `fit_shared`, `shared_intervals`,
`plot_comparison` ([](comparison.md)).

## Radial distributions, peaks, angles

```python
from feffgpu.analysis.rdf import RDFAccumulator, AngleAccumulator
from feffgpu.analysis.peaks import analyze_peak

acc = RDFAccumulator(symbols, pairs=[("Zn", "C")], r_max=8.0, bin_width=0.01, shells=[("Zn", "C", 1.8, 3.0)], device="cpu")
for frame in reader.iter_frames(selection):
    acc.add(frame.positions, frame.cell)
res = acc.result()
n = res["pairs"]["Zn-C"]["pairs_per_center_per_A"]
peak = analyze_peak(res["r"], n, 1.8, 3.0, fit="both", exact=res["shells"][0])
peak["integration"], peak["gaussian"], peak["skewed"]
```

`feffgpu.analysis.shell_table`, `partial_rdfs`, `bond_angles` and the other functions of `feffgpu.analysis.pairs`
work on arrays of configurations held in memory.

## Convergence

```python
from feffgpu.analysis.convergence import read_log, thermo_summary, block_average, spectrum_convergence

log = read_log("run.ener")                    # CP2K .ener, ASE MD log, or CSV
summary = thermo_summary(log, skip=1000.0)    # skip in the log's time unit
c = spectrum_convergence(r["k"], r["chi_frames"], weights=None, window={"kmin": 3, "kmax": 14, "dk": 1, "kweight": 2, "rmin": 1, "rmax": 5})
```

## The dynamics runner

```python
from feffgpu.md.dynamics import make_calculator, run_dynamics

run_dynamics(load_config("dynamics.toml"))     # writes [system] trajectory and [dynamics] log
calc = make_calculator("mace_mp", model="medium", device="cuda")   # or "emt", or "module:function"
```

## Reference

The modules, with every function and class: `feffgpu.md.config`, `feffgpu.md.spectrum`, `feffgpu.md.paths`,
`feffgpu.md.potentials`, `feffgpu.md.run`, `feffgpu.md.cache`, `feffgpu.md.average`, `feffgpu.md.compare`,
`feffgpu.md.analyze`, `feffgpu.md.dynamics`, `feffgpu.md.plots`, `feffgpu.md.cli`, `feffgpu.io.trajectory`,
`feffgpu.analysis.rdf`, `feffgpu.analysis.peaks`, `feffgpu.analysis.convergence`. Their docstrings (NumPy style, with
units) are the detailed reference.
