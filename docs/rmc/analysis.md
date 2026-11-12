# Analysis of a run

The analysis commands read the files of a finished run ([](outputs.md)). `RUN` is a run's output directory (all its
samples; choose one with `--sample LABEL`) or one sample's subdirectory. Every command accepts `--json`. The generic
tools behind them are in `feffgpu.analysis` and work on any configurations (frames of a trajectory too).

## Shell numbers: `rmc-exafs shells`

```bash
rmc-exafs shells runs/fit --shell Zn-O:1.8:2.4 --shell Zn-Zn:2.9:3.5
rmc-exafs shells runs/seed1 runs/seed2 runs/seed3 --shell Zn-O:1.8:2.4 --json
```

For every stated shell (center element, neighbor element, range of distances): N (pairs per center), R (mean
distance), sigma2 (variance: the MSRD) and c3 (third cumulant), by integration of the pair distribution of the
positions; nothing is fitted. Reported for the best state, with the spread over the states of each run, and with
several runs, the spread of their best states over the runs ([](uncertainty.md)).

Python: `feffgpu.rmc.analysis.shell_statistics([dirs], [("Zn", "O", 1.8, 2.4)])`.

## Pairs at impossible distances: `rmc-exafs short-pairs`

```bash
rmc-exafs short-pairs runs/fit                                   # the run's closest distances
rmc-exafs short-pairs runs/fit --threshold Zn-O:1.85 --threshold O-O:2.4
```

For every pair of elements: the pairs below the threshold per center atom, the share of center atoms that have one,
and the closest such pair, in the best state and in the whole population.

## Pair distributions: `rmc-exafs rdf`

```bash
rmc-exafs rdf runs/fit --center Zn --rmax 6 --out runs/fit/rdf
```

The partial pair distributions of the center element with every element (pairs per center per Å; the integral over a
range is a coordination number), over all states (`--which population`) or the best state. `--out` writes them to an
npz file per sample.

## Bond angles: `rmc-exafs angles`

```bash
rmc-exafs angles runs/fit --center Zn --neighbors O,O --ranges 1.8:2.4,1.8:2.4
```

The angles B-A-C at the center A between neighbors B and C within the stated distance ranges: their number, mean,
standard deviation and histogram (1 degree bins; plain counts, not divided by the sine of the angle).

## Figures: `rmc-exafs plot`

```bash
rmc-exafs plot runs/fit --out figures
```

`fits.png`: for every edge, k^w chi(k) in the window and |chi(R)|, data in black, the start state and the final best
state; every edge with its own window as stored in `spectra.npz`. `pairs.png`: the partial pair distributions of every
absorbing element, the start lattice filled, the final states as lines.

## Which distances make which features: `rmc-exafs signal-map`

```bash
rmc-exafs signal-map runs/fit --components O:1.5,2.2,2.6,4.0 --band 1:2 --band 2:3 --band 3:4 --out figures
```

The calculated spectrum of the best state is split into the single scattering of distance components (one element,
one range of distances between consecutive boundaries) and the multiple-scattering remainder. For every part and every
R band of chi(R): its share `Re <part, model> / <model, model>`. The shares of a band add up to one, interference
included; a negative share cancels signal in that band. The forward model is rebuilt from the run's
`config.resolved.toml`, so the structure, spectrum and phase files must still be where the run found them. The figure
shows the pair distribution of every component against the distance and |chi(R)| of every part; R is the distance
axis of the Fourier transform, and a feature sits below the distance that makes it by the scattering phase.

Python: `feffgpu.analysis.signal_map.component_spectra(edge, positions, components)` and `band_shares(...)` for any
`feffgpu.exafs.edge.EdgeModel` and configurations.

## Reading a run in Python

```python
from feffgpu.rmc import read_run, read_sample
from feffgpu.rmc.analysis import load_states

run = read_run("runs/fit")                         # run.json: shared S0^2 and E0 shifts, intervals
s = read_sample("runs/fit/zno")                    # report, history, state, spectra (dicts of arrays)
st = load_states("runs/fit/zno")                   # symbols, cell, population (P, C, n, 3), best (C, n, 3), ideal, misfits
```
