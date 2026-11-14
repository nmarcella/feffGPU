# Gradient refinement

`rmc-exafs refine` refines the atoms of a structure by gradient descent on the exact spectra, with the gradient of the
multiple scattering as well as of the single scattering. Any elements, one or several spectra (several edges of one
sample), periodic cells and particles. It has no population: one structure (or a few copies of it) moves downhill
from its start. It needs a few hundred steps where a random search needs many thousands of generations.

Which engine for what:

- **The refinement** when the spectra are to be fitted from a start structure and the atoms keep their sites. It is
  the route for metals and for anything with strong multiple scattering.
- **The population search** ([](search.md)) for EvAX's own rules, for exchanges and hops of atoms between sites, and
  for several samples that share S0² and the energy shift. Its optional gradient steps hold the multiple scattering
  fixed while the atoms move. That is enough where multiple scattering is weak; where it is strong those steps stop
  improving the fit (see "Why the multiple-scattering gradient matters" below), and the refinement should be used.

Two code paths do the same thing. A structure of one element with one spectrum fitted by the R-factor is refined by
`feffgpu.rmc.GpuRmc` (several copies, a hard core, no displacement limit). Everything else is refined by
`feffgpu.rmc.EdgeRefinement`: several elements, several spectra, the wavelet objectives, with the displacement limit
and the closest distances of `[constraints]`.

## The method

An ensemble of `copies` copies of the cell (every atom an absorber) is refined so that the ensemble-averaged spectrum
matches the target in R space:

- **Single scattering** is exact at every step for every pair within the cutoff: the pair histogram times the two-leg
  signal of the path stage (`feffgpu.exafs.PairTables`), differentiable in the positions.
- **Multiple scattering** (paths of three to `nleg` legs) is exact too: FEFF's path finder ([](../citing.md)) enumerates the paths of every
  absorber of every copy (found again every `refresh_every` steps) and the path stage evaluates them; its value in the
  misfit is the mean over all absorbers of their last full pass (every `value_every` steps). Its gradient, too costly
  for every absorber at every step, comes from a rotating subset of `ms_absorbers` absorbers accumulated into a running
  average whose memory is `ms_memory` cycles through all absorbers, after one exact pass at the start: in expectation
  the exact gradient.
- **Which paths.** By default FEFF's filter is applied to every path of the structure as it is
  (`criteria_rule = "feff"`), again every `refresh_every` steps. With `criteria_rule = "evax"` in the edge block the
  paths are chosen once on the start lattice by EvAX's rule ([](evax_differences.md)): whole groups of paths, by the
  size of their signal. The list is then the same for every atom and every step, which also removes the small jumps
  of the misfit when paths enter or leave the list. For a structure that starts as a crystal this is the better
  choice.
- **The step** (`optimizer = "sgd"`): steepest descent with the step normalized to a root-mean-square move of
  `step_size` per atom, decaying on a cosine to `step_size x final_fraction`; a step is taken only if the misfit
  (single scattering live, multiple scattering as held) does not rise, else it halves and is tried again. Adam
  (`"adam"`) is available for comparison; it normalizes per atom and moves atoms that have no reason to move.
- **The hard core**: a harmonic penalty of stiffness `core_stiffness` (eV/Å²) on every pair closer than
  `core_distance`.

S0² and the E0 shift come from `[amplitude]` and are held: the declared `s02` and `de0` (`mode = "fixed"`, the
default), or the values that an earlier run of `rmc-exafs run` found, for example on a reference material
(`mode = "from_run"`). The refinement never fits them; with `mode = "fit"` it stops with a message.

With several elements or several spectra (`EdgeRefinement`):

- **The misfit** of each spectrum is its own `objective` (the R-factor, EvAX's wavelet residual or the Cauchy
  wavelet), with its S0² and energy shift on the model. The refinement lowers the sum over the spectra, each with
  the same weight.
- **Single scattering** comes from the pair histograms of the edge model, split by neighbor element.
- **The displacement limit** `[constraints] max_displacement` (default 0.4 Å, EvAX's) holds every atom within that
  distance of its start site; 0 switches it off. `[constraints] closest_distance` adds springs for the pairs it
  names; `core_distance` is the closest distance of every other pair of elements.
- **Only `optimizer = "sgd"`** (the accepted-step rule); `copies` may be more than one.

## Why the multiple-scattering gradient matters

The gradient steps of the population search move the atoms on the single-scattering gradient while the multiple
scattering of the state is held. When the atoms have moved, the held multiple scattering is no longer theirs. In a
metal the multiple scattering between 4 and 6 Å is as large as the single scattering and reacts faster to disorder,
so the recomputed misfit is worse than the one the steps believed in. Two cases from the development of this
package: for cuprite (Cu2O, 384 atoms, a calculation of EvAX's library) the search with 20 gradient steps per
generation reached a wavelet misfit of 0.49 in three generations and then accepted no further trial, while the
refinement reached 0.034 in 300 steps and 12 minutes (EvAX's own fit: 0.037); for an fcc copper foil the search
stalled in the same way. A test (`tests/rmc/test_refine_edges.py`) shows the same on bcc tungsten.

## The configuration

```toml
[[sample]]
label = "cu"
structure = "Cu_fcc.vasp"
supercell = [4, 4, 4]

  [[sample.edge]]
  name = "Cu"
  absorber = "Cu"
  chi = "Cu.chi"
  phase = "phase_Cu.pkl"
  window = { kmin = 3.0, kmax = 16.0, dk = 1.0, kweight = 2, rmin = 1.5, rmax = 5.5 }
  rpath = 6.0
  nleg = 4

[amplitude]
s02 = 0.9
de0 = 1.0

[refine]
copies = 4
steps = 500
step_size = 0.003
core_distance = 2.1
```

`rmc-exafs example refine` prints a complete file. The first `[[sample]]` is used, with all its `[[sample.edge]]`
blocks. Outputs: `refine.json` and `refine.npz` in `<output>/<label>/` ([](outputs.md)).

A structure with two elements and two edges differs only in the edge blocks and the constraints:

```toml
[[sample]]
label = "particle"
structure = "start.extxyz"

  [[sample.edge]]
  name = "Pd_K"
  absorber = "Pd"
  chi = "Pd_K.chi"
  phase = "phase_Pd_K.pkl"
  potentials = { Au = 1, Pd = 2 }
  window = { kmin = 3.0, kmax = 11.0, dk = 1.0, kweight = 2, rmin = 1.0, rmax = 6.0 }
  rpath = 7.0

  [[sample.edge]]
  name = "Au_L3"
  absorber = "Au"
  chi = "Au_L3.chi"
  phase = "phase_Au_L3.pkl"
  potentials = { Au = 1, Pd = 2 }
  window = { kmin = 3.0, kmax = 11.0, dk = 1.0, kweight = 2, rmin = 1.0, rmax = 6.0 }
  rpath = 7.0

[amplitude]
mode = "fixed"

[constraints]
max_displacement = 0.5

[refine]
copies = 1
steps = 300
ms_absorbers = 64
core_distance = 2.0
start_kick = 0.01
```

## What to expect and what not

The start is the only prior: the copies stay near it where the spectrum does not pull them away. Started from a block
with too little disorder, the refinement recovers the pair variance of a synthetic target to about 10 % (the tests do
it for bcc W). The dependence of the result on the start is the method's uncertainty: repeat with other starts
(`start_kick`, `seed`) and report the spread. A gradient refinement moves the atoms only as far as the spectra ask
for: where the spectra say little (outer shells of a disordered particle, light neighbors far away) the widths stay
close to those of the start, while a random search spreads the atoms there. Shell distances are much less
affected than shell widths. For one element with the R-factor (`GpuRmc`) there is no limit on displacements: check
the pair distributions (`feffgpu.analysis`) for pairs at impossible distances.

Python:

```python
from feffgpu.rmc import GpuRmc
rmc = GpuRmc(phase, k, target, e0, cell, window=(3.0, 16.0), r_range=(1.5, 5.5), rmax=6.0, critpw=2.0, r_min=2.1)
positions, history = rmc.refine(start, steps=500, lr=0.003)
chi, misfit = rmc.spectrum_only(positions)

from feffgpu.rmc import EdgeRefinement, build_sample, load_config
cfg = load_config("rmc.toml")
sample = build_sample(cfg.sample[0], cfg)                       # the model cell and one EdgeModel per spectrum
ref = EdgeRefinement(sample.edges, sample.model.positions, sample.model.symbols, sample.model.cell, max_displacement=0.4, core_distance=1.5)
positions, history = ref.refine(sample.model.positions[None], steps=300)
spectra, misfits = ref.spectra(positions)
```
