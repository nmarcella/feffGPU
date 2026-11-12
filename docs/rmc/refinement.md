# Gradient refinement

`rmc-exafs refine` refines copies of a cell of one element by gradient descent on the exact spectrum. It is the
simpler of the two engines: no population, no lattice, no limit on the displacements; the objective is the R-space
misfit plus a hard core, nothing else. The population search ([](search.md)) handles any structure and every
constraint; the refinement is for one-element structures (metals) and for checks of the forward model.

## The method

An ensemble of `copies` copies of the cell (every atom an absorber) is refined so that the ensemble-averaged spectrum
matches the target in R space:

- **Single scattering** is exact at every step for every pair within the cutoff: the pair histogram times the two-leg
  signal of the path stage (`feffgpu.exafs.PairTables`), differentiable in the positions.
- **Multiple scattering** (three- and four-leg paths) is exact too: FEFF's path finder ([](../citing.md)) enumerates the paths of every
  absorber of every copy (found again every `refresh_every` steps) and the path stage evaluates them; its value in the
  misfit is the mean over all absorbers of their last full pass (every `value_every` steps). Its gradient, too costly
  for every absorber at every step, comes from a rotating subset of `ms_absorbers` absorbers accumulated into a running
  average whose memory is `ms_memory` cycles through all absorbers, after one exact pass at the start: in expectation
  the exact gradient.
- **The step** (`optimizer = "sgd"`): steepest descent with the step normalized to a root-mean-square move of
  `step_size` per atom, decaying on a cosine to `step_size x final_fraction`; a step is taken only if the misfit
  (single scattering live, multiple scattering as held) does not rise, else it halves and is tried again. Adam
  (`"adam"`) is available for comparison; it normalizes per atom and moves atoms that have no reason to move.
- **The hard core**: a harmonic penalty of stiffness `core_stiffness` (eV/Å²) on every pair closer than
  `core_distance`.

The target follows EvAX's convention: the measured chi divided by S0², compared with the model evaluated at the E0
shift. Both come from `[amplitude]` (`s02`, `de0`) and are held.

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

`rmc-exafs example refine` prints a complete file. The first `[[sample]]` and its first edge are used; the structure
must hold the absorber only. Outputs: `refine.json` and `refine.npz` in `<output>/<label>/` ([](outputs.md)).

## What to expect and what not

The start is the only prior: the copies stay near it where the spectrum does not pull them away. Started from a block
with too little disorder, the refinement recovers the pair variance of a synthetic target to about 10 % (the tests do
it for bcc W). The dependence of the result on the start is the method's uncertainty: repeat with other starts
(`start_kick`, `seed`) and report the spread. There is no limit on displacements: check the pair distributions
(`feffgpu.analysis`) for pairs at impossible distances.

Python:

```python
from feffgpu.rmc import GpuRmc
rmc = GpuRmc(phase, k, target, e0, cell, window=(3.0, 16.0), r_range=(1.5, 5.5), rmax=6.0, critpw=2.0, r_min=2.1)
positions, history = rmc.refine(start, steps=500, lr=0.003)
chi, misfit = rmc.spectrum_only(positions)
```
