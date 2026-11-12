# Python API by task

The generated reference of every function is in [](../api/index.md). This page shows the common tasks.

## A FEFF input to chi(k)

```python
from feffgpu import run

out = run("feff.inp", device="cuda")
out["k"], out["chi"]        # numpy arrays on FEFF's grid, 0.05 to 30 1/Å; S0² = 1, no Debye-Waller factor
out["counts"]               # {number of legs: number of paths}
out["timing"]               # {stage: seconds}
out["pot"], out["phase"]    # the content of FEFF's pot.pad and phase.pad
out["kept"]                 # the paths: {nleg: (coords, ipots, degeneracy, importance, atom indices)}
out["input"]                # the parsed input cards
```

`run(..., kgrid=k)` evaluates on your own k points. `run(..., pot=pot, phase=phase)` skips the stages whose results
you already have.

## Reading an input and a phase file

```python
from feffgpu import read_feff_inp, read_phase_pad

P = read_feff_inp("feff.inp")           # dict: positions P["rat_ang"] (Å), potential index per atom P["iphat"], cutoff P["rmax"], ...
phase = read_phase_pad("phase.pad")     # a phase file written by FEFF 8.5L
```

```python
from feffgpu.cli import build_phase

phase, info = build_phase("feff.inp", xkmax=24.5, device="cuda")     # potentials + phase shifts on the GPU
```

## The paths of one absorber in your own structure

```python
import numpy as np
from feffgpu import GPUPathFinder

finder = GPUPathFinder(phase, device="cuda", rmax=6.0, nlegmax=4, critpw=2.0)
kept, ref = finder.enumerate(positions, cell, absorber, ipot)
```

- `positions`: (n, 3) in Å; `cell`: a cubic length, a 3 × 3 matrix with the lattice vectors as rows, or `None` for
  a cluster; `absorber`: the index of the absorbing atom; `ipot`: (n,) the scatterer potential index of every atom
  (the absorber's own entry is replaced by 0).
- `kept[nleg] = (coords, ipots, deg, importance, atoms)`: for the paths with `nleg` legs, the coordinates
  (B, nleg, 3) with the absorber at the origin, the potential index of each atom of the path, the degeneracy, the
  importance, and the atom indices.

## The signal of paths on your own k grid

```python
import torch
from feffgpu import GPUPathStage

stage = GPUPathStage(phase, device="cuda")                 # complex64; dtype=torch.complex128 on request
k = torch.arange(0, 20.0001, 0.05)                         # 1/Å
S = stage.linear_matrix(k.numpy())                         # interpolation from the phase file's grid to k
chi = torch.zeros(len(k), device="cuda")
for nleg, (coords, ipots, deg, _imp, _atoms) in kept.items():
    cchi, reff = stage.cchi(coords, ipots)                 # FEFF's complex path signal on the phase file's grid
    chi += (deg[:, None] * stage.chi_fine(cchi, reff, k.numpy(), S, "feff")).sum(0)
```

To apply an energy shift ΔE0, evaluate at `q = sqrt(k² − 0.2625 ΔE0)` in place of `k`. S0² and Debye-Waller
factors are left to the caller, as in FEFF's `chi.dat`.

## The spectrum of whole configurations

For many absorbers at once use the forward model of `feffgpu.exafs`, which MD-EXAFS and RMC-EXAFS are built on:

```python
import numpy as np, torch
from feffgpu.exafs import PairTables, MultipleScattering, k_grid

k = k_grid(20.0)
pairs = PairTables(phase, k, np.zeros(len(k)), e0=0.0, box=cell, rmax=6.0, ms=False, device="cuda")
pos = torch.tensor(positions, device="cuda", dtype=torch.float32)[None]      # (copies, atoms, 3)
chi_ss = pairs.single_scattering(pos)                                         # every atom an absorber, every pair within 6 Å

ms = MultipleScattering(phase, k, 0.0, cell, rmax=6.0, critpw=2.0, nlegmax=4, device="cuda")
ms.enumerate(pos)                                                             # FEFF's path finder for every absorber
chi_ms = ms.full_pass(pos.reshape(-1, 3)).mean(0)                             # three- and four-leg paths
chi = chi_ss + chi_ms
```

With several elements, give `ipot` (the scatterer potential of every atom) and `absorbers` (the indices of the
absorbing atoms) to `MultipleScattering.enumerate`; `feffgpu.exafs.EdgeModel` wraps both parts for one absorption
edge of a structure with several elements.

## Comparing with a measured spectrum

```python
from feffgpu.exafs import FitWindow, SpectrumComparison, fit_shared, shared_intervals, read_chi, to_grid

kd, cd = read_chi("measured.chi")
window = FitWindow(kmin=3.0, kmax=14.0, dk=2.0, kweight=2, rmin=1.0, rmax=5.0)
cmp = SpectrumComparison("Zn", k, to_grid(k, kd, cd), window, fit_s02=True, fit_e0=True, device="cuda")
groups = [[(cmp, chi, 1.0)]]                # one sample with one spectrum
fit = fit_shared(groups, s02_bounds=(0.5, 1.5), de0_bounds=(-8.0, 8.0))
intervals = shared_intervals(groups)        # 90 % intervals and the sentence that says how they were obtained
print(cmp.s02, cmp.de0, cmp.r_factor(chi))
```

## Pair analysis

```python
from feffgpu.analysis import shell_table, partial_rdfs, short_pairs, bond_angles

rows = shell_table(positions, symbols, cell, [("Zn", "C", 1.8, 2.6)])     # N, R, MSRD, third cumulant by integration
```

`positions` may be (n, 3) or (configurations, n, 3). See [](../md/analysis.md) and [](../rmc/analysis.md).
