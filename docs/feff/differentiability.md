# Differentiability

The path stage and the forward model are written in PyTorch and are differentiable in the atom coordinates. The
gradient of any scalar function of a spectrum with respect to every coordinate comes from one backward pass.

## The gradient of a misfit

```python
import numpy as np, torch
from feffgpu.exafs import PairTables, FitWindow, k_grid, r_factor_torch

k = k_grid(20.0)
pairs = PairTables(phase, k, np.zeros(len(k)), e0=0.0, box=cell, rmax=6.0, ms=False, device="cuda")
pos = torch.tensor(positions, device="cuda", dtype=torch.float32)[None].requires_grad_(True)
window = FitWindow(kmin=3.0, kmax=16.0, dk=1.0, kweight=2, rmin=1.0, rmax=5.0)

chi = pairs.single_scattering(pos)                                   # (NK,)
misfit = r_factor_torch(chi[None], target, pairs.kt, window)[0]      # target: (NK,) tensor on the same grid
misfit.backward()
forces = -pos.grad                                                   # (1, atoms, 3): how each atom should move to lower the misfit
```

For multiple scattering, `MultipleScattering.chi_absorbers(pos_flat, absorbers)` returns spectra with a gradient,
and `MultipleScattering.full_gradient(pos_flat, loss)` gives the exact gradient of a loss of the mean spectrum in
passes of absorbers, so that memory stays bounded.

## What the gradient is used for

- **Gradient refinement** (`feffgpu.rmc.GpuRmc`): atom positions are moved along the gradient of the misfit.
- **Gradient steps inside the reverse Monte Carlo search**: after each random move, a few steps along the gradient
  ([](../rmc/search.md)).

## Details

- Path lists hold atom indices. While a list is held, the signal and its gradient follow the atoms exactly. The
  choice of paths (which paths pass the importance filter) is not differentiable; it changes only when the lists
  are found again.
- The pair histogram of `PairTables` uses linear interpolation between grid points of 0.002 Å, so the single
  scattering is continuous and piecewise differentiable in the positions.
- Gradients through long path lists use checkpointing: memory is traded for a second forward pass per batch.
- The path stage runs in complex64 by default. Use `dtype=torch.complex128` in `GPUPathStage` for checks of
  gradients by finite differences.
- The potentials and phase shifts are not differentiated: the phase file is held fixed while atoms move
  ([](../concepts/potentials.md)).
