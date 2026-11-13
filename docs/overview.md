# Overview

## What the package is

`feffgpu` calculates EXAFS spectra from atom positions with the theory of FEFF 8.5L, on an NVIDIA GPU.

FEFF computes the spectrum of one absorbing atom in a cluster in four steps: the scattering potentials of the atoms,
the phase shifts of each potential, the list of scattering paths that matter, and the signal of each path. `feffgpu`
does the same four steps in PyTorch. The first two give a *phase file* that depends on the chemistry and only weakly
on the exact positions; the last two depend on the positions and are fast. This split is what makes two things
practical:

- the spectrum of **every absorbing atom of every frame** of a molecular dynamics trajectory (MD-EXAFS), without
  writing one FEFF input per atom;
- **moving atoms until the spectrum fits** a measurement (RMC-EXAFS), with the multiple scattering of every model
  structure computed from its own atoms, and with gradients of the misfit with respect to every coordinate.

The three tools share one set of conventions (those of xraylarch, and so of Athena and Artemis), one forward model
and one set of analysis functions. They are described in [](concepts/conventions.md).

## What it is not

- It calculates **EXAFS only**. XANES, full multiple-scattering spectra, polarization, and spin are not ported.
- It keeps scattering paths of **up to eight legs** in MD-EXAFS and RMC-EXAFS (the default there is four) and up to nine,
  FEFF's own limit, for a FEFF input.
- It needs an **NVIDIA GPU** with a CUDA build of PyTorch. It runs on a CPU for small inputs, slowly.
- FEFF's own Debye-Waller models are not ported. In this package disorder comes from atom positions: from the
  frames of a trajectory or from the states of a reverse Monte Carlo model.
- It is not a data reduction program. Measured spectra are read as chi(k); background removal and normalization are
  done before, for example with xraylarch.

The full list is in [](feff/limits.md).

## How it was validated

Every stage reproduces FEFF's own intermediate files, and the whole chain was compared with FEFF 8.5L on random
environments drawn over elements, edges, structures and input cards. The results are in
[](validation/index.md). Literature values are not used as a yardstick anywhere in this package: validation is
against FEFF 8.5L itself and against synthetic targets with known answers.

## Where to go next

| You want to | Read |
|---|---|
| run a FEFF input on the GPU | [](quickstart.md) |
| get the spectrum of a trajectory | [](md/quickstart.md) |
| fit a structure to measured spectra | [](rmc/quickstart.md) |
| use the path stage inside your own code | [](feff/python_api.md), [](feff/differentiability.md) |
| run on several GPUs or a cluster | [](parallel.md) |
| let an agent run the tools | [](agents.md) |
| contribute | [](contributing.md) |
