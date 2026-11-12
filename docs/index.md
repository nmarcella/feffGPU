# feffgpu

`feffgpu` is FEFF 8.5L's EXAFS calculation rewritten in PyTorch for NVIDIA GPUs, with two tools built on it:

- **feffgpu** itself: a FEFF input file to chi(k), stage by stage as FEFF does it (potentials with the
  self-consistency loop, phase shifts, the path finder, the path stage), validated against FEFF 8.5L. The path
  stage runs on batches of hundreds of thousands of paths and is differentiable in the atom positions.
- **MD-EXAFS**: the EXAFS spectrum of a molecular dynamics trajectory, averaged over frames and absorbing atoms,
  with its decomposition by path type, radial distribution tools, and a comparison with a measured spectrum.
- **RMC-EXAFS**: reverse Monte Carlo fitting. The atoms of a model structure are moved until its calculated
  spectra fit the measured ones, by a population search that follows the rules of EvAX and by gradient
  refinement.

EXAFS is the extended X-ray absorption fine structure. chi(k) is the EXAFS signal as a function of the
photoelectron wave number k. FEFF is the program that calculates chi(k) from atom positions.

One installation gives all three:

```bash
pip install feffgpu
feffgpu doctor             # check the installation
```

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item-card} feffgpu in 5 minutes
:link: quickstart
:link-type: doc
A FEFF input to chi(k) on the GPU.
:::

:::{grid-item-card} MD-EXAFS in 10 minutes
:link: md/quickstart
:link-type: doc
A trajectory to its averaged spectrum.
:::

:::{grid-item-card} RMC-EXAFS in 15 minutes
:link: rmc/quickstart
:link-type: doc
A structure fitted to a spectrum.
:::
::::

`feffgpu` exists because FEFF exists, and its reverse Monte Carlo search follows EvAX. If you use this package,
please cite that work: see [](citing.md).

```{toctree}
:caption: Start
:maxdepth: 1

overview
installation
quickstart
```

```{toctree}
:caption: Concepts
:maxdepth: 1

concepts/pipeline
concepts/paths
concepts/potentials
concepts/conventions
concepts/what_exafs_determines
```

```{toctree}
:caption: feffgpu
:maxdepth: 1

feff/command_line
feff/python_api
feff/differentiability
feff/limits
```

```{toctree}
:caption: MD-EXAFS
:maxdepth: 1
:glob:

md/quickstart
md/*
```

```{toctree}
:caption: RMC-EXAFS
:maxdepth: 1
:glob:

rmc/quickstart
rmc/*
```

```{toctree}
:caption: Running at scale
:maxdepth: 1

parallel
agents
```

```{toctree}
:caption: Validation
:maxdepth: 1

validation/index
validation/random_environments
```

```{toctree}
:caption: Reference
:maxdepth: 1

api/index
contributing
changelog
citing
license
```
