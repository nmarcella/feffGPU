# Potentials of an edge

Every path signal needs the scattering potentials and the phase shifts of the atoms: a phase file
([](../concepts/potentials.md)). In MD-EXAFS each `[[edge]]` chooses explicitly where its phase file comes from:

```toml
[[edge]]
element = "Zn"
potentials = "reference"      # "reference" (default) | "per_frame" | "file"
```

The path signals are always computed from the actual positions of every frame. Only the potentials are shared between
frames, which is an approximation: see "What is and is not validated" below.

## `file`: your own phase file

```toml
potentials = "file"
phase = "phase_Zn.pkl"        # a pickle from `feffgpu phase cluster.inp phase_Zn.pkl`, or a phase.pad written by FEFF
```

- Potential 0 of the file must be the absorbing element; otherwise the run stops with a message.
- Every element of the trajectory gets the first scatterer potential with its atomic number. When that is not the
  right one (two potentials for one element, for example), give the map: `ipots = { Zn = 1, C = 2 }`.
- The spectrum's `kmax` must not exceed the k range of the phase file.
- A pickle is a Python object file: load only phase files that you made yourself.

This mode computes nothing before the frames. It is the mode of the tests and of the example, which use the phase
files of the repository.

## `reference`: computed once from a cluster of the trajectory

```toml
potentials = "reference"
reference_frame = 0           # default: the first selected frame
reference_atom = 17           # default: the first atom of the absorbing element
cluster_radius = 7.0          # Å
scf_radius = 4.0              # Å; 0 (the default) means no self-consistency, as FEFF without an SCF card
cards = ["EXCHANGE 0 0.0 0.0"]   # optional extra FEFF cards
```

The run cuts a cluster of radius `cluster_radius` around the reference atom of the reference frame (with periodic
images when the radius exceeds half the cell), writes a FEFF input for it, and computes the potentials and phase
shifts on the GPU with `feffgpu.cli.build_phase` (free atoms from the free-atom table, overlapped potentials, the
self-consistency loop if `scf_radius > 0`, phase shifts up to `phase_kmax`, by default `kmax + 2`). In the input,
potential 0 is the absorber and the scatterer potentials 1, 2, ... are the elements of the cluster in alphabetical
order (the rule of `md_exafs` 2.0); every atom of every frame gets the potential of its element. Every element of
the trajectory must have at least one atom in the cluster besides the absorber.

The cluster's input, the phase file and a summary are kept in the output folder:

```text
mdexafs_out/potentials/Zn/reference.feff.inp   the FEFF input of the cluster
mdexafs_out/potentials/Zn/reference.pkl        the phase file
mdexafs_out/potentials/Zn/reference.json       frame, atom, cluster size, potentials, time, settings
```

A later run, a continued run and every shard reuse the phase file while the settings that made it are unchanged. This
mode uses the free-atom table that ships with the package.

## `per_frame`: computed again at intervals

```toml
potentials = "per_frame"
potentials_every = 10         # new potentials every 10 selected frames
```

The selected frames are cut into blocks of `potentials_every`; the first frame of each block defines the cluster for
the whole block (the same calculation as `reference`, done more often). The phase files are kept as
`potentials/<edge>/frame_<index>.pkl`. Use it to see whether the potentials change along the trajectory, for example
when the local structure changes.

## Costs

The potentials are the slow part, the paths the fast part. A small cluster without self-consistency takes seconds;
the self-consistency loop takes tens of seconds and more, growing quickly with `scf_radius`
([](../concepts/potentials.md)). `reference` pays this once per edge; `per_frame` pays it once per block. The cache
also stores the two-leg tables of every phase file used (2 potentials x 2501 distances x 401 k points x 4 bytes,
about 8 MB before compression, for two scatterer potentials, `rpath` 6 Å, `pair_rmin` 1 Å and `kmax` 20), so a
`per_frame` run with many blocks has a larger cache.

On a cluster, compute the phase files once before the job array starts, so that the shards do not each compute them:

```bash
md-exafs run config.toml --potentials-only
```

## What is and is not validated

Validated, by the core test suite against FEFF 8.5L: the potentials, the phase shifts, the path finder and the path
stage for a given input ([](../validation/index.md)). Validated by the MD-EXAFS tests: the spectrum of a frame equals
an independent calculation with the path finder and path stage of every absorber, for a given phase file; a small
`reference` calculation runs and its phase file is kept and reused.

Not validated: how much the averaged spectrum depends on the choice between the modes, on the reference frame and
atom, on the cluster radius, or on self-consistency. These numbers have not been measured for this package, and they
depend on the material. To measure them for your system, run the same trajectory with two or three different
`reference_frame` or `reference_atom` values, or with `per_frame`, and compare the averaged spectra, for example with
`md-exafs compare` against the same measured spectrum, or with the R-factor between the two averages.
