# Potentials and phase files

## What a phase file is

Before any path can be computed, FEFF needs the scattering potential of every kind of atom and, from it, the
phase shifts of the photoelectron's partial waves at every energy. A *kind of atom* is a "unique potential" of the
FEFF input: potential 0 is the absorbing atom (with its core hole), potentials 1, 2, ... are the scatterers,
usually one per element.

A *phase file* holds these results: the energy grid, the energy reference, the phase shifts of every potential and
the absorber's dipole matrix elements. It is the content of FEFF's `phase.pad`. In `feffgpu` it is the object
`feffgpu.PhaseData`, written by

```bash
feffgpu phase feff.inp phase.pkl
```

or read from a `phase.pad` that FEFF wrote (`feffgpu.read_phase_pad`).

## How it is computed

1. **Free atoms.** FEFF's atomic code gives the electron density of each isolated atom. `feffgpu` reads these
   results from the free-atom table ([](../installation.md)) in place of running the atomic code.
2. **Overlapped potentials.** The atomic densities are overlapped in the cluster, the muffin-tin radii and the
   interstitial level are set, and the exchange-correlation self-energy (Hedin-Lundqvist by default) is added.
3. **Self-consistency.** With an `SCF` card, the densities are iterated to self-consistency. Each iteration needs
   the full multiple-scattering Green's function of the cluster inside the self-consistency radius at every energy
   of a contour. This is the heaviest part of the whole calculation and it grows quickly with that radius.
4. **Phase shifts.** The radial Dirac equation is solved for every potential and partial wave at every energy.

## Why one phase file serves many configurations

The potentials depend on the chemistry and on the average surroundings of an atom, and only weakly on the exact
positions of its neighbors. The path signals depend strongly on the positions. MD-EXAFS and RMC-EXAFS therefore
compute a phase file once, from a representative cluster, and reuse it for every frame or model structure, while the
paths are always computed from the actual positions.

This is an approximation. Its size depends on how much the local environment changes between configurations. The
tools let you choose: MD-EXAFS can recompute the potentials at intervals (its `potentials` setting), and any tool
accepts a phase file that you made yourself.

## The potential index of an element

The `POTENTIALS` block of the `feff.inp` that made a phase file fixes which index belongs to which element:

```text
POTENTIALS
*   ipot   Z   label
    0     30   Zn        the absorber
    1     30   Zn        Zn as a scatterer
    2      6   C         C as a scatterer
```

When a structure or a trajectory is given to MD-EXAFS or RMC-EXAFS, every atom gets the scatterer index of its
element (here Zn → 1, C → 2). The configuration files of the tools state this mapping.

## Choosing the reference cluster

- Use the same absorbing element and edge as the measurement.
- Include every element that occurs as a neighbor.
- Take a cluster that is typical of the material: for a crystal, the ideal structure; for a disordered system, a
  cut from an equilibrated frame.
- The self-consistency radius should hold a few neighbor shells. A larger radius costs much more time and memory.

## Costs

The potentials are the slow part and the path stage is the fast part. On the 12 GB card used for development, the
self-consistency loop took about 30 s for a cluster radius of 5 Å and about 300 s for 8 Å in clusters of a
thousand atoms; the path stage evaluated about 20,000 four-leg paths per second with gradients. Your numbers will
differ with the GPU and the input.
