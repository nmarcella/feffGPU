# Paths and multiple scattering

## Paths

In FEFF's theory the EXAFS of an absorbing atom is a sum over *scattering paths*. A path starts at the absorber,
visits one or more neighbors, and returns. A path with one neighbor has two legs (out and back) and is called
*single scattering*. Paths with two or three neighbors have three or four legs and are called *multiple scattering*.

Each path contributes an oscillation in k whose frequency is set by half its total length and whose amplitude and
phase come from the scattering properties of the atoms on it. `feffgpu` keeps paths of two, three and four legs.

## The path finder

`feffgpu.GPUPathFinder` lists the paths of one absorber within a cutoff (FEFF's `RPATH`: half the total path
length, in Å), as FEFF's `path` module does:

- every sequence of neighbors within the cutoff is a candidate;
- each candidate gets FEFF's plane-wave importance estimate, and candidates below the `CRITERIA` threshold (a
  percentage of the largest single-scattering path) are dropped;
- paths that are equivalent by symmetry are merged, and the number merged is the path's *degeneracy*.

It works in a periodic cell of any shape with the minimum-image convention, or in a cluster. The cell must be at
least twice as wide as the cutoff. Each atom carries the index of its scattering potential in the phase file (see
[](potentials.md)).

## The path stage

`feffgpu.GPUPathStage` computes the signal of batches of paths with FEFF's algorithm (the Rehr-Albers separable
representation of the propagators, to second order), on the GPU. It is differentiable in the atom coordinates,
which gives the force that a misfit of the spectrum exerts on every atom.

## Single scattering without a path list

A two-leg path depends only on the distance of the pair and on the scatterer's potential. `feffgpu` therefore
tabulates the exact two-leg signal once per scatterer potential on a fine distance grid (0.002 Å), and gets the
single-scattering spectrum of a whole configuration as the histogram of its pair distances times the table
(`feffgpu.exafs.PairTables`). Every pair within the cutoff is included; there is no importance filter and no
sampling of absorbers. This costs milliseconds for a thousand atoms and is exact to the interpolation between grid
points.

## Multiple scattering of every absorber

For multiple scattering there is no such shortcut. `feffgpu.exafs.MultipleScattering` runs the path finder for
every absorber of every configuration and evaluates all three- and four-leg paths with the path stage. A path list
stores atom indices, not coordinates, so the signal of a list always follows the atoms; only the *choice* of paths
is held until the list is found again.

## What disorder does

FEFF's chi(k) for one static cluster has no damping from disorder. In a measurement, thermal motion and static
disorder spread every path length, which damps the oscillation more strongly at high k. In this package that
spread is not a parameter of a path: it comes from the atom positions. Averaging the spectra of many frames of a
trajectory, or of the atoms of a disordered model, gives the damping, including its non-Gaussian part.

## Limits

Paths of more than four legs are not computed. Five- and six-leg paths exist, most strongly in close-packed and
collinear arrangements. Their contribution is left out, and its size depends on the structure: it has not been
measured for this package. To estimate it for a given cluster, run FEFF 8.5L itself with `NLEG 4` and with a larger
`NLEG` and compare. If a FEFF input asks for `NLEG` above 4, `feffgpu` uses 4.
