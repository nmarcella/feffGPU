# Trajectories

MD-EXAFS reads an existing trajectory frame by frame (`feffgpu.io.trajectory`). A frame is the element of every
atom, the Cartesian positions (Å) and the periodic cell (three lattice vectors, or none for a cluster). Frames are
produced one at a time: a trajectory with tens of thousands of frames is never held in memory. OVITO is not needed.

## Formats

| Format | `format` | Read by | Read frame by frame | Tested |
|---|---|---|---|---|
| extended xyz | `extxyz` (guessed from `.xyz`, `.extxyz`) | ASE's extended xyz reader | yes: one scan stores where each frame starts | yes, 10,000 frames |
| CP2K xyz, plain xyz | `extxyz` (or `xyz`) | the same reader (the comment line of CP2K is read as text) | yes | yes |
| LAMMPS text dump | `lammps-dump-text` (guessed from the content, `.lammpstrj`, `.dump`) | ASE's chunk reader of LAMMPS dumps | yes | yes, 10,000 frames |
| VASP XDATCAR | `vasp-xdatcar` (guessed from the name) | a reader of this package | yes, constant and variable cells | yes |
| ASE trajectory | `traj` | `ase.io.iread` | yes (random access) | yes |
| other formats of ASE | ASE's name | `ase.io.iread` | depends on ASE's reader | no |
| DCD, XTC, TRR, NetCDF and other binary formats | MDAnalysis's name | MDAnalysis (`reader = "mdanalysis"`) | yes | only the message shown when MDAnalysis is missing |

Notes:

- ASE's own `iread` builds every frame of a LAMMPS dump before it yields the first one, and ASE's XDATCAR and plain
  xyz readers read the whole file. This package avoids all three (it calls ASE's chunk reader for LAMMPS dumps,
  reads XDATCAR files itself and sends plain xyz files to the extended xyz reader). For another ASE format, check
  whether its reader streams before you give it a very large file; converting to extended xyz always works.
- The MDAnalysis reader is optional (`pip install 'feffgpu[mda]'`). Give the topology file in `[system] topology`
  when the trajectory file does not name the atoms. It has not been run in the development environment, where
  MDAnalysis is not installed.
- The test suite generates a 10,000-frame extended xyz file and a 10,000-frame LAMMPS dump and checks that reading
  them keeps the traced memory at a few megabytes, far below what the frames take when held, and that a LAMMPS dump
  yields its first frames before the reader reaches a damaged end of the file.

## Numeric atom types

LAMMPS writes numeric types. `[atoms]` maps them to elements:

```toml
[atoms]
1 = "Zn"
2 = "C"
```

For LAMMPS dumps the map applies to the `type` column; ASE sorts the atoms of a dump by their id, so the order does
not depend on how LAMMPS wrote them. For other files the map applies to the atomic numbers that ASE read (a type
written as `1` is read as hydrogen). A type that is missing from the map is an error that names it. The form
`Zn = 1` of `md_exafs` 2.0 is accepted too.

## Cells

The distances between atoms use the minimum-image convention with the cell (fractional rounding, so any cell shape
works).

**Static cell (NVT).** Give it once in `[lattice]`, in one of three ways; it is used for every frame and any cell
stored in the file is ignored:

```toml
[lattice]
a = 13.13                    # lengths (Å) ...
b = 13.13
c = 13.13
alpha = 90.0                 # ... and angles (degrees); b along the xy plane, as in crystallography
beta = 90.0
gamma = 90.0
# a = [13.13, 0.0, 0.0]      # or three lattice vectors (Å), as in md_exafs 2.0
# b = [0.0, 13.13, 0.0]
# c = [0.0, 0.0, 13.13]
# matrix = [[13.13, 0, 0], [0, 13.13, 0], [0, 0, 13.13]]    # or the vectors as rows of a matrix
```

**Cell of every frame (NPT).** `dynamic = true` reads the cell stored in each frame: the `Lattice` key of extended
xyz, the box of a LAMMPS dump (orthogonal or triclinic), the header of a variable-cell XDATCAR, the cell of an ASE
trajectory. A frame without a cell is an error. When `[lattice]` gives no static cell, the cells of the frames are
used.

**Clusters.** `pbc = [false, false, false]`: no cell and no periodic images. Mixed periodicity (a slab periodic in
two directions) is not supported; put enough vacuum around the slab and give a full cell instead.

**Small cells.** The EXAFS calculation needs every path to be found in one copy of the cell: the cell's smallest
width must be at least twice `rpath`. When it is not, the run stops with a message; `replicate = [2, 2, 2]` (or as
needed) repeats the cell for the calculation, and the absorbing atoms are still taken from the original cell only.
The RDF tools need no such setting: they include periodic images as far as the largest distance asks.

Positions may be wrapped into the cell or not (unwrapped coordinates of LAMMPS, for example): only the minimum-image
displacements are used.

## Which frames

```toml
[frames]
start = 0            # the first frame of the file is 0
stop = 1000          # excluded, as in Python's range; leave it out for the end of the file
step = 10
absorbers_per_frame = "all"     # or a number
seed = 0
```

The selected frames are `range(start, stop, step)`, clipped to the end of the file. They depend only on these
settings: a run split into shards uses exactly the same frames ([](shards.md)). A number of absorbers per frame
draws a random subset of the absorbing atoms of each frame from a random stream keyed by the seed, the frame index
and the element, so the subset of a frame is the same whichever shard computes it.

## What is checked

- Every frame must hold the same atoms in the same order. A frame with a different number, order or kind of atoms is
  an error (simulations with a changing number of atoms are not supported).
- Every element of the trajectory must have a scatterer potential in each edge's phase file ([](potentials.md)).
- Pairs closer than the start of the single-scattering table (`pair_rmin`, 1 Å by default) are counted and reported;
  lower `pair_rmin` for systems with shorter bonds.
