# The FEFF pipeline on the GPU

The code is a stage-by-stage re-implementation of FEFF 8.5L's EXAFS path (`rdinp`, `pot`, `xsph`, `path`, `genfmt`,
`ff2chi`) in PyTorch. Each stage reproduces FEFF's intermediate files, which is how it was validated: `PotData`
against `pot.pad`, `PhaseData` against `phase.pad`, the path list against `paths.dat`, chi against `chi.dat`.

**Input (`inp.py`).** FEFF's card parser for the EXAFS cards, producing the same quantities `rdinp` writes (potentials,
atom list with potential indices, edge, exchange model, SCF settings, path cutoff and criteria, k range).

**Potentials (`pot.py`, `scf.py`, `fms.py`).** The free atoms come from a table of FEFF's own `ATOM` results (one
compressed file per element, every core hole), so the Dirac-Fock atomic code is not re-implemented; everything after
that is: the overlapped (Mattheiss) potentials and densities, the muffin-tin radii and interstitial, the Hedin-Lundqvist
self-energy, and FEFF's self-consistency loop, which needs the full multiple-scattering Green's function of the
cluster inside `r_scf` at every energy of the contour (`fms.py`, the heaviest part, batched on the GPU). Ions and
unfrozen f electrons call FEFF's atomic code through the optional library.

**Phase shifts (`xsph.py`, `kernels.py`).** FEFF's energy mesh, the radial Dirac equation per potential and partial
wave, phase shifts to FEFF's partial-wave limit (or higher, with the patched library for validation), and the dipole
matrix elements with the core-hole final state. On a GPU the sequential loops of the radial solver run as compiled
kernels; with a Triton version they were not released with, the package tests them against the plain code before it
uses them ([](../feff/limits.md)).

**Path finder (`pathfinder.py`).** All 2-, 3- and 4-leg paths of an absorber within the path cutoff, with FEFF's
plane-wave importance estimate, the `CRITERIA` filter relative to the shortest single-scattering path, and the merging
of symmetry-equivalent paths into degeneracies. Works in a periodic cell of any shape (minimum image) or in a
cluster, with the potential index of every atom.

**Path stage (`genfmt.py`, `stage.py`).** The Rehr-Albers separable representation of the propagators to second order
(`iorder` 2), rotation matrices between legs, the sum over angular momenta, for batches of paths at once and with
autograd through the coordinates. `chain.py` adds the paths up on FEFF's fine k grid; `stage.linear_matrix` gives the
interpolation to any other grid, with the energy shift applied in momentum.

**Precision and validation.** The potentials and phases are kept in float64 where FEFF is; the path stage runs in
complex64 by default (relative error ~1e-6 in chi, enough for any fit) and complex128 on request. The random-environment
validation ([](../validation/random_environments.md)) compares chi, path counts and degeneracies, the Fermi level and the
partial-wave limits with FEFF 8.5L for 34 environments drawn at random over elements, edges, structures and cards.
