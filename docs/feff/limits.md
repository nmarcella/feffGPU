# Precision and limits

## What is implemented

| FEFF module | In `feffgpu` | Notes |
|---|---|---|
| `rdinp` | `feffgpu.inp.read_feff_inp` | the EXAFS cards of FEFF 8.5L: TITLE, EDGE / HOLE, S02, CONTROL, PRINT, SCF, EXCHANGE, EXAFS, RPATH, NLEG, CRITERIA, POTENTIALS, ATOMS, FOLP / AFOLP, INTERSTITIAL, JUMPRM, NOHOLE, RGRID, ION, UNFREEZEF and related cards |
| `pot` (with `ATOM`) | `feffgpu.pot`, `feffgpu.scf`, `feffgpu.fms` | free atoms from the table; overlapped potentials, Hedin-Lundqvist exchange, the self-consistency loop with its full multiple-scattering Green's function; the result equals FEFF's `pot.pad` |
| `xsph` | `feffgpu.xsph`, `feffgpu.kernels` | energy mesh, self-energy, phase shifts and dipole matrix elements per potential; the result equals FEFF's `phase.pad` |
| `path` | `feffgpu.pathfinder` | two-, three- and four-leg paths within RPATH with FEFF's importance filter and degeneracy merging; any periodic cell or none |
| `genfmt` and `ff2chi` | `feffgpu.genfmt`, `feffgpu.stage`, `feffgpu.chain` | the signals of batches of paths (rotation matrices, Rehr-Albers separable propagators to second order) and their sum |

## Limits

- **EXAFS only.** XANES and full multiple-scattering spectra, polarization and ellipticity, and spin-dependent
  potentials are not ported. A card for them raises an error that names the card.
- **Edges.** K, L and M edges.
- **Paths of two, three and four legs.** `NLEG` above 4 is treated as 4 ([](../concepts/paths.md)).
- **No Debye-Waller models.** FEFF's `SIG2`, `DEBYE` and `CORRELATIONS` are not ported. Disorder comes from atom
  positions.
- **Ions and unfrozen f electrons** (`ION`, `UNFREEZEF`) need FEFF's own atomic code through the optional library
  ([](../installation.md)).
- **Periodic cells** must be at least twice as wide as the path cutoff (minimum-image convention).
- **NVIDIA GPUs.** The code runs on CUDA devices. It runs on a CPU for small inputs, slowly. Several GPUs: see
  [](../parallel.md).

## Precision

- The potentials and phase shifts are computed in float64 where FEFF uses double precision.
- On a GPU the two sequential loops of the radial solver run as compiled kernels (`feffgpu.kernels`, written with
  Triton, the kernel compiler that PyTorch installs). They repeat the plain PyTorch code operation by operation and
  agree with it to rounding (about 1e-14). A kernel is compiled by whatever Triton version is installed, and a new
  compiler can change what a kernel computes (see "Known differences" below). The package therefore carries a list of
  the Triton versions its kernels were tested with before the release (3.1.0 and 3.8.0 for this version):
  - with one of those versions the kernels are used as they are, and nothing is tested at run time;
  - with any other version the kernels are compared with the plain code on a small built-in problem before they are
    used, once per process (about a quarter of a second);
  - `feffgpu doctor` runs that test on request, whatever the version.

  If the kernels do not reproduce the plain code, the calculation stops with an error that says so. The package does
  not change to the slower plain code on its own; `FEFFGPU_KERNELS=0` selects the plain code on purpose (same
  results, potentials and phase shifts take longer).
- The path stage runs in complex64 by default, which gives chi to a relative error of about 1e-6, and in
  complex128 on request (`GPUPathStage(phase, dtype=torch.complex128)`; `feffgpu.run` uses complex128).
- The single scattering of `feffgpu.exafs.PairTables` interpolates the exact two-leg signal on a distance grid of
  0.002 Å. A test compares it with the path stage for every absorber (agreement better than 0.2 % of the largest
  signal in the k range tested).

## Known differences from FEFF 8.5L

- **One partial wave at an integer boundary.** FEFF's number of partial waves at each energy is the integer part
  of a product with k. Where that product lands on an integer, FEFF and `feffgpu` can round to different sides, and
  FEFF then keeps one more partial wave at that energy. In the tungsten test set this happens at one energy
  (k = 32 1/Å, the partial wave l = 47, a scattering amplitude of 1.3e-3). The effect on chi(k) is far below the
  agreement reported in [](../validation/index.md).

- **A kernel compiler that changed a result (fixed in 0.2.0).** Triton 3.8, which comes with PyTorch 2.14, compiles
  one loop pattern differently from Triton 3.1: a group of variables that start as the same value and are shifted
  inside a loop keep their start value. The outward kernel of version 0.1.0 used that pattern in its start-up steps.
  With the newer compiler its wave functions were off by a few percent from the sixth radial point on. Most of that
  cancels in the phase shifts: in the sweep over all elements ([](../validation/elements.md)) chi(k) still agreed with
  FEFF 8.5L to an R-factor of 1.9e-6 in the median (9e-4 at most). The Fermi level of self-consistent potentials
  moved by about 3e-5 hartree (up to 8e-4), and in at least one case (palladium K) the self-consistency loop then
  stopped at another iteration than FEFF's. Version 0.2.0 does not use the pattern, tests the kernels as described
  above, and tests the Fermi level against FEFF's on the GPU. With the older compiler version 0.1.0 was not affected.

## Phase files are pickles

`feffgpu phase` writes a Python pickle. Loading a pickle runs code: load only phase files that you made yourself or
that come from someone you trust. FEFF's `phase.pad` can be used instead and is a plain data file.
