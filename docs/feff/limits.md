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

## Phase files are pickles

`feffgpu phase` writes a Python pickle. Loading a pickle runs code: load only phase files that you made yourself or
that come from someone you trust. FEFF's `phase.pad` can be used instead and is a plain data file.
