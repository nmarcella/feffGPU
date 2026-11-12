# Validation

Validation in this project means two things only: agreement with FEFF 8.5L itself, and recovery of known answers
from synthetic targets. Values from the literature are not used as a yardstick.

## Against FEFF 8.5L

```{figure} agreement_by_element_and_edge.png
:alt: The R-factor between FEFF 8.5L and feffgpu per element and edge, and the two spectra of every environment drawn over each other

FEFF 8.5L (black) and feffgpu (blue, on top) for the 34 random environments of the validation, one panel per absorbing
element and edge; above, the R-factor between the two codes. The spectra are stored in `data/validation/agreement.npz`
and `tools/plot_validation_agreement.py` redraws the figure.
```

**Random environments.** The whole chain (potentials, phase shifts, paths, chi) was compared with FEFF 8.5L on
environments drawn at random over elements (absorbers from C to Au, Z = 6 to 79; neighbors up to Z = 80), edges (K, L2,
L3, M5), structures (lattices and random
packings, with disorder, vacancies, strain and surfaces) and input cards. The median R-factor between the two codes is 1.3e-6; 33 of 34 are below
2e-5; the one above (2.4e-3) is a random packing with an `ION` card. The number of paths, weighted by degeneracy, is
identical in 29 of the 34; in the other 5 the codes differ by one path at the importance threshold. The Fermi levels
agree to 2e-6 hartree or better. The table is on the next page, [](random_environments.md). The tool that made the comparison, `tools/validate_random.py`, runs FEFF's own modules on
every case and compares; it needs a FEFF 8.5L build ([](../installation.md)).

**Regression cases.** Four of those environments are stored in `data/validation/` with FEFF's own outputs. The test
`tests/test_validation_cases.py` runs the whole chain on each and requires an R-factor below 2e-5 in the analysis
window and the same degeneracy-weighted path counts as FEFF. They cover symmetry-degenerate paths on an ideal
lattice, ground-state exchange, an ionized absorber and an M5 edge.

**Stage by stage.** With the optional FEFF library, tests compare intermediate quantities with FEFF's own routines:
the free-atom orbitals and overlapped potentials (`tests/test_pot.py`), the energy mesh, the self-energy, the phase
shifts and the dipole matrix elements (`tests/test_xsph.py`), and the path list (`tests/test_pathfinder.py`). One
known difference is described in [](../feff/limits.md).

## The forward model of configurations

`tests/test_shared_layer.py`:

- the single scattering from pair tables equals the two-leg paths of every absorber through the path finder and
  the path stage, within the interpolation error of the table's 0.002 Å grid;
- the multiple-scattering spectra per absorber agree between a full pass and a subset evaluation, with several
  elements, an absorber subset and a sheared cell;
- the two forms of the R-factor agree, and the Hanning window has xraylarch's definition;
- a planted S0² and E0 shift are recovered by the joint fit, and the reported intervals contain them.

## MD-EXAFS

The tests in `tests/md/` compare the spectrum of a frame with an independent calculation through the path finder
and path stage, check that the decomposition by path type sums to the total, that a trajectory of Gaussian
displacements reproduces the analytic damping within the sampling error, and that a run split into shards merges
to the same result as a single run. See [](../md/quickstart.md) and the pages after it.

## RMC-EXAFS

The tests in `tests/rmc/` check each rule of the search against its definition, recover a planted disorder from a
synthetic target made by the forward model, and check that the library reproduces the reference implementation it
was moved from, with the same random seed. See [](../rmc/quickstart.md) and the pages after it.

## What has not been measured yet

- The effect of paths with more than four legs on a spectrum.
- The accuracy of one reference phase file for all frames of a trajectory against potentials computed for every
  absorber ([](../concepts/potentials.md)).
- Scaling with the number of GPUs ([](../parallel.md)).

These are stated here so that nobody takes them as settled.

## Running the tests

```bash
pip install -e ".[test]"
python -m pytest -m "not gpu"            # without a GPU: conventions, analysis, configuration, the agent interface
python -m pytest                         # with a GPU: everything except the oracle tests
FEFFGPU_POTPH_LIB=/path/to/libfeff8lpotph.so FEFF_HOME=/path/to/feff85exafs/local_install python -m pytest   # with the FEFF library too
```

## Record of the full run for version 0.2.0

Every test, with a GPU, the FEFF 8.5L library and xraylarch, in a fresh environment (Python 3.12, PyTorch 2.14 with
CUDA 12.6, xraylarch 2026.3, one 12 GB NVIDIA GPU), on 2026-11-12: 168 passed, none skipped, none failed, in 8 minutes.

Tests carry the markers `gpu` (needs CUDA), `oracle` (needs the FEFF library) and `slow`. The hosted continuous
integration has no GPU and runs the first line; the GPU and oracle tests are run by hand before each release and
their log is kept with the release notes.
