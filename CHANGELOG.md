# Changelog

All notable changes to this project are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/),
and the versions follow [Semantic Versioning](https://semver.org/).

## 0.2.0 (unreleased)

The first public version: one package with three tools.

### Added

- **MD-EXAFS** (`feffgpu.md`, command `md-exafs`): the EXAFS spectrum of a molecular dynamics trajectory averaged
  over frames and absorbing atoms, from one TOML file; any trajectory format that ASE reads, no OVITO; decomposition
  by path type; an HDF5 cache; shards for several GPUs and nodes; comparison with a measured spectrum; radial
  distribution, peak-fit and convergence tools.
- **RMC-EXAFS** (`feffgpu.rmc`, command `rmc-exafs`): reverse Monte Carlo fitting by a population search that
  follows EvAX's rules and by gradient refinement, from one TOML file; several samples and edges in one fit with
  S0² and the E0 shift shared per edge; constraints; analysis tools.
- **Shared layer**: `feffgpu.exafs` (xraylarch's conventions for the window, the Fourier transform and the
  R-factor; reading measured spectra; the forward model of whole configurations; `SpectrumComparison`; the joint fit
  of S0² and the E0 shift with intervals), `feffgpu.io` (structures and trajectories), `feffgpu.analysis` (pair
  distributions, shell tables, pairs at impossible distances, bond angles), `feffgpu.parallel` (devices, shards,
  keyed random streams: the access points for several GPUs), `feffgpu.config_tools` (validated TOML configurations),
  `feffgpu.load_phase` (a phase file from a `phase.pad` or a pickle).
- **One command** `feffgpu` with the subcommands `run`, `phase`, `doctor`, `md`, `rmc`, `mcp`.
  `feffgpu doctor` checks the installation. `--json` output on the commands that print results.
- **MCP server** (`feffgpu-mcp`, extra `feffgpu[mcp]`): tools for agents to validate configurations, start runs as
  jobs, follow them, and read results and documentation, inside one workspace directory.
- **Documentation** for Read the Docs, with the API reference, and `llms.txt` / `llms-full.txt` for agents.
- `AGENTS.md`, `CONTRIBUTING.md`, a code of conduct, continuous integration for the tests that need no GPU.
- **Tutorials** (`tutorials/`): notebooks for Google Colab or a local Jupyter. MD-EXAFS step by step on a short
  synthetic trajectory of bcc iron; MD-EXAFS for a trajectory that you upload; RMC-EXAFS from a finished EvAX
  calculation (translation of the parameter file, checks against EvAX's output files, a fit and the comparison of
  the fitted structures; it needs the files of the EvAX example, which the repository does not hold).
  `tools/run_tutorials.py` runs the notebooks and stores their outputs; the extra `feffgpu[notebooks]` installs what
  that needs.
- How to translate an EvAX parameter file, in "Differences from EvAX".
- A figure of the agreement with FEFF 8.5L for every environment of the validation, by element and edge
  (`docs/validation/agreement_by_element_and_edge.png`, drawn by `tools/plot_validation_agreement.py` from
  `data/validation/agreement.npz`), and tests of the window, the Fourier transform and the R-factor against xraylarch
  itself (`tests/test_larch_conventions.py`).
- **Validation element by element and edge by edge**: every element from hydrogen to einsteinium with each of its K,
  L3 and M5 edges (267 cases) through FEFF 8.5L and through `feffgpu` (`tools/validate_elements.py`,
  `tools/plot_validation_elements.py`, `data/validation/elements.npz`, the documentation page "Every element, edge by
  edge", and the figure on the main page). Median R-factor between the two codes 1.7e-6; 264 of 267 below 2e-5.
- **A test of the compiled GPU kernels against the plain PyTorch code.** The kernels are released with the list of
  Triton versions they were tested with (3.1.0 and 3.8.0). With one of those nothing is tested at run time; with any
  other version the kernels are compared with the plain code on a small built-in problem once per process (about a
  quarter of a second) before they are used, and `feffgpu doctor` runs the test on request. If the kernels differ, the
  calculation stops with an error; there is no automatic change to slower code. `FEFFGPU_KERNELS=0` selects the plain
  code on purpose. `tests/test_kernels.py` and the stored validation cases now also test the Fermi level against
  FEFF's.

### Changed

- The free-atom table ships inside the package (81 MB, packed without loss; the arrays are identical byte for byte to
  the table of version 0.1.0). Nothing has to be downloaded after `pip install`, and the commands `feffgpu fetch-atoms`
  and `feffgpu-fetch-atoms` are gone. `FEFFGPU_ATOMS` still points the package at another table.
- Python 3.11 or newer. New dependencies: `ase`, `h5py`, `matplotlib`, `pydantic`.
- `feffgpu.compare.r_factor` takes the sill width `dk` of the window (default 1 1/Å, as before).
- User errors of the commands end with one message that says what to do.

### Fixed

- The potentials stage could fail with an out-of-memory error in a process that had used the GPU before and whose
  freed memory was still held in PyTorch's cache, because parts of the calculation take GPU memory outside that cache.
  The cache is now released before the potentials are computed.
- In some PyTorch builds a batched linear solve on the CPU fails after `torch.set_num_threads()` was called before the
  first linear-algebra call of the process. The potentials stage now falls back to solving one matrix at a time and
  checks the result.
- A population search with EvAX's wavelet misfit (`objective = "wavelet"`) stopped at its first evaluation with an
  error about number types. The wavelet misfit is now returned in double precision, like the R-factor.
- `feffgpu.compare.hanning` and `r_factor` returned "not a number" for a window without sills (`dk = 0`, EvAX's
  default window). They now give the rectangular window, as the torch form does.

- **The outward GPU kernel of the radial solver gave different numbers under Triton 3.8** (the kernel compiler that
  comes with PyTorch 2.14) than under Triton 3.1: the newer compiler handles a group of loop variables that start as
  the same value and are shifted inside the loop differently. chi(k) still agreed with FEFF 8.5L to an R-factor of
  about 2e-6 in the median (9e-4 at most over all elements), but the Fermi level of self-consistent potentials was off
  by about 3e-5 hartree (up to 8e-4). The kernel no longer uses that pattern and agrees with the plain code to
  rounding under both compilers. The sweep over all elements found it.

### Notes

- A known difference from FEFF 8.5L at one partial wave is documented under "Precision and limits".
- Several GPUs in one run are not implemented yet; MD-EXAFS can be split over GPUs and nodes with shards.

## 0.1.0 (2026-10-12)

- FEFF 8.5L's EXAFS pipeline on the GPU in PyTorch: the input reader, the potentials with the self-consistency
  loop, the phase shifts, the path finder up to four legs, the path stage, FEFF's binary files, the commands
  `feffgpu`, `feffgpu-phase` and `feffgpu-fetch-atoms`; validated against FEFF 8.5L on random environments.
