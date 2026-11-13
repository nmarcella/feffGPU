# AGENTS.md: working on feffgpu with a coding agent

This file is for agents (and people) who change the code. To *use* the software from an agent, see
`docs/agents.md` (the MCP server).

## What this repository is

`feffgpu` is FEFF 8.5L's EXAFS calculation in PyTorch for NVIDIA GPUs, with MD-EXAFS (`feffgpu.md`) and RMC-EXAFS
(`feffgpu.rmc`) built on it. EXAFS = extended X-ray absorption fine structure; chi(k) = the EXAFS signal against
the photoelectron wave number k; FEFF = the program that calculates chi(k) from atom positions; RMC = reverse Monte
Carlo; EvAX = the established RMC-EXAFS program whose rules the search follows.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126    # or .../whl/cpu on a machine without a GPU
pip install -e ".[test,docs,mcp,larch]"
feffgpu doctor               # what is installed, which GPUs, what is missing
```

## Tests

```bash
python -m pytest -m "not gpu" -q        # no GPU: conventions, analysis, configuration, agent interface (minutes)
python -m pytest -q                     # with a GPU: everything except the oracle tests
python -m pytest tests/rmc -q           # one tool
ruff check src tests
sphinx-build -W -b html docs docs/_build/html
```

Markers: `gpu`, `oracle` (needs the FEFF 8.5L library, `fortran/README.md`), `slow`. One GPU job at a time on a
shared machine. Do not start long fits or benchmarks as tests.

## Rules that are not negotiable

1. **Validation against FEFF 8.5L stays green** (`tests/test_validation_cases.py`, `tests/test_pot.py`,
   `tests/test_xsph.py`, `tests/test_pathfinder.py`). A change that alters a number FEFF defines is wrong.
2. **A move or rewrite does not change numbers.** Same inputs and seed, same result. Prove it with a test.
3. **Two yardsticks only**: FEFF 8.5L, and synthetic targets with known answers. No literature values in tests or
   documentation.
4. **No measured data in the repository.** Tests and examples build structures in code and use the phase files in
   `tests/data/W/` and `data/validation/`.
5. **Shared tools live once.** Anything MD-EXAFS and RMC-EXAFS both need goes in `feffgpu.exafs`, `feffgpu.io`,
   `feffgpu.analysis` or `feffgpu.parallel`.
6. **Conventions are xraylarch's** (`docs/concepts/conventions.md`): Hanning window by `kmin`, `kmax`, `dk`; R-factor
   on the R window; S0² and the E0 shift on the model, one value per edge, shared between samples.
7. **Never fix a sample's coordination number as a result.** Every fitted number gets an uncertainty and a sentence
   on how it was obtained.
8. **Every public function takes `device`.** Independent work items go through `feffgpu.parallel.run_items`; their
   random numbers come from `feffgpu.parallel.keyed_rng`. Results must not depend on the number of GPUs. (The RMC
   search still draws from one generator per sample in a fixed order; moving it to keyed streams changes every
   number of a run and is a separate, announced step. Do not do it in passing.)
9. **Do not modify FEFF's Fortran** (the patch in `fortran/`) without asking the maintainers.
10. Plain English, American spelling, terms defined at first use, NumPy-style docstrings with units.
11. **Never call `torch.set_num_threads()`** in the package or in a test. In some PyTorch builds it breaks batched
    linear solves on the CPU for the rest of the process (`tests/test_fms_solve.py` shows it in a fresh process).
12. **A compiled kernel is only as right as its compiler.** `src/feffgpu/kernels.py` repeats the plain code of
    `xsph.py` operation by operation, and `feffgpu.xsph.kernel_difference` must stay at rounding (about 1e-14).
    `kernels.TESTED_TRITON` lists the Triton versions for which that was tested; with any other version the package
    runs the test itself before it uses the kernels (`kernels_usable`). Add a version to the list only after
    `tests/test_kernels.py` passed with it on a GPU. Do not add an automatic change to slower code: a failed test
    stops the calculation. In a kernel, never fill a history by shifting variables that start as the same value
    (`d0 = d1; d1 = d2; ...` after `d0 = d1 = d2 = x`): Triton 3.8 compiles that wrongly. Change the plain code and
    the kernel together.

## Map of the package

```
src/feffgpu/
  inp.py          FEFF input cards -> parameters                     (FEFF's rdinp)
  pot.py scf.py fms.py   free atoms from the table, overlapped potentials, self-consistency loop   (FEFF's pot)
  xsph.py kernels.py     energy mesh, self-energy, phase shifts, matrix elements                    (FEFF's xsph)
  pathfinder.py   paths of one absorber, importance filter, degeneracies (FEFF's path)
  genfmt.py       the path signal, literal port, the reference       (FEFF's genfmt)
  stage.py        the path signal batched on the GPU, differentiable (what everything else calls)
  chain.py        feff.inp -> chi(k): `feffgpu.run`
  pad.py          FEFF's pot.pad / phase.pad readers; PhaseData, PotData; load_phase (a phase.pad or a pickle)
  cell.py         periodic cells and the minimum image (torch)
  compare.py      R-factor in numpy
  atoms_table.py  the free-atom table that ships in data/atoms (one packed file per element), its loader and packer
  potph_lib.py onepath.py   bindings to the optional FEFF 8.5L library (oracle tests, ions)
  cli.py          the `feffgpu` command and its subcommands
  config_tools.py SHARED by the two tools' TOML configurations: base model, the window model, error messages
                  with the path of the field, reading and writing TOML
  exafs/          SHARED: transform.py (grid, window, FT, R-factor, FitWindow), data.py (measured chi(k)),
                  forward.py (PairTables, MultipleScattering, PathSpectra), comparison.py (SpectrumComparison),
                  amplitude.py (joint fit of S0² and the E0 shift), edge.py (one absorption edge of a structure),
                  wavelet.py (EvAX's wavelet residual, optional), plots.py
  io/             SHARED: structure.py, trajectory.py, evax.py, lattice.py
  analysis/       SHARED: pairs.py (distances, shell tables, short pairs, angles), rdf.py, peaks.py,
                  convergence.py, plots.py
  parallel/       SHARED: devices, shards, keyed random streams, run_items (the access points for several GPUs)
  md/             MD-EXAFS: configuration, frames -> spectra, HDF5 cache, shards and merge, comparison, commands
  rmc/            RMC-EXAFS: configuration, model structure, constraints, moves, the EvAX-rule search, gradient
                  refinement, outputs, commands
  mcp/            the MCP server (server.py: tools as plain methods; jobs.py: background jobs in a workspace)
tests/            test_*.py (core, shared layer, agent interface), tests/md, tests/rmc
docs/             Sphinx + MyST sources (Read the Docs); docs/md and docs/rmc are the tools' user guides
examples/         runnable examples that generate their own synthetic inputs
tools/            build_atoms_table.py, validate_random.py (need the FEFF 8.5L library)
fortran/          the patch for feff85exafs and how to build the optional library
data/validation/  four random environments with FEFF's own outputs (regression cases)
```

## Where to change what

| Task | Start here |
|---|---|
| a FEFF card is missing | `inp.py` (parser), then the stage that uses it; add a validation case |
| the path signal | `stage.py`; `genfmt.py` is the reference it must match |
| a new misfit or window option | `exafs/transform.py`, `exafs/comparison.py`; document in `docs/concepts/conventions.md` |
| a new trajectory format | `io/trajectory.py` |
| a new analysis of structures | `analysis/` (plain arrays in, results out), then a subcommand in `md/cli.py` or `rmc/cli.py` |
| a new move or constraint of the RMC search | `rmc/moves.py`, `rmc/constraints.py`; if it differs from EvAX, add it to `docs/rmc/evax_differences.md` |
| a configuration key | the pydantic model in `md/config.py` or `rmc/config.py` (shared pieces in `config_tools.py`), with a description; regenerate the reference page (`docs/md/_make_config_reference.py`, `docs/rmc/_make_config_reference.py`) |
| a new MCP tool | a method of `Tools` in `mcp/server.py` and its name in `TOOL_NAMES`; a test in `tests/test_mcp.py` |
| several GPUs | `parallel/__init__.py` (`run_items`); keep the equality test one GPU against several |

## Machine-readable entry points

- `feffgpu doctor --json`, `md-exafs schema`, `rmc-exafs schema`, `feffgpu-mcp --list-tools`.
- Output schemas: `docs/md/outputs.md`, `docs/rmc/outputs.md`.
- The built documentation publishes `llms.txt` and `llms-full.txt`.

## Before you finish a change

- `ruff check src tests` and `python -m pytest -m "not gpu" -q` pass.
- If numerical code changed: the GPU tests were run and their last lines are in the pull request.
- Documentation updated in the same change; `sphinx-build -W` passes.
- `CHANGELOG.md` has an entry.
