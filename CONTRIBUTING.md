# Contributing

Contributions are welcome: bug reports, corrections, new features, documentation. This page says how to set up,
how the tests and the validation work, and what a change must satisfy.

## Setup

```bash
git clone https://github.com/nmarcella/feffGPU.git
cd feffGPU
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126     # the CUDA build that matches your driver
pip install -e ".[test,docs,mcp,larch]"      # larch: the tests that compare the conventions with xraylarch
feffgpu doctor
```

Without a GPU you can still work on everything that the tests marked "not gpu" cover: install the CPU build of
PyTorch (`--index-url https://download.pytorch.org/whl/cpu`).

## Tests

```bash
python -m pytest -m "not gpu"       # no GPU needed; this is what the hosted continuous integration runs
python -m pytest                    # with a GPU
```

Markers: `gpu` (needs CUDA), `oracle` (needs the optional FEFF 8.5L library, `fortran/README.md`), `slow`.
The hosted runners have no GPU. A pull request that touches numerical code must say in its description that the
GPU tests were run, and paste the last lines of the output.

## How validation works

There are two yardsticks, and no others:

1. **FEFF 8.5L itself.** The stages reproduce FEFF's intermediate files, and the whole chain is compared with FEFF
   on stored random environments (`tests/test_validation_cases.py`, `data/validation/`). These tests must stay
   green. If a change alters a number that FEFF defines, the change is wrong unless FEFF is.
2. **Synthetic targets with known answers.** A structure is built in code, its spectrum is made by the forward
   model, and the tool must recover what was planted.

Values from the literature are not used as a yardstick in tests or documentation.

When code is moved or rewritten, the same inputs and seed must give the same numbers as before. Say in the pull
request how you checked.

## Proposing a difference from FEFF or EvAX

`feffgpu` follows FEFF 8.5L, and the reverse Monte Carlo search follows EvAX as closely as possible. A deliberate
difference is allowed when it has a reason, a switch that restores the original rule where that is possible, a
test, and an entry in the documentation (`docs/feff/limits.md` for FEFF, `docs/rmc/evax_differences.md` for EvAX).
Open an issue first.

## Conventions of the code

- The conventions of the physics are in `docs/concepts/conventions.md` and are implemented once in `feffgpu.exafs`.
  MD-EXAFS and RMC-EXAFS share that layer, `feffgpu.io`, `feffgpu.analysis` and `feffgpu.parallel`. Put a tool that
  both could use into the shared layer, not into one of them.
- Every public function takes a `device`; nothing assumes `cuda:0`. Independent work items go through
  `feffgpu.parallel.run_items`, and their random numbers come from `feffgpu.parallel.keyed_rng`, so that results do
  not depend on the number of GPUs.
- NumPy-style docstrings on public functions and classes, with units. The API reference is generated from them.
- Configuration is a validated TOML file (pydantic models with a description for every field).
- Commands have `--json` where they print a result. Output files have a documented schema.
- Never fix a sample's coordination number as a result. Report an uncertainty with every fitted number and say how
  it was obtained.
- Plain English, American spelling, terms defined at first use.
- Line length up to 140. `ruff check` must pass.

## Data

Do not add measured data to the repository. Tests and examples build their structures in code and use the phase
files that are already there. If an example needs new data, open an issue first.

## Documentation

```bash
pip install -e ".[docs]"
sphinx-build -W -b html docs docs/_build/html
```

The build must pass with warnings as errors. Write the documentation of a feature in the same pull request as the
feature. Tutorials are not executed during the build: they show stored output, and the page says which command
regenerates it.

## Pull requests

- One topic per pull request, with tests.
- The description says what changed, why, and how it was tested (with the GPU or without).
- By contributing you agree that your contribution is released under the MIT License of this project.

## Versions and releases

The version number is not written in any file: it comes from git when the package is built or installed
(`[tool.setuptools_scm]` in `pyproject.toml`). At a release tag `v0.3.0` the version is `0.3.0`; seven commits later
it is `0.3.1.dev7+g<commit>`. So every merged pull request has its own version, and
`pip install --upgrade git+https://github.com/nmarcella/feffGPU.git` installs it.

A release, by a maintainer:

1. Give the section of `CHANGELOG.md` its version and date, and set `version` and `date-released` in `CITATION.cff`.
2. On GitHub, publish a release with a new tag `vX.Y.Z` on `main`. The workflow `release.yml` builds the package,
   checks that the built version is the tag, and uploads it to PyPI through PyPI's trusted publishing (no token).
3. Read the Docs builds the documentation of every push by itself.

An upload to PyPI cannot be replaced: a mistake needs a new version number.

## Agents

If you work with a coding agent, point it at `AGENTS.md`.
