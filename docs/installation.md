# Installation

## Requirements

- Linux.
- Python 3.11 or newer.
- An NVIDIA GPU and its driver. `feffgpu` was developed on a 12 GB card; the memory needed grows with the number of
  atoms inside the self-consistency radius and with the number of paths per batch.
- A CUDA build of PyTorch 2.1 or newer that matches the driver.

## Install

Install PyTorch first, so that you get the CUDA build that fits your driver, then `feffgpu`:

```bash
python -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126     # choose the CUDA version of your driver
pip install feffgpu
```

The table at <https://pytorch.org/get-started/locally/> lists the index URL for each CUDA version. `nvidia-smi`
shows the driver version and the highest CUDA version it supports.

From a source checkout, for development:

```bash
git clone https://github.com/nmarcella/feffGPU.git
cd feffGPU
pip install -e ".[test]"
```

Optional parts are installed as extras:

| Extra | Adds | Needed for |
|---|---|---|
| `feffgpu[mace]` | MACE potentials | running new molecular dynamics with `md-exafs dynamics` |
| `feffgpu[mda]` | MDAnalysis | reading large binary trajectory formats |
| `feffgpu[larch]` | xraylarch | cross-checks against xraylarch, data reduction |
| `feffgpu[mcp]` | the MCP SDK | the server for agents, [](agents.md) |
| `feffgpu[docs]` | Sphinx and its extensions | building this documentation |
| `feffgpu[test]` | pytest | running the tests |

## The free-atom table

The potentials stage needs FEFF's atomic results for every element (Z = 1 to 99) as a neighbor and as an absorber
with each core hole. They are stored in a table of 100 MB that is too large for the package itself. Download it
once:

```bash
feffgpu fetch-atoms
```

The download is checked against a SHA-256 checksum and unpacked into the installed package. To keep it elsewhere,
for example on a shared file system:

```bash
feffgpu fetch-atoms /data/feffgpu_atoms
export FEFFGPU_ATOMS=/data/feffgpu_atoms
```

On a machine without network access, download
`https://github.com/nmarcella/feffGPU/releases/download/atoms-v1/feffgpu_atoms_v1.tar.gz` elsewhere, copy it, and run

```bash
feffgpu fetch-atoms --from-file feffgpu_atoms_v1.tar.gz
```

The table is needed to compute potentials and phase files. It is not needed to run MD-EXAFS or RMC-EXAFS with a
phase file that already exists.

## Check the installation

```bash
feffgpu doctor
```

prints the versions, the GPUs with their free memory, the state of the free-atom table, and the optional parts. It
ends with `OK` or with a list of problems and exits with code 1 if something required is missing. `feffgpu doctor
--json` gives the same as JSON.

## The optional FEFF 8.5L library

`feffgpu` needs FEFF's own Fortran code for three things only:

- inputs with `ION` or `UNFREEZEF` cards (ions and unfrozen f electrons are not in the free-atom table);
- rebuilding the free-atom table (`tools/build_atoms_table.py`);
- the tests that compare intermediate quantities with FEFF (`pytest -m oracle`).

`fortran/README.md` in the repository says how to build the library from the open-source `feff85exafs` release with
a patch that raises FEFF's partial-wave limit. Point `FEFF_HOME` (a directory with `lib/`) or `FEFFGPU_POTPH_LIB`
(the file `libfeff8lpotph.so`) at it.

## Environment variables

| Variable | Meaning |
|---|---|
| `FEFFGPU_ATOMS` | directory of the free-atom table (default: inside the package) |
| `FEFFGPU_POTPH_LIB` | path to `libfeff8lpotph.so` (optional) |
| `FEFF_HOME` | a FEFF 8.5L build directory with `bin/` and `lib/` (optional) |
| `FEFF_LMAX` | force a partial-wave limit instead of FEFF's rule |
| `CUDA_VISIBLE_DEVICES` | the GPUs that PyTorch, and so `feffgpu`, can see |
| `PYTORCH_CUDA_ALLOC_CONF` | set to `expandable_segments:True` for large runs to reduce memory fragmentation |
| `FEFFGPU_DOCS` | folder of the documentation sources, for the documentation tools of the MCP server |

## Troubleshooting

`this PyTorch is a CPU-only build`
: `pip` installed the CPU wheel. Install the CUDA build with the index URL above.

`PyTorch was built for CUDA ... but no NVIDIA GPU is usable`
: Check `nvidia-smi`. Check that `CUDA_VISIBLE_DEVICES` is not empty. A driver older than the CUDA build also gives
  this message: install the PyTorch build for an older CUDA version.

`the free-atom table is not installed`
: Run `feffgpu fetch-atoms`, or set `FEFFGPU_ATOMS`.

`card ... is not supported by the GPU FEFF`
: The input uses a FEFF card outside the EXAFS calculation (for example `XANES`, `POLARIZATION`, `SPIN`). See
  [](feff/limits.md).

`CUDA out of memory`
: The self-consistency loop of the potentials needs memory that grows with the number of atoms inside the
  self-consistency radius: reduce the `SCF` radius or the cluster. In MD-EXAFS and RMC-EXAFS reduce the batch size
  (see their configuration pages) and set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

`Intel oneMKL ERROR: Parameter 6 was incorrect on entry to ZLASWP`, or `Pivots given to lu_solve must all be greater or equal to 1`
: A problem of some PyTorch builds, not of your input: after `torch.set_num_threads()` was called before the first
  linear-algebra call of a process, batched linear solves on the CPU fail. `feffgpu` then solves one matrix at a time
  and continues (the first message may still be printed). To avoid it in your own code, set the number of threads
  with the environment variable `OMP_NUM_THREADS` before Python starts and do not call `torch.set_num_threads()`.

`the periodic cell ... must exceed twice the path cutoff`
: Paths are found with the minimum-image convention, which needs a cell at least twice as wide as the path cutoff.
  Use a larger supercell or a smaller cutoff.
