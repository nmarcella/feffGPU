# Tutorials

Notebooks that run on [Google Colab](https://colab.research.google.com), which provides NVIDIA GPUs, or on your own
computer.

| Notebook | What it shows | |
|---|---|---|
| [01_md_exafs_iron.ipynb](01_md_exafs_iron.ipynb) | MD-EXAFS step by step on a short synthetic trajectory of bcc iron: the configuration, the run, the averaged spectrum and its parts, the check that enough frames were used, the radial distribution, and the comparison with a spectrum | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nmarcella/feffGPU/blob/main/tutorials/01_md_exafs_iron.ipynb) |
| [02_md_exafs_your_trajectory.ipynb](02_md_exafs_your_trajectory.ipynb) | The same steps for a trajectory that you upload, with an optional comparison with your measured spectrum | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nmarcella/feffGPU/blob/main/tutorials/02_md_exafs_your_trajectory.ipynb) |
| [03_rmc_exafs_from_evax.ipynb](03_rmc_exafs_from_evax.ipynb) | RMC-EXAFS from a finished EvAX calculation of a copper foil: the parameter file translated keyword by keyword, three checks against the files EvAX wrote, a fit with `rmc-exafs refine` (the gradient refinement, the default engine), and the two fitted structures side by side. **Needs the files of the EvAX example**, which are not in this repository | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nmarcella/feffGPU/blob/main/tutorials/03_rmc_exafs_from_evax.ipynb) |

The two MD-EXAFS notebooks are stored with the outputs of a run on the development machine (a 12 GB GPU), so they
can be read here without running anything. The RMC-EXAFS notebook is stored without outputs: they would show the
measured spectrum of the EvAX example, which this repository does not hold.

## On Google Colab

1. Click a badge above.
2. Choose a GPU: menu *Runtime → Change runtime type → T4 GPU*. The free plan of Colab usually offers this GPU, but
   it does not guarantee one at every moment.
3. Run the cells from the top. The first cell downloads this repository (about 100 MB) and installs the package.

Colab deletes the files of a session when it ends. The last cell of each notebook downloads the results.

## On your own computer

```bash
git clone https://github.com/nmarcella/feffGPU.git
cd feffGPU
pip install -e ".[notebooks]"        # the package, and the tools that run notebooks
pip install jupyterlab               # or any other program that opens notebooks
cd tutorials
jupyter lab
```

The notebooks then use the checkout they sit in, and write into `tutorials/fe_tutorial`, `tutorials/my_run` and
`tutorials/cu_evax`.

## The data

Nothing in this folder was measured. The RMC-EXAFS notebook reads the files of an EvAX example from a folder that
you name (`in/` with `start.dat`, `Cu.p1` and `foils_not_aligned.chi`; `out/` with EvAX's output files).

- `data/Fe_bcc_300K.extxyz`: 60 frames of 128 iron atoms in a bcc block, 0.5 ps apart. The atoms are joined to
  their first and second neighbors by springs and moved with a thermostat at 300 K. The forces are not a model of
  iron; the file only has the features of a real trajectory.
- `data/Fe_synthetic_spectrum.chi`: the averaged spectrum of a second, independent and ten times longer trajectory
  of the same kind, multiplied by S0² = 0.80, shifted by ΔE0 = +2.0 eV, with a little noise. It stands in for a
  measured spectrum.

Both files are made by [`make_fe_trajectory.py`](make_fe_trajectory.py).

## For maintainers

`python tools/run_tutorials.py` runs every notebook in a temporary copy of this folder and stores it with its
outputs; paths of the machine are shortened to `...`. `tests/test_tutorials.py` checks the stored notebooks and, on
a GPU, runs them. The RMC-EXAFS notebook runs only when the environment variable `FEFFGPU_EVAX_EXAMPLE` names the
folder of the EvAX example, and it is never stored with outputs (`--out FOLDER` writes an executed copy elsewhere).
