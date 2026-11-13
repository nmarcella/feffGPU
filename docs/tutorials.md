# Tutorials

The tutorials are notebooks in the folder [`tutorials/`](https://github.com/nmarcella/feffGPU/tree/main/tutorials)
of the repository. They run on [Google Colab](https://colab.research.google.com), which provides NVIDIA GPUs, or on
your own computer. The MD-EXAFS notebooks are stored with the outputs of a run, so they can also be read without
running anything.

| Notebook | What it shows | |
|---|---|---|
| [MD-EXAFS from a trajectory: a worked example with iron](https://github.com/nmarcella/feffGPU/blob/main/tutorials/01_md_exafs_iron.ipynb) | Every step on a short synthetic trajectory of bcc iron: the configuration, the run, the averaged spectrum and its parts, the check that enough frames were used, the radial distribution, the comparison with a spectrum | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nmarcella/feffGPU/blob/main/tutorials/01_md_exafs_iron.ipynb) |
| [MD-EXAFS with your own trajectory](https://github.com/nmarcella/feffGPU/blob/main/tutorials/02_md_exafs_your_trajectory.ipynb) | The same steps for a trajectory that you upload, with an optional comparison with your measured spectrum | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nmarcella/feffGPU/blob/main/tutorials/02_md_exafs_your_trajectory.ipynb) |
| [RMC-EXAFS from an EvAX project: copper foil](https://github.com/nmarcella/feffGPU/blob/main/tutorials/03_rmc_exafs_from_evax.ipynb) | A finished EvAX calculation taken over: the parameter file translated keyword by keyword ([](rmc/evax_differences.md)), three checks against the files EvAX wrote, a fit with `rmc-exafs refine` (the gradient refinement, the default engine), and the two fitted structures side by side. It needs the files of the EvAX example, which the repository does not hold, and it is stored without outputs | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nmarcella/feffGPU/blob/main/tutorials/03_rmc_exafs_from_evax.ipynb) |

## On Google Colab

1. Click a badge.
2. Choose a GPU: menu *Runtime → Change runtime type → T4 GPU*. The free plan of Colab usually offers this GPU, but
   it does not guarantee one at every moment.
3. Run the cells from the top. The first cell downloads the repository (about 100 MB) and installs the package;
   `feffgpu doctor` in the second cell reports whether the GPU is seen.

Colab deletes the files of a session when it ends; the last cell of each notebook downloads the results.

## On your own computer

```bash
git clone https://github.com/nmarcella/feffGPU.git
cd feffGPU
pip install -e ".[notebooks]"        # the package, and the tools that run notebooks
pip install jupyterlab               # or any other program that opens notebooks
cd tutorials
jupyter lab
```

## The data of the tutorials

Nothing in the repository's tutorial data was measured. The iron trajectory (60 frames of 128 atoms) comes from springs between
neighbors and a thermostat at 300 K; the forces are not a model of iron. The spectrum it is compared with is the
averaged spectrum of a second trajectory of the same kind with a known S0² and E0 shift put in. Both files are made
by `tutorials/make_fe_trajectory.py`.

The RMC-EXAFS notebook reads the input and output files of an EvAX calculation from a folder that you provide.

The command-line versions of the same material are [](md/quickstart.md), [](rmc/quickstart.md) and the folder
`examples/`.
