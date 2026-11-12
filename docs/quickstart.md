# feffgpu in 5 minutes

This page takes a FEFF input file to chi(k) on the GPU. You need the installation of [](installation.md).

## From the command line

Any `feff.inp` written for an EXAFS calculation with FEFF 8 works. The repository has one for body-centered cubic
tungsten (K edge, self-consistent potentials, paths to four legs):

```bash
feffgpu tests/data/W/feff.inp --out chi.dat
```

```text
wrote chi.dat; stages (s): input 0.0, potentials 14.9, phases 3.7, paths 0.1, chi 0.4; total 19.1
```

(The times are from one run on a 12 GB RTX 3080 Ti; most of the time is the self-consistency loop of the potentials.)
`chi.dat` has two columns, k (1/Å) and chi, with FEFF's convention for `chi.dat`: the amplitude factor S0² is 1 and there is no
Debye-Waller factor. The header gives the number of paths per number of legs, the Fermi level and the muffin-tin
zero.

## From Python

```python
from feffgpu import run

out = run("tests/data/W/feff.inp", device="cuda")
k, chi = out["k"], out["chi"]            # FEFF's grid, 0.05 to 30 1/Å
print(out["counts"])                     # paths per number of legs
print(out["timing"])                     # seconds per stage
phase = out["phase"]                     # the phase file: potentials and phase shifts
```

`out["pot"]` and `out["phase"]` equal FEFF's `pot.pad` and `phase.pad`; `out["kept"]` holds the paths (coordinates,
potential indices, degeneracies) per number of legs.

## A phase file for the other tools

MD-EXAFS and RMC-EXAFS take a *phase file*: the potentials and phase shifts of your material, computed once from a
FEFF input of a representative cluster.

```bash
feffgpu phase feff.inp phase.pkl
```

The `POTENTIALS` block of that `feff.inp` defines the potential index of every element: 0 is the absorber, 1 and
up are the scatterers. The same indices are used when a structure or a trajectory is given to the other tools.
A phase file written by FEFF itself (`phase.pad`) can be used in place of the pickle.

```{warning}
`phase.pkl` is a Python pickle. Loading a pickle runs code, so load only phase files that you made yourself or that
come from someone you trust. FEFF's own `phase.pad` is a plain data file and has no such risk.
```

## Next

- The stages and what they produce: [](concepts/pipeline.md).
- The Python interface by task: [](feff/python_api.md).
- The spectrum of a trajectory: [](md/quickstart.md). Fitting a structure: [](rmc/quickstart.md).
