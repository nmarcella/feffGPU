# The FEFF 8.5L library (optional)

`feffgpu` needs FEFF's own Fortran only for three things: inputs with `ION` or `UNFREEZEF` cards (ions and unfrozen
f electrons are not in the free-atom table, so their atomic calculation runs through FEFF's code), rebuilding the
free-atom table (`tools/build_atoms_table.py`), and the oracle tests that compare intermediate quantities with FEFF
(`tests/test_pot.py`, `tests/test_xsph.py`, `tests/test_pathfinder.py`). Everything else runs without it.

FEFF 8.5L is not included in this repository (it is copyright University of Washington). Build it from the open-source
release:

```bash
git clone https://github.com/xraypy/feff85exafs.git
cd feff85exafs
git apply /path/to/feffGPU/fortran/feff85exafs_ltot80.patch     # raises the partial-wave limit ltot to 80 (FEFF ships 24)
scons                                                                  # feff85exafs builds with SCons and gfortran; see its README
```

The patch only enlarges fixed array dimensions (`ltot`, and what depends on it) so that FEFF's own partial-wave
rule can be followed at high k and for heavy elements; the validation in `docs/` was made against this build. After
the build, point the package at it:

```bash
export FEFF_HOME=/path/to/feff85exafs/local_install        # a directory with bin/ (feff8l_rdinp, feff8l_pot, ...) and lib/ (libfeff8lpotph.so, libfeff8lpath.so)
# or only the library:
export FEFFGPU_POTPH_LIB=/path/to/libfeff8lpotph.so
```

With `FEFF_HOME` set, `tools/validate_random.py` can regenerate the random-environment validation (it runs FEFF's six
modules on every case and compares), and the oracle tests run.
