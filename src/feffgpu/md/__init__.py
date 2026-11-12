"""MD-EXAFS: the EXAFS spectrum of a molecular dynamics trajectory, averaged over frames and absorbing atoms.

The tool reads an existing trajectory frame by frame (``feffgpu.io.trajectory``), computes the spectrum of every
frame with the GPU forward model of this package (single scattering from pair tables, multiple scattering from
FEFF's path finder and path stage), stores every frame's spectrum and its parts by path type in an HDF5 cache, and
computes averages, comparisons with a measured spectrum and structural analyses from there. It continues the
original MD-EXAFS program (``md_exafs`` 2.0, https://github.com/Joint-Photon-Sciences-Institute/MD-EXAFS) without
OVITO, FEFF input trees, a FEFF executable or a database.

Modules:

- ``config``: the TOML configuration (pydantic models), validation, the JSON Schema, example files;
- ``spectrum``: ``EdgeSpectra``, the spectrum of one frame for one edge, by path type;
- ``paths``: path-type names;
- ``potentials``: the phase file of each edge (``file``, ``reference``, ``per_frame``);
- ``run``: the loop over frames, shards, continuation;
- ``cache``: the HDF5 cache and the merge of shards;
- ``average``: averages, spread, standard errors and convergence from the cache;
- ``compare``: comparison with measured spectra (S0^2 and the E0 shift on the model, the R-factor);
- ``analyze``: radial distributions, peaks, angles and the thermodynamic log of a trajectory;
- ``dynamics``: an optional runner that makes a trajectory with an ASE calculator;
- ``plots``: figures;
- ``cli``: the ``md-exafs`` command.
"""
from .config import MDExafsConfig, load_config, validate_file, json_schema, example_configs, output_directory
from .spectrum import EdgeSpectra, FrameSpectrum, choose_absorbers
from .paths import PathTypes, path_type_name
from .run import run, load_run
from .cache import merge_shards
from .average import average_cache, edge_average, write_average

__all__ = ["MDExafsConfig", "load_config", "validate_file", "json_schema", "example_configs", "output_directory", "EdgeSpectra",
           "FrameSpectrum", "choose_absorbers", "PathTypes", "path_type_name", "run", "load_run", "merge_shards", "average_cache",
           "edge_average", "write_average"]
