"""RMC-EXAFS: reverse Monte Carlo fitting of EXAFS spectra on the GPU.

The atoms of a model structure are moved until the calculated spectra (the exact forward model of
``feffgpu.exafs.edge.EdgeModel``: single scattering of every pair, multiple scattering of every absorber's paths) fit the
measured ones. Two engines:

- ``EvaxSearch`` (``feffgpu.rmc.search``): a population search that follows the rules of EvAX (Timoshenko and
  Kuzmin): tournament selection, crossover, uniform random moves, cooled acceptance, adaptive spectrum weights,
  the displacement limit and closest distances; with additions that are switched on in the configuration (gradient
  steps, exchanges, hops, several samples with S0^2 and the E0 shift shared per edge). Any structure, any number of
  samples and edges.
- ``GpuRmc`` (``feffgpu.rmc.refine``): gradient refinement of the positions of copies of a one-element cell.

Command: ``rmc-exafs`` (``feffgpu.rmc.cli``). One TOML file per run (``feffgpu.rmc.config``).

Typical use from Python::

    from feffgpu.rmc import run_search
    summary = run_search("config.toml")
"""
from .config import (RmcConfig, RunSettings, SampleSettings, EdgeSettings, WindowSettings, AmplitudeSettings, SearchSettings, MoveSettings,
                     HopSettings, ConstraintSettings, RefineSettings, load_config, json_schema, example_configs, validate_file, output_directory)
from .structure import ModelCell, SiteTable, own_sites, sublattice_sites, fluorite_sites, read_site_file, ArrangementStatistics
from .constraints import Box, ClosestDistances
from .rules import EdgeWeights, Annealing, evax_selection, site_crossover
from .moves import start_population, uniform_step, exchange_atoms, hop_atoms, fire_relax, GradientObjective
from .paths import StatePaths
from .search import RmcSample, EvaxSearch
from .refine import GpuRmc
from .run import build_sample, build_samples, run_search, run_refine
from .outputs import METHOD, read_run, read_sample, sample_dirs

__all__ = ["RmcConfig", "RunSettings", "SampleSettings", "EdgeSettings", "WindowSettings", "AmplitudeSettings", "SearchSettings", "MoveSettings",
           "HopSettings", "ConstraintSettings", "RefineSettings", "load_config", "json_schema", "example_configs", "validate_file", "output_directory",
           "ModelCell", "SiteTable", "own_sites", "sublattice_sites", "fluorite_sites", "read_site_file", "ArrangementStatistics", "Box",
           "ClosestDistances", "EdgeWeights", "Annealing", "evax_selection", "site_crossover", "start_population", "uniform_step", "exchange_atoms",
           "hop_atoms", "fire_relax", "GradientObjective", "StatePaths", "RmcSample", "EvaxSearch", "GpuRmc", "build_sample", "build_samples",
           "run_search", "run_refine", "METHOD", "read_run", "read_sample", "sample_dirs"]
