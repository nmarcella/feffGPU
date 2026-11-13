"""RMC-EXAFS: reverse Monte Carlo fitting of EXAFS spectra on the GPU.

The atoms of a model structure are moved until the calculated spectra (the exact forward model of
``feffgpu.exafs.edge.EdgeModel``: single scattering of every pair, multiple scattering of every absorber's paths) fit the
measured ones. Two engines:

- ``GpuRmc`` and ``EdgeRefinement`` (``feffgpu.rmc.refine``), the default (``rmc-exafs refine``): gradient refinement of
  the positions on the exact spectra, with the gradient of the multiple scattering: one element against one
  spectrum, or any elements against several spectra.
- ``EvaxSearch`` (``feffgpu.rmc.search``, ``rmc-exafs run``): a population search that follows the rules of EvAX
  (Timoshenko and Kuzmin): tournament selection, crossover, uniform random moves, cooled acceptance, adaptive spectrum
  weights, the displacement limit and closest distances; with additions that are switched on in the configuration
  (gradient steps, exchanges, hops, several samples with S0^2 and the E0 shift shared per edge).

``EvaxCalculation`` (``feffgpu.rmc.evax``, ``rmc-exafs from-evax``) reads an EvAX calculation and writes the inputs for
the same problem.

Command: ``rmc-exafs`` (``feffgpu.rmc.cli``). One TOML file per run (``feffgpu.rmc.config``).

Typical use from Python::

    from feffgpu.rmc import run_refine, run_search
    summary = run_refine("config.toml")        # the [refine] section
    summary = run_search("config.toml")        # the [search] section
"""
from .config import (RmcConfig, RunSettings, SampleSettings, EdgeSettings, WindowSettings, AmplitudeSettings, SearchSettings, MoveSettings,
                     HopSettings, ConstraintSettings, RefineSettings, load_config, json_schema, example_configs, validate_file, output_directory)
from .structure import ModelCell, SiteTable, own_sites, sublattice_sites, fluorite_sites, read_site_file, ArrangementStatistics
from .constraints import Box, ClosestDistances
from .rules import EdgeWeights, Annealing, evax_selection, site_crossover
from .moves import start_population, uniform_step, exchange_atoms, hop_atoms, fire_relax, GradientObjective
from .paths import StatePaths
from .search import RmcSample, EvaxSearch
from .refine import GpuRmc, EdgeRefinement
from .run import build_sample, build_samples, run_search, run_refine
from .outputs import METHOD, read_run, read_sample, sample_dirs
from .evax import EvaxCalculation, EvaxSpectrum, read_parameters, scf_radius

__all__ = ["RmcConfig", "RunSettings", "SampleSettings", "EdgeSettings", "WindowSettings", "AmplitudeSettings", "SearchSettings", "MoveSettings",
           "HopSettings", "ConstraintSettings", "RefineSettings", "load_config", "json_schema", "example_configs", "validate_file", "output_directory",
           "ModelCell", "SiteTable", "own_sites", "sublattice_sites", "fluorite_sites", "read_site_file", "ArrangementStatistics", "Box",
           "ClosestDistances", "EdgeWeights", "Annealing", "evax_selection", "site_crossover", "start_population", "uniform_step", "exchange_atoms",
           "hop_atoms", "fire_relax", "GradientObjective", "StatePaths", "RmcSample", "EvaxSearch", "GpuRmc", "EdgeRefinement", "build_sample", "build_samples",
           "run_search", "run_refine", "METHOD", "read_run", "read_sample", "sample_dirs", "EvaxCalculation", "EvaxSpectrum", "read_parameters",
           "scf_radius"]
