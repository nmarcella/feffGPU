"""Reading structures and trajectories, without OVITO.

- ``feffgpu.io.structure``: element symbols, cells, POSCAR / xyz / ASE readers, supercells, potential indices.
- ``feffgpu.io.trajectory``: streaming trajectory readers (ASE formats, LAMMPS dumps, XDATCAR; MDAnalysis optional).
- ``feffgpu.io.evax``: the files of an EvAX run.
- ``feffgpu.io.lattice``: ideal bcc and fcc blocks for examples and tests.
"""
from .structure import (SYMBOLS, Z_OF, cell_from_parameters, read_poscar, supercell, read_xyz, read_structure, read_potentials,
                        ipot_table, parse_ipots)
from .evax import read_cell_file, read_evax_params, read_evax_rdf
from .lattice import bcc_block, fcc_block

__all__ = ["SYMBOLS", "Z_OF", "cell_from_parameters", "read_poscar", "supercell", "read_xyz", "read_structure", "read_potentials",
           "ipot_table", "parse_ipots", "read_cell_file", "read_evax_params", "read_evax_rdf", "bcc_block", "fcc_block"]
