"""Analysis of atomic configurations, shared by MD-EXAFS (frames of a trajectory) and RMC-EXAFS (states of a model).

``feffgpu.analysis.pairs``: pair distances, partial pair distributions, shell tables (N, R, MSRD, third cumulant by
integration), pairs at impossible distances, bond-angle distributions.
"""
from .pairs import (minimum_image, pair_distances, distances_in_range, cumulants, shell_numbers, shell_table, partial_rdfs,
                    short_pairs, bond_angles, angle_histogram, registered_pair_distances)
from .plots import plot_pair_distributions

__all__ = ["minimum_image", "pair_distances", "distances_in_range", "cumulants", "shell_numbers", "shell_table", "partial_rdfs",
           "short_pairs", "bond_angles", "angle_histogram", "registered_pair_distances", "plot_pair_distributions"]
