"""Path types: the names under which the spectrum of a frame is decomposed.

A scattering path starts at the absorbing atom, visits one or more scattering atoms and returns. For paths of up to
four legs the type is the
sequence of elements along the path, absorber first, written with dashes: ``"Zn-C"`` is single scattering from a C
neighbor of a Zn absorber, ``"Zn-C-C"`` double scattering through two C atoms, ``"Zn-C-Zn-C"`` a four-leg path. A path
and its time reverse (the same atoms visited in the opposite order) give the same signal; FEFF's path finder keeps
one of the two with degeneracy 2. Their two spellings name one type: of ``"Zn-C-Zn"`` and ``"Zn-Zn-C"`` the type is
the one whose scatterer sequence comes first in alphabetical order (``"Zn-C-Zn"``), and either spelling may be used
in a selection. In a four-leg path the middle atom may be the absorbing atom itself; it is named by its element.

Paths of five and more legs are few and weak, and their element sequences are many: they are gathered into one type
for each number of legs, named ``"Zn-(5 legs)"``, ``"Zn-(6 legs)"`` and so on.

The list of types of an edge is fixed before any frame is read (every combination of the system's elements up to the
largest number of legs), so every frame and every shard of a run stores the same types in the same order.
"""
from __future__ import annotations
import re
from itertools import product
import numpy as np
import torch

__all__ = ["canonical_sequence", "path_type_name", "long_type_name", "PathTypes"]

_LONG = re.compile(r"-\((\d+) legs\)$")


def long_type_name(absorber, nleg):
    """The name of the type that gathers every path of ``nleg`` legs (five or more) of an absorber: ``"Zn-(5 legs)"``."""
    return f"{absorber}-({int(nleg)} legs)"


def canonical_sequence(scatterers):
    """The scatterer sequence of a path or of its time reverse, whichever comes first alphabetically (a tuple)."""
    s = tuple(str(e) for e in scatterers)
    r = tuple(reversed(s))
    return min(s, r)


def path_type_name(absorber, scatterers):
    """The canonical name of a path type, for example ``path_type_name("Zn", ["Zn", "C"]) == "Zn-C-Zn"``."""
    return "-".join((str(absorber),) + canonical_sequence(scatterers))


class PathTypes:
    """The path types of one absorbing element in a system, and the lookup from potential sequences to types.

    Parameters
    ----------
    absorber : str
        Element of the absorbing atoms.
    ss_elements : list of str
        Elements that have a scatterer potential (the neighbors seen in single scattering), in the order of their
        potential index.
    elements : list of str
        Every element of the system (the absorber's element included); multiple-scattering types are built from these.
    nleg : int
        Largest number of legs (2 to 8).
    element_of_potential : dict
        ``{potential index: element}`` for the scatterer potentials (``0`` is added as the absorber).

    Attributes
    ----------
    ss : list of str
        Single-scattering types, one per scatterer potential.
    ms : list of str
        Multiple-scattering types (three legs, then four legs, alphabetical within each group; then one type for each
        larger number of legs).
    names : list of str
        ``ss + ms``: the order of the stored parts.
    """

    def __init__(self, absorber, ss_elements, elements, nleg, element_of_potential):
        self.absorber = str(absorber)
        self.nleg = int(nleg)
        if not 2 <= self.nleg <= 8:
            raise ValueError(f"nleg must be between 2 and 8 (got {nleg})")
        self.ss_elements = [str(e) for e in ss_elements]
        self.elements = sorted({str(e) for e in elements} | {self.absorber})
        self.ss = [f"{self.absorber}-{e}" for e in self.ss_elements]
        ms3 = sorted({path_type_name(self.absorber, s) for s in product(self.elements, repeat=2)}) if self.nleg >= 3 else []
        ms4 = sorted({path_type_name(self.absorber, s) for s in product(self.elements, repeat=3)}) if self.nleg >= 4 else []
        self.long = [long_type_name(self.absorber, n) for n in range(5, self.nleg + 1)]      # one type per number of legs from five on
        self.ms = ms3 + ms4 + self.long
        self.names = self.ss + self.ms
        self.element_of_potential = {0: self.absorber, **{int(p): str(e) for p, e in element_of_potential.items()}}
        self._lut = {}

    def legs_of(self, name):
        """Number of legs of a type."""
        m = _LONG.search(name)
        return int(m.group(1)) if m else len(name.split("-"))

    def long_index(self, nleg):
        """The index in ``self.ms`` of the type that gathers the paths of ``nleg`` legs (five or more)."""
        return self.ms.index(long_type_name(self.absorber, nleg))

    def normalize(self, name):
        """The canonical spelling of a type name written by a user (either orientation); raises ``KeyError`` with
        the list of known types if the name is not a type of this edge."""
        if str(name).strip() in self.long:
            return str(name).strip()
        parts = [p.strip() for p in str(name).split("-")]
        if len(parts) < 2 or parts[0] != self.absorber:
            raise KeyError(f"path type {name!r}: the first element must be the absorber {self.absorber!r}; "
                           f"known types: {', '.join(self.names)}")
        canon = path_type_name(parts[0], parts[1:])
        if canon not in self.names:
            raise KeyError(f"path type {name!r} is not a type of the {self.absorber} edge; known types: {', '.join(self.names)}")
        return canon

    def lookup(self, nleg, n_potentials, device="cpu"):
        """A tensor that maps the scatterer potentials of a path (one index per scatterer) to the index of its type
        in ``self.ms``; shape ``(n_potentials,) * (nleg - 1)``. Combinations that name no type map to -1."""
        key = (int(nleg), int(n_potentials), str(device))
        if key in self._lut:
            return self._lut[key]
        m = int(nleg) - 1
        lut = np.full((n_potentials,) * m, -1, dtype=np.int64)
        index = {name: j for j, name in enumerate(self.ms)}
        for combo in product(range(n_potentials), repeat=m):
            if any(p not in self.element_of_potential for p in combo):
                continue
            name = path_type_name(self.absorber, [self.element_of_potential[p] for p in combo])
            lut[combo] = index.get(name, -1)
        t = torch.as_tensor(lut, device=device)
        self._lut[key] = t
        return t
