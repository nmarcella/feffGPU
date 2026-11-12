"""The model cell of a reverse Monte Carlo fit: the atoms, the periodic cell, and the lattice sites of the atoms.

A model cell is a start structure (element symbols, Cartesian positions in A, lattice vectors). Following EvAX, the
**lattice site of an atom is its position in the start structure**: the displacement limit of the search (EvAX's box)
is measured from it, and EvAX's crossover exchanges the displacements that two states have on the same site.

Exchanges and hops change which atom sits on which site, never the element of an atom index. Two kinds of sites
exist:

- the own site of an atom: its start position. An exchange of two atoms swaps their sites.
- the sites of a hop sublattice: a list of sites for the atoms of one element, some of them vacant (no atom sits
  there). A hop moves an atom of that element onto a vacant site. The list can be read from a file
  (``read_site_file``) or made by a built-in generator (``fluorite_sites``: the anion sublattice of the fluorite
  structure).

``SiteTable`` holds all sites of a model cell. Every configuration of the search carries ``site_of`` (n,), the
index in the table of the site of every atom.
"""
from __future__ import annotations
import itertools
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

__all__ = ["SiteTable", "ModelCell", "own_sites", "sublattice_sites", "fluorite_sites", "read_site_file", "minimum_image",
           "ArrangementStatistics"]


def minimum_image(rel, cell):
    """Minimum image of displacements ``rel`` (..., 3) in ``cell`` ((3, 3) rows the lattice vectors, or None)."""
    if cell is None:
        return rel
    c = np.asarray(cell, float)
    f = rel @ np.linalg.inv(c)
    return (f - np.round(f)) @ c


@dataclass
class SiteTable:
    """The lattice sites of a model cell.

    Attributes
    ----------
    positions : numpy.ndarray, shape (m, 3)
        Site positions (A): the fixed start lattice.
    ids : numpy.ndarray of int, shape (m,)
        The identifier of every site. The crossover draws one random number per identifier, so that two states
        exchange the displacements of the atoms that sit on the same site.
    n_ids : int
        Number of identifiers (at least ``m``; a lattice generator may leave identifiers unused).
    start_site_of : numpy.ndarray of int, shape (n,)
        The site of every atom in the start structure.
    hop_element : str or None
        The element of the hop sublattice, if there is one.
    hop_sites : numpy.ndarray of int, shape (m_h,)
        Table indices of the sites of the hop sublattice (empty without one).
    hop_neighbors : numpy.ndarray of int, shape (m_h, K)
        For every hop site, the hop sites (local indices into ``hop_sites``) whose atom may hop onto it when it is
        vacant, in a fixed order, padded with -1.
    """
    positions: np.ndarray
    ids: np.ndarray
    n_ids: int
    start_site_of: np.ndarray
    hop_element: str | None = None
    hop_sites: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    hop_neighbors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), int))

    @property
    def m(self):
        return len(self.positions)

    def occupancy(self, site_of):
        """Per site, the atom on it (-1: vacant), for one configuration ``site_of`` (n,)."""
        out = np.full(self.m, -1, int)
        out[np.asarray(site_of)] = np.arange(len(site_of))
        return out

    def vacant_hop_sites(self, site_of):
        """Local indices of the vacant sites of the hop sublattice in one configuration."""
        occupied = np.zeros(self.m, bool)
        occupied[np.asarray(site_of)] = True
        return np.nonzero(~occupied[self.hop_sites])[0]

    def describe(self):
        if self.hop_element is None:
            return f"{self.m} sites (each atom's start position)"
        vac = len(self.vacant_hop_sites(self.start_site_of))
        return f"{self.m} sites; hop sublattice of {self.hop_element}: {len(self.hop_sites)} sites, {vac} vacant at the start"


def own_sites(positions):
    """Every atom on its own site (its start position); identifiers in atom order."""
    positions = np.asarray(positions, float)
    n = len(positions)
    return SiteTable(positions.copy(), np.arange(n), n, np.arange(n))


def _distances(a, b, cell, chunk=512):
    """Minimum-image distances between the points a (m, 3) and b (k, 3), computed in chunks of rows: (m, k)."""
    out = np.empty((len(a), len(b)))
    for s in range(0, len(a), chunk):
        out[s:s + chunk] = np.linalg.norm(minimum_image(b[None, :, :] - a[s:s + chunk, None, :], cell), axis=-1)
    return out


def _neighbor_lists(sites, cell, radius):
    """For every site, the other sites within ``radius`` (A), nearest first (ties by index), padded with -1."""
    d = _distances(sites, sites, cell)
    np.fill_diagonal(d, np.inf)
    lists = [np.array(sorted(np.nonzero(d[i] <= radius + 1e-9)[0], key=lambda j: (round(d[i, j], 6), j)), int) for i in range(len(sites))]
    K = max((len(v) for v in lists), default=0)
    out = np.full((len(sites), K), -1, int)
    for i, v in enumerate(lists):
        out[i, :len(v)] = v
    return out


def _assign(atom_pos, sites, cell, tolerance):
    """Greedy assignment of atoms to free sites: the closest atom-site pairs first, each atom to its nearest free site
    within ``tolerance`` (A). Returns the site index of every atom."""
    d = _distances(atom_pos, sites, cell)
    taken = np.zeros(len(sites), bool)
    out = np.full(len(atom_pos), -1, int)
    for q in np.argsort(d.min(1), kind="stable"):
        for s in np.argsort(d[q], kind="stable"):
            if d[q, s] > tolerance:
                break
            if not taken[s]:
                taken[s], out[q] = True, s
                break
        if out[q] < 0:
            raise ValueError(f"an atom at {np.round(atom_pos[q], 3).tolist()} finds no free site within {tolerance:.3f} A "
                             f"(nearest {d[q].min():.3f} A): the site list does not match the structure")
    return out


def sublattice_sites(symbols, positions, cell, element, site_positions, radius=None, tolerance=None):
    """A site table with a hop sublattice given as a list of sites.

    Parameters
    ----------
    symbols : array of str, shape (n,)
    positions : array, shape (n, 3)
        The start structure (A).
    cell : array (3, 3) or None
    element : str
        The element of the hop sublattice. Its atoms are assigned to their nearest free site of the list; the sites
        without an atom are vacant.
    site_positions : array, shape (m_h, 3)
        Every site of the sublattice, occupied or vacant (A).
    radius : float, optional
        Two sites are neighbors (an atom may hop between them) within this distance (A). Default: 1.1 times the
        nearest distance between two sites.
    tolerance : float, optional
        Largest distance of an atom from its site (A). Default: half the nearest site distance.

    Returns
    -------
    SiteTable
        The atoms of other elements keep their own sites (first in the table, in atom order), then the sublattice.
    """
    symbols = np.asarray(symbols).astype(str)
    positions = np.asarray(positions, float)
    sub = np.asarray(site_positions, float).reshape(-1, 3)
    if len(sub) < 2:
        raise ValueError(f"the hop sublattice of {element} needs at least two sites")
    d = _distances(sub, sub, cell)
    np.fill_diagonal(d, np.inf)
    nearest = float(d.min())
    if nearest < 1e-3:
        raise ValueError(f"the hop sublattice of {element} lists two sites at the same place")
    radius = 1.1 * nearest if radius is None else float(radius)
    tolerance = 0.5 * nearest if tolerance is None else float(tolerance)
    hop_atoms = np.nonzero(symbols == element)[0]
    if len(hop_atoms) == 0:
        raise ValueError(f"the structure has no {element} atom for the hop sublattice")
    if len(hop_atoms) > len(sub):
        raise ValueError(f"{len(hop_atoms)} {element} atoms but only {len(sub)} sites in the hop sublattice")
    others = np.nonzero(symbols != element)[0]
    table = np.concatenate([positions[others], sub])
    start = np.empty(len(symbols), int)
    start[others] = np.arange(len(others))
    start[hop_atoms] = len(others) + _assign(positions[hop_atoms], sub, cell, tolerance)
    m = len(table)
    return SiteTable(table, np.arange(m), m, start, element, len(others) + np.arange(len(sub)), _neighbor_lists(sub, cell, radius))


def fluorite_sites(symbols, positions, cell, anion, cells=None):
    """The site table of a fluorite structure whose anion sublattice holds vacancies (the built-in site generator).

    The cations sit on the face-centered cubic sites ``t + n h`` (``h`` half the conventional cell edge, ``n`` integer
    with an even sum) and the anions on the simple cubic sites ``t + (m + 1/2) h``; ``t`` is the start position of the
    first cation. Every anion site is listed; the sites without an anion are vacant. An anion hops onto a vacant site
    from one of its six nearest sites.

    Parameters
    ----------
    symbols, positions, cell
        The start structure: a cubic cell of ``cells`` x ``cells`` x ``cells`` conventional fluorite cells.
    anion : str
        The element of the anion sublattice. Every other atom must sit on a cation site.
    cells : int, optional
        Conventional cells along each edge (default: from the number of cations, four per conventional cell).

    Returns
    -------
    SiteTable
        The cations keep their own sites; site identifiers follow the grid of the lattice (cation sites
        ``(c0 L + c1) L + c2``, anion sites ``L^3 + (m0 L + m1) L + m2``, ``L = 2 cells``).
    """
    symbols = np.asarray(symbols).astype(str)
    positions = np.asarray(positions, float)
    if cell is None:
        raise ValueError("the fluorite site generator needs a periodic cell")
    c = np.asarray(cell, float)
    edge = float(c[0, 0])
    if np.abs(c - np.diag(np.diag(c))).max() > 1e-6 or np.abs(np.diag(c) - edge).max() > 1e-6:
        raise ValueError("the fluorite site generator needs a cubic cell (equal edges along x, y and z)")
    cat = np.nonzero(symbols != anion)[0]
    an = np.nonzero(symbols == anion)[0]
    if len(cat) == 0 or len(an) == 0:
        raise ValueError(f"the fluorite site generator needs cations and {anion} anions")
    if cells is None:
        cells = int(round((len(cat) / 4) ** (1 / 3)))
    cells = int(cells)
    if 4 * cells ** 3 != len(cat):
        raise ValueError(f"{len(cat)} cations do not fill {cells}^3 conventional fluorite cells (4 cations per cell)")
    L = 2 * cells
    h = edge / L
    t0 = positions[cat[0]]
    u = (positions[cat] - t0) / h
    nn = np.round(u)
    off = np.linalg.norm(minimum_image((u - nn) * h, c), axis=-1)
    if off.max() > 0.25 * h or (np.mod(nn.astype(int).sum(-1), 2) != 0).any():
        raise ValueError("the cations are not on the face-centered cubic sites of a fluorite lattice (the generator needs the ideal start structure)")
    cc = np.mod(nn.astype(int), L)
    cat_ids = (cc[:, 0] * L + cc[:, 1]) * L + cc[:, 2]
    if len(set(cat_ids.tolist())) != len(cat):
        raise ValueError("two cations sit on the same site of the fluorite lattice")
    g = np.stack(np.meshgrid(np.arange(L), np.arange(L), np.arange(L), indexing="ij"), -1).reshape(-1, 3)
    an_sites = t0 + (g + 0.5) * h
    m_an = np.mod(np.round((positions[an] - t0) / h - 0.5).astype(int), L)
    an_grid = (m_an[:, 0] * L + m_an[:, 1]) * L + m_an[:, 2]
    if len(set(an_grid.tolist())) != len(an):
        raise ValueError(f"two {anion} atoms sit on the same anion site of the fluorite lattice")
    d_an = np.linalg.norm(minimum_image(positions[an] - an_sites[an_grid], c), axis=-1)
    if d_an.max() > 0.25 * h:
        raise ValueError(f"an {anion} atom is {d_an.max():.3f} A from its anion site (the generator needs the ideal start structure)")
    e1 = np.array([o for o in itertools.product([-1, 0, 1], repeat=3) if sum(abs(v) for v in o) == 1])
    gn = np.mod(g[:, None, :] + e1[None], L)
    neighbors = (gn[..., 0] * L + gn[..., 1]) * L + gn[..., 2]                                  # (L^3, 6) local indices, in the generator's order
    table = np.concatenate([positions[cat], an_sites])
    ids = np.concatenate([cat_ids, L ** 3 + np.arange(L ** 3)])
    start = np.empty(len(symbols), int)
    start[cat] = np.arange(len(cat))
    start[an] = len(cat) + an_grid
    return SiteTable(table, ids, 2 * L ** 3, start, str(anion), len(cat) + np.arange(L ** 3), neighbors)


def read_site_file(path):
    """The sites of a hop sublattice from a file (A).

    ``.txt``, ``.dat`` or ``.csv``: three columns of Cartesian coordinates per line (lines starting with ``#`` are
    skipped). Any other file: a structure file that ASE reads (POSCAR, CIF, extended xyz, ...), every atom a site.
    """
    path = Path(path)
    if path.suffix.lower() in (".txt", ".dat", ".csv"):
        a = np.loadtxt(path, comments="#", delimiter="," if path.suffix.lower() == ".csv" else None, ndmin=2)
        if a.shape[1] < 3:
            raise ValueError(f"{path}: three columns (x y z in A) per site expected")
        return a[:, :3].astype(float)
    from ..io.structure import read_structure
    return read_structure(path)[1]


class ModelCell:
    """A start structure for reverse Monte Carlo: symbols, positions, cell, and its lattice sites.

    Parameters
    ----------
    symbols : array of str, shape (n,)
    positions : array, shape (n, 3)
        Cartesian positions (A) of the start structure, the fixed start lattice.
    cell : array (3, 3) or None
        Lattice vectors as rows (A); None for a cluster.
    name : str
    sites : SiteTable, optional
        Default: every atom on its own site.
    """

    def __init__(self, symbols, positions, cell, name="", sites=None):
        self.symbols = np.asarray(symbols).astype(str)
        self.positions = np.asarray(positions, float)
        if self.positions.shape != (len(self.symbols), 3):
            raise ValueError("positions must be (n, 3) for n symbols")
        self.cell = None if cell is None else np.asarray(cell, float)
        self.name = str(name)
        self.sites = own_sites(self.positions) if sites is None else sites

    @classmethod
    def from_file(cls, path, supercell=(1, 1, 1), format=None):
        """Read a structure with ASE (POSCAR, CIF, extended xyz, ...) and replicate it ``supercell`` times."""
        from ..io.structure import read_structure, supercell as replicate
        sym, pos, cell = read_structure(path, format=format)
        n = [int(v) for v in supercell]
        if any(v < 1 for v in n):
            raise ValueError(f"supercell must be three positive integers (got {list(supercell)})")
        if n != [1, 1, 1]:
            if cell is None:
                raise ValueError(f"{path}: a supercell needs a periodic cell")
            sym, pos, cell = replicate(sym, pos, cell, n)
        return cls(sym, pos, cell, name=Path(path).stem)

    @property
    def n(self):
        return len(self.symbols)

    @property
    def elements(self):
        """The element names, sorted (the order of the species codes)."""
        return sorted(set(self.symbols.tolist()))

    @property
    def species_code(self):
        """(n,) the index of every atom's element in ``elements``."""
        names = self.elements
        return np.array([names.index(s) for s in self.symbols])

    def counts(self):
        return {e: int((self.symbols == e).sum()) for e in self.elements}

    def with_hop_sites(self, element, site_positions=None, generator=None, radius=None, tolerance=None):
        """This cell with a hop sublattice for ``element``: from ``site_positions`` (m, 3), or from the built-in generator
        ``"fluorite"``. Returns a new ModelCell."""
        if generator == "fluorite":
            sites = fluorite_sites(self.symbols, self.positions, self.cell, element)
        elif site_positions is not None:
            sites = sublattice_sites(self.symbols, self.positions, self.cell, element, site_positions, radius=radius, tolerance=tolerance)
        else:
            raise ValueError("give site_positions or generator='fluorite'")
        return ModelCell(self.symbols, self.positions, self.cell, self.name, sites)

    def describe(self):
        cell = "no periodic cell" if self.cell is None else "cell " + " x ".join(f"{v:.3f}" for v in np.linalg.norm(self.cell, axis=1)) + " A"
        return f"{self.n} atoms (" + ", ".join(f"{c} {e}" for e, c in self.counts().items()) + f"), {cell}; {self.sites.describe()}"


class ArrangementStatistics:
    """Statistics of the arrangement of atoms on sites, for configurations of one model cell.

    Parameters
    ----------
    model : ModelCell
    exchange : (str, str), optional
        The two elements that exchange. Their start sites form the exchange sublattice; two of its sites are neighbors
        within 1.05 times its nearest site distance.

    Notes
    -----
    ``__call__(site_of)`` with ``site_of`` (C, n) returns the averages over the C cells:

    - ``atoms_off_start_site``: atoms not on their start site;
    - ``sites_changed``: occupied sites whose element differs from the start;
    - with a hop sublattice: ``vacancies``, ``vacancies_moved`` (vacant sites that were occupied at the start) and
      ``vacant_neighbors_of_a_vacancy`` (vacant neighbor sites per vacant site, with ``vacant_neighbors_random``, the
      value for a random arrangement);
    - with an exchange pair (A, B): ``fraction_A`` (the share of A on the exchange sublattice),
      ``unlike_neighbors_of_A`` and ``warren_cowley_A`` (``1 - P(B next to A) / x_B``: 0 for a random arrangement,
      above 0 when like neighbors are preferred), the same for B; with a hop sublattice also ``A_around_vacancies``
      (the share of A among the exchange sites nearest to a vacant site).
    """

    def __init__(self, model, exchange=None):
        self.model = model
        s = model.sites
        self.start = s.start_site_of
        self.exchange = tuple(exchange) if exchange else None
        self.ex_sites = None
        if self.exchange:
            ex_atoms = np.nonzero(np.isin(model.symbols, self.exchange))[0]
            if len(ex_atoms) >= 2:
                self.ex_sites = s.start_site_of[ex_atoms]
                pos = s.positions[self.ex_sites]
                d = _distances(pos, pos, model.cell)
                np.fill_diagonal(d, np.inf)
                self.ex_nb = d <= 1.05 * d.min()
                if s.hop_element is not None and len(s.hop_sites):
                    dh = _distances(s.positions[s.hop_sites], pos, model.cell)
                    self.hop_ex_nb = dh <= 1.05 * dh.min()
        if s.hop_element is not None and len(s.hop_sites) > 1:
            ph = s.positions[s.hop_sites]
            d = _distances(ph, ph, model.cell)
            np.fill_diagonal(d, np.inf)
            self.hop_nb = d <= 1.05 * d.min()

    def __call__(self, site_of):
        model, s = self.model, self.model.sites
        site_of = np.asarray(site_of).reshape(-1, model.n)
        sym = model.symbols
        occ0 = s.occupancy(self.start)
        el0 = np.where(occ0 >= 0, sym[np.clip(occ0, 0, None)], "")
        rows = []
        for so in site_of:
            r = {"atoms_off_start_site": float((so != self.start).sum())}
            occ = s.occupancy(so)
            el = np.where(occ >= 0, sym[np.clip(occ, 0, None)], "")
            both = (occ >= 0) & (occ0 >= 0)
            r["sites_changed"] = float((both & (el != el0)).sum())
            if s.hop_element is not None and len(s.hop_sites):
                vac = occ[s.hop_sites] < 0
                r["vacancies"] = float(vac.sum())
                r["vacancies_moved"] = float((vac & (occ0[s.hop_sites] >= 0)).sum())
                if hasattr(self, "hop_nb") and vac.any():
                    r["vacant_neighbors_of_a_vacancy"] = float(self.hop_nb[vac][:, vac].sum(1).mean())
                    z = float(self.hop_nb.sum(1).mean())
                    r["vacant_neighbors_random"] = (vac.sum() - 1) / max(len(vac) - 1, 1) * z
            if self.ex_sites is not None:
                a, b = self.exchange
                e = el[self.ex_sites]
                is_a, is_b = e == a, e == b
                ab = is_a | is_b
                x_a = float(is_a.sum() / max(ab.sum(), 1))
                r[f"fraction_{a}"] = x_a
                for name, mine, other, x_other in ((a, is_a, is_b, 1 - x_a), (b, is_b, is_a, x_a)):
                    nb_all = (self.ex_nb & ab[None]).sum(1)
                    unlike = (self.ex_nb & other[None]).sum(1) / np.maximum(nb_all, 1)
                    p = float(unlike[mine].mean()) if mine.any() else float("nan")
                    r[f"unlike_neighbors_of_{name}"] = p
                    r[f"warren_cowley_{name}"] = 1.0 - p / x_other if x_other > 0 else float("nan")
                if s.hop_element is not None and len(s.hop_sites) and hasattr(self, "hop_ex_nb"):
                    vac = occ[s.hop_sites] < 0
                    if vac.any():
                        around = self.hop_ex_nb[vac]
                        r[f"{a}_around_vacancies"] = float(((around & is_a[None]).sum(1) / np.maximum((around & ab[None]).sum(1), 1)).mean())
            rows.append(r)
        keys = rows[0].keys()
        return {k: float(np.mean([row[k] for row in rows if k in row])) for k in keys}
