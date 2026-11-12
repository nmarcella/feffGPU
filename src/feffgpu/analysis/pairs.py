"""Pair statistics of atomic configurations: distances, partial radial distributions, shell tables, pairs at
impossible distances, and bond-angle distributions.

These functions take plain arrays and work for a trajectory (frames) and for a reverse Monte Carlo model (states)
alike: ``pos`` is (C, n, 3) over configurations (or (n, 3) for one), ``symbols`` (n,) the element of every atom, and
``cell`` the periodic cell (a cubic length, three lengths, a 3 x 3 matrix with the lattice vectors as rows, or None
for a cluster). Distances use the minimum image, exact while they stay below half the smallest width of the cell.

Shell numbers are obtained by plain integration of the pair histogram in a stated range: N is the number of pairs
per center atom in the range, R their mean distance, sigma2 their variance (the mean-square relative displacement,
MSRD), C3 their third cumulant. Nothing is fitted.
"""
from __future__ import annotations
import numpy as np

__all__ = ["minimum_image", "pair_distances", "distances_in_range", "cumulants", "shell_numbers", "shell_table", "partial_rdfs",
           "short_pairs", "bond_angles", "angle_histogram", "registered_pair_distances"]


def _as_frames(pos):
    pos = np.asarray(pos, float)
    return pos[None] if pos.ndim == 2 else pos


def _as_cell(cell):
    """None, a float (cubic), or a (3, 3) array."""
    if cell is None:
        return None
    c = np.asarray(cell, float)
    if c.ndim == 0:
        return float(c)
    if c.ndim == 1:
        return np.diag(c)
    return c


def minimum_image(rel, cell):
    """Minimum image of displacement vectors ``rel`` (..., 3) in the cell (numpy).

    ``cell`` is None (returns ``rel``), a cubic length, three lengths, or a 3 x 3 matrix with the lattice vectors as
    rows (fractional rounding)."""
    c = _as_cell(cell)
    if c is None:
        return rel
    if isinstance(c, float):
        return rel - c * np.round(rel / c)
    f = rel @ np.linalg.inv(c)
    return (f - np.round(f)) @ c


def pair_distances(pos, cell, idx_a, idx_b):
    """Minimum-image distances between two index sets.

    Parameters
    ----------
    pos : array, shape (C, n, 3) or (n, 3)
    cell : None, float, (3,) or (3, 3)
    idx_a, idx_b : arrays of int
        The center atoms and the neighbor atoms.

    Returns
    -------
    numpy.ndarray, shape (C, len(idx_a), len(idx_b))
        The distance of an atom to itself (an index in both sets) is ``inf``.
    """
    pos = _as_frames(pos)
    idx_a, idx_b = np.asarray(idx_a), np.asarray(idx_b)
    out = np.empty((pos.shape[0], len(idx_a), len(idx_b)))
    block = max(1, int(4e6 // max(len(idx_b), 1)))                       # centers per block: about 100 MB of temporaries
    for c in range(pos.shape[0]):
        pb = pos[c][idx_b]
        for s in range(0, len(idx_a), block):
            rel = minimum_image(pb[None, :, :] - pos[c][idx_a[s:s + block]][:, None, :], cell)
            out[c, s:s + block] = np.linalg.norm(rel, axis=-1)
    same = idx_a[:, None] == idx_b[None, :]
    if same.any():
        out[:, same] = np.inf
    return out


def distances_in_range(pos, symbols, cell, center, neighbor, r_min, r_max):
    """All center-neighbor distances with ``r_min <= r < r_max``, over all configurations.

    ``center`` and ``neighbor`` are element symbols (``neighbor`` may be a list of symbols). Returns the distances
    (1-D) and the number of center atoms summed over the configurations (the divisor of a coordination number)."""
    pos = _as_frames(pos)
    symbols = np.asarray(symbols)
    ia = np.nonzero(symbols == center)[0]
    ib = np.nonzero(np.isin(symbols, np.atleast_1d(neighbor)))[0]
    if len(ia) == 0 or len(ib) == 0:
        return np.zeros(0), pos.shape[0] * len(ia)
    d = pair_distances(pos, cell, ia, ib)
    return d[(d >= r_min) & (d < r_max)], pos.shape[0] * len(ia)


def cumulants(r):
    """Mean, variance, third cumulant, standardized skewness and excess kurtosis of a sample of distances.

    Returns a dict with the keys ``R`` (A), ``sigma2`` (A^2), ``c3`` (A^3), ``skew``, ``kurt`` and ``n`` (the sample
    size). An empty sample gives NaN."""
    r = np.asarray(r, float).ravel()
    if r.size == 0:
        return {"R": float("nan"), "sigma2": float("nan"), "c3": float("nan"), "skew": float("nan"), "kurt": float("nan"), "n": 0}
    m = r.mean(); d = r - m
    v = (d ** 2).mean(); c3 = (d ** 3).mean()
    return {"R": float(m), "sigma2": float(v), "c3": float(c3), "skew": float(c3 / v ** 1.5) if v > 0 else 0.0,
            "kurt": float((d ** 4).mean() / v ** 2 - 3) if v > 0 else 0.0, "n": int(r.size)}


def shell_numbers(dist, n_centers, r_min, r_max):
    """Shell numbers by integration: the distances of ``dist`` inside ``r_min <= r < r_max``.

    Returns ``{"N", "R", "sigma2", "c3", "skew"}`` with N = pairs per center (``n_centers`` is the number of center
    atoms summed over the configurations)."""
    dist = np.asarray(dist, float)
    d = dist[(dist >= r_min) & (dist < r_max)]
    c = cumulants(d)
    return {"N": len(d) / max(n_centers, 1), "R": c["R"], "sigma2": c["sigma2"], "c3": c["c3"], "skew": c["skew"]}


def shell_table(pos, symbols, cell, shells):
    """Coordination number, mean distance, MSRD and third cumulant of each shell, by integration of the pair
    histogram (nothing fitted).

    Parameters
    ----------
    pos, symbols, cell
        The configurations (see the module docstring).
    shells : list of (center, neighbor, r_min, r_max)
        One row per shell: element symbols and the range of distances (A) that defines the shell.

    Returns
    -------
    list of dict
        One per shell: ``center``, ``neighbor``, ``r_min``, ``r_max``, ``N``, ``R``, ``sigma2``, ``c3``, ``skew``.
    """
    rows = []
    for center, neighbor, r_min, r_max in shells:
        d, n_centers = distances_in_range(pos, symbols, cell, center, neighbor, r_min, r_max)
        row = {"center": center, "neighbor": neighbor, "r_min": float(r_min), "r_max": float(r_max)}
        row.update(shell_numbers(d, n_centers, r_min, r_max))
        rows.append(row)
    return rows


def partial_rdfs(pos, cell, centers, symbols, elements, rmax=10.0, dr=0.01):
    """Partial pair distributions of chosen center atoms with each element.

    Parameters
    ----------
    pos : array, shape (C, n, 3)
    cell : None, float, (3,) or (3, 3)
    centers : array of int
        Indices of the center (absorbing) atoms.
    symbols : array of str, shape (n,)
    elements : list of str
        Neighbor elements, one distribution each.
    rmax, dr : float
        Range and bin width (A).

    Returns
    -------
    r : numpy.ndarray, shape (nbin,)
        Bin centers.
    g : dict
        ``{element: pairs per center per A}`` (the integral over a range is a coordination number).
    dist : dict
        ``{element: the distances below rmax}`` and ``{element + "_count": number of centers}``, for
        ``shell_numbers`` and ``cumulants``.
    """
    pos = _as_frames(pos)
    symbols = np.asarray(symbols)
    centers = np.asarray(centers)
    nb = int(round(rmax / dr))
    r = (np.arange(nb) + 0.5) * dr
    g, dist = {}, {}
    for el in elements:
        j = np.nonzero(symbols == el)[0]
        d = pair_distances(pos, cell, centers, j)
        d = d[d <= rmax]
        h, _ = np.histogram(d, bins=nb, range=(0.0, rmax))
        g[el] = h / (pos.shape[0] * len(centers) * dr)
        dist[el] = d
        dist[el + "_count"] = pos.shape[0] * len(centers)
    return r, g, dist


def short_pairs(pos, symbols, cell, thresholds):
    """Pairs at distances that do not exist in the chemistry.

    A reverse Monte Carlo model without limits on the closest distances contains some pairs that no real material
    has; a trajectory from a poor potential can too. This counts them.

    Parameters
    ----------
    thresholds : dict
        ``{"A-B": distance}``: below this distance (A) an A-B pair is counted as impossible. A is the center.

    Returns
    -------
    dict
        ``{"A-B": {"threshold", "per_center", "share_of_centers", "closest"}}``: pairs below the threshold per A
        atom, the share of A atoms that have at least one, and the closest A-B distance found.
    """
    pos = _as_frames(pos)
    symbols = np.asarray(symbols)
    out = {}
    for key, th in thresholds.items():
        a, b = key.split("-")
        ia, ib = np.nonzero(symbols == a)[0], np.nonzero(symbols == b)[0]
        if len(ia) == 0 or len(ib) == 0:
            continue
        d = pair_distances(pos, cell, ia, ib)
        below = d < th
        out[key] = {"threshold": float(th), "per_center": float(below.sum() / (pos.shape[0] * len(ia))),
                    "share_of_centers": float(below.any(2).mean()), "closest": float(d.min())}
    return out


def bond_angles(pos, symbols, cell, center, neighbors, ranges):
    """Angles B-A-C between two neighbors B and C of a center atom A.

    Parameters
    ----------
    center : str
        Element of the center atoms A.
    neighbors : (str, str)
        Elements of B and C (they may be the same element).
    ranges : ((r_min, r_max), (r_min, r_max))
        The A-B and A-C distance ranges (A) that define a neighbor.

    Returns
    -------
    numpy.ndarray
        All angles in degrees, over all centers and configurations. With B and C the same element each unordered
        pair of neighbors is counted once.
    """
    pos = _as_frames(pos)
    symbols = np.asarray(symbols)
    eb, ec = neighbors
    (blo, bhi), (clo, chi_) = ranges
    ia = np.nonzero(symbols == center)[0]
    ib, ic = np.nonzero(symbols == eb)[0], np.nonzero(symbols == ec)[0]
    same = eb == ec and (blo, bhi) == (clo, chi_)
    out = []
    for c in range(pos.shape[0]):
        p = pos[c]
        for i in ia:
            vb = minimum_image(p[ib] - p[i], cell); db = np.linalg.norm(vb, axis=-1)
            mb = (db >= blo) & (db < bhi) & (ib != i)
            ub = vb[mb] / db[mb][:, None]
            if same:
                if len(ub) < 2:
                    continue
                cosang = ub @ ub.T
                iu = np.triu_indices(len(ub), 1)
                out.append(cosang[iu])
            else:
                vc = minimum_image(p[ic] - p[i], cell); dc = np.linalg.norm(vc, axis=-1)
                mc = (dc >= clo) & (dc < chi_) & (ic != i)
                if not mb.any() or not mc.any():
                    continue
                uc = vc[mc] / dc[mc][:, None]
                cosang = ub @ uc.T
                if eb == ec:                                           # same element, different ranges: drop an atom paired with itself
                    cosang = cosang[ib[mb][:, None] != ic[mc][None, :]]
                out.append(np.ravel(cosang))
    if not out:
        return np.zeros(0)
    return np.degrees(np.arccos(np.clip(np.concatenate(out), -1.0, 1.0)))


def angle_histogram(angles, bin_width=1.0):
    """Histogram of angles (degrees) from 0 to 180: bin centers and plain counts (no division by sin(angle))."""
    edges = np.arange(0.0, 180.0 + 1e-9, bin_width)
    h, _ = np.histogram(angles, bins=edges)
    return 0.5 * (edges[:-1] + edges[1:]), h


def registered_pair_distances(positions, sites, a, box, shells):
    """Distances of lattice-registered pairs of a cubic lattice block, for tests with a known answer.

    ``positions`` (C, n, 3); ``sites`` (n, 3) the ideal sites (A) on a grid of spacing ``a / 2``; ``box`` the cubic
    length; ``shells`` ``{label: [offsets in units of a / 2]}``. Returns ``{label: distances}`` of every pair
    (site, site + offset) over all copies."""
    half = a / 2
    s_int = np.round(sites / half).astype(int)
    m = int(round(box / half))
    index = {tuple(v): i for i, v in enumerate(s_int)}
    out = {}
    for sh, offs in shells.items():
        pairs = np.array([(i, index[tuple((v + np.array(o)) % m)]) for i, v in enumerate(s_int) for o in offs])
        rel = positions[:, pairs[:, 1]] - positions[:, pairs[:, 0]]
        rel -= box * np.round(rel / box)
        out[sh] = np.linalg.norm(rel, axis=-1).reshape(-1)
    return out
