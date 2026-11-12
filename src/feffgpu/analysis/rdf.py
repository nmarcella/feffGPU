"""Partial radial distribution functions (RDFs) accumulated configuration by configuration.

``RDFAccumulator`` takes one configuration at a time (positions, cell), so a trajectory of any length is processed
without holding it in memory. Distances are histogrammed with torch on any device (``"cpu"`` or a GPU). The cell may
change from one configuration to the next (NPT), may have any shape, and may be smaller than twice the largest
distance: periodic images are included as far as needed. Clusters (no cell) work too.

For every center-neighbor element pair the result holds:

- ``pairs_per_center_per_A``: the number of neighbors per center atom per A, whose integral over a distance range is
  a coordination number;
- ``g``: the same normalized by the ideal-gas expectation, ``g(r) = n(r) / (rho_B 4 pi r^2 dr)`` with the exact shell
  volume and ``rho_B = (N_B - delta_AB) / V`` averaged over the configurations (periodic systems only), so that
  ``g -> 1`` at large distances;
- ``coordination``: the running coordination number, the integral of ``pairs_per_center_per_A`` from zero.

Shell numbers (``shells``) are exact: the number of pairs per center, the mean distance, the variance (MSRD) and
the third cumulant of the distances in a stated range, from power sums of the distances themselves (not of the
histogram). They are the numbers of ``feffgpu.analysis.shell_table``, without holding the distances in memory.

Projected RDFs count only the bonds whose direction lies within a stated angle of a crystal direction (or of its
opposite), optionally of every cubic-symmetry equivalent of it, as in ``md_exafs`` 2.0.
"""
from __future__ import annotations
import math
import numpy as np
import torch

__all__ = ["RDFAccumulator", "cubic_directions", "solid_angle_fraction", "image_shifts", "moments_to_shell", "AngleAccumulator"]


def cubic_directions(direction):
    """The cubic-symmetry equivalents of a direction (permutations of the components and sign changes), as unit
    vectors (m, 3); a direction and its opposite are both included."""
    d = np.asarray(direction, float)
    n = np.linalg.norm(d)
    if n == 0:
        raise ValueError("a direction must not be zero")
    d = d / n
    out = set()
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        for sx in (1, -1):
            for sy in (1, -1):
                for sz in (1, -1):
                    v = np.array([sx * d[perm[0]], sy * d[perm[1]], sz * d[perm[2]]])
                    out.add(tuple(np.round(v, 9) + 0.0))
    return np.array(sorted(out))


def solid_angle_fraction(directions, cos_threshold, samples=400000):
    """The fraction of all directions that lie within the cones ``|u . d| >= cos_threshold`` around any of the given
    unit vectors (both senses), by a uniform Fibonacci grid of directions on the sphere."""
    dirs = np.asarray(directions, float).reshape(-1, 3)
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    i = np.arange(samples) + 0.5
    z = 1 - 2 * i / samples
    phi = math.pi * (1 + 5 ** 0.5) * i
    rho = np.sqrt(1 - z * z)
    u = np.stack([rho * np.cos(phi), rho * np.sin(phi), z], 1)
    inside = np.zeros(samples, bool)
    for s in range(0, len(dirs), 16):
        inside |= (np.abs(u @ dirs[s:s + 16].T) >= cos_threshold).any(1)
    return float(inside.mean())


def image_shifts(cell, r_max):
    """The lattice translations (S, 3) that, added to minimum-image displacements, reach every image within
    ``r_max``: only the zero translation when ``r_max`` is at most half the cell's smallest width."""
    c = np.asarray(cell, float)
    vol = abs(np.linalg.det(c))
    widths = [vol / np.linalg.norm(np.cross(c[(i + 1) % 3], c[(i + 2) % 3])) for i in range(3)]
    reach = [int(math.floor(r_max / w + 0.5)) for w in widths]
    grid = np.stack(np.meshgrid(*[np.arange(-m, m + 1) for m in reach], indexing="ij"), -1).reshape(-1, 3)
    grid = grid[np.argsort(np.abs(grid).sum(1), kind="stable")]                    # the zero translation first
    return grid @ c


def moments_to_shell(count, s1, s2, s3, ref, n_centers):
    """Shell numbers from the power sums of ``x = r - ref`` over the pairs of a range.

    Returns
    -------
    dict
        ``N`` (pairs per center), ``R`` (A), ``sigma2`` (A^2), ``c3`` (A^3), ``skew``, ``n_pairs``; NaN for an empty
        range. The variance and third cumulant are those of the distances (population values, as
        ``feffgpu.analysis.cumulants``)."""
    if count <= 0:
        return {"N": 0.0, "R": math.nan, "sigma2": math.nan, "c3": math.nan, "skew": math.nan, "n_pairs": 0}
    m1, m2, m3 = s1 / count, s2 / count, s3 / count
    var = max(m2 - m1 * m1, 0.0)
    c3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
    return {"N": count / max(n_centers, 1), "R": ref + m1, "sigma2": var, "c3": c3, "skew": c3 / var ** 1.5 if var > 0 else 0.0,
            "n_pairs": int(round(count))}


class RDFAccumulator:
    """Partial RDFs, shell numbers and projected RDFs accumulated over configurations.

    Parameters
    ----------
    symbols : array of str, shape (n,)
        Element of every atom (the same in every configuration).
    pairs : list of tuple, optional
        ``(center, neighbor)`` element symbols; default every ordered pair of the elements present.
    r_max : float
        Largest distance (A).
    bin_width : float
        Bin width (A).
    shells : list of tuple, optional
        ``(center, neighbor, r_min, r_max)``: ranges whose exact shell numbers are wanted (``r_min <= r < r_max``, as ``shell_table``).
    projections : list of dict, optional
        Each with ``center``, ``neighbor``, ``direction`` (3 numbers), ``angle`` (degrees, default 30), ``symmetry``
        (bool, cubic equivalents), and optionally ``r_min``, ``r_max`` for its shell numbers and ``label``.
    device : str
        Torch device of the histogram accumulation.
    max_pairs : int
        Center-neighbor pairs per batch (bounds the memory).
    """

    def __init__(self, symbols, pairs=None, r_max=8.0, bin_width=0.01, shells=None, projections=None, device="cpu", max_pairs=4_000_000):
        self.symbols = np.asarray(symbols)
        elements = sorted(set(str(s) for s in self.symbols))
        self.pairs = [tuple(p) for p in pairs] if pairs else [(a, b) for a in elements for b in elements]
        for a, b in self.pairs:
            if a not in elements or b not in elements:
                raise ValueError(f"pair {a}-{b}: the system has no {a if a not in elements else b} atom (elements: {elements})")
        self.r_max, self.dr = float(r_max), float(bin_width)
        self.nb = int(math.ceil(self.r_max / self.dr - 1e-9))
        self.edges = self.dr * np.arange(self.nb + 1)
        self.device = torch.device(device)
        self.max_pairs = int(max_pairs)
        self.index = {el: np.nonzero(self.symbols == el)[0] for el in elements}
        self.counts = {p: torch.zeros(self.nb, dtype=torch.float64, device=self.device) for p in self.pairs}
        self.ideal = {p: 0.0 for p in self.pairs}                           # sum over configurations of N_A (N_B - delta) / V
        self.shells = [(str(a), str(b), float(lo), float(hi)) for a, b, lo, hi in (shells or [])]
        for a, b, lo, hi in self.shells:
            if (a, b) not in self.pairs:
                self.pairs.append((a, b))
                self.counts[(a, b)] = torch.zeros(self.nb, dtype=torch.float64, device=self.device)
                self.ideal[(a, b)] = 0.0
            if hi > self.r_max + 1e-9:
                raise ValueError(f"shell {a}-{b} {lo}-{hi} A reaches beyond r_max ({self.r_max} A)")
        self.sums = [np.zeros(4) for _ in self.shells]
        self.proj = []
        for i, p in enumerate(projections or []):
            d = np.asarray(p["direction"], float)
            dirs = cubic_directions(d) if p.get("symmetry") else d[None] / np.linalg.norm(d)
            cos_th = math.cos(math.radians(float(p.get("angle", 30.0))))
            lo = float(p.get("r_min") if p.get("r_min") is not None else 0.0)
            hi = float(p.get("r_max") if p.get("r_max") is not None else self.r_max)
            label = p.get("label") or (f"{p['center']}-{p['neighbor']} along [{' '.join(f'{v:g}' for v in p['direction'])}]"
                                       + (" and equivalents" if p.get("symmetry") else ""))
            for el in (p["center"], p["neighbor"]):
                if el not in elements:
                    raise ValueError(f"projection {label}: the system has no {el} atom")
            self.proj.append({"label": label, "center": p["center"], "neighbor": p["neighbor"], "directions": dirs, "cos": cos_th,
                              "angle": float(p.get("angle", 30.0)), "symmetry": bool(p.get("symmetry", False)), "r_min": lo, "r_max": hi,
                              "fraction": solid_angle_fraction(dirs, cos_th),
                              "counts": torch.zeros(self.nb, dtype=torch.float64, device=self.device), "sums": np.zeros(4), "ideal": 0.0})
        self.frames = 0
        self.periodic = None

    def _pair_distances(self, pos, cell, ia, ib, want_vectors):
        """Yield (distances, vectors or None) for batches of centers, every image within r_max, self pairs excluded."""
        dev = self.device
        P = torch.as_tensor(pos, device=dev, dtype=torch.float64)
        pb = P[torch.as_tensor(ib, device=dev)]
        ib_t = torch.as_tensor(ib, device=dev)
        if cell is not None:
            C = torch.as_tensor(np.asarray(cell, float), device=dev, dtype=torch.float64)
            Cinv = torch.linalg.inv(C)
            shifts = torch.as_tensor(image_shifts(cell, self.r_max), device=dev, dtype=torch.float64)
        else:
            shifts = torch.zeros(1, 3, device=dev, dtype=torch.float64)
        per = max(1, self.max_pairs // max(len(ib) * len(shifts), 1))
        for s in range(0, len(ia), per):
            a = torch.as_tensor(ia[s:s + per], device=dev)
            rel = pb[None, :, :] - P[a][:, None, :]                                   # (B, nB, 3)
            if cell is not None:
                f = rel @ Cinv
                rel = (f - torch.round(f)) @ C
            same = a[:, None] == ib_t[None, :]
            for j, t in enumerate(shifts):
                v = rel + t
                d = torch.linalg.norm(v, dim=-1)
                m = d < self.r_max
                if j == 0:
                    m &= ~same
                yield d[m], (v[m] if want_vectors else None)

    @torch.no_grad()
    def add(self, positions, cell=None):
        """Add one configuration: positions (n, 3) in A and its cell (3 x 3, rows the lattice vectors) or None."""
        pos = np.asarray(positions, float)
        if pos.shape != (len(self.symbols), 3):
            raise ValueError(f"positions must be ({len(self.symbols)}, 3)")
        periodic = cell is not None
        if self.periodic is None:
            self.periodic = periodic
        elif self.periodic != periodic:
            raise ValueError("all configurations must be periodic, or all clusters")
        vol = abs(float(np.linalg.det(np.asarray(cell, float)))) if periodic else None
        for (a, b) in self.pairs:
            ia, ib = self.index[a], self.index[b]
            if periodic:
                self.ideal[(a, b)] += len(ia) * (len(ib) - (1 if a == b else 0)) / vol
            shell_ids = [i for i, sh in enumerate(self.shells) if sh[0] == a and sh[1] == b]
            proj_ids = [i for i, p in enumerate(self.proj) if p["center"] == a and p["neighbor"] == b]
            for d, v in self._pair_distances(pos, cell, ia, ib, want_vectors=bool(proj_ids)):
                if d.numel() == 0:
                    continue
                idx = torch.clamp((d / self.dr).long(), max=self.nb - 1)
                self.counts[(a, b)] += torch.bincount(idx, minlength=self.nb).to(torch.float64)
                for i in shell_ids:
                    _a, _b, lo, hi = self.shells[i]
                    x = d[(d >= lo) & (d < hi)] - 0.5 * (lo + hi)
                    if x.numel():
                        self.sums[i] += np.array([float(x.numel()), float(x.sum()), float((x * x).sum()), float((x ** 3).sum())])
                for i in proj_ids:
                    p = self.proj[i]
                    u = v / d[:, None]
                    D = torch.as_tensor(p["directions"], device=self.device, dtype=torch.float64)
                    keep = (torch.abs(u @ D.T) >= p["cos"] - 1e-12).any(1)
                    dk = d[keep]
                    if dk.numel():
                        bins = torch.clamp((dk / self.dr).long(), max=self.nb - 1)
                        p["counts"] += torch.bincount(bins, minlength=self.nb).to(torch.float64)
                        x = dk[(dk >= p["r_min"]) & (dk < p["r_max"])] - 0.5 * (p["r_min"] + p["r_max"])
                        if x.numel():
                            p["sums"] += np.array([float(x.numel()), float(x.sum()), float((x * x).sum()), float((x ** 3).sum())])
        for p in self.proj:
            if periodic:
                na, nb_ = len(self.index[p["center"]]), len(self.index[p["neighbor"]])
                p["ideal"] += na * (nb_ - (1 if p["center"] == p["neighbor"] else 0)) / vol
        self.frames += 1

    def result(self):
        """The accumulated RDFs.

        Returns
        -------
        dict
            ``r`` (bin centers, A), ``edges``, ``frames``, ``bin_width``, ``r_max``; ``pairs``: per ``"A-B"`` label
            ``counts``, ``pairs_per_center_per_A``, ``g`` (None for clusters), ``coordination``, ``centers`` (atoms of
            A), ``neighbors`` (atoms of B); ``shells``: per requested range ``center``, ``neighbor``, ``r_min``,
            ``r_max``, ``N``, ``R``, ``sigma2``, ``c3``, ``skew``, ``n_pairs``; ``projections``: per projection
            ``label``, ``directions``, ``angle``, ``solid_angle_fraction``, ``pairs_per_center_per_A``, ``g`` and
            ``shell`` (its numbers in its range).
        """
        if self.frames == 0:
            raise ValueError("no configuration was added")
        F = self.frames
        r = 0.5 * (self.edges[1:] + self.edges[:-1])
        shell_vol = 4.0 / 3.0 * math.pi * (self.edges[1:] ** 3 - self.edges[:-1] ** 3)
        out = {"r": r, "edges": self.edges, "frames": F, "bin_width": self.dr, "r_max": self.r_max, "pairs": {}, "shells": [],
               "projections": []}
        for (a, b) in self.pairs:
            c = self.counts[(a, b)].cpu().numpy()
            na = len(self.index[a])
            ppa = c / (F * na * self.dr)
            g = c / (self.ideal[(a, b)] * shell_vol) if self.periodic and self.ideal[(a, b)] > 0 else None
            out["pairs"][f"{a}-{b}"] = {"center": a, "neighbor": b, "counts": c, "pairs_per_center_per_A": ppa, "g": g,
                                        "coordination": np.cumsum(c) / (F * na), "centers": int(na), "neighbors": int(len(self.index[b]))}
        for (a, b, lo, hi), s in zip(self.shells, self.sums):
            row = {"center": a, "neighbor": b, "r_min": lo, "r_max": hi}
            row.update(moments_to_shell(s[0], s[1], s[2], s[3], 0.5 * (lo + hi), F * len(self.index[a])))
            out["shells"].append(row)
        for p in self.proj:
            c = p["counts"].cpu().numpy()
            na = len(self.index[p["center"]])
            g = c / (p["ideal"] * shell_vol * p["fraction"]) if self.periodic and p["ideal"] > 0 and p["fraction"] > 0 else None
            sh = {"r_min": p["r_min"], "r_max": p["r_max"]}
            sh.update(moments_to_shell(p["sums"][0], p["sums"][1], p["sums"][2], p["sums"][3], 0.5 * (p["r_min"] + p["r_max"]), F * na))
            out["projections"].append({"label": p["label"], "center": p["center"], "neighbor": p["neighbor"], "directions": p["directions"],
                                       "angle": p["angle"], "symmetry": p["symmetry"], "solid_angle_fraction": p["fraction"],
                                       "pairs_per_center_per_A": c / (F * na * self.dr), "g": g, "coordination": np.cumsum(c) / (F * na),
                                       "shell": sh})
        return out


class AngleAccumulator:
    """Bond-angle distributions B-A-C accumulated over configurations with ``feffgpu.analysis.bond_angles``.

    Parameters
    ----------
    symbols : array of str, shape (n,)
    specs : list of dict
        Each with ``center``, ``neighbors`` (two elements), ``ranges`` (two (r_min, r_max) pairs, A) and optionally
        ``bin_width`` (degrees, default 1).
    """

    def __init__(self, symbols, specs):
        self.symbols = np.asarray(symbols)
        self.specs = []
        for s in specs:
            bw = float(s.get("bin_width", 1.0))
            edges = np.arange(0.0, 180.0 + 1e-9, bw)
            label = f"{s['neighbors'][0]}-{s['center']}-{s['neighbors'][1]}"
            self.specs.append({"label": label, "center": s["center"], "neighbors": tuple(s["neighbors"]),
                               "ranges": tuple(tuple(r) for r in s["ranges"]), "edges": edges, "counts": np.zeros(len(edges) - 1), "n": 0})
        self.frames = 0

    def add(self, positions, cell=None):
        """Add one configuration (positions (n, 3) in A, cell or None)."""
        from .pairs import bond_angles
        for s in self.specs:
            ang = bond_angles(np.asarray(positions, float), self.symbols, cell, s["center"], s["neighbors"], s["ranges"])
            h, _ = np.histogram(ang, bins=s["edges"])
            s["counts"] += h
            s["n"] += len(ang)
        self.frames += 1

    def result(self):
        """Per distribution: ``label``, ``angle`` (bin centers, degrees), ``counts``, ``per_center`` (angles per center
        atom per configuration in each bin), ``angles`` (total number), ``mean`` (degrees)."""
        out = []
        for s in self.specs:
            centers = 0.5 * (s["edges"][1:] + s["edges"][:-1])
            n_center = int((self.symbols == s["center"]).sum()) * max(self.frames, 1)
            tot = s["counts"].sum()
            out.append({"label": s["label"], "center": s["center"], "neighbors": list(s["neighbors"]),
                        "ranges": [list(r) for r in s["ranges"]],
                        "angle": centers, "counts": s["counts"].copy(), "per_center": s["counts"] / n_center, "angles": int(s["n"]),
                        "mean": float((centers * s["counts"]).sum() / tot) if tot > 0 else math.nan})
        return out
