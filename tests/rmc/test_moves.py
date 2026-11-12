"""The moves on general structures: exchanges of two elements on any structure, hops onto vacant sites of a site list
(the built-in fluorite generator and a file of sites), the start population, EvAX's uniform step, FIRE. CPU only."""
import numpy as np
import pytest
import torch
from rmc_helpers import fluorite_toy, bcc
from feffgpu.rmc import (ModelCell, fluorite_sites, sublattice_sites, read_site_file, exchange_atoms, hop_atoms, start_population, uniform_step,
                         fire_relax, ArrangementStatistics)
from feffgpu.rmc.moves import HopTables

DEV = "cpu"


def check_occupancy(sites, so, element_atoms):
    """One atom per site, every atom of the hop element on a site of the sublattice."""
    for row in so.reshape(-1, so.shape[-1]):
        row = row.tolist()
        assert len(set(row)) == len(row)
        assert set(row[i] for i in element_atoms) <= set(sites.hop_sites.tolist())


def test_hops_keep_the_occupancy_consistent():
    sym, pos, cell = fluorite_toy(0, 8)
    sites = fluorite_sites(sym, pos, cell, "C")
    c_atoms = np.nonzero(sym == "C")[0]
    P = 3
    p = torch.tensor(np.repeat(pos[None, None], P, 0), dtype=torch.float32)
    so = torch.tensor(sites.start_site_of)[None, None].repeat(P, 1, 1)
    gen = torch.Generator()
    gen.manual_seed(0)
    hops = HopTables(sites, DEV)
    total = 0
    for _ in range(50):
        done, skipped = hop_atoms(p, so, torch.ones(P, dtype=torch.bool), hops, 2, 0.05, gen)
        total += done
        check_occupancy(sites, so.numpy(), c_atoms)
        for q in range(P):
            assert len(sites.vacant_hop_sites(so[q, 0].numpy())) == 8                    # the number of vacancies never changes
            rel = p[q, 0].numpy()[c_atoms] - sites.positions[so[q, 0].numpy()[c_atoms]]
            d = np.linalg.norm(rel - 11.0 * np.round(rel / 11.0), axis=-1)                # minimum image: a site and its atom may be images
            assert d.max() < 0.5                                                          # every hopped atom sits at its new site (plus the jitter)
    assert total == 50 * P * 2
    # the neighbors of a fluorite anion site: its six nearest anion sites
    nb = sites.positions[sites.hop_sites][sites.hop_neighbors[0]] - sites.positions[sites.hop_sites][0]
    d = np.linalg.norm(nb - 11.0 * np.round(nb / 11.0), axis=-1)
    assert np.allclose(d, 2.75)


def test_exchanges_on_any_structure():
    sym, pos, cell = bcc(3, 3.2)
    sym = np.where(np.arange(len(sym)) % 3 == 0, "Mo", "W")                               # a bcc alloy: exchanges need no special lattice
    model = ModelCell(sym, pos, cell)
    P = 2
    p = torch.tensor(np.repeat(pos[None, None], P, 0), dtype=torch.float32)
    so = torch.tensor(model.sites.start_site_of)[None, None].repeat(P, 1, 1)
    gen = torch.Generator()
    gen.manual_seed(1)
    ia, ib = torch.tensor(np.nonzero(sym == "W")[0]), torch.tensor(np.nonzero(sym == "Mo")[0])
    assert exchange_atoms(p, so, torch.tensor([True, False]), ia, ib, 5, gen) == 5
    assert torch.equal(p[1], torch.tensor(pos[None], dtype=torch.float32))               # the state not flagged is untouched
    occ = model.sites.occupancy(so[0, 0].numpy())
    elem = sym[occ]
    assert (elem == "Mo").sum() == (sym == "Mo").sum() and (elem != sym).sum() > 0        # the composition is kept, the arrangement changed
    assert np.allclose(p[0, 0].numpy(), model.sites.positions[so[0, 0].numpy()], atol=1e-6)   # every atom on its (new) site
    stats = ArrangementStatistics(model, ("W", "Mo"))
    s0, s1 = stats(model.sites.start_site_of[None]), stats(so[0].numpy())
    assert s0["atoms_off_start_site"] == 0 and s1["atoms_off_start_site"] > 0 and s1["sites_changed"] == s1["atoms_off_start_site"]


def test_hops_from_a_file_of_sites_on_a_structure_without_a_generator(tmp_path):
    """A simple cubic metal with C atoms on some of its octahedral sites: the octahedral sites come from a file."""
    a, L = 3.0, 3
    grid = np.stack(np.meshgrid(*[np.arange(L)] * 3, indexing="ij"), -1).reshape(-1, 3) * a
    octa = grid + np.array([a / 2, a / 2, a / 2])
    np.savetxt(tmp_path / "sites.txt", octa, header="x y z (A)")
    rng = np.random.default_rng(2)
    filled = np.sort(rng.choice(len(octa), 10, replace=False))
    sym = np.array(["Fe"] * len(grid) + ["C"] * len(filled))
    pos = np.concatenate([grid, octa[filled] + rng.normal(0, 0.05, (len(filled), 3))])
    cell = np.eye(3) * a * L
    sites = sublattice_sites(sym, pos, cell, "C", read_site_file(tmp_path / "sites.txt"))
    assert len(sites.hop_sites) == len(octa) and len(sites.vacant_hop_sites(sites.start_site_of)) == len(octa) - len(filled)
    assert (sites.hop_neighbors >= 0).sum(1).min() == 6                                   # the nearest octahedral sites, at a
    p = torch.tensor(pos[None, None], dtype=torch.float32)
    so = torch.tensor(sites.start_site_of)[None, None]
    gen = torch.Generator()
    gen.manual_seed(3)
    done, skipped = hop_atoms(p, so, torch.tensor([True]), HopTables(sites, DEV), 4, 0.0, gen)
    assert done + skipped == 4 and done > 0
    check_occupancy(sites, so.numpy(), np.nonzero(sym == "C")[0])
    model = ModelCell(sym, pos, cell, sites=sites)
    st = ArrangementStatistics(model)(so[0].numpy())
    assert st["vacancies"] == len(octa) - len(filled) and st["vacancies_moved"] > 0


def test_start_population_and_the_uniform_step():
    base = torch.zeros(1, 50, 3)
    movable = torch.ones(50)
    movable[:5] = 0                                                                       # atoms no edge sees do not move
    gen = torch.Generator()
    gen.manual_seed(4)
    g = start_population(base, 6, 0.1, movable, gen, "gaussian")
    u = start_population(base, 6, 0.1, movable, gen, "uniform")
    assert g.shape == (6, 1, 50, 3) and float(g[:, :, :5].abs().max()) == 0.0 and float(u[:, :, :5].abs().max()) == 0.0
    assert 0.08 < float(g[:, :, 5:].std()) < 0.12
    assert float(u.abs().max()) <= 0.1 / np.sqrt(2) + 1e-7                                # EvAX's start: uniform within +- d / sqrt(2)
    s = uniform_step(base[None].repeat(4, 1, 1, 1), 0.005, movable, gen)
    assert float(s.abs().max()) <= 0.005 and float(s[:, :, :5].abs().max()) == 0.0 and float(s.abs().mean()) > 0.002
    with pytest.raises(ValueError):
        start_population(base, 2, 0.1, movable, gen, "other")


class Wells:
    """Harmonic wells around fixed centers: E = 0.5 k |x - c|^2 per atom."""

    def __init__(self, centers, k=2.0):
        self.c, self.k = centers, k

    def energy_forces(self, pos):
        d = pos - self.c
        return 0.5 * self.k * (d ** 2).sum((1, 2)), -self.k * d


def test_fire_minimizes_and_keeps_the_projection():
    gen = torch.Generator()
    gen.manual_seed(5)
    c = torch.randn(3, 20, 3, generator=gen)
    x0 = c + 0.5 * torch.randn(3, 20, 3, generator=gen)
    x, E, F, took = fire_relax(Wells(c), x0, 200, fmax=1e-4)
    assert took < 200 and float((x - c).abs().max()) < 1e-3
    home = x0.clone()
    proj = lambda p: home + (p - home) * torch.clamp(0.1 / (p - home).norm(dim=-1, keepdim=True).clamp(min=1e-12), max=1.0)   # noqa: E731
    x, E, F, took = fire_relax(Wells(c), x0, 50, project=proj)
    assert float((x - home).norm(dim=-1).max()) <= 0.1 + 1e-5                             # the constraint holds after every step
    assert float(Wells(c).energy_forces(x)[0].sum()) < float(Wells(c).energy_forces(x0)[0].sum())
