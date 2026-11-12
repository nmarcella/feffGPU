"""The displacement limit (EvAX's box) on the fixed start lattice, and the closest distances. CPU only.

The limit holds every atom to its site of the start lattice, which is never refitted: a shift of all atoms together
is a displacement of every atom, and no translation makes an atom jump to another site (the defect of a limit that
re-estimated the origin of the lattice from the atoms and lost its parity)."""
import numpy as np
import pytest
import torch
from rmc_helpers import fluorite_toy, bcc
from feffgpu.rmc import Box, ClosestDistances, fluorite_sites, own_sites, exchange_atoms, hop_atoms
from feffgpu.rmc.moves import HopTables

DEV = "cpu"


def toy(copies=3):
    sym, pos, cell = fluorite_toy(0, 8)
    sites = fluorite_sites(sym, pos, cell, "C")
    ideal = torch.tensor(np.repeat(pos[None], copies, 0), dtype=torch.float32)
    site = torch.tensor(sites.start_site_of)[None].repeat(copies, 1)
    return sym, cell, sites, ideal, site


def test_box_holds_every_atom_to_its_start_site():
    sym, cell, sites, ideal, site = toy()
    box = Box(sites.positions, 0.4, cell, DEV)
    gen = torch.Generator()
    gen.manual_seed(0)
    pos = ideal + (0.3 * torch.randn(ideal.shape, generator=gen)).clamp(-1.0, 1.0)
    out = box.project(pos, site)
    before, after = (pos - ideal).norm(dim=-1), (out - ideal).norm(dim=-1)
    assert float(before.max()) > 0.8                                     # the test moved atoms far beyond the limit
    assert float(after.max()) <= 0.4 + 1e-4                              # EvAX's rule: the distance from the atom's own start position
    inside = before <= 0.4
    assert torch.equal(out[inside], pos[inside])                         # atoms inside the limit are not touched at all
    assert torch.allclose(after[~inside], torch.full_like(after[~inside], 0.4), atol=1e-4)
    assert torch.allclose(box.project(out, site), out, atol=1e-5)        # one pass is enough: the lattice does not move
    frac, largest = box.at_limit(out, site)
    assert abs(frac - float((before >= 0.4 - 1e-3).float().mean())) < 2e-3 and largest <= 0.4 + 1e-4


def test_a_shift_of_all_atoms_together_counts_as_a_displacement():
    sym, cell, sites, ideal, site = toy()
    box = Box(sites.positions, 0.2, cell, DEV)
    out = box.project(ideal + torch.tensor([0.3, 0.0, 0.0]), site)       # no pair distance changed; every atom is 0.3 A from its site
    assert torch.allclose((out - ideal).mean(1), torch.tensor([0.2, 0.0, 0.0]).expand(ideal.shape[0], 3), atol=1e-4)
    gen = torch.Generator()
    gen.manual_seed(1)
    pos = ideal.clone()
    for _ in range(100):                                                 # 100 steps that each push all atoms the same way
        pos = box.project(pos + torch.tensor([0.05, 0.03, 0.0]) + 0.01 * torch.randn(pos.shape, generator=gen), site)
    assert float((pos - ideal).norm(dim=-1).max()) <= 0.2 + 1e-4
    assert box.at_limit(pos, site)[0] > 0.9


@pytest.mark.parametrize("shift", [(2.75, 0.0, 0.0), (1.5, 0.2, -0.4), (2.75, 2.75, 0.0)])
def test_no_translation_makes_an_atom_jump_to_another_site(shift):
    """A rigid translation by half the conventional cell along x (a vector between cation sites of the wrong parity) is a
    displacement of every atom by that vector; the projection moves every atom by the same vector back to the limit."""
    sym, cell, sites, ideal, site = toy(copies=2)
    box = Box(sites.positions, 0.4, cell, DEV)
    v = torch.tensor(shift)
    d = box.displacement(ideal + v, site)
    assert torch.allclose(d, v.expand_as(d), atol=1e-4)                  # the same displacement for every atom: no site was swapped
    out = box.project(ideal + v, site)
    expect = ideal + v * min(1.0, 0.4 / float(v.norm()))
    assert torch.allclose(out, expect, atol=1e-4)
    L = float(cell[0, 0])
    image = ideal + L * torch.tensor([1.0, -2.0, 0.0])                   # periodic images are the same sites
    assert float(box.displacement(image, site).abs().max()) < 1e-4
    assert torch.allclose(box.project(image, site), image)


def test_exchanges_and_hops_move_atoms_to_other_sites_of_the_same_lattice():
    sym, cell, sites, ideal, site = toy()
    box = Box(sites.positions, 0.4, cell, DEV)
    gen = torch.Generator()
    gen.manual_seed(2)
    P, (C, n, _) = 2, ideal.shape
    pop = ideal[None].repeat(P, 1, 1, 1) + 0.02 * torch.randn(P, C, n, 3, generator=gen)
    so = site[None].repeat(P, 1, 1)
    every = torch.ones(P, dtype=torch.bool)
    zn, cu = torch.tensor(np.nonzero(sym == "Zn")[0]), torch.tensor(np.nonzero(sym == "Cu")[0])
    assert exchange_atoms(pop, so, every, zn, cu, 3, gen) == P * C * 3
    hops, skipped = hop_atoms(pop, so, every, HopTables(sites, DEV), 3, 0.05, gen)
    assert hops == P * C * 3 and skipped == 0
    flat, sf = pop.view(-1, n, 3), so.view(-1, n)
    assert float((flat - ideal.repeat(P, 1, 1)).norm(dim=-1).max()) > 2.0   # atoms now sit on sites that are not their start sites
    assert float(box.displacement(flat, sf).norm(dim=-1).max()) < 0.4       # and the limit measures them from those sites
    assert torch.equal(box.project(flat, sf), flat)


def test_the_fluorite_generator_needs_the_ideal_start_and_any_start_works_without_it():
    sym, pos, cell = fluorite_toy(0, 8)
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="ideal start"):
        fluorite_sites(sym, pos + rng.normal(0, 0.9, pos.shape), cell, "C")
    with pytest.raises(ValueError, match="do not fill"):
        fluorite_sites(sym[1:], pos[1:], cell, "C")
    moved = pos + rng.normal(0, 0.2, pos.shape)                              # a relaxed start: every atom's start position is its site
    s = own_sites(moved)
    box = Box(s.positions, 0.1, cell, DEV)
    site = torch.tensor(s.start_site_of)[None]
    p = torch.tensor(moved[None], dtype=torch.float32)
    assert float(box.displacement(p, site).abs().max()) == 0.0


def test_closest_distances():
    sym, pos, cell = bcc(2, 3.0)
    sym = np.array(["W", "Mo"] * (len(sym) // 2))
    cd = ClosestDistances(sym, {"W-Mo": 2.7, "W-W": 2.5, "Fe-W": 9.0}, cell, spring=40.0, device=DEV)   # Fe is not in the structure: skipped
    assert cd.as_dict() == {"Mo-W": 2.7, "W-W": 2.5}
    p = torch.tensor(pos[None], dtype=torch.float32)
    r_nn = 3.0 * np.sqrt(3) / 2                                              # 2.598 A: W-Mo pairs below 2.7, W-W pairs at 3.0 are fine
    E, n_below = cd.energy(p)
    n_wmo = int(sum(1 for i in range(len(sym)) for j in range(i + 1, len(sym)) if {sym[i], sym[j]} == {"W", "Mo"}
                    and np.linalg.norm((pos[i] - pos[j]) - 6.0 * np.round((pos[i] - pos[j]) / 6.0)) < 2.6))
    assert n_below == n_wmo > 0
    assert float(E[0]) == pytest.approx(0.5 * 40.0 * n_wmo * (2.7 - r_nn) ** 2, rel=1e-4)
    assert float(cd.worst(p)[0]) == pytest.approx(2.7 - r_nn, abs=1e-5)
    assert ClosestDistances(sym, {}, cell, device=DEV).worst(p).tolist() == [0.0]
    with pytest.raises(ValueError):
        ClosestDistances(sym, {"WMo": 2.0}, cell, device=DEV)
