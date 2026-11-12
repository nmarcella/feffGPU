"""EvAX's rules as implemented in feffgpu.rmc.rules: the selection and the replacement of a bad child, the crossover by
lattice site, the spectrum weights, and the acceptance with its cooling. CPU only."""
import math
import numpy as np
import pytest
import torch
from rmc_helpers import fluorite_toy
from feffgpu.rmc import Box, EdgeWeights, Annealing, evax_selection, site_crossover, fluorite_sites, exchange_atoms, hop_atoms
from feffgpu.rmc.moves import HopTables

DEV = "cpu"


def boxed_parents(seed, copies=2):
    """Two parents (C, n, 3) of the fluorite toy (16 Zn, 16 Cu, 56 C on 64 anion sites), every atom within 0.4 A of its site."""
    sym, pos, cell = fluorite_toy(0, 8)
    sites = fluorite_sites(sym, pos, cell, "C")
    ideal = torch.tensor(np.repeat(pos[None], copies, 0), dtype=torch.float32)
    box = Box(sites.positions, 0.4, cell, DEV)
    site = torch.tensor(sites.start_site_of)[None].repeat(copies, 1)
    gen = torch.Generator()
    gen.manual_seed(seed)
    A_ = box.project(ideal + 0.12 * torch.randn(ideal.shape, generator=gen), site)
    B_ = box.project(ideal + 0.12 * torch.randn(ideal.shape, generator=gen), site)
    species = torch.tensor([sorted(set(sym)).index(s) for s in sym])
    return sym, sites, box, gen, A_, B_, site, species, cell


def tables(sites):
    return torch.tensor(sites.positions, dtype=torch.float32), torch.tensor(sites.ids), sites.n_ids


def test_site_crossover_is_evaxs_atom_rule_when_the_arrangements_are_equal():
    sym, sites, box, gen, A_, B_, site, species, cell = boxed_parents(0)
    C, n, _ = A_.shape
    pos_t, ids_t, n_ids = tables(sites)
    mask = torch.rand(C, n_ids, generator=gen) > 0.5
    c1, c2, taken = site_crossover(A_, B_, site, site, pos_t, ids_t, n_ids, species, cell=box.cell, mask=mask)
    ids = ids_t[site]
    for c in range(C):
        assert len(set(ids[c].tolist())) == n                                             # one atom per site
    m = torch.gather(mask, 1, ids)                                                        # the draw per site is a draw per atom
    assert torch.allclose(c1, torch.where(m[..., None], B_, A_), atol=1e-5)               # EvAX: child 1 takes parent 2's displacement of the drawn atoms
    assert torch.allclose(c2, torch.where(m[..., None], A_, B_), atol=1e-5)               # and child 2 parent 1's, of the same atoms
    assert taken == int(m.sum()) and 0.3 * C * n < taken < 0.7 * C * n
    assert box.at_limit(c1, site)[1] <= 0.4 + 1e-4 and box.at_limit(c2, site)[1] <= 0.4 + 1e-4   # the children are inside the limit


def test_site_crossover_stays_valid_when_the_arrangements_differ():
    sym, sites, box, gen, A_, B_, site, species, cell = boxed_parents(1)
    C, n, _ = A_.shape
    pos_t, ids_t, n_ids = tables(sites)
    Bp, sB = B_[None].clone(), site[None].clone()                                         # parent 2 gets exchanges and hops: another arrangement
    every = torch.ones(1, dtype=torch.bool)
    zn, cu = torch.tensor(np.nonzero(sym == "Zn")[0]), torch.tensor(np.nonzero(sym == "Cu")[0])
    assert exchange_atoms(Bp, sB, every, zn, cu, 4, gen) == 4 * C
    hops, skipped = hop_atoms(Bp, sB, every, HopTables(sites, DEV), 4, 0.05, gen)
    assert hops == 4 * C and skipped == 0
    B_, sB = box.project(Bp[0], sB[0]), sB[0]
    assert not torch.equal(ids_t[site], ids_t[sB])
    c1, c2, taken = site_crossover(A_, B_, site, sB, pos_t, ids_t, n_ids, species, cell=box.cell, gen=gen)
    for child, s in ((c1, site), (c2, sB)):
        assert box.at_limit(child, s)[1] <= 0.4 + 1e-4                                    # inside the limit, on its own parent's arrangement
    from feffgpu.rmc.constraints import nearest_sites
    dA, dB = A_ - nearest_sites(pos_t, box.cell, A_, site), B_ - nearest_sites(pos_t, box.cell, B_, sB)
    d1, d2 = c1 - nearest_sites(pos_t, box.cell, c1, site), c2 - nearest_sites(pos_t, box.cell, c2, sB)
    sp = species.tolist()

    def check(d_child, d_own, s_own, d_other, s_other):
        took = unlike = 0
        for c in range(C):
            at = {int(s_): j for j, s_ in enumerate(s_other[c].tolist())}
            for i in range(n):
                j = at.get(int(s_own[c, i]))
                own = bool(torch.allclose(d_child[c, i], d_own[c, i], atol=1e-5))
                other = j is not None and bool(torch.allclose(d_child[c, i], d_other[c, j], atol=1e-5))
                assert own or other                                                       # its own displacement, or that of the other parent's atom on the same site
                if j is None or sp[j] != sp[i]:
                    assert own                                                            # a site vacant or of another element in the other parent is not exchanged
                    unlike += 1
                took += int(other and not own)
        return took, unlike

    took1, unlike1 = check(d1, dA, site, dB, sB)
    took2, unlike2 = check(d2, dB, sB, dA, site)
    assert took1 == taken and took1 > 0.3 * C * n and took2 > 0.3 * C * n
    assert unlike1 > 0 and unlike2 > 0                                                    # the test had such sites


def test_selection_is_evaxs_tournament_and_a_bad_child_gives_way_to_its_unbred_parent():
    P = 8
    pop = torch.arange(P, dtype=torch.float32)[:, None].repeat(1, 3)                      # state i is the vector (i, i, i)
    M = torch.tensor([0.5, 0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4])
    picks = torch.tensor([[0, 2, 4, 0], [1, 3, 5, 2], [1, 4, 6, 6], [1, 5, 7, 6]])        # pair iv: parent 1 from picks[0:2, iv], parent 2 from picks[2:4, iv]
    rho = torch.tensor([0.9, 0.9, 0.2, 0.7])                                              # pairs 0, 1, 3 may breed; pair 0 has one parent twice
    calls, seen = [], []

    def crossover(i, j):
        calls.append((i, j))
        return pop[i] + 100, pop[j] + 200

    def bad(kids, parents):
        seen.append(list(parents))
        return kids[:, 0] == 100

    parents, new, kept, st = evax_selection(pop, M, None, crossover, bad, picks=picks, rho=rho)
    assert parents.tolist() == [1, 1, 3, 5, 5, 7, 0, 6]                                   # the better of each two picks
    assert calls == [(3, 5), (0, 6)]                                                      # the same parent twice does not breed; rho 0.2 does not breed
    assert seen == [[3, 5, 0, 6]]                                                         # every child with the parent whose arrangement it keeps
    # slot 6: the child (100) is bad -> the unbred selected parent (state 0), not the slot's previous occupant (state 6)
    assert new[:, 0].tolist() == [1, 1, 103, 205, 5, 7, 0, 206]
    assert kept == [2, 3, 7]
    assert st == {"bred_pairs": 2, "children": 4, "children_rejected": 1, "distinct_parents": 6}
    assert pop[:, 0].tolist() == list(range(P))                                           # the picks read the old population; it is not changed
    parents, _, _, _ = evax_selection(pop[:2], torch.tensor([1.0, 1.0]), None, crossover, lambda kids, par: kids[:, 0] < 0,
                                      picks=torch.tensor([[0], [1], [1], [0]]), rho=torch.tensor([0.0]))
    assert parents.tolist() == [0, 1]                                                     # a tie goes to the first pick
    with pytest.raises(ValueError):
        evax_selection(pop[:3], M[:3], None, crossover, bad)


def test_spectrum_weights_follow_evaxs_rule():
    w = EdgeWeights(2)
    assert w.w.tolist() == [0.5, 0.5]
    rng = np.random.default_rng(0)
    d_all = []
    for _ in range(3):                                                                    # three generations of 6 moves each
        R0 = torch.tensor(rng.uniform(0.01, 0.05, (6, 2)))
        d = rng.normal(0, 1, (6, 2)) * np.array([0.004, 0.001])                           # the first spectrum changes four times as much
        w.update(R0 + torch.tensor(d), R0)
        d_all.append(np.abs(d))
    avg = np.concatenate(d_all).mean(0)                                                   # the mean over all proposed moves
    expect = (1 / avg) / (1 / avg).sum()
    assert np.allclose(w.avg.numpy(), avg, rtol=1e-9) and w.num == 18
    assert np.allclose(w.w.numpy(), expect, rtol=1e-9) and abs(float(w.w.sum()) - 1) < 1e-12
    assert w.w[1] > w.w[0]                                                                # the spectrum that changes less weighs more
    st = w.state()                                                                        # the saved state continues the same sequence
    w2 = EdgeWeights(2, weights=st["weights"], average_delta=st["average_delta"], num=st["num"])
    R0 = torch.tensor(rng.uniform(0.01, 0.05, (6, 2)))
    R1 = R0 + 0.002
    assert w.update(R1, R0) == w2.update(R1, R0) and torch.allclose(w.w, w2.w, rtol=1e-12) and w2.num == 24
    kept = w.w.clone()                                                                    # a restart: the weights stay until the next update, which takes its own moves only
    w.restart()
    assert w.avg is None and w.num == 0 and torch.equal(w.w, kept)
    R0 = torch.tensor(rng.uniform(0.01, 0.05, (6, 2)))
    d = rng.normal(0, 1, (6, 2)) * np.array([0.001, 0.003])
    w.update(R0 + torch.tensor(d), R0)
    avg = np.abs(d).mean(0)
    assert w.num == 6 and np.allclose(w.avg.numpy(), avg, rtol=1e-9) and np.allclose(w.w.numpy(), (1 / avg) / (1 / avg).sum(), rtol=1e-9)
    valid = torch.tensor([True, False, True, True, False, True])                          # moves refused by a constraint do not count
    R2 = R0 + torch.tensor(rng.normal(0, 0.002, (6, 2)))
    w3 = EdgeWeights(2)
    w3.update(R2, R0, valid)
    assert w3.num == 4 and np.allclose(w3.avg.numpy(), (R2 - R0).abs()[valid].mean(0).numpy())
    one = EdgeWeights(1)                                                                  # one spectrum keeps the weight 1
    assert one.update(torch.tensor([[0.02], [0.03]]), torch.tensor([[0.025], [0.02]])) is False and one.w.tolist() == [1.0]


def test_acceptance_is_evaxs_rule_with_a_linear_cooling():
    an = Annealing(0.8, 500)
    assert an.target(1) == pytest.approx(0.8 * (1 - 1 / 500)) and an.target(250) == pytest.approx(0.4) and an.target(499) == pytest.approx(0.8 / 500)
    assert an.target(500) == 0.0 and an.target(600) == 0.0
    assert an.logp(250) == pytest.approx(math.log(0.4)) and an.logp(500) == -math.inf
    assert Annealing(0.0, 500).logp(1) == -math.inf and Annealing(-1.0, 500).target(1) == 0.0   # acceptance <= 0: worse moves never
    delta = torch.tensor([-0.2, 0.0, 0.1, 0.1, 0.3], dtype=torch.float64)
    rho = torch.tensor([0.99, 0.99, 0.3, 0.6, 0.01], dtype=torch.float64)
    acc = an.accept(delta, 250, rho)
    mean = float(delta.abs().mean())                                                      # the first generation's own mean
    assert an.average_delta == pytest.approx(mean) and an.num == 5
    expect = [True, True] + [0.4 ** (float(d) / mean) >= float(r) for d, r in zip(delta[2:], rho[2:])]
    assert expect == [True, True, True, False, True] and acc.tolist() == expect
    delta2 = torch.tensor([0.5, -0.1, 0.02], dtype=torch.float64)                         # the running mean is over all proposed moves
    an.accept(delta2, 251, torch.ones(3, dtype=torch.float64))
    assert an.average_delta == pytest.approx(float(torch.cat([delta, delta2]).abs().mean())) and an.num == 8
    assert an.accept(torch.tensor([-1e-3, 0.0, 1e-9, 0.5], dtype=torch.float64), 500, torch.zeros(4, dtype=torch.float64)).tolist() == [True, True, False, False]
    gen = torch.Generator()                                                               # a worse move of mean size is accepted with the target probability
    gen.manual_seed(0)
    an2 = Annealing(0.8, 500)
    frac = float(an2.accept(torch.full((20000,), 0.01, dtype=torch.float64), 100, torch.rand(20000, generator=gen, dtype=torch.float64)).double().mean())
    assert abs(frac - 0.8 * (1 - 100 / 500)) < 0.015
    an4 = Annealing(0.8, 500)                                                             # a restart forgets a large early change; the count is not changed
    an4.accept(torch.full((16,), -27.0, dtype=torch.float64), 1, torch.ones(16, dtype=torch.float64))
    small, rho = torch.full((16,), 0.001, dtype=torch.float64), torch.full((16,), 0.5, dtype=torch.float64)
    assert an4.accept(small, 2, rho).all() and an4.average_delta == pytest.approx((27.0 + 0.001) / 2)
    an4.restart()
    assert an4.average_delta is None and an4.num == 0 and an4.target(250) == pytest.approx(0.4)
    mixed = torch.tensor([0.001] * 15 + [0.004], dtype=torch.float64)
    acc = an4.accept(mixed, 250, rho)
    assert an4.num == 16 and an4.average_delta == pytest.approx(float(mixed.mean()))
    assert acc.tolist() == [0.4 ** (float(d) / float(mixed.mean())) >= 0.5 for d in mixed] and acc[:15].sum() == 0 and not acc[15]
    an5 = Annealing(0.8, 500)                                                             # refused moves are not accepted and stay out of the mean
    valid = torch.tensor([True, False, True])
    acc = an5.accept(torch.tensor([-0.1, -0.2, 0.3], dtype=torch.float64), 10, torch.zeros(3, dtype=torch.float64), valid)
    assert acc.tolist() == [True, False, True] and an5.num == 2 and an5.average_delta == pytest.approx(0.2)
    st = an2.state()                                                                      # the saved state continues
    an3 = Annealing(st["acceptance"], st["froze_in"], st["average_delta"], st["num"])
    assert an3.average_delta == an2.average_delta and an3.num == 20000 and an3.target(100) == an2.target(100)
