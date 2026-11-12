"""The free-atom table that ships with the package: complete, readable, and packed without loss."""
import io
import lzma
import numpy as np
import pytest


def test_the_packaged_table_is_complete_and_readable(monkeypatch):
    monkeypatch.delenv("FEFFGPU_ATOMS", raising=False)
    from feffgpu import atoms_table as T
    st = T.table_status()
    assert st == {"files": 99, "complete": True}, (st, T.table_dir())
    w = T.load_element(74)
    assert 0 in set(int(h) for h in w["holes"]) and 1 in set(int(h) for h in w["holes"])       # W as a neighbor and with a K hole
    assert w["h1_dgc"].dtype == np.float64 and w["h1_dgc"].shape[0] == 251
    assert T.load_element(120) is None and T.element_file(74).endswith("Z074.npz.xz")


def test_the_potentials_stage_reads_it(monkeypatch):
    monkeypatch.delenv("FEFFGPU_ATOMS", raising=False)
    from feffgpu import pot
    a = pot._from_table(74, 1, 0)
    assert a is not None and a["norb"] > 0 and a["rho"].shape == (251,)
    assert pot._from_table(74, 40, 0) is None                                                   # a hole that the element does not have


def test_packing_keeps_every_array_and_a_plain_file_takes_precedence(tmp_path):
    from feffgpu import atoms_table as T
    rng = np.random.default_rng(0)
    arrays = {"holes": np.array([0, 1]), "h0_rho": rng.normal(size=(251,)), "h1_dgc": rng.normal(size=(251, 30)), "h1_scalars": np.array([np.nan, 1e-310, -0.0, 4.0])}
    src, dst = tmp_path / "plain", tmp_path / "packed"
    src.mkdir()
    np.savez_compressed(src / "Z005.npz", **arrays)
    T.pack_table(src, dst, quiet=True)
    back = np.load(io.BytesIO(lzma.decompress((dst / "Z005.npz.xz").read_bytes())))
    assert all(back[k].tobytes() == arrays[k].tobytes() and back[k].dtype == arrays[k].dtype for k in arrays)
    assert T.table_status(dst) == {"files": 1, "complete": False}
    assert T.load_element(5, str(dst))["h1_dgc"].shape == (251, 30)
    np.savez(dst / "Z005.npz", holes=np.array([7]))                                               # a table you built yourself
    assert list(T.load_element(5, str(dst))["holes"]) == [7]
    with pytest.raises(SystemExit):
        T.pack_table(tmp_path, dst, quiet=True)
