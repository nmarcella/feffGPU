"""The Triton kernels of the radial solver against the torch code that they replace on the GPU.

A kernel is compiled by whatever Triton is installed, so a new compiler can change what a kernel computes without a
line of this package changing. That happened: Triton 3.8 compiles a group of loop variables that start as the same
value and are shifted inside the loop (d0 = d1, d1 = d2, ...) so that they keep their start value, where Triton 3.1
compiled the loop as written. The outward kernel used that pattern for its Runge-Kutta start; its results were then
off by a few percent in the wave function, which mostly cancels in the phase shifts but moved the Fermi level of the
self-consistent potentials by about 3e-5 hartree (up to 8e-4). The kernel no longer uses the pattern. The kernels are released
with a list of Triton versions they were tested with (`kernels.TESTED_TRITON`); with another version the package
compares them with the torch code once per process before it uses them (`feffgpu.xsph.kernels_usable`), and a failed
comparison stops the calculation: it does not change to the slower code on its own. These tests keep all of that in
place, and `test_kernels_reproduce_the_torch_code` is the test to run before a version is added to the list."""
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def test_kernels_are_not_used_on_the_cpu():
    from feffgpu import xsph
    assert xsph.kernels_usable(torch.zeros(3, dtype=torch.float64)) is False


@pytest.mark.gpu
@needs_gpu
def test_kernels_reproduce_the_torch_code():
    pytest.importorskip("triton")
    from feffgpu.xsph import kernel_difference
    outward, inward = kernel_difference("cuda")
    assert outward < 1e-11 and inward < 1e-11, (outward, inward)


FRESH = {"checked": False, "usable": False, "note": ""}


@pytest.mark.gpu
@needs_gpu
def test_a_tested_triton_version_is_used_without_a_test(monkeypatch):
    """With a Triton version of `kernels.TESTED_TRITON` nothing is compared at run time."""
    triton = pytest.importorskip("triton")
    from feffgpu import kernels, xsph

    def must_not_run(dev="cuda"):
        raise AssertionError("the comparison ran although the Triton version is a tested one")

    monkeypatch.setattr(xsph, "USE_KERNELS", True)
    monkeypatch.setattr(xsph, "_KERNELS", dict(FRESH))
    monkeypatch.setattr(kernels, "TESTED_TRITON", (triton.__version__,))
    monkeypatch.setattr(xsph, "kernel_difference", must_not_run)
    assert xsph.kernels_usable(torch.zeros(3, dtype=torch.float64, device="cuda")) is True
    assert "tested with" in xsph.kernel_status()["note"]


@pytest.mark.gpu
@needs_gpu
def test_another_triton_version_is_tested_once_per_process(monkeypatch):
    pytest.importorskip("triton")
    from feffgpu import kernels, xsph
    calls = []
    real = xsph.kernel_difference
    monkeypatch.setattr(xsph, "USE_KERNELS", True)
    monkeypatch.setattr(xsph, "_KERNELS", dict(FRESH))
    monkeypatch.setattr(kernels, "TESTED_TRITON", ("0.0.0",))
    monkeypatch.setattr(xsph, "kernel_difference", lambda dev="cuda": calls.append(dev) or real(dev))
    t = torch.zeros(3, dtype=torch.float64, device="cuda")
    assert xsph.kernels_usable(t) is True and xsph.kernels_usable(t) is True
    assert len(calls) == 1
    assert "tested now" in xsph.kernel_status()["note"]


@pytest.mark.gpu
@needs_gpu
def test_the_installed_triton_passes_the_test_on_request():
    """What `feffgpu doctor` runs: the comparison itself, whatever the Triton version."""
    pytest.importorskip("triton")
    from feffgpu import xsph
    if not xsph.USE_KERNELS:
        pytest.skip("the kernels are switched off (FEFFGPU_KERNELS=0)")
    status = xsph.kernel_status("cuda", test=True)
    assert status["usable"] is True and "reproduce the plain code" in status["note"], status


@pytest.mark.gpu
@needs_gpu
def test_a_failed_test_stops_the_calculation(monkeypatch):
    """No automatic change to the slower code: kernels that do not reproduce the plain code stop the calculation
    with a message that says what to do, at every call, and the comparison itself runs once."""
    pytest.importorskip("triton")
    from feffgpu import kernels, xsph
    calls = []
    monkeypatch.setattr(xsph, "USE_KERNELS", True)
    monkeypatch.setattr(xsph, "_KERNELS", dict(FRESH))
    monkeypatch.setattr(kernels, "TESTED_TRITON", ("0.0.0",))
    monkeypatch.setattr(xsph, "kernel_difference", lambda dev="cuda": calls.append(dev) or (3.0e-2, 1.0e-15))
    t = torch.zeros(3, dtype=torch.float64, device="cuda")
    for _ in range(2):
        with pytest.raises(xsph.KernelCheckError, match="do not reproduce the plain code") as err:
            xsph.kernels_usable(t)
        assert "FEFFGPU_KERNELS=0" in str(err.value)
    assert len(calls) == 1
    assert xsph.kernel_status("cuda")["usable"] is False                                            # the report does not raise
    monkeypatch.setattr(xsph, "USE_KERNELS", False)                                                 # the deliberate switch
    assert xsph.kernels_usable(t) is False


@pytest.mark.gpu
@pytest.mark.slow
@needs_gpu
def test_potentials_are_the_same_with_and_without_the_kernels(monkeypatch):
    """Self-consistent potentials of a stored case on the GPU, with the kernels and with the torch code: the Fermi
    level and the interstitial potential agree to rounding, and both agree with FEFF's own pot.pad."""
    pytest.importorskip("triton")
    from feffgpu import xsph
    from feffgpu.chain import run
    from feffgpu.pad import read_pot_pad
    case = ROOT / "data/validation/A014"
    ref = read_pot_pad(case / "pot.pad")
    monkeypatch.setattr(xsph, "USE_KERNELS", True)
    with_kernels = run(case / "feff.inp", device="cuda")["pot"]
    monkeypatch.setattr(xsph, "USE_KERNELS", False)
    without = run(case / "feff.inp", device="cuda")["pot"]
    assert abs(with_kernels.xmu - without.xmu) < 1e-10, (with_kernels.xmu, without.xmu)
    assert abs(with_kernels.vint - without.vint) < 1e-10
    assert abs(with_kernels.xmu - ref.xmu) < 2e-6, with_kernels.xmu - ref.xmu
