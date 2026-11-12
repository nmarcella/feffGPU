"""The batched linear solve of the full multiple-scattering matrix (feffgpu.fms.solve_batch).

In some PyTorch builds, changing the number of threads (torch.set_num_threads) before the first linear-algebra call of
a process makes the batched LU factorization on the CPU return invalid pivots, and a batched solve raises. solve_batch
must still return the correct solution. The check runs in a fresh process, because the thread count must be changed
before the first linear-algebra call, and because no test may change it inside the test process."""
import subprocess
import sys
import torch

CODE = """
import torch
torch.set_num_threads(max(1, torch.get_num_threads() // 2))
from feffgpu.fms import solve_batch
g = torch.Generator().manual_seed(0)
a = torch.randn(17, 279, 279, dtype=torch.complex128, generator=g)
b = torch.randn(17, 279, 9, dtype=torch.complex128, generator=g)
x = solve_batch(a, b)
print("RESIDUAL", float((a @ x - b).abs().max()))
"""


def test_batched_solve_is_correct_after_a_thread_count_change():
    out = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-1500:]
    assert float(out.stdout.split("RESIDUAL")[1]) < 1e-9, out.stdout[-300:]


def test_batched_solve_equals_one_matrix_at_a_time_and_reports_a_singular_matrix():
    from feffgpu.fms import solve_batch
    g = torch.Generator().manual_seed(1)
    a = torch.randn(5, 40, 40, dtype=torch.complex128, generator=g)
    b = torch.randn(5, 40, 3, dtype=torch.complex128, generator=g)
    x = solve_batch(a, b)
    one = torch.stack([torch.linalg.solve(a[i], b[i]) for i in range(5)])
    assert float((x - one).abs().max()) < 1e-10 and float((a @ x - b).abs().max()) < 1e-10
    a[2] = 0.0
    try:
        solve_batch(a, b)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a singular matrix must raise")
