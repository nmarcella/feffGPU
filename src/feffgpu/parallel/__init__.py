"""Devices, shards and random streams: the access points for several GPUs.

Today every run uses one GPU. The pieces that a multi-GPU run needs are already here and are used by the single-GPU
code, so that adding a pool of worker processes later changes no result:

- ``resolve_devices`` turns the ``devices`` setting of a configuration (``"all"``, a count, a list of indices, or a
  device string) into a list of device strings, and ``primary_device`` picks the one that is used today.
- ``Shard`` splits a list of independent work items (frames of a trajectory, inputs) into ``count`` parts in a fixed
  way, so that a run can be divided over the jobs of a scheduler array, each on its own GPU, and merged afterwards.
- ``keyed_rng`` and ``keyed_generator`` give random streams that depend only on the seed and on the keys of the work
  item (for example the frame number), never on the order in which items are computed. Results therefore do not
  depend on the number of shards or workers.
- ``run_items`` is the one place where work items are executed. It runs them in order on the primary device; a pool
  with one worker process per GPU will replace its body.
"""
from __future__ import annotations
import hashlib
import warnings
from dataclasses import dataclass
import numpy as np
import torch

__all__ = ["resolve_devices", "primary_device", "Shard", "keyed_seed", "keyed_rng", "keyed_generator", "run_items"]


def resolve_devices(spec="cuda"):
    """The device strings that a ``devices`` setting names.

    Parameters
    ----------
    spec : str, int or list of int
        ``"all"``: every visible CUDA device (``CUDA_VISIBLE_DEVICES`` is respected, because torch only sees those);
        an integer n: the first n visible devices; a list of indices: those devices; a device string such as
        ``"cuda"``, ``"cuda:1"`` or ``"cpu"``: that device.

    Returns
    -------
    list of str
        At least one entry. Falls back to ``["cpu"]`` with a warning when CUDA is asked for and not available.
    """
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if isinstance(spec, str) and spec not in ("all",):
        if spec.startswith("cuda") and n_gpu == 0:
            warnings.warn("CUDA is not available: running on the CPU, which is slow (feffgpu is written for NVIDIA GPUs)")
            return ["cpu"]
        return [spec]
    if n_gpu == 0:
        warnings.warn("CUDA is not available: running on the CPU, which is slow (feffgpu is written for NVIDIA GPUs)")
        return ["cpu"]
    if spec == "all":
        idx = list(range(n_gpu))
    elif isinstance(spec, int):
        if spec < 1:
            raise ValueError("devices must name at least one device")
        if spec > n_gpu:
            raise ValueError(f"devices = {spec}, but only {n_gpu} CUDA device(s) are visible")
        idx = list(range(spec))
    else:
        idx = [int(i) for i in spec]
        bad = [i for i in idx if i < 0 or i >= n_gpu]
        if bad or not idx:
            raise ValueError(f"devices = {list(spec)}: valid indices are 0 to {n_gpu - 1}")
    return [f"cuda:{i}" for i in idx]


def primary_device(spec="cuda", quiet=False):
    """The device that a run uses today: the first of ``resolve_devices(spec)``.

    When the setting names more than one device, a notice says that the run uses one (a pool over several GPUs is
    not implemented yet; split the work with shards instead)."""
    devices = resolve_devices(spec)
    if len(devices) > 1 and not quiet:
        print(f"feffgpu: {len(devices)} devices requested; this version runs on one ({devices[0]}). To use several GPUs, split the run "
              "into shards (one job per GPU) and merge the results.", flush=True)
    return devices[0]


@dataclass(frozen=True)
class Shard:
    """Part ``index`` (from 1) of ``count`` parts of a list of independent work items."""
    index: int = 1
    count: int = 1

    def __post_init__(self):
        if self.count < 1 or not (1 <= self.index <= self.count):
            raise ValueError(f"shard {self.index}/{self.count}: the index runs from 1 to the count")

    @classmethod
    def parse(cls, text):
        """``"3/16"`` -> ``Shard(3, 16)``; ``None`` or ``""`` -> the whole run."""
        if not text:
            return cls()
        try:
            i, n = str(text).split("/")
            return cls(int(i), int(n))
        except ValueError as err:
            raise ValueError(f"a shard is written INDEX/COUNT, for example 3/16 (got {text!r})") from err

    def select(self, items):
        """The items of this shard: every ``count``-th item starting at ``index - 1``. The union over all shards is
        the full list and no item is in two shards."""
        return list(items)[self.index - 1::self.count]

    def __str__(self):
        return f"{self.index}/{self.count}"


def keyed_seed(seed, *keys):
    """A 63-bit seed that depends only on ``seed`` and ``keys`` (integers or strings)."""
    h = hashlib.blake2b(repr((int(seed),) + tuple(keys)).encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") >> 1


def keyed_rng(seed, *keys):
    """A numpy random generator for one work item, independent of the order in which items are computed."""
    return np.random.default_rng(keyed_seed(seed, *keys))


def keyed_generator(seed, *keys, device="cpu"):
    """A torch random generator for one work item, independent of the order in which items are computed."""
    gen = torch.Generator(device=device)
    gen.manual_seed(keyed_seed(seed, *keys))
    return gen


def run_items(function, items, devices="cuda", quiet=False):
    """Run ``function(item, device)`` for every item and return the results in the order of ``items``.

    This is the access point for several GPUs. Today the items run one after the other on the primary device, with
    no overhead; a pool with one worker process per device will replace this body. ``function`` must not depend on
    the order of execution (use ``keyed_rng`` for random numbers)."""
    device = primary_device(devices, quiet=quiet)
    return [function(item, device) for item in items]
