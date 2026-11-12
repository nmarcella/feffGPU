# Several GPUs and clusters

## Where the package stands

**Every run uses one GPU today.** A pool of worker processes, one per GPU, is planned and is not implemented yet.
What exists now:

- every function and command takes a device, and nothing assumes the first GPU;
- **MD-EXAFS can already use several GPUs and several nodes through shards**: independent parts of the frames,
  each computed by its own job on its own GPU, merged afterwards;
- the pieces that a pool needs are in place and are used by the single-GPU code (`feffgpu.parallel`), so that
  results will not change when the pool arrives.

## Choosing the device

Command-line tools take `--device` (`cuda`, `cuda:1`, `cpu`). Configuration files of MD-EXAFS and RMC-EXAFS have

```toml
[run]
devices = "cuda"        # or "cuda:1", or "all", or a number of GPUs, or a list of indices [0, 2]
```

`devices` may name several GPUs. A run then prints a notice and uses the first of them. `CUDA_VISIBLE_DEVICES` is
respected, because PyTorch only sees the GPUs it lists; this is the simplest way to place a job on a given GPU:

```bash
CUDA_VISIBLE_DEVICES=2 md-exafs run config.toml
```

## Shards: MD-EXAFS on several GPUs or nodes

The frames of a trajectory are independent. `--shard INDEX/COUNT` computes one part of them:

```bash
CUDA_VISIBLE_DEVICES=0 md-exafs run config.toml --shard 1/4 &
CUDA_VISIBLE_DEVICES=1 md-exafs run config.toml --shard 2/4 &
CUDA_VISIBLE_DEVICES=2 md-exafs run config.toml --shard 3/4 &
CUDA_VISIBLE_DEVICES=3 md-exafs run config.toml --shard 4/4 &
wait
md-exafs merge mdexafs_out
```

Shard `i` of `n` takes every `n`-th selected frame starting at the `i`-th. Each shard writes its own file; `merge`
combines them. The merged result equals a single run: frames are stored by their index, averages are taken in frame
order, and a random choice that belongs to a frame (a subset of absorbers) comes from a random stream keyed by the
seed and the frame number, not by the order of computation. A test checks that one shard and three shards give
the same result.

On a cluster with SLURM this is a job array, one array task per shard and one GPU each, followed by a merge job.
A template is in `examples/cluster/` of the repository. See [](md/shards.md) for the details and the state of
testing.

A stopped shard continues from its own file: frames that are already there are skipped.

## RMC-EXAFS

An RMC-EXAFS run uses one GPU. Independent runs (different random seeds, different inputs of a scan) are separate
jobs and can each have their own GPU; running several seeds is also how the uncertainty of an RMC result is
estimated ([](rmc/uncertainty.md)).

## Many FEFF inputs

Separate `feffgpu run` jobs, one GPU each.

## Memory

- The self-consistency loop of the potentials grows with the number of atoms inside the self-consistency radius.
- The path stage works in batches. MD-EXAFS and RMC-EXAFS take their batch sizes from the free memory of the GPU
  or from the configuration.
- For large runs set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- `feffgpu doctor` shows the free memory of each GPU.

## For developers: the access points

`feffgpu.parallel` holds what a multi-GPU pool will use:

| Name | Purpose |
|---|---|
| `resolve_devices(spec)` | the device strings named by a `devices` setting |
| `primary_device(spec)` | the device used today (the first), with a notice if several were named |
| `Shard`, `Shard.parse("3/16")`, `Shard.select(items)` | a fixed split of independent work items |
| `keyed_rng(seed, *keys)`, `keyed_generator(seed, *keys)` | random streams that depend on the work item, not on the order of computation |
| `run_items(function, items, devices)` | the one place where independent work items are executed |

`run_items` runs the items in order on the primary device, with no overhead. A pool with one worker process per
GPU will replace its body. The rules for code that goes through it: a work item must not depend on the order of
execution; its random numbers come from `keyed_rng`; sums over items are taken in a fixed order. With these rules
one GPU and several GPUs give the same result.

Measured scaling with the number of GPUs will be published here when the pool exists. No scaling numbers are
claimed now.
