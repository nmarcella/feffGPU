# Several GPUs and nodes: shards

The frames of a trajectory are independent of one another. MD-EXAFS splits a run into shards: each shard computes
part of the frames, on its own GPU or node, and writes its own file; a merge combines the files into the cache of the
whole run. This works today on one machine with several GPUs and on a cluster. A pool of worker processes inside one
run (several GPUs used by one command) is not implemented yet; see "What comes later".

## Running shards

```bash
md-exafs run config.toml --potentials-only        # phase files first, once (potentials = "reference" or "per_frame")
CUDA_VISIBLE_DEVICES=0 md-exafs run config.toml --shard 1/4 &
CUDA_VISIBLE_DEVICES=1 md-exafs run config.toml --shard 2/4 &
CUDA_VISIBLE_DEVICES=2 md-exafs run config.toml --shard 3/4 &
CUDA_VISIBLE_DEVICES=3 md-exafs run config.toml --shard 4/4 &
wait
md-exafs merge mdexafs_out
md-exafs average config.toml
```

Shard `i` of `n` takes every `n`-th selected frame starting with the `i`-th (`feffgpu.parallel.Shard`). It writes
`chi.shard-<i>-of-<n>.h5` and `report.shard-<i>-of-<n>.json` in the output folder. `md-exafs merge` checks that every
shard is there, that all were made with the same settings, and that together they hold every selected frame exactly
once, and then writes `chi.h5` with the rows in frame order. After a merge, `average`, `compare`, `plot` and
`convergence` work as after a whole run.

The phase files of `potentials = "reference"` and `"per_frame"` are computed by the first process that needs them and
kept in the output folder. When shards start at the same time, several of them may compute the same phase file; the
file is written atomically and the result is the same up to rounding. `--potentials-only` computes them once before
the shards start and avoids this.

## Why the merged result equals a single run

- Frames are stored with their index, and every average is taken in frame order, so the order in which shards finish
  does not matter.
- A random choice that belongs to a frame (the subset of absorbers when `absorbers_per_frame` is a number) comes from
  a random stream keyed by the seed, the frame index and the element (`feffgpu.parallel.keyed_rng`), not from the
  order of computation.
- The loop over frames goes through `feffgpu.parallel.run_items`, the single place where independent work items are
  executed.

A test runs the same configuration as one whole run and as three shards, merges the shards, and checks that every
stored array and the averages agree to floating-point tolerance (on the GPU, sums accumulated with atomic operations
can differ in the last bits). Another test checks that the absorber subsets are the same when the frames are split
into two shards.

## Stopping and continuing

A run (or a shard) writes the cache in blocks of `[run] frames_per_block` frames. When it stops, by an error, a time
limit or a signal, starting the same command again continues: the frames already in the file are skipped, and rows of
a block that was not completed are dropped. A cache is continued only with the settings that made it; with other
settings the run stops and says which settings differ.

## A SLURM job array

`examples/cluster/md_exafs_array.sbatch` in the repository is a template: one job computes the phase files, an array of
jobs computes one shard each on one GPU, and a last job merges the shards and writes the averages. Submit it with the
dependencies shown at the top of the file. **The template has not been run on a cluster yet**; check the partition,
account, modules and time limits of your cluster before using it, and test it on a small selection of frames first.

## What comes later

`feffgpu.parallel.run_items` runs the frames one after the other on the first device of `[run] devices`. A pool with
one worker process per GPU will replace its body, so that one command can use several GPUs of a node; the shards,
the keyed random streams and the merge are the parts that make the results independent of the number of workers.
Until then, use shards: one per GPU.
