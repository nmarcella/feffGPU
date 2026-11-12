#!/bin/bash
# MD-EXAFS on a SLURM cluster: one shard of the frames per array task, one GPU each, then a merge job.
#
# THIS TEMPLATE HAS NOT BEEN RUN ON A CLUSTER YET. Adapt the partition, account, modules, memory and time limits to
# your cluster, and test it on a small frame selection ([frames] stop) before a long run.
#
# Submit the three stages with dependencies (from the folder of config.toml):
#
#   NSHARDS=16
#   pot=$(sbatch --parsable md_exafs_array.sbatch potentials config.toml)
#   arr=$(sbatch --parsable --dependency=afterok:$pot --array=1-$NSHARDS md_exafs_array.sbatch shard config.toml $NSHARDS)
#   sbatch --dependency=afterok:$arr md_exafs_array.sbatch merge config.toml
#
# Stage "potentials" computes the phase files of every edge once (nothing to do for potentials = "file").
# Stage "shard" computes shard SLURM_ARRAY_TASK_ID of NSHARDS: every NSHARDS-th selected frame. A shard that is stopped
# (time limit, preemption) continues from its own file when it is submitted again.
# Stage "merge" combines the shard files into chi.h5 and writes the averages (and the comparison, if configured).
#
#SBATCH --job-name=md-exafs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=md-exafs-%x-%A_%a.out
##SBATCH --partition=gpu
##SBATCH --account=your_account

set -euo pipefail

stage=${1:?usage: md_exafs_array.sbatch potentials|shard|merge CONFIG [NSHARDS]}
config=${2:?the configuration file}

# Load the environment that has feffgpu installed, for example:
# module load cuda
# source /path/to/your/venv/bin/activate

case "$stage" in
  potentials)
    md-exafs run "$config" --potentials-only
    ;;
  shard)
    nshards=${3:?the number of shards}
    md-exafs run "$config" --shard "${SLURM_ARRAY_TASK_ID}/${nshards}"
    ;;
  merge)
    md-exafs merge "$config"
    md-exafs average "$config"
    if grep -Eq '^\[\[?compare\]\]?' "$config"; then
      md-exafs compare "$config"
    fi
    ;;
  *)
    echo "unknown stage: $stage (use potentials, shard or merge)" >&2
    exit 2
    ;;
esac
