#!/usr/bin/env bash
#SBATCH --account=CRANMER-SL3-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=run
#SBATCH --output=logs/%x-%j.out
# Main fit on CSD3 (2026-10-29): tempered SMC, 1200 members, to beta = 1, sweep cap 40, flat distance range, seed 1; then a 400-step
# emcee drift check (40 walkers). Each job runs up to 12 h (SL3) and resumes from the last saved stage / emcee step: submit a chain
# (from $HOME, so logs go to $HOME/logs), each job after the previous one ends (--dependency=afterany:<jobid>). A job started after the
# run has finished exits after a few minutes (everything cached).
# Code: a fixed checkout $HOME/ocen_main (git worktree of $HOME/ocen_stream at the run's commit) on PYTHONPATH, ahead of the editable
# install in $HOME/ocen_stream/.venv, so $HOME/ocen_stream can change during the run. The run's provenance refuses changed sources and a
# changed environment.
set -euo pipefail
module purge     # not rhel8/default-amp: it loads CUDA 11.4; jax and cupy bring CUDA 12 via pip
W=/rds/user/$USER/hpc-work/ocen; OUT=$W/runs/smc_main_n1200; CODE=$HOME/ocen_main
cd $CODE; echo "code $(git log --oneline -1)  node $(hostname)  $(date)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
export PYTHONPATH=$CODE/src
$HOME/ocen_stream/.venv/bin/python -c "import ocen_stream; print('ocen_stream from', ocen_stream.__file__)"
$HOME/ocen_stream/.venv/bin/python -u -m ocen_stream.smc --inputs $W/inputs/gpu-inputs-20261027 --out $OUT --n 1200 --seed 1 \
    --max-sweeps 40 --polish-steps 400 --walkers 40
echo "done $(date)"
