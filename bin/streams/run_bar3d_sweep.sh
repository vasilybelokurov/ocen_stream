#!/bin/bash
# 3D bar sweep (bin/streams/bar3d_sweep.py): NW worker processes x OMP_NUM_THREADS threads; extra args passed through.
# Usage: bin/streams/run_bar3d_sweep.sh [extra args]   (default 7 x 2 = 14 cores)
cd "$(dirname "$0")/../.."
source <venv>/bin/activate
NW=${NW:-7}; export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
mkdir -p results/streams/logs
for i in $(seq 0 $((NW-1))); do
  python bin/streams/bar3d_sweep.py --worker $i --nworkers $NW "$@" > results/streams/logs/bar3d_w$i.log 2>&1 &
done
wait
echo "all workers done"
