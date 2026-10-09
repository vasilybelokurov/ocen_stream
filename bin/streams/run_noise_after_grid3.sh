#!/bin/bash
# Wait for grid 3 to finish, then run the chi-KDE noise test (OMP_NUM_THREADS=4).
cd "$(dirname "$0")/../.."
while pgrep -f "spray_bar_grid2.py" >/dev/null; do sleep 20; done
source <venv>/bin/activate
export OMP_NUM_THREADS=4
python bin/streams/noise_test.py --models 34.5,16,1.4 34.5,24,1.2 --seeds 2 3 4 --double > results/streams/logs/noise_test.log 2>&1
