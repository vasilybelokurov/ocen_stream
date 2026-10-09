#!/bin/bash
# Wait for the free-angle fit, then run the conditional likelihood scans (OMP_NUM_THREADS=4).
cd "$(dirname "$0")/../.."
while pgrep -f "fit_bar_ocen.py" >/dev/null; do sleep 20; done
source <venv>/bin/activate
export OMP_NUM_THREADS=4
python bin/streams/scan_best_fit.py > results/streams/logs/scan_best_fit.log 2>&1
