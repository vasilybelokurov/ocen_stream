#!/bin/bash
# Extended bar grid (grid 3): Omega_b 33-40.5 (1.5) x angle 16-28 (4) x amplitude 1.0-1.6 (0.2); 96 points, 12 reused from grid 2.
# Scored with the robust chi-KDE likelihood, the sky-conditional score and the overshoot fraction. OMP_NUM_THREADS=4.
cd "$(dirname "$0")/../.."
source <venv>/bin/activate
export OMP_NUM_THREADS=4
python bin/streams/spray_bar_grid2.py --nrel 8000 --omega 33 34.5 36 37.5 39 40.5 --angle 16 20 24 28 --amp 1.0 1.2 1.4 1.6 \
  --summary spray_bar_grid3.json > results/streams/logs/spray_bar_grid3.log 2>&1
