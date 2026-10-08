#!/bin/bash
# Rotating-bar test for model A (prescribed potential, fitted rotation, 1955.58 Myr, baumgardt frame): Hunter+2024 axisymmetric
# control, then the Hunter+2024 barred MW at constant pattern speeds 33, 37.5, 41 km/s/kpc, bar angle 28 deg today.
# Sequential, OMP_NUM_THREADS=8.
cd "$(dirname "$0")/../.."
source <venv>/bin/activate
export OMP_NUM_THREADS=8
S=results/streams/spin/A_nodm.json
python bin/streams/run_prescribed.py --model A_nodm --out results/streams/hunter_axi/A_nodm --mw hunter24_axi --spin $S \
  > results/streams/logs/hunter_axi_A.log 2>&1; tail -1 results/streams/logs/hunter_axi_A.log
for om in 33 37.5 41; do
  python bin/streams/run_prescribed.py --model A_nodm --out results/streams/hunter_bar$om/A_nodm --mw hunter24_bar --bar-omega $om --spin $S \
    > results/streams/logs/hunter_bar${om}_A.log 2>&1; tail -1 results/streams/logs/hunter_bar${om}_A.log
done
