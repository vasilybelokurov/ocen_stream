#!/bin/bash
# DB98 Model 1 with Ibata+2019's solar frame and omega Cen distance (frames.py: ibata19), fitted rotation, 1955.58 Myr,
# three DM models, sequential, OMP_NUM_THREADS=8.
cd "$(dirname "$0")/../.."
source <venv>/bin/activate
export OMP_NUM_THREADS=8
for spec in "A_nodm A_nodm" "B_dm_phot B_dm_phot" "C_dm5x_shell B_dm_phot"; do
  set -- $spec
  python bin/streams/run_prescribed.py --model $1 --ics results/nbody/$2/ics.npz --out results/streams/db98_ibata/$1 \
    --spin results/streams/spin/$1.json --mw configs/potentials/DB98_Model1.ini --frame ibata19 > results/streams/logs/db98_ibata_$1.log 2>&1
  tail -1 results/streams/logs/db98_ibata_$1.log
done
