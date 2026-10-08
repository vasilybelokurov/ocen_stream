#!/bin/bash
# Rotation/host/time tests for the three DM models (prescribed potential, 200k star tracers), sequential, 10 threads.
#  T1 maxrot:      McMillan17, 1955.58 Myr, every counter-rotating tracer flipped (results/streams/spin_max)
#  T2 db98_rot:    DB98 Model 1, 1955.58 Myr, data-fitted rotation (results/streams/spin)
#  T3 db98_rot_5g: DB98 Model 1, 5000 Myr, data-fitted rotation
cd "$(dirname "$0")/../.."
source <venv>/bin/activate
export OMP_NUM_THREADS=10
DB=configs/potentials/DB98_Model1.ini
run() {  # tag model ics spin mw tback
  python bin/streams/run_prescribed.py --model $2 --ics results/nbody/$3/ics.npz --out results/streams/$1/$2 \
    --spin results/streams/$4/$2.json --mw $5 --tback $6 > results/streams/logs/$1_$2.log 2>&1
  tail -1 results/streams/logs/$1_$2.log
}
for spec in "A_nodm A_nodm" "B_dm_phot B_dm_phot" "C_dm5x_shell B_dm_phot"; do set -- $spec; run prescribed_maxrot $1 $2 spin_max McMillan17 1955.58; done
for spec in "A_nodm A_nodm" "B_dm_phot B_dm_phot" "C_dm5x_shell B_dm_phot"; do set -- $spec; run db98_rot $1 $2 spin $DB 1955.58; done
for spec in "A_nodm A_nodm" "B_dm_phot B_dm_phot" "C_dm5x_shell B_dm_phot"; do set -- $spec; run db98_rot_5g $1 $2 spin $DB 5000; done
