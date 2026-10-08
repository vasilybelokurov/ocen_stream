#!/bin/bash
# Queue (2026-10-08, approved): finish the running A tupd=4 check (pid 17304), then tupd convergence
# A tupd=2, B tupd=2 and 1 (first 300 Myr), then the four validation runs (bin/streams/run_validation.sh).
cd "$(dirname "$0")/../.."
source <venv>/bin/activate
while kill -0 17304 2>/dev/null; do sleep 30; done
echo "A_tupd4 finished"
python bin/streams/run_restricted.py --ics results/nbody/A_nodm/ics.npz --out results/streams/checks/A_nodm_tupd2 --tstop 300 --tupd 2 --snap 50
for u in 2 1; do
  python bin/streams/run_restricted.py --ics results/nbody/B_dm_phot/ics.npz --out results/streams/checks/B_dm_phot_tupd$u --tstop 300 --tupd $u --snap 50
done
echo "CHECKS DONE"
bin/streams/run_validation.sh
