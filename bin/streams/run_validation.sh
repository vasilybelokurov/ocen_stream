#!/bin/bash
# Validation runs of the restricted N-body against the live runs (same ICs, same orbit start, 1955.58 Myr):
# models A (no DM) and B (photometric-mass DM), each with the refitted and the frozen satellite potential.
# Deeply bound core frozen (r_max(E) < 30 pc): ~12 min per run. Skips runs that already finished.
set -e
cd "$(dirname "$0")/../.."
source <venv>/bin/activate
TUPD=${TUPD:-2}; RF=${RF:-30}
for m in A_nodm B_dm_phot; do
  for v in refit frozen; do
    out=results/streams/validation/${m}_${v}_rf$RF
    if [ -f $out/snap_today.npz ]; then echo "skip $out"; continue; fi
    extra=""; [ $v = frozen ] && extra="--frozen"
    python bin/streams/run_restricted.py --ics results/nbody/$m/ics.npz --out $out --tupd $TUPD --snap 50 --rfreeze $RF $extra
  done
done
echo ALL DONE
