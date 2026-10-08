#!/bin/bash
# Validation runs of the restricted N-body against the live runs (same ICs, same orbit start, 1955.58 Myr):
# models A (no DM) and B (photometric-mass DM), each with the refitted and the frozen satellite potential.
# ~1.5-2 h each, run sequentially (each uses all cores).
set -e
cd "$(dirname "$0")/../.."
source <venv>/bin/activate
TUPD=${TUPD:-2}
for m in A_nodm B_dm_phot; do
  python bin/streams/run_restricted.py --ics results/nbody/$m/ics.npz --out results/streams/validation/${m}_refit --tupd $TUPD
  python bin/streams/run_restricted.py --ics results/nbody/$m/ics.npz --out results/streams/validation/${m}_frozen --tupd $TUPD --frozen
done
echo ALL DONE
