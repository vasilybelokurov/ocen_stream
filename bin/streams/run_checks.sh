#!/bin/bash
# Numerical checks of the restricted N-body runner (approved 2026-10-08), run sequentially:
#  isolation 100 Myr for A and B; update-interval convergence on the orbit (first 300 Myr) for A and B.
set -e
cd "$(dirname "$0")/../.."
source <venv>/bin/activate
for m in A_nodm B_dm_phot; do
  python bin/streams/run_restricted.py --ics results/nbody/$m/ics.npz --out results/streams/checks/${m}_iso --no-host --tstop 100 --snap 10
done
for m in A_nodm B_dm_phot; do
  for u in 4 2 1; do
    python bin/streams/run_restricted.py --ics results/nbody/$m/ics.npz --out results/streams/checks/${m}_tupd$u --tstop 300 --tupd $u --snap 10
  done
done
echo ALL DONE
