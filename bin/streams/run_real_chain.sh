#!/bin/bash
# Real-data chain (2026-10-19): 3 angle-stratified waves + direct per-angle profile, measurement priors on (user).
# Usage: bin/streams/run_real_chain.sh   (from the repository root; 8 workers; ~14 h)
set -e
export PYTHONPATH=src
BASE=results/streams/rescore_real_all_20261019
PREV=()
for k in 1 2 3; do
  O=results/streams/search_real_wave${k}_20261019
  OMP_NUM_THREADS=4 python bin/streams/search_wave.py --catalogue real --box data --base $BASE --outdir $O --seed-base 1700000 --prior ${PREV[@]:+--prev ${PREV[@]}}
  OMP_NUM_THREADS=1 python bin/streams/stream_batch.py run --outdir $O --workers 8 > $O/run.log 2>&1
  PREV+=($O)
done
OMP_NUM_THREADS=1 python bin/streams/profile_nm.py --catalogue real --box data --outdir results/streams/profile_real_20261019 \
  --start-from results/streams/emulator_round1_20261018 results/streams/emulator_round2_20261018 results/streams/emulator_check_20261018 ${PREV[@]} \
  --angles 12.5 17.5 22.5 25 27.5 32.5 --workers 8 --budget 30 --prior --seed-base 1900000
