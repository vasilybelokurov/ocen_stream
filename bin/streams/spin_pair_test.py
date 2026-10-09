#!/usr/bin/env python3
"""Rotation test A (spray with cluster rotation, release_ic spin term): the two best grid4 models at 32000 release epochs, seeds 1-3,
with spin, paired with the no-spin sprays of seed_test.py (same seeds -> common random numbers). Prints lnL(spin) - lnL(no spin)
per model and seed, and dlnL(16 - 20 deg) with and without spin. Sprays: results/streams/seed_test/*_spin.npz.
Usage: OMP_NUM_THREADS=8 python bin/streams/spin_pair_test.py
"""
import json, time
import numpy as np
from grid4_common import ROOT, make_spray, score, SPIN

MODELS = [(34.5, 16., 1.4, 5.6), (34.5, 20., 1.2, 5.6)]; out = []
for seed in (1, 2, 3):
    for om, an, am, dist in MODELS:
        t0 = time.time(); tag = f"om{om:g}_an{an:g}_am{am:g}_d{dist:g}"
        f = ROOT/f"results/streams/seed_test/{tag}_n32000_s{seed}_spin.npz"
        if f.exists():
            m = dict(np.load(f))
        else:
            m = make_spray(om, an, am, dist, nrel=32000, seed=seed, spin=SPIN); np.savez_compressed(f, **m)
        L1 = score(m)["total"]; L0 = score(dict(np.load(ROOT/f"results/streams/seed_test/{tag}_n32000_s{seed}.npz")))["total"]
        out.append(dict(tag=tag, seed=seed, lnL_spin=L1, lnL_nospin=L0))
        print(f"{tag} seed {seed}: lnL spin {L1:.1f}  no spin {L0:.1f}  diff {L1-L0:+.1f} [{time.time()-t0:.0f} s]", flush=True)
(ROOT/"results/plot_data/spin_pair_test.json").write_text(json.dumps(out, indent=1))
for key in ("lnL_nospin", "lnL_spin"):
    dd = [out[2*i][key]-out[2*i+1][key] for i in range(3)]
    print(f"{key}: dlnL(16-20) per seed {np.round(dd, 1)} mean {np.mean(dd):.1f}")
