#!/usr/bin/env python3
"""grid4 rerun with cluster rotation in the spray (release_ic spin term; spin profile results/streams/spin/A_nodm_vrot.json):
same 448 points, 8000 release epochs, seed 1 as grid4 (common random numbers -> paired with grid4). Sprays in
results/streams/spray_grid4_spin/; summary results/plot_data/grid4_spin.json; per-star lnL results/plot_data/grid4_spin_perstar.npz.
Resumable. Usage: OMP_NUM_THREADS=8 python bin/streams/grid4_spin.py
"""
import itertools, json, time
import numpy as np
from grid4_common import ROOT, make_spray, score, SPIN, du

OMEGA = [33., 34., 34.5, 35., 35.5, 36., 37.]; ANGLE = [16., 20., 24., 28.]; AMP = [1.0, 1.2, 1.4, 1.6]; DIST = [5.45, 5.50, 5.55, 5.60]
outd = ROOT/"results/streams/spray_grid4_spin"; outd.mkdir(parents=True, exist_ok=True); res, P, L = [], [], []
for om, an, am, dist in itertools.product(OMEGA, ANGLE, AMP, DIST):
    t0 = time.time(); tag = f"om{om:g}_an{an:g}_am{am:g}_d{dist:g}"; f = outd/f"{tag}.npz"
    if f.exists():
        m = dict(np.load(f))
    else:
        m = make_spray(om, an, am, dist, spin=SPIN); np.savez_compressed(f, **m)
    s = score(m); P.append((om, an, am, dist)); L.append(s["per_star"])
    res.append(dict(omega=om, angle=an, amp=am, d=dist, lnL=s["total"], K=s["K"], seconds=time.time()-t0))
    print(f"{tag:28s} lnL {s['total']:10.1f}  [{time.time()-t0:.0f} s]", flush=True)
    (ROOT/"results/plot_data/grid4_spin.json").write_text(json.dumps(dict(spin="results/streams/spin/A_nodm_vrot.json", grid=res)))
np.savez_compressed(ROOT/"results/plot_data/grid4_spin_perstar.npz", params=np.array(P), lnL=np.array(L), u=du)
