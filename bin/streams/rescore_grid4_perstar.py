#!/usr/bin/env python3
"""Re-score all grid4 sprays with score_conditional and store per-star lnL (448 x N members) for the block bootstrap.
Output: results/plot_data/grid4_perstar.npz (params (448, 4): omega, angle, amp, d; lnL (448, N); u = members' phi1).
Usage: python bin/streams/rescore_grid4_perstar.py
"""
import itertools, time
import numpy as np
from grid4_common import ROOT, du, score

OMEGA = [33., 34., 34.5, 35., 35.5, 36., 37.]; ANGLE = [16., 20., 24., 28.]; AMP = [1.0, 1.2, 1.4, 1.6]; DIST = [5.45, 5.50, 5.55, 5.60]
P, L = [], []; t0 = time.time()
for i, (om, an, am, dist) in enumerate(itertools.product(OMEGA, ANGLE, AMP, DIST)):
    m = dict(np.load(ROOT/f"results/streams/spray_grid4/om{om:g}_an{an:g}_am{am:g}_d{dist:g}.npz"))
    P.append((om, an, am, dist)); L.append(score(m)["per_star"])
    if i % 20 == 0:
        print(f"{i+1}/448  {time.time()-t0:.0f} s", flush=True)
np.savez_compressed(ROOT/"results/plot_data/grid4_perstar.npz", params=np.array(P), lnL=np.array(L), u=du)
print("done", time.time()-t0)
