#!/usr/bin/env python3
"""Sensitivity of the grid4 ranking to the debris age cut of score_conditional (age_max = 500, 700, 1000 Myr; sprays release over
the last 1000 Myr). Re-scores the stored grid4 sprays (no new integrations); prints the best point and the angle / Omega_b profiles
per cut. Output results/plot_data/agecut_rescore.json.
Usage: python bin/streams/agecut_rescore.py
"""
import itertools, json
import numpy as np
from grid4_common import ROOT, score

OMEGA = [33., 34., 34.5, 35., 35.5, 36., 37.]; ANGLE = [16., 20., 24., 28.]; AMP = [1.0, 1.2, 1.4, 1.6]; DIST = [5.45, 5.50, 5.55, 5.60]
P = np.array(list(itertools.product(OMEGA, ANGLE, AMP, DIST))); out = {}
for amax in (500., 700., 1000.):
    L = np.array([score(dict(np.load(ROOT/"results/streams/spray_grid4/om{:g}_an{:g}_am{:g}_d{:g}.npz".format(*p))), age_max=amax)["total"] for p in P])
    i = int(np.argmax(L)); prof = {n: {f"{v:g}": float(L[P[:, k] == v].max()-L[i]) for v in np.unique(P[:, k])} for k, n in enumerate(["omega", "angle", "amp", "d"])}
    out[f"{amax:g}"] = dict(best=P[i].tolist(), lnL=float(L[i]), profiles=prof, lnL_all=L.tolist())
    print(f"age_max {amax:g}: best {P[i]} lnL {L[i]:.1f}; angle profile {[round(v) for v in prof['angle'].values()]}; "
          f"Omega profile {[round(v) for v in prof['omega'].values()]}; amp {[round(v) for v in prof['amp'].values()]}", flush=True)
(ROOT/"results/plot_data/agecut_rescore.json").write_text(json.dumps(dict(params=P.tolist(), cuts=out)))
