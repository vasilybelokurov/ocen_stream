#!/usr/bin/env python3
"""Footprint pmra/pmdec medians along chord phi1 for the T1 solar-frame x distance runs (sun_dist_bar_grids.py) vs members
(median per 2-deg bin, bootstrap 1-sigma). Output plots/sun_dist_T1_medians.png."""
import json
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from grid4_common import ROOT, d, du

R = [r for r in json.loads((ROOT/"results/plot_data/sun_dist_bar_grids.json").read_text()) if r["test"] == "T1"]
cen = np.array(R[0]["phi1"]); edges = np.append(cen-1, cen[-1]+1); ib = np.digitize(du, edges)-1; rng = np.random.default_rng(0)
fig, ax = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
for col, mod in enumerate(("W", "R16")):
    for row, q in enumerate(("pmra", "pmdec")):
        a = ax[row, col]
        for j in range(len(cen)):
            x = d[q][ib == j]
            if len(x) >= 10:
                bs = [np.median(rng.choice(x, len(x))) for _ in range(200)]; a.errorbar(cen[j], np.median(x), np.std(bs), fmt="ko", ms=4, zorder=5)
        for r in R:
            if r["model"] != mod:
                continue
            y = np.array([np.nan if v is None else v for v in r["med"][q]])
            a.plot(cen, y, ls="-" if r["frame"] == "baumgardt" else "--", color={5.43: "C0", 5.6: "C1", 5.8: "C2"}[r["dist"]],
                   label=f"{r['frame']} d={r['dist']:g}  lnL {r['lnL']:.0f}")
        a.set_ylabel(f"{q} [mas/yr]"); a.set_title(f"{mod}: " + ("28 deg, size 1.15, Om 36, amp 1.2" if mod == "W" else "16 deg, Om 34.5, amp 1.4"))
        a.grid(alpha=.3)
    ax[1, col].set_xlabel("chord phi1 [deg]"); ax[0, col].legend(fontsize=7)
fig.tight_layout(); fig.savefig(ROOT/"plots/sun_dist_T1_medians.png", dpi=110); print("saved")
