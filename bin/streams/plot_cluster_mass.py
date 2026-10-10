#!/usr/bin/env python3
"""Footprint PM medians vs chord phi1 for the omega Cen mass test (cluster_mass_test.py; seeds 1-3, mean of medians) vs members.
Output plots/cluster_mass_medians.png."""
import json
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from grid4_common import ROOT, d, du

R = json.loads((ROOT/"results/plot_data/cluster_mass_test.json").read_text())
cen = np.array(R[0]["phi1"]); edges = np.append(cen-1, cen[-1]+1); ib = np.digitize(du, edges)-1; rng = np.random.default_rng(0)
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for a, q in zip(ax, ("pmra", "pmdec")):
    for j in range(len(cen)):
        x = d[q][ib == j]
        if len(x) >= 10:
            a.errorbar(cen[j], np.median(x), np.std([np.median(rng.choice(x, len(x))) for _ in range(200)]), fmt="ko", ms=4, zorder=5)
    for mf, c in ((0.7, "C0"), (1.0, "C1"), (1.5, "C3")):
        y = np.nanmean([[np.nan if v is None else v for v in r["med"][q]] for r in R if r["mass_factor"] == mf], 0)
        lnl = np.mean([r["lnL"] for r in R if r["mass_factor"] == mf])
        a.plot(cen, y, color=c, label=f"M x {mf:g}  <lnL> {lnl:.0f}")
    a.set_xlabel("chord phi1 [deg]"); a.set_ylabel(f"{q} [mas/yr]"); a.grid(alpha=.3)
ax[0].legend(); fig.suptitle("omega Cen mass x 0.7 / 1 / 1.5 (W model: 28 deg, size 1.15, Om 36, amp 1.2, d 5.6)"); fig.tight_layout()
fig.savefig(ROOT/"plots/cluster_mass_medians.png", dpi=110); print("saved")
