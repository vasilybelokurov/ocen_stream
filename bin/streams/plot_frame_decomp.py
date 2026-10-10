#!/usr/bin/env python3
"""Footprint pmra/pmdec medians vs chord phi1 for the IC x projection frame factorial and the one-at-a-time frame controls
(frame_decomp_edges.py; diagonal cells from sun_dist_bar_grids.py T1, d 5.6) against member medians. Output plots/frame_decomp_medians.png."""
import json
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from grid4_common import ROOT, d, du

T1 = {(r["frame"], r["frame"]): r for r in json.loads((ROOT/"results/plot_data/sun_dist_bar_grids.json").read_text()) if r["test"] == "T1" and r["model"] == "W" and r["dist"] == 5.6}
F = json.loads((ROOT/"results/plot_data/frame_decomp_edges.json").read_text())
runs = [("IC baumgardt / proj baumgardt", T1[("baumgardt", "baumgardt")], "C0", "-"), ("IC ibata19 / proj ibata19", T1[("ibata19", "ibata19")], "C1", "-")]
runs += [(f"IC {r['ic']} / proj {r['proj']}", r, c, "--") for r, c in zip([r for r in F if r["test"] in "FC"], ("C2", "C3", "C4", "C5", "C6"))]
cen = np.array(runs[0][1]["phi1"]); edges = np.append(cen-1, cen[-1]+1); ib = np.digitize(du, edges)-1; rng = np.random.default_rng(0)
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for a, q in zip(ax, ("pmra", "pmdec")):
    for j in range(len(cen)):
        x = d[q][ib == j]
        if len(x) >= 10:
            a.errorbar(cen[j], np.median(x), np.std([np.median(rng.choice(x, len(x))) for _ in range(200)]), fmt="ko", ms=4, zorder=5)
    for lab, r, c, ls in runs:
        a.plot(cen, [np.nan if v is None else v for v in r["med"][q]], color=c, ls=ls, label=f"{lab}  lnL {r['lnL']:.0f}")
    a.set_xlabel("chord phi1 [deg]"); a.set_ylabel(f"{q} [mas/yr]"); a.grid(alpha=.3)
ax[0].legend(fontsize=7); fig.suptitle("W model (28 deg, size 1.15, Om 36, amp 1.2), d 5.6"); fig.tight_layout()
fig.savefig(ROOT/"plots/frame_decomp_medians.png", dpi=110); print("saved")
