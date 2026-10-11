#!/usr/bin/env python3
"""lnL in the (bar angle, Omega_b) plane for each bar size (T2 grid + grid-edge runs E), best over bar amplitude.
Inputs: results/plot_data/sun_dist_bar_grids.json (T2), results/plot_data/frame_decomp_edges.json (E; baumgardt frame).
All: baumgardt frame, d 5.6, catalogue PMs, 4x particles, seed 1.  Output: plots/T2_angle_omega.png
Usage: python bin/streams/plot_T2_angle_omega.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
rows = [r for r in json.loads((ROOT/"results/plot_data/sun_dist_bar_grids.json").read_text()) if r["test"] == "T2"]
rows += [dict(r, edge=True) for r in json.loads((ROOT/"results/plot_data/frame_decomp_edges.json").read_text())
         if r["test"] == "E" and r["ic"] == "baumgardt" and r["proj"] == "baumgardt"]
best = {}                                           # (size, an, om) -> (lnL, amp, edge)
for r in rows:
    k = (r["size"], r["an"], r["om"])
    if k not in best or r["lnL"] > best[k][0]:
        best[k] = (r["lnL"], r["am"], r.get("edge", False))
top = max(v[0] for v in best.values())
ANG = sorted({k[1] for k in best}); OM = sorted({k[2] for k in best}); SIZES = sorted({k[0] for k in best})

fig, ax = plt.subplots(1, len(SIZES), figsize=(5.2*len(SIZES), 4.8), sharey=True, constrained_layout=True)
for a, sz in zip(ax, SIZES):
    Z = np.full((len(OM), len(ANG)), np.nan)
    for (s, an, om), (l, am, ed) in best.items():
        if s != sz: continue
        i, j = OM.index(om), ANG.index(an); Z[i, j] = l-top
        a.text(j, i, f"{l-top:+.0f}\namp {am}" + (" *" if ed else ""), ha="center", va="center", fontsize=8)
    im = a.imshow(Z, origin="lower", cmap="viridis", vmin=-1500, vmax=0, aspect="auto")
    a.set_xticks(range(len(ANG))); a.set_xticklabels([f"{x:.0f}" for x in ANG]); a.set_yticks(range(len(OM)))
    a.set_yticklabels([f"{x:.0f}" for x in OM]); a.set_xlabel("bar angle [deg]"); a.set_title(f"bar size x{sz}")
ax[0].set_ylabel(r"$\Omega_b$ [km/s/kpc]")
fig.colorbar(im, ax=ax, label="lnL - best (best over amp)")
fig.suptitle("Bar angle x pattern speed, per bar length (d 5.6, catalogue PMs, 4x, seed 1; seed noise ~50-250; * = edge run; "
             "blank = not run)", fontsize=10)
fig.savefig(ROOT/"plots/T2_angle_omega.png", dpi=90); print("plots/T2_angle_omega.png")
