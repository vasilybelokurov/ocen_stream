#!/usr/bin/env python3
"""Plots for spray_bar_grid2: (1) lnL (total, sky, pm) and overshoot maps over (Omega_b, angle) per amplitude;
(2) particle overlays for the 3 best and 3 worst models by total lnL, plus the old-chi2 'best' (34.5, 32, 1.2): sky, pmra-b, pmdec-b,
v_los-b, trailing arm coloured by release age, chi-ordered track (black line), stream 54 members grey.
Usage: python bin/streams/plot_spray_bar_grid2.py -> plots/spray_bar_grid2_maps.png, plots/spray_bar_grid2_models.png
"""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"bin/streams"))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from spray_bar_grid2 import load_data
sys.path.insert(0, str(ROOT/'src'))
from ocen_dm.streams.score import age_ridge

G = json.loads((ROOT/"results/plot_data/spray_bar_grid2.json").read_text())["grid"]
om = sorted({x["omega"] for x in G}); an = sorted({x["angle"] for x in G}); am = sorted({x["amp"] for x in G})
best = max(x["score"]["total"] for x in G)
fig, ax = plt.subplots(4, len(am), figsize=(5*len(am), 15), squeeze=False)
for j, A in enumerate(am):
    for i, (key, lab) in enumerate((("total", "Delta lnL total"), ("sky", "Delta lnL sky"), ("pm", "Delta lnL pm"), ("overshoot", "overshoot fraction"))):
        Z = np.full((len(an), len(om)), np.nan)
        for x in G:
            if x["amp"] == A:
                v = x["overshoot"] if key == "overshoot" else x["score"][key]-max(y["score"][key] for y in G)
                Z[an.index(x["angle"]), om.index(x["omega"])] = v
        im = ax[i, j].imshow(Z, origin="lower", aspect="auto", cmap="viridis" if key != "overshoot" else "magma_r",
                             extent=(om[0]-0.75, om[-1]+0.75, an[0]-2, an[-1]+2))
        for x in G:
            if x["amp"] == A:
                v = x["overshoot"] if key == "overshoot" else x["score"][key]-max(y["score"][key] for y in G)
                ax[i, j].text(x["omega"], x["angle"], f"{v:.2f}" if key == "overshoot" else f"{v:.0f}", ha="center", va="center", fontsize=7, color="w")
        ax[i, j].set_title(f"amp {A:g}: {lab}", fontsize=9); ax[i, j].set_xlabel("Omega_b"); ax[i, j].set_ylabel("angle"); fig.colorbar(im, ax=ax[i, j])
fig.tight_layout(); fig.savefig(ROOT/"plots/spray_bar_grid2_maps.png", dpi=65)

d = load_data()
srt = sorted(G, key=lambda x: -x["score"]["total"])
sel = srt[:3] + [x for x in G if (x["omega"], x["angle"], x["amp"]) == (34.5, 32., 1.2)] + srt[-2:]
fig, ax = plt.subplots(len(sel), 4, figsize=(22, 4.4*len(sel)), squeeze=False)
for i, x in enumerate(sel):
    tag = f"om{x['omega']:g}_an{x['angle']:g}_am{x['amp']:g}"; m = np.load(ROOT/f"results/streams/spray_grid2/{tag}.npz")
    k = (m["b"] > 5) & (m["l"] < -10) & (m["l"] > -80)
    ct = age_ridge(m["age"], m["l"], m["b"], dict(pmra=m["pmra"], pmdec=m["pmdec"], vlos=m["vlos"], d=m["d"]))
    kw = dict(c=m["age"][k], cmap="viridis_r", vmin=0, vmax=1000, s=2)
    a = ax[i, 0]; a.plot(d["l"], d["b"], ".", color="0.6", ms=1.5, zorder=1); a.scatter(m["l"][k], m["b"][k], zorder=2, **kw)
    H, xe, ye = np.histogram2d(d["l"], d["b"], bins=(np.arange(-80, -9, 1.), np.arange(5, 51, 1.)))   # member density outline on top
    a.contour(0.5*(xe[1:]+xe[:-1]), 0.5*(ye[1:]+ye[:-1]), H.T, levels=[2, 10], colors=["k", "k"], linewidths=[0.6, 1.0], zorder=4)
    a.plot(ct["l"], ct["b"], "r-", lw=0.7, zorder=5); a.plot(ct["l"], ct["b"], "ro", ms=3, zorder=6)
    a.set_xlim(-10, -80); a.set_ylim(5, 50); a.set_xlabel("l"); a.set_ylabel("b")
    a.set_title(f"Om {x['omega']:g} ang {x['angle']:g} amp {x['amp']:g}: dlnL {x['score']['total']-best:.0f} (sky {x['score']['sky']:.0f}, pm {x['score']['pm']:.0f}); overshoot {x['overshoot']:.2f}", fontsize=8)
    for jj, (q, dq, yl) in enumerate((("pmra", d["pmra"], (-22, 0)), ("pmdec", d["pmdec"], (-15, -3)), ("vlos", d["v"], (120, 300)))):
        a = ax[i, jj+1]
        a.plot(d["b"], dq, "o" if q == "vlos" else ".", color="k" if q == "vlos" else "0.6", ms=5 if q == "vlos" else 1.5, mfc="w" if q == "vlos" else None, zorder=5 if q == "vlos" else 1)
        a.scatter(m["b"][k], m[q][k], zorder=2, **kw); a.plot(ct["b"], ct[q], "r-", lw=0.7, zorder=6); a.plot(ct["b"], ct[q], "ro", ms=3, zorder=7); a.set_xlim(10, 45); a.set_ylim(*yl); a.set_xlabel("b"); a.set_ylabel(q)
    for a in ax[i]:
        a.grid(alpha=0.3)
fig.suptitle("Grid-2 ranking: top 3 by lnL, the old-chi2 'best' (34.5, 32, 1.2), bottom 2. Colour: release age [0-1000 Myr]; red: ridge = 2D density peak per 15-Myr release-age bin, 0-450 Myr; black contours: member density (2 and 10 per deg^2)", fontsize=11)
fig.tight_layout(); fig.savefig(ROOT/"plots/spray_bar_grid2_models.png", dpi=62)
for x in srt[:8]:
    print(x["omega"], x["angle"], x["amp"], round(x["score"]["total"]-best), round(x["score"]["sky"]), round(x["score"]["pm"]), round(x["score"]["vlos"]), round(x["overshoot"], 2))
print("...")
for x in srt[-3:]:
    print(x["omega"], x["angle"], x["amp"], round(x["score"]["total"]-best), round(x["overshoot"], 2))
