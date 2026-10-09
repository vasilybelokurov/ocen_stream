#!/usr/bin/env python3
"""Plots for the Stage-2a spray grid (results/plot_data/spray_bar_grid.json): total and per-observable chi2 over
(Omega_b, angle) for each amplitude, and the three best grid points over the data track.
Usage: python bin/streams/plot_spray_bar_grid.py -> plots/spray_bar_grid_chi2.png, plots/spray_bar_grid_best.png
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

g = json.loads((ROOT/"results/plot_data/spray_bar_grid.json").read_text())["grid"]
data = json.loads((ROOT/"results/plot_data/stream54_track.json").read_text())["all"]
om = sorted({x["omega"] for x in g}); an = sorted({x["angle"] for x in g}); am = sorted({x["amp"] for x in g})
fig, ax = plt.subplots(4, len(am), figsize=(5*len(am), 15), squeeze=False)
for j, A in enumerate(am):
    for i, key in enumerate(("total", "l", "pmra", "pmdec")):
        Z = np.full((len(an), len(om)), np.nan)
        for x in g:
            if x["amp"] == A:
                Z[an.index(x["angle"]), om.index(x["omega"])] = x["chi2_total"] if key == "total" else x["chi2"][key]["chi2"]
        im = ax[i, j].imshow(np.log10(Z), origin="lower", aspect="auto", cmap="viridis_r",
                             extent=(om[0]-0.75, om[-1]+0.75, an[0]-2, an[-1]+2))
        for x in g:
            if x["amp"] == A:
                v = x["chi2_total"] if key == "total" else x["chi2"][key]["chi2"]
                ax[i, j].text(x["omega"], x["angle"], f"{v:.0f}", ha="center", va="center", fontsize=7, color="w")
        ax[i, j].set_title(f"amp {A:g}: chi2 {key}", fontsize=9); ax[i, j].set_xlabel("Omega_b [km/s/kpc]"); ax[i, j].set_ylabel("bar angle [deg]")
        fig.colorbar(im, ax=ax[i, j], label="log10 chi2")
fig.tight_layout(); fig.savefig(ROOT/"plots/spray_bar_grid_chi2.png", dpi=70)
best = sorted(g, key=lambda x: x["chi2_total"])[:3]
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
for jj, q in enumerate(("l", "pmra", "pmdec")):
    ax[jj].errorbar(data["b"], data[q]["mu"], yerr=data[q]["sig_mu"], fmt="o", color="k", ms=5, label="data")
    for c, x in zip(("C3", "C0", "C2"), best):
        ax[jj].errorbar(np.array(data["b"])+0.3*(best.index(x)+1), x["track"][q]["mu"], yerr=x["track"][q]["sig_mu"], fmt="^", color=c, ms=4,
                        label=f"Om {x['omega']:g}, ang {x['angle']:g}, amp {x['amp']:g}: chi2 {x['chi2_total']:.0f}")
    ax[jj].set_xlabel("b [deg]"); ax[jj].set_ylabel(q); ax[jj].grid(alpha=0.3)
ax[0].invert_yaxis(); ax[1].set_ylim(-16, -2); ax[2].set_ylim(-12, -5); ax[0].legend(fontsize=7)
fig.suptitle("Spray grid: three best (Omega_b, angle, amplitude), model A, 1.96 Gyr, vs stream 54 track", fontsize=11)
fig.tight_layout(); fig.savefig(ROOT/"plots/spray_bar_grid_best.png", dpi=80)
for x in best:
    print(x["omega"], x["angle"], x["amp"], round(x["chi2_total"]), {q: (round(v["chi2"]), v["nbins"], v["masked"]) for q, v in x["chi2"].items()})
