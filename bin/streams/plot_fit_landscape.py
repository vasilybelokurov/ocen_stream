#!/usr/bin/env python3
"""Likelihood landscape from the optimiser logs (fit_bar_ocen*.json): 1D panels of Delta lnpost vs each parameter (upper envelope
~ profile likelihood) and 2D pair panels coloured by Delta lnpost. Optimiser samples (non-uniform coverage), not a sampled
posterior. Both logs share the same likelihood (robust chi-KDE, 8000 release epochs, seed 1); fixed-angle points at 28 deg.
Usage: python bin/streams/plot_fit_landscape.py -> plots/fit_landscape.png
"""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

L = []
for f, tag in (("fit_bar_ocen.json", "fixed 28"), ("fit_bar_ocen_freeangle.json", "free angle")):
    p = ROOT/"results/plot_data"/f
    if p.exists():
        for x in json.loads(p.read_text())["log"]:
            L.append(dict(x, angle=x.get("angle", 28.), run=tag))
P = ("angle", "omega", "amp", "d", "pmra", "pmdec"); lab = dict(angle="bar angle [deg]", omega="Omega_b [km/s/kpc]", amp="bar amplitude",
                                                                  d="distance [kpc]", pmra="pmra [mas/yr]", pmdec="pmdec [mas/yr]")
lp = np.array([x["lnpost"] for x in L]); best = lp.max(); dl = lp-best
X = {k: np.array([x[k] for x in L]) for k in P}; free = np.array([x["run"] == "free angle" for x in L])
n = len(P); fig, ax = plt.subplots(n, n, figsize=(3.2*n, 3.0*n))
for i in range(n):
    for j in range(n):
        a = ax[i, j]
        if j > i:
            a.axis("off"); continue
        if i == j:
            a.scatter(X[P[i]][~free], dl[~free], s=8, c="0.5", label="fixed 28 deg"); a.scatter(X[P[i]][free], dl[free], s=8, c="C3", label="free angle")
            a.set_ylim(-3000, 50); a.set_ylabel("Delta lnpost")
            if i == 0:
                a.legend(fontsize=7)
        else:
            o = np.argsort(dl); sc = a.scatter(X[P[j]][o], X[P[i]][o], c=dl[o], s=10, cmap="viridis", vmin=-2000, vmax=0)
            k = np.argmax(dl); a.plot(X[P[j]][k], X[P[i]][k], "r*", ms=12)
            a.set_ylabel(lab[P[i]])
        a.set_xlabel(lab[P[j]]); a.tick_params(labelsize=7)
fig.colorbar(sc, ax=ax, fraction=0.02, label="Delta lnpost (optimiser samples)")
fig.suptitle(f"Likelihood landscape from the optimiser logs ({len(L)} evaluations; best lnpost {best:.0f}); red star = best", fontsize=11)
fig.savefig(ROOT/"plots/fit_landscape.png", dpi=65, bbox_inches="tight"); print("plots/fit_landscape.png")
