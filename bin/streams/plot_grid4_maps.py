#!/usr/bin/env python3
"""2D profile maps of the 4D grid (results/plot_data/grid4.json): for each parameter pair, the maximum of lnL (score_conditional)
over the other two, shown as Delta lnL from the global maximum. Output plots/grid4_maps.png.
Usage: python bin/streams/plot_grid4_maps.py
"""
import itertools, json
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
g = json.loads((ROOT/"results/plot_data/grid4.json").read_text())["grid"]
names = ["omega", "angle", "amp", "d"]; lab = [r"$\Omega_b$ [km/s/kpc]", "bar angle [deg]", "bar amplitude", r"$d_{\omega Cen}$ [kpc]"]
a = np.array([[r[n] for n in names]+[r["lnL"]] for r in g]); L = a[:, 4]-a[:, 4].max()
fig, ax = plt.subplots(2, 3, figsize=(15, 9)); ax = ax.ravel()
for k, (i, j) in enumerate(itertools.combinations(range(4), 2)):
    xi, yj = np.unique(a[:, i]), np.unique(a[:, j]); Z = np.full((len(yj), len(xi)), np.nan)
    for p, x in enumerate(xi):
        for q, y in enumerate(yj):
            Z[q, p] = L[(a[:, i] == x) & (a[:, j] == y)].max()
    im = ax[k].imshow(Z, origin="lower", aspect="auto", cmap="viridis", vmin=-1000, vmax=0)
    for p in range(len(xi)):
        for q in range(len(yj)):
            ax[k].text(p, q, f"{Z[q, p]:.0f}", ha="center", va="center", fontsize=7, color="w" if Z[q, p] < -500 else "k")
    ax[k].set_xticks(range(len(xi)), [f"{v:g}" for v in xi]); ax[k].set_yticks(range(len(yj)), [f"{v:g}" for v in yj])
    ax[k].set_xlabel(lab[i]); ax[k].set_ylabel(lab[j])
fig.colorbar(im, ax=ax, label=r"$\Delta\ln L$ (max over the other two parameters)", shrink=0.8)
b = a[np.argmax(a[:, 4])]
fig.suptitle(f"4D grid, conditional KDE likelihood (3681 members); best: Omega_b {b[0]:g}, angle {b[1]:g}, amp {b[2]:g}, d {b[3]:g}; "
             f"PMs fixed (-3.2223, -6.7517)")
out = ROOT/"plots/grid4_maps.png"; fig.savefig(out, dpi=110, bbox_inches="tight"); print(out)
