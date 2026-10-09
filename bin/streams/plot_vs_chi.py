#!/usr/bin/env python3
"""Spray particles (trailing arm, chi > 0, release age < 700 Myr) vs the Gibbons phase chi, coloured by release age, with the
model track along chi: median (line) and 16-84% band of each observable in 5 kpc Myr bins of chi (bins with >= 15 particles).
Usage: python bin/streams/plot_vs_chi.py om34.5_an24_am1.2 [more tags] -> plots/vs_chi_<tag>.png
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

for tag in sys.argv[1:]:
    m = np.load(ROOT/f"results/streams/spray_grid2/{tag}.npz")
    k = (m["chi"] > 0) & (m["age"] < 700)
    chi, age = m["chi"][k], m["age"][k]
    edges = np.arange(0, 250.1, 5.); cen = 0.5*(edges[1:]+edges[:-1])
    fig, ax = plt.subplots(2, 3, figsize=(19, 10))
    for a, (q, lab, yl) in zip(ax.ravel(), (("l", "l [deg]", (-75, -5)), ("b", "b [deg]", (-5, 45)), ("pmra", "pmra [mas/yr]", (-22, 0)),
                                             ("pmdec", "pmdec [mas/yr]", (-16, -3)), ("vlos", "v_los [km/s]", (50, 300)), ("d", "distance [kpc]", (2.5, 6.5)))):
        y = m[q][k]
        sc = a.scatter(chi, y, c=age, s=2, cmap="viridis_r", vmin=0, vmax=700, zorder=1)
        med, lo, hi = (np.full(len(cen), np.nan) for _ in range(3))
        for i, (e0, e1) in enumerate(zip(edges[:-1], edges[1:])):
            s = (chi >= e0) & (chi < e1)
            if s.sum() >= 15:
                lo[i], med[i], hi[i] = np.percentile(y[s], [16, 50, 84])
        a.fill_between(cen, lo, hi, color="r", alpha=0.18, zorder=2); a.plot(cen, med, "r-", lw=2, zorder=3)
        a.set_xlim(0, 250); a.set_ylim(*yl); a.set_xlabel("Gibbons phase chi [kpc Myr]"); a.set_ylabel(lab); a.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax, label="release age [Myr]", fraction=0.02)
    fig.suptitle(f"Spray {tag} (model A, 1.96 Gyr): trailing-arm particles vs Gibbons phase; red = track (median, 16-84%) along chi", fontsize=11)
    fig.savefig(ROOT/f"plots/vs_chi_{tag}.png", dpi=75, bbox_inches="tight"); print(f"plots/vs_chi_{tag}.png")
