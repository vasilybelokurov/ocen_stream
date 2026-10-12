#!/usr/bin/env python3
"""Footprint median tracks (pmra, pmdec, v_los, d vs phi1) of the LMC test sprays (results/plot_data/lmc_spray_test.jsonl,
bin/streams/lmc_spray_test.py), seed 1; one row per bar model, colours = M_LMC. Output plots/lmc_spray_test.png.
Usage: python bin/streams/plot_lmc_spray_test.py"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
R = [json.loads(l) for l in (ROOT/"results/plot_data/lmc_spray_test.jsonl").read_text().splitlines()]
R = [r for r in R if r["seed"] == 1]
mods = sorted({(r["an"], r["om"], r["size"]) for r in R}); Q = ("pmra", "pmdec", "vlos", "d")
lab = dict(pmra="pmra [mas/yr]", pmdec="pmdec [mas/yr]", vlos="v_los [km/s]", d="distance [kpc]")
fig, ax = plt.subplots(len(mods), 4, figsize=(15, 3.2*len(mods)), constrained_layout=True, squeeze=False)
for i, mo in enumerate(mods):
    rs = sorted([r for r in R if (r["an"], r["om"], r["size"]) == mo], key=lambda r: r["mass"])
    for r, c in zip(rs, plt.cm.viridis(np.linspace(0, 0.9, len(rs)))):
        for j, q in enumerate(Q):
            y = np.array([np.nan if v is None else v for v in r["med"][q]])
            ax[i, j].plot(r["phi1"], y, "-o", ms=2, color=c, label=f"M_LMC {r['mass']:.1e}, lnL {r['lnL']:.0f}")
    for j, q in enumerate(Q):
        ax[i, j].set(xlabel="phi1 (chord frame) [deg]", ylabel=lab[q], title=f"bar {mo[0]:g} deg, Omega {mo[1]:g}, size {mo[2]:g}")
    ax[i, 0].legend(fontsize=7)
fig.suptitle("omega Cen trailing debris (< 700 Myr, |phi2| < 6 deg): footprint medians per 2-deg phi1 bin, seed 1, 8000 epochs")
fig.savefig(ROOT/"plots/lmc_spray_test.png", dpi=110); print("plots/lmc_spray_test.png")
