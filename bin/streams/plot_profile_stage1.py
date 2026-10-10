#!/usr/bin/env python3
"""Plots for profile stage 1 (profile_stage1.py):
  plots/profile_s1_params.png  lnL (mean of 3 seeds, error bar = seed sd/sqrt 3) vs each nuisance parameter, 28 vs 16 deg;
  plots/profile_s1_best.png    best 28-deg and best 16-deg points (seed 1): trailing debris (age < 700 Myr) as 2D histograms in
                               chord phi1 vs dphi2 / pmra / pmdec, members overplotted.
Usage: python bin/streams/plot_profile_stage1.py"""
import json
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from grid4_common import ROOT, project_gc, GC, du, dx, d

R = json.loads((ROOT/"results/plot_data/profile_stage1.json").read_text())
P = {a: [r for r in R if r["an"] == a] for a in (28., 16.)}
labs = dict(om="Omega_b [km/s/kpc]", am="bar amplitude", size="bar size", dist="omega Cen d [kpc]", pmra="pmra [mas/yr]", pmdec="pmdec [mas/yr]")
fig, ax = plt.subplots(2, 3, figsize=(14, 7.5), sharey=True); ymax = max(r["lnL"] for r in R)
for a, (k, lab) in zip(ax.flat, labs.items()):
    for an, c in ((28., "C3"), (16., "C0")):
        x = [r[k] for r in P[an]]; y = [r["lnL"] for r in P[an]]; e = [np.std(r["lnL_seeds"], ddof=1)/np.sqrt(3) for r in P[an]]
        a.errorbar(x, y, e, fmt="o", color=c, ms=4, alpha=.8, label=f"{an:g} deg")
    a.set_xlabel(lab); a.set_ylim(ymax-3500, ymax+300); a.grid(alpha=.3)
    if k == "dist":
        a.axvspan(5.38, 5.48, color="g", alpha=.15, label="catalogue 5.43+-0.05")
    if k in ("pmra", "pmdec"):
        a.axvline(-3.2499 if k == "pmra" else -6.7461, color="g", ls="--")
for a in ax[:, 0]:
    a.set_ylabel("lnL (mean of 3 seeds)")
ax[0, 0].legend(fontsize=8); ax[1, 0].legend(fontsize=8); fig.suptitle("Profile stage 1: 40 shared nuisance points, 1x particles")
fig.tight_layout(); fig.savefig(ROOT/"plots/profile_s1_params.png", dpi=110)

from scipy.ndimage import gaussian_filter
fig, ax = plt.subplots(3, 2, figsize=(13, 10.5), sharex=True)
for col, an in enumerate((28., 16.)):
    b = max(P[an], key=lambda r: r["lnL"])
    ms = [dict(np.load(ROOT/f"results/streams/profile_s1/{b['tag']}_s{s}.npz")) for s in (1, 2, 3)]
    m = {q: np.concatenate([z[q] for z in ms]) for q in ms[0]}
    mu, mx = project_gc(m["l"], m["b"], GC); k = (m["age"] < 700) & (m["chi"] > 0)
    for row, (my, dy, lab, lim) in enumerate(((mx, dx, "dphi2 [deg]", (-6, 6)), (m["pmra"], d["pmra"], "pmra [mas/yr]", (-22, -1)),
                                              (m["pmdec"], d["pmdec"], "pmdec [mas/yr]", (-14, -4)))):
        a = ax[row, col]; xr = (du.min()-1, du.max()+1)
        H, xe, ye = np.histogram2d(mu[k], my[k], bins=(90, 120), range=(xr, lim)); H = gaussian_filter(H, (1.0, 1.5))
        Hc = H/np.maximum(H.max(1, keepdims=True), 1e-12)          # conditional on phi1: each column scaled to its peak
        a.plot(du, dy, ".", color="0.55", ms=1.5, alpha=.5, zorder=1)
        a.contour(0.5*(xe[1:]+xe[:-1]), 0.5*(ye[1:]+ye[:-1]), Hc.T, levels=[0.1, 0.3, 0.6, 0.9], colors=["C0", "C2", "C1", "C3"], linewidths=1.2, zorder=2)
        a.set_ylabel(lab); a.set_ylim(lim)
    ax[0, col].set_title(f"{an:g} deg  Om {b['om']:.1f}  amp {b['am']:.2f}  size {b['size']:.2f}  d {b['dist']:.2f}  lnL {b['lnL']:.0f}", fontsize=10)
    ax[2, col].set_xlabel("chord phi1 [deg]")
fig.suptitle("contours: model density per phi1 column (0.1/0.3/0.6/0.9 of peak; seeds 1-3 pooled); grey: members", fontsize=10)
fig.tight_layout(); fig.savefig(ROOT/"plots/profile_s1_best.png", dpi=110); print("saved")
