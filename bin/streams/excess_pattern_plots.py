#!/usr/bin/env python3
"""Figures for the excess-pattern test (bin/streams/excess_pattern_test.py; 2026-10-21).
plots/excess_pattern_recovery.png: recovered fraction (lnL - published)/(fitted - published) per variant, 25 and 12.5 deg.
plots/excess_pattern_debris.png  : 25-deg debris (chi > 0, age < 700 Myr, coloured by release age) vs members for the fitted bar,
                                   the published bar, the best static excess (phase 90) and the co-rotating excess at phase 45.
Usage: PYTHONPATH=src python bin/streams/excess_pattern_plots.py
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from grid4_common import d, du, dx, project_gc, GC

R = json.loads((ROOT/"results/plot_data/excess_pattern_test.json").read_text())["runs"]
order = ["co-rotating ph0", "co-rotating ph45", "co-rotating ph90", "static ph0", "static ph45", "static ph90", "static ph135",
         "Omega22.5 ph0", "Omega22.5 ph90", "counter-rotating ph0"]
fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
for j, (lab, col) in enumerate((("25", "C3"), ("12.5", "C0"))):
    rr = {r["name"]: r for r in R if r["angle_label"] == lab}
    ax.bar(np.arange(len(order))+0.2*(2*j-1), [rr[k]["recovered"] for k in order], 0.4, color=col, label=f"{lab} deg")
ax.axhline(1, color="k", lw=0.8, ls="--"); ax.axhline(0, color="k", lw=0.8); ax.axhspan(0.3, 0.7, color="0.92", zorder=0)
ax.text(len(order)-0.5, 0.5, "undecided\n(30-70%)", ha="right", va="center", fontsize=8)
ax.set_xticks(np.arange(len(order))); ax.set_xticklabels([k.replace(" ph", "\nphase ") for k in order], fontsize=8)
ax.set(ylabel="recovered fraction of the gap\n(1 = inflated bar, 0 = published bar alone)", ylim=(-1.8, 1.2),
       title="Published bar + the inflated bar's excess as a separate pattern (own pattern speed / phase), fixed best-fit parameters")
ax.legend(); fig.savefig(ROOT/"plots/excess_pattern_recovery.png", dpi=110); print("plots/excess_pattern_recovery.png")
runs = [("25 fitted bar", "a25_fitted_bar"), ("25 published bar", "a25_published_bar"), ("25 static excess, phase 90", "a25_static_ph90"),
        ("25 co-rotating excess, phase 45", "a25_co-rotating_ph45")]
fig, ax = plt.subplots(4, 4, figsize=(20, 15), sharex=True, constrained_layout=True)
for j, (title, rid) in enumerate(runs):
    m = dict(np.load(ROOT/f"results/streams/excess_pattern_20261021/runs/{rid}.npz")); u, x = project_gc(m["l"], m["b"], GC)
    k = (m["chi"] > 0) & (m["age"] < 700) & (u > -6) & (u < 32); o = np.argsort(-m["age"][k])
    rec = [r for r in R if r["run_id"] == rid][0]
    for i, (q, dq, yl, lim) in enumerate((("x", dx, r"$\Delta\phi_2$ [deg]", (-12, 15)), ("pmra", d["pmra"], r"$\mu_{\alpha*}$", (-28, 0)),
                                          ("pmdec", d["pmdec"], r"$\mu_\delta$", (-17, -2)), ("vlos", d["v"], r"$v_{\rm los}$", (-50, 300)))):
        val = x if q == "x" else m[q]
        sc = ax[i, j].scatter(u[k][o], val[k][o], c=m["age"][k][o], s=0.3, cmap="turbo", vmin=0, vmax=700, rasterized=True)
        ok = np.isfinite(dq); ax[i, j].plot(du[ok], dq[ok], "k.", ms=1.3 if q != "vlos" else 5, alpha=0.45)
        ax[i, j].set(ylim=lim, xlim=(-6, 32), ylabel=yl if j == 0 else None)
    ax[0, j].set_title(f"{title}\nrecovered {rec['recovered']:+.2f}, branch {rec['branch_all']:.2f}, far-end density {rec['far_end_density_ratio']:.2f}", fontsize=10)
    ax[3, j].set_xlabel(r"$\phi_1$ [deg]")
fig.colorbar(sc, ax=ax, shrink=0.4, label="release age [Myr ago]")
fig.savefig(ROOT/"plots/excess_pattern_debris.png", dpi=80); print("plots/excess_pattern_debris.png")
