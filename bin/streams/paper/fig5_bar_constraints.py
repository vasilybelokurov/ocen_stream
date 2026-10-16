#!/usr/bin/env python3
"""Paper Fig. 5: what the stream says about the bar. Three-dimensional sweep over present bar angle (16-36 deg), pattern
speed (30-40 km/s/kpc) and bar size factor S (0.9-1.5), bar amplitude 1.2, omega Cen at d = 5.6 kpc and catalogue proper
motions, 8000 release epochs, seed 1 (bin/streams/bar3d_sweep.py; 770 sprays, results/plot_data/bar3d_sweep/). Bars that
reach beyond corotation (S x 4.25 kpc > R_CR) are excluded.

(a) Delta ln L relative to the best sweep point, maximised over S at each (angle, pattern speed). Points: best S.
(b) the same maximised over pattern speed and S, against angle; labels give (Omega_b, S) of the best point.
Shaded: literature bar angle 27 +- 2 deg (Wegg & Gerhard 2013; Bland-Hawthorn & Gerhard 2016) and pattern speeds
33.29 +- 1.81 (Clarke & Gerhard 2022) and 39.0 +- 3.5 km/s/kpc (Portail et al. 2017). Grey band in (b): +-150, the
typical seed-to-seed scatter of ln L at this resolution. ln L differences are not calibrated (members treated as
independent) and are used only to rank models. Output: plots/paper_fig5_bar_constraints.pdf/.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"bin/streams/paper"))
import numpy as np

from paper_style import PAGE, setup

plt = setup()


def main():
    rows = [json.loads(l) for f in sorted((ROOT/"results/plot_data/bar3d_sweep").glob("w*.jsonl")) for l in f.open() if l.strip()]
    rows = [r for r in rows if r.get("lnL") is not None]
    ang = np.array([r["an"] for r in rows]); om = np.array([r["om"] for r in rows]); sz = np.array([r["size"] for r in rows])
    lnl = np.array([r["lnL"] for r in rows]); best = lnl.max(); dl = lnl-best
    A, O = np.unique(ang), np.unique(om)
    grid = np.full((len(O), len(A)), np.nan); bsz = np.full_like(grid, np.nan)
    for i, o in enumerate(O):
        for j, a in enumerate(A):
            k = (ang == a) & (om == o)
            if k.any():
                grid[i, j] = dl[k].max(); bsz[i, j] = sz[k][np.argmax(dl[k])]
    fig, ax = plt.subplots(1, 2, figsize=(PAGE, 2.9), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.42))
    da, do = np.diff(A).mean(), np.diff(O).mean()
    im = ax[0].pcolormesh(np.append(A-da/2, A[-1]+da/2), np.append(O-do/2, O[-1]+do/2), np.clip(grid, -2000, 0), cmap="viridis",
                          vmin=-2000, vmax=0, shading="flat")
    cb = fig.colorbar(im, ax=ax[0], pad=0.02); cb.set_label(r"$\Delta\ln L$ (max over $S$)")
    for i, o in enumerate(O):
        for j, a in enumerate(A):
            if np.isfinite(grid[i, j]) and grid[i, j] > -600:
                ax[0].text(a, o, f"{bsz[i, j]:.1f}", ha="center", va="center", fontsize=4.8, color="w" if grid[i, j] < -300 else "k")
    ax[0].axvspan(25, 29, color="w", alpha=0.18, lw=0)
    for mu, lab in ((33.29, "Clarke & Gerhard 2022"), (39.0, "Portail et al. 2017")):
        ax[0].axhline(mu, color="w", lw=0.7, ls="--")
        ax[0].text(36.8, mu+0.15, lab, color="w", fontsize=5.5, ha="right", va="bottom")
    ax[0].set_ylim(O[0]-do/2, O[-1]+do/2)
    ax[0].set(xlabel=r"present bar angle $\phi_{\rm b}$ [deg]", ylabel=r"$\Omega_{\rm b}$ [km s$^{-1}$ kpc$^{-1}$]")
    prof = np.nanmax(grid, axis=0)
    ax[1].axhspan(-150, 0, color="0.85", lw=0); ax[1].axvspan(25, 29, color="#d95f02", alpha=0.15, lw=0)
    ax[1].plot(A, prof, "ko-", ms=3, lw=0.9)
    for j, a in enumerate(A):
        k = ang == a; i = np.argmax(dl[k])
        ax[1].annotate(f"{om[k][i]:.0f}\n{sz[k][i]:.1f}", (a, prof[j]), xytext=(0, 9 if j % 2 else -15), textcoords="offset points",
                       fontsize=5, ha="center", va="center", linespacing=0.9)
    ax[1].set(xlabel=r"present bar angle $\phi_{\rm b}$ [deg]", ylabel=r"$\Delta\ln L$ (max over $\Omega_{\rm b}$, $S$)", ylim=(-1500, 120))
    ax[1].text(27, 60, "literature", ha="center", fontsize=6, color="#d95f02")
    ax[1].text(0.97, 0.06, r"labels: $\Omega_{\rm b}$, $S$ of the best point", transform=ax[1].transAxes, ha="right", fontsize=5.5)
    for k, a in enumerate(ax):
        a.text(0.03 if k == 0 else 0.95, 0.96, "ab"[k], transform=a.transAxes, va="top", ha="left" if k == 0 else "right",
               fontweight="bold", color="w" if k == 0 else "k")
    for ext in ("pdf", "png"):
        fig.savefig(ROOT/f"plots/paper_fig5_bar_constraints.{ext}")
    print(ROOT/"plots/paper_fig5_bar_constraints.pdf", "points", len(rows), "best", rows[int(np.argmax(lnl))]["tag"])


if __name__ == "__main__":
    main()
