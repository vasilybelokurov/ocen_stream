#!/usr/bin/env python3
"""Paper Fig. 1: the data. Stream-54 members (STREAMFINDER DR3, Ibata et al. 2024; the 3681 used in the fits) on the sky,
in the stream frame, and their proper motions and line-of-sight velocities along the stream.

phi1, phi2: great-circle frame through omega Cen (results/plot_data/stream54_gc_frame_chord.json); dphi2 is the offset from
the 4th-order polynomial track used by the likelihood. Running medians in 2-deg phi1 bins with bootstrap errors (data only).
Open squares: proper motions of the strongest on-stream overdensity in an alternative Gaia DR3 selection without orbit
templates (bin/streams/m5_independent_selection.py; same footprint and CMD locus as the members), where it exceeds 4 sigma.
Output: plots/paper_fig1_data.pdf/.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"bin/streams")); sys.path.insert(0, str(ROOT/"bin/streams/paper")); sys.path.insert(0, str(ROOT/"src"))
import numpy as np

from grid4_common import GC, d, du, dx
from ocen_stream.cluster import galactic
from ocen_stream.path import gc_rotation, to_stream
from paper_style import COLOURS, PAGE, setup

plt = setup()


def running(x, y, edges, nboot=400, seed=1):
    rng = np.random.default_rng(seed); c, m, e = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        k = (x >= a) & (x < b) & np.isfinite(y)
        if k.sum() < 10:
            continue
        yy = y[k]; bs = np.median(rng.choice(yy, (nboot, yy.size)), axis=1)
        c.append(0.5*(a+b)); m.append(np.median(yy)); e.append(np.std(bs))
    return np.array(c), np.array(m), np.array(e)


def main():
    l0, b0 = galactic(); l0 = l0-360. if l0 > 180. else l0          # same wrapping as the members' l
    alt = json.loads((ROOT/"results/plot_data/m5_independent_selection.json").read_text())
    fig = plt.figure(figsize=(PAGE, 5.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.15, 2], hspace=0.22)
    top = outer[0].subgridspec(1, 2, wspace=0.3); bottom = outer[1].subgridspec(2, 1, hspace=0.06)
    ax_sky = fig.add_subplot(top[0]); ax_f = fig.add_subplot(top[1])
    ax_pa = fig.add_subplot(bottom[0]); ax_pd = fig.add_subplot(bottom[1], sharex=ax_pa)
    sc = ax_sky.scatter(d["l"], d["b"], c=du, s=1.2, lw=0, cmap="viridis", rasterized=True)
    ax_sky.plot(l0, b0, "*", color="k", ms=8, mec="w", mew=0.5)
    ax_sky.annotate(r"$\omega$ Cen", (l0, b0), xytext=(4, -9), textcoords="offset points", fontsize=7)
    ax_sky.set(xlabel=r"$\ell$ [deg]", ylabel=r"$b$ [deg]", xlim=(-20, -72), ylim=(12, 43), aspect="equal")
    cb = fig.colorbar(sc, ax=ax_sky, fraction=0.05, pad=0.02); cb.set_label(r"$\phi_1$ [deg]", fontsize=7); cb.ax.tick_params(labelsize=6)
    has_v = np.isfinite(d["v"])
    ax_f.scatter(du, dx, s=1.2, lw=0, color=COLOURS["grey"], rasterized=True)
    ax_f.scatter(du[has_v], dx[has_v], s=9, marker="o", facecolor="none", edgecolor="C3", lw=0.7, label=f"with $v_{{\\rm los}}$ ({has_v.sum()})")
    ax_f.axhline(0, color="k", lw=0.5)
    ax_f.set(xlabel=r"$\phi_1$ [deg]", ylabel=r"$\Delta\phi_2$ [deg]", ylim=(-9, 9))
    ax_f.legend(loc="lower right", handletextpad=0.2)
    edges = np.arange(-2., 28.1, 2.)
    for ax, q, lab, key in ((ax_pa, "pmra", r"$\mu_{\alpha*}$ [mas yr$^{-1}$]", "ridge_pmra"),
                            (ax_pd, "pmdec", r"$\mu_\delta$ [mas yr$^{-1}$]", "ridge_pmdec")):
        ax.scatter(du, d[q], s=1.0, lw=0, color=COLOURS["grey"], alpha=0.6, rasterized=True)
        c, m, e = running(du, d[q], edges)
        ax.errorbar(c, m, e, fmt="o", ms=3, color=COLOURS["data"], capsize=0, lw=0.8, label="members, median")
        sig = np.array(alt["peak_sig"]); ok = sig >= 4
        ax.plot(np.array(alt["phi1"])[ok], np.array(alt[key])[ok], "s", ms=4.5, mfc="none", mec=COLOURS["alt"], mew=0.9,
                label=r"alternative selection ($>4\sigma$)")
        ax.set_ylabel(lab)
    ax_pa.set_ylim(-22, 0); ax_pd.set_ylim(-14, -4)
    ax_pa.legend(loc="lower left", ncol=2); plt.setp(ax_pa.get_xticklabels(), visible=False)
    ax_pd.set_xlabel(r"$\phi_1$ [deg]"); ax_pa.set_xlim(-3, 28)
    ins = ax_pd.inset_axes([0.70, 0.56, 0.28, 0.40])
    ins.errorbar(du[has_v], d["v"][has_v], d["e_v"][has_v], fmt="o", ms=2, color="C3", lw=0.6, capsize=0)
    ins.set_xlabel(r"$\phi_1$", fontsize=6, labelpad=1); ins.set_ylabel(r"$v_{\rm los}$ [km s$^{-1}$]", fontsize=6, labelpad=1)
    ins.tick_params(labelsize=5.5, pad=1)
    for k, a in enumerate((ax_sky, ax_f, ax_pa, ax_pd)):
        a.text(0.02, 0.96, "abcd"[k], transform=a.transAxes, va="top", fontweight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(ROOT/f"plots/paper_fig1_data.{ext}")
    print(ROOT/"plots/paper_fig1_data.pdf", "N members", len(du), "with v_los", has_v.sum())


if __name__ == "__main__":
    main()
