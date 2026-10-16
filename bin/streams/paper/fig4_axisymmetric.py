#!/usr/bin/env python3
"""Paper Fig. 4: no axisymmetric host reproduces the proper-motion turn.

Hosts: the Hunter et al. (2024) axisymmetric model with omega Cen's distance and proper motions refitted (no bar), and, at
that same cluster state, its variants (halo flattening q = 0.8, 1.2; disc mass x0.8, x1.2; v_c(R0) = 220, 240 km/s) and the
McMillan (2017) model (bin/streams/paper/paper_models.py variants; 32000 release epochs, seed 4). Dashed: the barred
16-deg solution. (a) model track across the stream: median Delta phi2 of the debris in 1-deg phi1 bins (|Delta phi2| < 4).
(b, c) model proper motions predicted at the members' sky positions (median of particles within 0.5 deg; as Fig. 3),
binned like the data; black points: data medians. Output: plots/paper_fig4_axisymmetric.pdf/.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"bin/streams")); sys.path.insert(0, str(ROOT/"bin/streams/paper")); sys.path.insert(0, str(ROOT/"src"))
import numpy as np

from fig2_models import load
from fig3_residuals import binned, predict
from grid4_common import GC, d, du, dx, project_gc
from paper_style import COLOURS, COLUMN, setup

plt = setup()
VARIANTS = [("halo_q0p8", r"halo $q=0.8$", "#a6cee3"), ("halo_q1p2", r"halo $q=1.2$", "#1f78b4"),
            ("disc_x0p8", r"disc $\times0.8$", "#b2df8a"), ("disc_x1p2", r"disc $\times1.2$", "#33a02c"),
            ("vc220", r"$v_{\rm c}=220$", "#fb9a99"), ("vc240", r"$v_{\rm c}=240$", "#e31a1c"),
            ("mcmillan17", "McMillan (2017)", "#ff7f00")]


def load_variant(name):
    m = dict(np.load(ROOT/f"results/streams/paper/variant_{name}_seed4.npz"))
    k = (m["chi"] > 0) & (m["age"] < 700.)
    mu, mx = project_gc(m["l"][k], m["b"][k], GC)
    return dict(phi1=mu, dphi2=mx, pmra=m["pmra"][k], pmdec=m["pmdec"][k])


def track(m, edges):
    c, med = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        k = (m["phi1"] >= a) & (m["phi1"] < b) & (np.abs(m["dphi2"]) < 4)
        if k.sum() >= 20:
            c.append(0.5*(a+b)); med.append(np.median(m["dphi2"][k]))
    return np.array(c), np.array(med)


def main():
    rec = json.loads((ROOT/"results/plot_data/paper_models.json").read_text())
    rng = np.random.default_rng(3)
    edges2 = np.arange(-2., 28.1, 2.); edges1 = np.arange(-2., 28.1, 1.)
    fig, ax = plt.subplots(3, 1, figsize=(COLUMN, 6.0), sharex=True, gridspec_kw=dict(hspace=0.07, height_ratios=[0.8, 1, 1]))
    models = [(n, lab, col, load_variant(n)) for n, lab, col in VARIANTS]
    models = [("null_fit", "Hunter axisymmetric", COLOURS["null"], load("null_fit"))]+models
    for name, lab, col, m in models+[("bar16", r"bar, $16^\circ$", COLOURS["bar16"], load("bar16"))]:
        ls = "--" if name == "bar16" else "-"; lw = 1.3 if name in ("null_fit", "bar16") else 0.8
        c, t = track(m, edges1); ax[0].plot(c, t, ls, color=col, lw=lw)
        pred = predict(m)
        for i, q in enumerate(("pmra", "pmdec")):
            c, med, _ = binned(du, pred[q], edges2, rng)
            ax[i+1].plot(c, med, ls, color=col, lw=lw, label=lab)
    k = np.abs(dx) < 4
    c, t = track(dict(phi1=du[k], dphi2=dx[k]), edges1); ax[0].plot(c, t, "ko", ms=2.5)
    for i, q in enumerate(("pmra", "pmdec")):
        c, med, err = binned(du, d[q], edges2, rng)
        ax[i+1].errorbar(c, med, err, fmt="ko", ms=2.8, capsize=0, lw=0.8, zorder=5)
    ax[0].set_ylabel(r"$\Delta\phi_2$ [deg]"); ax[0].set_ylim(-3.5, 2.5)
    ax[1].set_ylabel(r"$\mu_{\alpha*}$ [mas yr$^{-1}$]"); ax[2].set_ylabel(r"$\mu_\delta$ [mas yr$^{-1}$]")
    ax[2].set_xlabel(r"$\phi_1$ [deg]"); ax[1].set_xlim(-3, 28)
    ax[1].legend(loc="lower left", fontsize=5.6, ncol=2, columnspacing=0.6, handlelength=1.6)
    for j, a in enumerate(ax):
        a.text(0.97, 0.94, "abc"[j], transform=a.transAxes, va="top", ha="right", fontweight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(ROOT/f"plots/paper_fig4_axisymmetric.{ext}")
    print(ROOT/"plots/paper_fig4_axisymmetric.pdf")
    for n, r in rec.get("variants", {}).items():
        print(n, round(r["lnL"], 1))


if __name__ == "__main__":
    main()
