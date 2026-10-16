#!/usr/bin/env python3
"""Paper Fig. 3: where along the stream each model succeeds or fails.

(a, b) Proper-motion residuals, data minus model, in 2-deg phi1 bins. The model prediction for a member is the median
proper motion of the model particles within 0.5 deg of the member's (phi1, dphi2); observed proper motions never select
particles. Bins: median over members with a prediction, bootstrap error.
(c) Cumulative difference in per-member ln L along phi1 between each barred solution and the refitted no-bar model.
Models as in Fig. 2 (seed 4, 32000 release epochs). Output: plots/paper_fig3_residuals.pdf/.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"bin/streams")); sys.path.insert(0, str(ROOT/"bin/streams/paper")); sys.path.insert(0, str(ROOT/"src"))
import numpy as np
from scipy.spatial import cKDTree

from fig2_models import MODELS, load
from grid4_common import d, du, dx
from paper_style import COLOURS, COLUMN, setup

plt = setup()
COL = dict(null_fit=COLOURS["null"], bar28=COLOURS["bar28"], bar16=COLOURS["bar16"])
SHORT = dict(null_fit="no bar", bar28=r"bar, $28^\circ$", bar16=r"bar, $16^\circ$")


def predict(m, radius=0.5):
    tree = cKDTree(np.column_stack((m["phi1"], m["dphi2"])))
    nb = tree.query_ball_point(np.column_stack((du, dx)), radius)
    out = {q: np.full(len(du), np.nan) for q in ("pmra", "pmdec")}
    for i, idx in enumerate(nb):
        if len(idx) >= 5:
            for q in out:
                out[q][i] = np.median(m[q][idx])
    return out


def binned(x, y, edges, rng):
    c, med, err = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        k = (x >= a) & (x < b) & np.isfinite(y)
        if k.sum() < 15:
            continue
        yy = y[k]; c.append(0.5*(a+b)); med.append(np.median(yy))
        err.append(np.std(np.median(rng.choice(yy, (300, yy.size)), axis=1)))
    return np.array(c), np.array(med), np.array(err)


def main():
    rng = np.random.default_rng(2)
    edges = np.arange(-2., 28.1, 2.)
    fig, ax = plt.subplots(3, 1, figsize=(COLUMN, 5.6), sharex=True, gridspec_kw=dict(hspace=0.07))
    ref = np.load(ROOT/"results/streams/paper/null_fit_seed4_perstar.npy"); order = np.argsort(du)
    for j, (name, _) in enumerate(MODELS):
        pred = predict(load(name))
        for i, q in enumerate(("pmra", "pmdec")):
            c, med, err = binned(du, d[q]-pred[q], edges, rng)
            ax[i].errorbar(c+0.25*(j-1), med, err, fmt="o-", ms=2.5, lw=0.8, color=COL[name], capsize=0, label=SHORT[name])
        if name != "null_fit":
            dl = np.load(ROOT/f"results/streams/paper/{name}_seed4_perstar.npy")-ref
            ax[2].plot(du[order], np.cumsum(dl[order]), color=COL[name], lw=1.0, label=SHORT[name]+" minus no bar")
    for a in ax[:2]:
        a.axhline(0, color="k", lw=0.5)
    ax[0].set_ylabel(r"$\Delta\mu_{\alpha*}$ [mas yr$^{-1}$]"); ax[1].set_ylabel(r"$\Delta\mu_\delta$ [mas yr$^{-1}$]")
    ax[2].set_ylabel(r"cumulative $\Delta\ln L$"); ax[2].set_xlabel(r"$\phi_1$ [deg]"); ax[2].axhline(0, color="k", lw=0.5)
    ax[0].legend(loc="lower left", ncol=3, columnspacing=0.8, handletextpad=0.3); ax[2].legend(loc="upper left")
    ax[0].set_xlim(-3, 28)
    for k, a in enumerate(ax):
        a.text(0.97, 0.94, "abc"[k], transform=a.transAxes, va="top", ha="right", fontweight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(ROOT/f"plots/paper_fig3_residuals.{ext}")
    print(ROOT/"plots/paper_fig3_residuals.pdf")


if __name__ == "__main__":
    main()
