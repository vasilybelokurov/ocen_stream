#!/usr/bin/env python3
"""Paper Fig. 2: data versus three models along the stream: the axisymmetric Hunter et al. (2024) host with omega Cen's
distance and proper motions refitted (no bar), and the barred host at the refined 28-deg and 16-deg solutions.

Model: particle-spray debris (trailing arm, released in the last 700 Myr, Gibbons chi > 0: the particles the likelihood
uses), 32000 release epochs, seed 4 (bin/streams/paper/paper_models.py display). Grey scale: log particle density per
panel; black points: members. Column titles: ln L relative to the 16-deg solution, mean and scatter over seeds 4-6.
Output: plots/paper_fig2_models.pdf/.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"bin/streams")); sys.path.insert(0, str(ROOT/"bin/streams/paper")); sys.path.insert(0, str(ROOT/"src"))
import numpy as np
from matplotlib.colors import LogNorm

from grid4_common import GC, d, du, dx, project_gc
from paper_style import PAGE, setup

plt = setup()
MODELS = [("null_fit", "No bar (state refitted)"), ("bar28", r"Bar, $\phi_{\rm b}=28^\circ$"), ("bar16", r"Bar, $\phi_{\rm b}=16^\circ$")]


def load(name):
    m = dict(np.load(ROOT/f"results/streams/paper/{name}_seed4.npz"))
    k = (m["chi"] > 0) & (m["age"] < 700.)
    mu, mx = project_gc(m["l"][k], m["b"][k], GC)
    return dict(phi1=mu, dphi2=mx, pmra=m["pmra"][k], pmdec=m["pmdec"][k], vlos=m["vlos"][k])


def main():
    rec = json.loads((ROOT/"results/plot_data/paper_models.json").read_text())["display"]
    ref = np.array([rec["bar16"]["lnL"][s] for s in ("4", "5", "6")])
    rows = [("dphi2", r"$\Delta\phi_2$ [deg]", (-6, 6)), ("pmra", r"$\mu_{\alpha*}$ [mas yr$^{-1}$]", (-22, 0)),
            ("pmdec", r"$\mu_\delta$ [mas yr$^{-1}$]", (-14, -4)), ("vlos", r"$v_{\rm los}$ [km s$^{-1}$]", (160, 270))]
    data = dict(dphi2=dx, pmra=d["pmra"], pmdec=d["pmdec"], vlos=d["v"])
    fig, ax = plt.subplots(4, 3, figsize=(PAGE, 7.0), sharex=True, sharey="row", gridspec_kw=dict(hspace=0.06, wspace=0.05))
    xe = np.linspace(-3, 28, 125)
    for j, (name, title) in enumerate(MODELS):
        m = load(name); r = rec[name]; p = r["params"]
        dl = np.array([r["lnL"][s] for s in ("4", "5", "6")])-ref
        head = title+"\n"+(r"$\Delta\ln L=0$" if name == "bar16" else rf"$\Delta\ln L={dl.mean():.0f}\pm{dl.std(ddof=1):.0f}$")
        head += rf", $D={p['dist']:.2f}$ kpc"
        ax[0, j].set_title(head, fontsize=7.5)
        for i, (q, lab, lim) in enumerate(rows):
            ye = np.linspace(*lim, 90)
            h, _, _ = np.histogram2d(m["phi1"], m[q], (xe, ye))
            if h.max() > 0:
                ax[i, j].pcolormesh(xe, ye, h.T, cmap="Greys", norm=LogNorm(vmin=1, vmax=h.max()), rasterized=True)
            ok = np.isfinite(data[q])
            ax[i, j].scatter(du[ok], data[q][ok], s=(5 if q == "vlos" else 0.6), lw=0, color="#d95f02", alpha=0.9 if q == "vlos" else 0.5,
                             rasterized=True)
            ax[i, j].set_ylim(*lim)
            if j == 0:
                ax[i, j].set_ylabel(lab)
        ax[-1, j].set_xlabel(r"$\phi_1$ [deg]")
    ax[0, 0].set_xlim(-3, 28)
    ax[0, 2].text(0.97, 0.06, "members", color="#d95f02", transform=ax[0, 2].transAxes, ha="right", fontsize=7)
    ax[0, 2].text(0.97, 0.16, "model debris", color="0.3", transform=ax[0, 2].transAxes, ha="right", fontsize=7)
    for ext in ("pdf", "png"):
        fig.savefig(ROOT/f"plots/paper_fig2_models.{ext}")
    print(ROOT/"plots/paper_fig2_models.pdf")


if __name__ == "__main__":
    main()
