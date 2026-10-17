#!/usr/bin/env python3
"""Where the seed noise of the pooled stream score comes from (2026-10-17).

Input: results/plot_data/particle_convergence_20261017_v2.json (pooled per-star ln L, models A = refined 28 deg and
B = refined 16 deg, seeds 1-5). For each member: seed standard deviation of its per-star ln L(A) - ln L(B).
(a, b) members in (phi1, Delta phi2) and (phi1, mu_delta) coloured by that sd at 128k epochs; (c) seed sd of the summed
A - B against release epochs: actual, and sqrt(sum of per-star variances) (what independent per-star noise would give);
(d) cumulative share of the per-star variance against phi1, at 32k and 128k.
Output: plots/score_noise_source_20261017.png
Usage: PYTHONPATH=src python bin/streams/plot_score_noise_source.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from grid4_common import d, du, dx

rec = json.loads((ROOT/"results/plot_data/particle_convergence_20261017_v2.json").read_text())
R = {(r["model"], r["nrel"], r["seed"]): r for r in rec["runs"]}
N = sorted({r["nrel"] for r in rec["runs"]}); S = sorted({r["seed"] for r in rec["runs"]})


def diff(n):
    return np.array([np.asarray(R[("A", n, s)]["scores"]["pooled"]["per_star"])-np.asarray(R[("B", n, s)]["scores"]["pooled"]["per_star"]) for s in S])


fig, ax = plt.subplots(1, 4, figsize=(22, 4.8), gridspec_kw=dict(wspace=0.5))
D = diff(N[-1]); sd = D.std(0, ddof=1)
o = np.argsort(sd)
for a, y, lab in ((ax[0], dx, r"$\Delta\phi_2$ [deg]"), (ax[1], d["pmdec"], r"$\mu_\delta$ [mas/yr]")):
    sc = a.scatter(du[o], y[o], c=sd[o], s=3, cmap="magma_r", vmin=0, vmax=np.percentile(sd, 99), lw=0)
    a.set(xlabel=r"$\phi_1$ [deg]", ylabel=lab, xlim=(-6, 30))
ax[0].set_ylim(-9, 22); ax[1].set_ylim(-15, -4)
cb = fig.colorbar(sc, ax=ax[1], fraction=0.05, pad=0.02); cb.set_label("per-star seed sd (128k)")
act = [diff(n).sum(1).std(ddof=1) for n in N]
ind = [np.sqrt(diff(n).var(0, ddof=1).sum()) for n in N]
ax[2].plot(N, act, "o-", label="actual seed sd of Σ (A − B)")
ax[2].plot(N, ind, "s--", label="if stars were independent: √Σ var")
ax[2].plot(N, act[0]*np.sqrt(N[0]/np.array(N)), "k:", label=r"$\propto N^{-1/2}$")
ax[2].set(xscale="log", yscale="log", xlabel="release epochs", ylabel="seed sd of Δln L (A − B)"); ax[2].legend(fontsize=7)
for n, c in ((32000, "C0"), (128000, "C3")):
    v = diff(n).var(0, ddof=1); oo = np.argsort(du)
    ax[3].plot(du[oo], np.cumsum(v[oo])/v.sum(), color=c, label=f"{n//1000}k")
ax3b = ax[3].twinx(); ax3b.hist(du, bins=np.arange(-6, 31, 1), color="0.85", zorder=0); ax3b.set_ylabel("members per 1°", color="0.5")
ax[3].set_zorder(ax3b.get_zorder()+1); ax[3].patch.set_visible(False)
ax[3].set(xlabel=r"$\phi_1$ [deg]", ylabel="cumulative share of per-star variance"); ax[3].legend(fontsize=7, loc="upper left")
for a, t in zip(ax, "abcd"):
    a.text(.03, .95, t, transform=a.transAxes, fontweight="bold", va="top")
fig.savefig(ROOT/"plots/score_noise_source_20261017.png", dpi=130, bbox_inches="tight")
print(ROOT/"plots/score_noise_source_20261017.png")
