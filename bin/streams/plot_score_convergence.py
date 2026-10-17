#!/usr/bin/env python3
"""Convergence of three stream-score weightings with release-epoch number, on the same sprays (2026-10-17).

Input: results/plot_data/particle_convergence_20261017_v2.json (particle_convergence.py with CONV_VERSION=_v2): models A
(refined 28), B (refined 16), C (27B long-bar track fit), E (A + axisymmetric variant), release epochs 4k-128k, seeds 1-5;
scores per spray: legacy (1/K over chi bins with >= 20 particles), pooled (one bin), chi200 (fixed 5-wide chi bins 0-200,
fixed weights). Rows: scores; columns: (a) seed sd of ln L per model and of the pair differences, with N^-1/2;
(b) mean ln L minus its value at 128k (bias); (c) pair differences A-B, A-C, A-E, mean +- sd over seeds.
Output: plots/score_convergence_20261017.png; numbers printed and in results/plot_data/score_convergence_20261017.json.
Usage: PYTHONPATH=src python bin/streams/plot_score_convergence.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
rec = json.loads((ROOT/"results/plot_data/particle_convergence_20261017_v2.json").read_text())
runs = rec["runs"]
N = sorted({r["nrel"] for r in runs}); S = sorted({r["seed"] for r in runs}); M = sorted({r["model"] for r in runs})
PAIRS = (("A", "B"), ("A", "C"), ("A", "E"))


def total(r, sc):
    return r["total"] if sc == "legacy" else r["scores"][sc]["total"]


R = {(r["model"], r["nrel"], r["seed"]): r for r in runs}
out = {}
fig, ax = plt.subplots(3, 3, figsize=(16, 12))
for i, sc in enumerate(("legacy", "pooled", "chi200")):
    o = out[sc] = dict(models={}, pairs={})
    for j, m in enumerate(M):
        t = np.array([[total(R[(m, n, s)], sc) for s in S] for n in N])
        o["models"][m] = dict(mean=t.mean(1).tolist(), sd=t.std(1, ddof=1).tolist())
        ax[i, 0].plot(N, t.std(1, ddof=1), "o-", color=f"C{j}", label=f"{m}")
        ax[i, 1].plot(N, t.mean(1)-t.mean(1)[-1], "o-", color=f"C{j}", label=m)
    for j, (a, b) in enumerate(PAIRS):
        dl = np.array([[total(R[(a, n, s)], sc)-total(R[(b, n, s)], sc) for s in S] for n in N])
        o["pairs"][f"{a}-{b}"] = dict(mean=dl.mean(1).tolist(), sd=dl.std(1, ddof=1).tolist())
        ax[i, 0].plot(N, dl.std(1, ddof=1), "s--", color=f"C{j+5}", label=f"{a}−{b}")
        ax[i, 2].errorbar(np.array(N)*(1+0.04*j), dl.mean(1), dl.std(1, ddof=1), fmt="s-", color=f"C{j+5}", capsize=2, label=f"{a}−{b}")
    ref = o["models"]["A"]["sd"][0]*np.sqrt(N[0]/np.array(N))
    ax[i, 0].plot(N, ref, "k:", label=r"$\propto N^{-1/2}$")
    ax[i, 0].set(xscale="log", yscale="log", ylabel=f"{sc}: seed sd")
    ax[i, 1].set(xscale="log", ylabel=f"{sc}: mean ln L − mean at 128k"); ax[i, 1].axhline(0, color="k", lw=.5)
    ax[i, 2].set(xscale="log", ylabel=f"{sc}: Δln L"); ax[i, 2].axhline(0, color="k", lw=.5)
    for a_ in ax[i]:
        a_.legend(fontsize=6.5)
for a_ in ax[-1]:
    a_.set_xlabel("release epochs")
ax[0, 0].set_title("seed scatter (models ●, pair differences ■)", fontsize=9)
ax[0, 1].set_title("bias: mean ln L relative to 128k", fontsize=9)
ax[0, 2].set_title("pair differences, mean ± sd over 5 seeds", fontsize=9)
fig.savefig(ROOT/"plots/score_convergence_20261017.png", dpi=130, bbox_inches="tight")
(ROOT/"results/plot_data/score_convergence_20261017.json").write_text(json.dumps(dict(nrel=N, scores=out), indent=1))
for sc, o in out.items():
    print(sc)
    for k, v in o["pairs"].items():
        print(f"   {k}: " + "  ".join(f"{n//1000}k {m_:+.0f}±{s_:.0f}" for n, m_, s_ in zip(N, v["mean"], v["sd"])))
    for k, v in o["models"].items():
        print(f"   {k}: sd " + " ".join(f"{s_:.0f}" for s_ in v["sd"]) + " | mean-128k " + " ".join(f"{m_-v['mean'][-1]:+.0f}" for m_ in v["mean"]))
