#!/usr/bin/env python3
"""Live progress of profile_nm.py chains (2026-10-19). (a) best-so-far jackknife ln L (128k) vs evaluation number per chain;
(b) current best per angle (both chains) vs angle, with the starting values, the truth check score and the truth angle;
(c) parameters of each angle's current best point (Omega, amplitude, distance, v_c) vs angle, truth marked.
Usage: python bin/streams/plot_profile_progress.py --outdir results/streams/profile_T2c0_20261019 --truth-score -22174 --truth-angle 27
"""
import argparse, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(); ap.add_argument("--outdir", required=True); ap.add_argument("--truth-score", type=float)
ap.add_argument("--truth-angle", type=float); ap.add_argument("--truth", default='{"omega": 35, "amp": 1.0, "dist": 5.43, "vc": 229}')
a = ap.parse_args(); od = ROOT/a.outdir; truth = json.loads(a.truth)
ch = defaultdict(list)
for f in (od/"runs").glob("a*_e*.json"):
    m = re.match(r"a([\d.]+)_(\d)_e(\d+)", f.stem); r = json.loads(f.read_text())
    ch[(float(m.group(1)), int(m.group(2)))].append((int(m.group(3)), r["total_jk"], r["params"]))
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
angs = sorted({k[0] for k in ch}); cm = plt.get_cmap("viridis"); col = {g: cm(i/max(1, len(angs)-1)) for i, g in enumerate(angs)}
best = {}
for (g, j), ev in sorted(ch.items()):
    ev.sort(); y = np.maximum.accumulate([e[1] for e in ev])
    ax[0].plot([e[0] for e in ev], y, "-" if j == 0 else "--", color=col[g], label=f"{g:g} deg, chain {j}")
    i = int(np.argmax([e[1] for e in ev]))
    if g not in best or ev[i][1] > best[g][0]:
        best[g] = (ev[i][1], ev[i][2], ev[0][1])
if a.truth_score:
    ax[0].axhline(a.truth_score, color="k", ls=":", lw=1)
ax[0].set(xlabel="evaluation", ylabel="best so far, ln L (128k, jackknife)", title="(a) chains"); ax[0].legend(fontsize=6, ncol=2)
g = np.array(sorted(best)); ax[1].plot(g, [best[x][0] for x in g], "o-", label="current best (128k)")
ax[1].plot(g, [best[x][2] for x in g], "s:", color="0.5", label="first evaluation (start)")
if a.truth_score:
    ax[1].axhline(a.truth_score, color="k", ls="--", label="truth (256k, independent)")
if a.truth_angle:
    ax[1].axvline(a.truth_angle, color="C3", lw=0.8)
ax[1].set(xlabel="bar angle [deg]", ylabel="ln L", title="(b) profile so far (128k ~40-125 below 256k)"); ax[1].legend(fontsize=7)
for k, mk in (("omega", "o"), ("amp", "s"), ("dist", "^"), ("vc", "v")):
    v = np.array([best[x][1][k] for x in g]); t = truth.get(k)
    ax[2].plot(g, v/t if t else v, mk+"-", label=f"{k} / truth")
ax[2].axhline(1, color="k", lw=0.6); ax[2].set(xlabel="bar angle [deg]", ylabel="parameter / truth", title="(c) best-point parameters")
ax[2].legend(fontsize=7)
n = sum(len(v) for v in ch.values()); fig.suptitle(f"{a.outdir}: {n} evaluations so far", fontsize=9)
fig.savefig(ROOT/"plots/profile_progress.png", dpi=110); print(ROOT/"plots/profile_progress.png", n)
