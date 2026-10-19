#!/usr/bin/env python3
"""Summary of the angle-stratified search on mock T2c0 (truth angle 27 deg; 2026-10-19).
(a) best measured jackknife ln L per angle bin after each wave (training set = wave 0); truth check score as a line.
(b) emulator angle profile (max over the other 8 parameters, L-BFGS) before the waves and after wave 3; truth angle.
(c) emulator prediction vs measured mean at the checked predicted maxima (all mocks/data so far), with 1:1 line.
Output: plots/search_T2c0_summary.png
Usage: OMP_NUM_THREADS=4 PYTHONPATH=src python bin/streams/plot_search_T2.py
"""
import sys, json, glob, hashlib, warnings
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
warnings.filterwarnings("ignore")
import search_wave as SW
from scipy.optimize import minimize
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

CAT = ROOT/"results/streams/mocks_20261018/T2/mock_wide_c0.npz"; BASE = ROOT/"results/streams/rescore_mockwide_T2c0_20261018"
W = [ROOT/"results/streams/search_T2c0_wave1_20261018", ROOT/"results/streams/search_T2c0_wave2_20261019", ROOT/"results/streams/search_T2c0_wave3_20261019"]
TRUTH, TANG = -22174., 27.
csha = hashlib.sha256(CAT.read_bytes()).hexdigest(); bins = [(10, 15), (15, 20), (20, 25), (25, 30), (30, 35)]


def best_per_bin(rows):
    out = []
    for lo, hi in bins:
        y = [r["y"] for r in rows if lo <= 10+25*r["unit"][0] < hi and r["purpose"] != SW.HELD]
        out.append(max(y))
    return out


def profile(em, g):
    a_s = np.arange(10.5, 35, 1.25); p = []
    for a in a_s:
        ua = (a-10)/25; bv = -np.inf
        for x0 in g["X"][np.argsort(-g["ybar"])[:8]]:
            r = minimize(lambda z: -em.predict(np.insert(z, 0, ua)[None], return_std=False)[0], x0[1:], method="L-BFGS-B", bounds=[(0, 1)]*8)
            bv = max(bv, -r.fun)
        p.append(bv)
    return a_s, np.array(p)


fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
for k in range(4):
    rows = SW.observations(BASE, W[:k], csha); b = best_per_bin(rows)
    ax[0].plot([0.5*(lo+hi) for lo, hi in bins], b, "o-", label=f"after wave {k}" if k else "training set")
ax[0].axhline(TRUTH, color="k", ls="--", label="truth (independent spray)"); ax[0].axvline(TANG, color="C3", lw=0.8)
ax[0].set(xlabel="bar angle bin [deg]", ylabel="best measured ln L (jackknife)", title="(a) search progress per angle bin"); ax[0].legend(fontsize=7)
for k, lab in ((0, "before the waves"), (3, "after wave 3")):
    g, s2, em = SW.fit(SW.observations(BASE, W[:k], csha)); a_s, p = profile(em, g)
    ax[1].plot(a_s, p, "-", label=lab)
ax[1].axhline(TRUTH, color="k", ls="--", label="truth score"); ax[1].axvline(TANG, color="C3", lw=0.8, label="truth angle")
ax[1].set(xlabel="bar angle [deg]", ylabel="emulator: max ln L over the rest", title="(b) emulator angle profile (mock T2)"); ax[1].legend(fontsize=7)
chk = [("real, round 1", -21326, 1100, -22094, 11), ("real, round 2", -22120, 600, -22771, 15), ("real, final", -21136, 530, -21569, 11),
       ("mock T2, wave 3", -21915, 370, -23836, 11)]
for lab, mu, sd, y, se in chk:
    ax[2].errorbar(mu, y, xerr=sd, yerr=se, fmt="o", capsize=2, label=lab)
lim = [-24200, -20800]; ax[2].plot(lim, lim, "k:", lw=0.8); ax[2].set(xlim=lim, ylim=lim, xlabel="emulator prediction (+- 1 sd)",
                                                                           ylabel="measured (8 or 4 seeds)", title="(c) checked predicted maxima")
ax[2].legend(fontsize=7)
fig.savefig(ROOT/"plots/search_T2c0_summary.png", dpi=120); print(ROOT/"plots/search_T2c0_summary.png")
