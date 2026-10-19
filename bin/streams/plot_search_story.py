#!/usr/bin/env python3
"""The mock-T2 story in one figure (2026-10-19): how the angle bias was found, diagnosed and addressed.
(a) emulator angle profile before the waves (truth 27 deg); (b) oracle ablation: max-angle error without / with one truth
observation (T2c0, T2c1); (c) best measured ln L per angle bin after each wave; (d) emulator profile after wave 3 vs the
measured check points at its per-angle maximisers; (e) direct Nelder-Mead profile so far (128k best per angle, live);
(f) emulator predictions vs measured values at checked predicted maxima. Truth score (independent 256k spray) -22174.
Output: plots/search_T2c0_story.png
Usage: OMP_NUM_THREADS=4 PYTHONPATH=src python bin/streams/plot_search_story.py
"""
import sys, json, glob, re, hashlib, warnings
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
warnings.filterwarnings("ignore")
import search_wave as SW
from scipy.optimize import minimize
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
CAT = ROOT/"results/streams/mocks_20261018/T2/mock_wide_c0.npz"; BASE = ROOT/"results/streams/rescore_mockwide_T2c0_20261018"
W = [ROOT/"results/streams/search_T2c0_wave1_20261018", ROOT/"results/streams/search_T2c0_wave2_20261019", ROOT/"results/streams/search_T2c0_wave3_20261019"]
TRUTH, TANG = -22174., 27.; csha = hashlib.sha256(CAT.read_bytes()).hexdigest(); bins = [(10, 15), (15, 20), (20, 25), (25, 30), (30, 35)]


def profile(em, g, a_s):
    out = []
    for a in a_s:
        ua = (a-10)/25; bv = -np.inf
        for x0 in g["X"][np.argsort(-g["ybar"])[:8]]:
            r = minimize(lambda z: -em.predict(np.insert(z, 0, ua)[None], return_std=False)[0], x0[1:], method="L-BFGS-B", bounds=[(0, 1)]*8)
            bv = max(bv, -r.fun)
        out.append(bv)
    return np.array(out)


fig, ax = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True); ax = ax.ravel(); a_s = np.arange(10.5, 35, 1.25)
g0, _, em0 = SW.fit(SW.observations(BASE, [], csha)); p0 = profile(em0, g0, a_s)
ax[0].plot(a_s, p0, "C0-"); ax[0].axvline(TANG, color="C3", label="truth angle"); ax[0].axhline(TRUTH, color="k", ls="--", label="truth score")
ax[0].axvline(a_s[np.argmax(p0)], color="C0", ls=":", label=f"emulator peak {a_s[np.argmax(p0)]:.0f} deg")
ax[0].set(title="(a) before: emulator profile misses the truth", xlabel="bar angle [deg]", ylabel="ln L"); ax[0].legend(fontsize=7)
lab = ["T2c0", "T2c1"]; before = [-15.7, -15.9]; after = [-1.6, -1.4]; x = np.arange(2)
ax[1].bar(x-0.18, before, 0.35, label="emulator as trained"); ax[1].bar(x+0.18, after, 0.35, label="+ one simulation at the truth")
ax[1].axhline(0, color="k", lw=0.6); ax[1].set_xticks(x, lab)
ax[1].set(title="(b) diagnosis: missing simulations near the peak", ylabel="max-angle error [deg]"); ax[1].legend(fontsize=7)
for k in range(4):
    rows = SW.observations(BASE, W[:k], csha)
    b = [max(r["y"] for r in rows if lo <= 10+25*r["unit"][0] < hi and r["purpose"] != SW.HELD) for lo, hi in bins]
    ax[2].plot([0.5*(lo+hi) for lo, hi in bins], b, "o-", label="training set" if k == 0 else f"after wave {k}")
ax[2].axhline(TRUTH, color="k", ls="--"); ax[2].axvline(TANG, color="C3", lw=0.8)
ax[2].set(title="(c) waves: best simulated model per angle bin", xlabel="bar angle bin [deg]", ylabel="best measured ln L"); ax[2].legend(fontsize=7)
g3, _, em3 = SW.fit(SW.observations(BASE, W, csha)); p3 = profile(em3, g3, a_s)
ax[3].plot(a_s, p3, "C1-", label="emulator after wave 3")
pc = {20: -23712, 25: -23801, 30: -24072, 33: -23211}; pe = {20: -22703, 25: -21934, 30: -22488, 33: -23278}
ax[3].plot(list(pc), list(pc.values()), "ks", label="measured at its maximisers (4 seeds)")
for a_ in pc:
    ax[3].plot([a_, a_], [pe[a_], pc[a_]], "k:", lw=0.8)
ax[3].axhline(TRUTH, color="k", ls="--"); ax[3].axvline(TANG, color="C3", lw=0.8)
ax[3].set(title="(d) after waves: profile shape is an artefact", xlabel="bar angle [deg]", ylabel="ln L"); ax[3].legend(fontsize=7)
best = {}
for f in (ROOT/"results/streams/profile_T2c0_20261019/runs").glob("a*_e*.json"):
    m = re.match(r"a([\d.]+)_(\d)_e(\d+)", f.stem); r = json.loads(f.read_text()); g_ = float(m.group(1))
    best[g_] = max(best.get(g_, -np.inf), r["total_jk"])
starts = {12.5: -23606, 17.5: -23770, 22.5: -22962, 27: -23151, 27.5: -23151, 32.5: -23569}
gg = sorted(best); n = len(list((ROOT/"results/streams/profile_T2c0_20261019/runs").glob("a*_e*.json")))
ax[4].plot(sorted(starts), [starts[k] for k in sorted(starts)], "s:", color="0.5", label="start (best from waves, 256k)")
ax[4].plot(gg, [best[k] for k in gg], "o-", color="C2", label=f"direct search so far (128k; {n} sprays)")
ax[4].axhline(TRUTH, color="k", ls="--", label="truth score"); ax[4].axvline(TANG, color="C3", lw=0.8)
ax[4].set(title="(e) direct per-angle search (running)", xlabel="bar angle [deg]", ylabel="ln L"); ax[4].legend(fontsize=7)
chk = [("real, round 1", -21326, 1100, -22094), ("real, round 2", -22120, 600, -22771), ("real, final", -21136, 530, -21569),
       ("T2, wave 3 max", -21915, 370, -23836)] + [(f"T2, {a_} deg", pe[a_], 400, pc[a_]) for a_ in pc]
for i, (l, mu, sd, y) in enumerate(chk):
    ax[5].errorbar(mu, y, xerr=sd, fmt="o", color=f"C{i % 10}", capsize=2, label=l)
lim = [-24300, -20900]; ax[5].plot(lim, lim, "k:"); ax[5].set(xlim=lim, ylim=lim, title="(f) emulator is over-optimistic at its maxima",
                                                             xlabel="emulator prediction (+-1 sd)", ylabel="measured")
ax[5].legend(fontsize=6)
fig.suptitle("Mock T2 (truth: bar angle 27 deg, score -22174): finding and fixing the angle bias", fontsize=11)
fig.savefig(ROOT/"plots/search_T2c0_story.png", dpi=110); print(ROOT/"plots/search_T2c0_story.png")
