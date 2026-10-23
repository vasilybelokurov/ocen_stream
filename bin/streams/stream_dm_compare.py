#!/usr/bin/env python3
"""Old vs new dm(phi1) of the omega Cen stream (2026-10-23).

Old: MS colour centring in 3 bins with the hand-marked PM windows (results/plot_data/gaia_cmd_mask.json: dm_fits, dm_relation,
capped at 0). New: Poisson CMD-template fit with the data-driven masks (results/plot_data/stream_dm_fit[_orbitstart].json),
full / MS-only / TO-SGB-only; open symbols = rejected bins (weak or null >= 2 sigma). Reference: omega Cen's orbit in McMillan17
(5 log10(d / 5.43 kpc)).
Usage: PYTHONPATH=src python bin/streams/stream_dm_compare.py -> plots/stream_dm_compare.png
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.argv = sys.argv[:1]
from stream_masks import cluster_orbit
from ocen_dm.cluster import OCEN_DISTANCE_KPC

L = lambda f: json.loads((ROOT/"results/plot_data"/f).read_text())
P, new, nob = L("gaia_cmd_mask.json"), L("stream_dm_fit.json"), L("stream_dm_fit_orbitstart.json")
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True); ph = np.linspace(-3, 31, 200)
O = cluster_orbit(); o = np.argsort(O["phi1"]); ax.plot(O["phi1"][o], 5*np.log10(O["dist"][o]/OCEN_DISTANCE_KPC), "-", color="0.75", lw=3, label="omega Cen orbit (McMillan17)")
ax.plot(ph, np.minimum(np.polyval(P["dm_relation"]["coef"], ph), 0.), "k--", label="old relation (MS colour centring, capped at 0)")
for k, v in P["dm_fits"].items():
    a0, a1 = map(float, k.split("-")); ax.errorbar(0.5*(a0+a1), v["dm"], yerr=v["err"], fmt="ks", ms=7, capsize=3, label="old: MS colour centring" if a0 == 14 else None)
for R, off, col, lab in ((new, 0., "C0", "new: CMD template fit (marked-start masks)"), (nob, 0.35, "C3", "new: CMD template fit (orbit-start masks)")):
    for i, r in enumerate(R["bins"]):
        x = r["phi1"]+off; e = [[r["all_dm"]-r["all_lo"]], [r["all_hi"]-r["all_dm"]]]
        ax.errorbar(x, r["all_dm"], yerr=e, fmt="o", color=col, mfc=col if r["accepted"] else "none", ms=7, capsize=3, label=lab if i == 0 else None)
    cf = R["relation"]["coef"]; ax.plot(ph, np.polyval(cf, ph), "-", color=col, lw=0.8, label=f"linear fit to accepted bins: {cf[1]:+.2f} {cf[0]:+.4f} $\\phi_1$")
for i, r in enumerate(new["bins"]):
    ax.plot(r["phi1"]-0.35, r["ms_dm"], "v", color="C0", ms=5, alpha=0.6, label="new, MS only" if i == 0 else None)
    ax.plot(r["phi1"]-0.35, r["tosgb_dm"], "^", color="C0", ms=5, alpha=0.6, label="new, TO/SGB only" if i == 0 else None)
ax.axhline(0, color="k", lw=0.4); ax.set(xlabel=r"$\phi_1$ [deg]", ylabel="dm (stream - cluster) [mag]", xlim=(-3, 31), ylim=(-1.4, 0.6),
                                          title="Distance-modulus offset along the stream: old vs new (open: rejected bins; error bars: 68%, ~2x too small per injections)")
ax.title.set_fontsize(10); ax.legend(fontsize=8, loc="lower left"); fig.savefig(ROOT/"plots/stream_dm_compare.png", dpi=100); print("plots/stream_dm_compare.png")
