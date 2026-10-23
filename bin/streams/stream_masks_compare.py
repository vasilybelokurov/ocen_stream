#!/usr/bin/env python3
"""Compare stream-mask tracks (2026-10-23): old fitter (cubic-spline smoothing, results/plot_data/stream_masks_v1_spline.json) vs the
new fitter (linear bridging + pooled refits of weak bins; stream_masks.json = marked start, stream_masks_orbitstart.json = start on
omega Cen's McMillan17 orbit, no anchoring), with the orbit itself.
Usage: PYTHONPATH=src python bin/streams/stream_masks_compare.py -> plots/stream_masks_new_fitter.png
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.argv = sys.argv[:1]
from stream_masks import cluster_orbit

L = lambda f: json.loads((ROOT/"results/plot_data"/f).read_text())
old, new, orb = L("stream_masks_v1_spline.json"), L("stream_masks.json"), L("stream_masks_orbitstart.json"); G = np.array(new["grid"]); O = cluster_orbit()
fig, ax = plt.subplots(2, 3, figsize=(16, 7), sharex=True, constrained_layout=True, gridspec_kw=dict(height_ratios=(2, 1)))
for j, (q, lab) in enumerate((("pmra", r"$\mu_{\alpha*}$ [mas/yr]"), ("pmdec", r"$\mu_\delta$ [mas/yr]"), ("dphi2", r"$\Delta\phi_2$ [deg]"))):
    c_old, c_new, c_orb = (np.array(x["tracks"][q]["c"]) for x in (old, new, orb))
    ax[0, j].plot(O["phi1"], O[q], "-", color="0.6", lw=3, label="omega Cen orbit (start of the orbit run)")
    ax[0, j].plot(G, c_old, "k--", label="old fitter (marked start)"); ax[0, j].plot(G, c_new, "C0-", lw=1.8, label="new fitter, marked start")
    ax[0, j].plot(G, c_orb, "C3-", lw=1.2, label="new fitter, orbit start")
    acc = [r["c"] for r in new["tables"][q] if r.get("accepted")]
    ax[0, j].plot(acc, np.interp(acc, G, c_new), "o", color="C0", ms=4)
    ax[1, j].plot(G, c_orb-c_new, "C3-", label="orbit start - marked start"); ax[1, j].plot(G, c_new-c_old, "C0--", label="new - old (marked start)")
    ax[1, j].axhspan(-0.5, 0.5, color="0.9"); ax[1, j].axhline(0, color="k", lw=0.5)
    ax[0, j].set(ylabel=lab, title=q); ax[1, j].set(xlabel=r"$\phi_1$ [deg]", ylabel="difference", ylim=(-2.5, 2.5))
    if q == "dphi2":
        ax[0, j].set_ylim(-9, 5)
ax[0, 0].legend(fontsize=8); ax[1, 0].legend(fontsize=8)
fig.suptitle("Stream mask tracks: new fitter (weak bins refitted on wider windows, straight-line bridging) from two starts; dots: accepted bins; grey band: +-0.5")
fig.savefig(ROOT/"plots/stream_masks_new_fitter.png", dpi=100)
