#!/usr/bin/env python3
"""Sky / pmra / pmdec mask tracks and widths at each iteration of stream_iterate.py (results/streams/iterate_<start>/iter<k>/
stream_masks.json), both starts (2026-10-23). Rows: track (with the k = 2 core band of the last round), change from the previous
round (grey band: median per-bin track error), intrinsic width.
Usage: python bin/streams/stream_mask_iterations.py -> plots/stream_mask_iterations.png
"""
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

Q = (("dphi2", r"$\Delta\phi_2$ [deg]"), ("pmra", r"$\mu_{\alpha*}$ [mas/yr]"), ("pmdec", r"$\mu_\delta$ [mas/yr]"))
fig, ax = plt.subplots(3, 3, figsize=(18, 11), sharex=True, constrained_layout=True, gridspec_kw=dict(height_ratios=(2, 1, 1)))
for st, ls in (("current", "-"), ("anchored", "--")):
    fs = sorted((ROOT/f"results/streams/iterate_{st}").glob("iter*/stream_masks.json")); prev = None
    for k, f in enumerate(fs):
        M = json.loads(f.read_text()); G = np.array(M["grid"]); col = plt.cm.viridis(k/max(len(fs)-1, 1))
        for j, (q, lab) in enumerate(Q):
            c, s = np.array(M["tracks"][q]["c"]), np.array(M["tracks"][q]["s"])
            ax[0, j].plot(G, c, ls, color=col, lw=1.2, label=f"{st} start, iter {k}")
            if prev is not None:
                ax[1, j].plot(G, c-np.array(prev["tracks"][q]["c"]), ls, color=col, lw=1.2)
            ax[2, j].plot(G, s, ls, color=col, lw=1.2)
            if k == len(fs)-1 and st == "current":
                ax[0, j].fill_between(G, c-2*s, c+2*s, color="0.85", lw=0, zorder=0)
                e = [r["mu_err"] for r in M["tables"][q] if np.isfinite(r["mu_err"])]
                ax[1, j].axhspan(-np.median(e), np.median(e), color="0.88", zorder=0)
        prev = M
for j, (q, lab) in enumerate(Q):
    ax[0, j].set(ylabel=lab, title=q); ax[1, j].set(ylabel="change from previous round"); ax[2, j].set(ylabel="intrinsic width", xlabel=r"$\phi_1$ [deg]")
    ax[1, j].axhline(0, color="k", lw=0.5)
ax[1, 0].set_ylim(-0.6, 0.6); ax[1, 1].set_ylim(-0.8, 0.8); ax[1, 2].set_ylim(-0.3, 0.3)
ax[0, 0].set_ylim(-4, 6); ax[0, 0].legend(fontsize=7, ncol=2)
fig.suptitle("Mask tracks per iteration (solid: start from the old dm line, dashed: from the anchored line; colour: round; grey: k = 2 core of the last round / median per-bin track error)")
fig.savefig(ROOT/"plots/stream_mask_iterations.png", dpi=95); print("plots/stream_mask_iterations.png")
