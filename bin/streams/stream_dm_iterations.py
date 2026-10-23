#!/usr/bin/env python3
"""dm detections and the quadratic dm(phi1) fitted at each iteration of stream_iterate.py (results/streams/iterate_<start>/iter<k>/
stream_dm_fit.json), one panel per start (2026-10-23).
Usage: python bin/streams/stream_dm_iterations.py -> plots/stream_dm_iterations.png
"""
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ph = np.linspace(-3, 31, 200); starts = ("anchored", "current")
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True, constrained_layout=True)
for a, st in zip(ax, starts):
    its = sorted((ROOT/f"results/streams/iterate_{st}").glob("iter*/stream_dm_fit.json"))
    for k, f in enumerate(its):
        F = json.loads(f.read_text()); col = plt.cm.viridis(k/max(len(its)-1, 1)); R = F.get("relation_poly")
        x = np.array([r["phi1"] for r in F["bins"]])+0.25*(k-1.5); y = np.array([r["all_dm"] for r in F["bins"]]); acc = np.array([r["accepted"] for r in F["bins"]])
        e = np.array([[r["all_dm"]-r["all_lo"] for r in F["bins"]], [r["all_hi"]-r["all_dm"] for r in F["bins"]]])
        a.errorbar(x[acc], y[acc], yerr=e[:, acc], fmt="o", color=col, ms=5, capsize=2); a.errorbar(x[~acc], y[~acc], yerr=e[:, ~acc], fmt="o", color=col, mfc="none", ms=5, capsize=2)
        if R:
            a.plot(ph, np.polyval(R["coef"], np.clip(ph, R["phi1_lo"], R["phi1_hi"])), "-", color=col, lw=1.6,
                   label=f"iter {k}: quadratic fitted to {len(R['accepted'])} accepted bins (chi2 {R['chi2']:.1f}/{R['dof']})")
    a.axhline(0, color="0.5", lw=0.5); a.set(xlabel=r"$\phi_1$ [deg]", title=f"start: {st} line", xlim=(-3, 31), ylim=(-1.4, 0.6)); a.legend(fontsize=8, loc="lower left")
ax[0].set_ylabel("dm (stream - cluster) [mag]")
fig.suptitle("dm per bin (filled: accepted, open: rejected) and the quadratic fed back into the CMD mask, per iteration")
fig.savefig(ROOT/"plots/stream_dm_iterations.png", dpi=100); print("plots/stream_dm_iterations.png")
