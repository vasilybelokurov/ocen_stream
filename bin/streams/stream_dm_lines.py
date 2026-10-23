#!/usr/bin/env python3
"""dm detections per phi1 bin (marked-start masks; results/plot_data/stream_dm_fit.json) and the smooth dm(phi1) lines: the one the
CMD mask uses now (gaia_cmd_mask.json dm_relation, capped at 0) and linear fits to each method's accepted bins (2026-10-23).
Usage: PYTHONPATH=src python bin/streams/stream_dm_lines.py -> plots/stream_dm_lines.png
"""
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
from gaia_cmd_mask import dm_of

P = json.loads((ROOT/"results/plot_data/gaia_cmd_mask.json").read_text()); F = json.loads((ROOT/"results/plot_data/stream_dm_fit.json").read_text())
b = [r for r in F["bins"]]; x = np.array([r["phi1"] for r in b]); acc = np.array([r["accepted"] for r in b]); ph = np.linspace(-3, 31, 200)
fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
P0 = json.loads((ROOT/"results/plot_data/gaia_cmd_mask_pre_iterate.json").read_text())
ax.plot(ph, dm_of(P0, ph), "k:", lw=1.5, label="old line (before iteration): +0.556 - 0.0544 $\\phi_1$, capped at 0")
R = P["dm_relation"]; ax.plot(ph, dm_of(P, ph), "k-", lw=2.5, label=f"iterated line used by the CMD mask: {R['slope']:+.4f} $\\phi_1$ (0 at the cluster, flat beyond {R['phi1_max']:g})")
t, tl, th = (np.array([r[k] for r in b]) for k in ("all_dm", "all_lo", "all_hi")); m, me = np.array([r["msc_dm"] for r in b]), np.array([r["msc_err"] for r in b])
for y, e, col, mk, lab, d in ((t, [t-tl, th-t], "C0", "o", "template fit (MS + TO, counts)", -0.2), (m, me, "C3", "s", "MS colour centring", 0.2)):
    ax.errorbar(x[acc]+d, y[acc], yerr=np.array(e)[:, acc] if np.ndim(e) == 2 else e[acc], fmt=mk, color=col, ms=7, capsize=3, label=lab)
    ax.errorbar(x[~acc]+d, y[~acc], yerr=np.array(e)[:, ~acc] if np.ndim(e) == 2 else e[~acc], fmt=mk, color=col, mfc="none", ms=7, capsize=3)
    w = 1/np.maximum(0.5*(np.array(e)[0]+np.array(e)[1]) if np.ndim(e) == 2 else e, 0.02); cf = np.polyfit(x[acc], y[acc], 1, w=w[acc])
    ax.plot(ph, np.polyval(cf, ph), "--", color=col, lw=1.5, label=f"linear fit ({lab.split(' (')[0]}): {cf[1]:+.2f} {cf[0]:+.4f} $\\phi_1$")
    print(lab, "fit", np.round(cf[::-1], 4))
ax.axhline(0, color="0.5", lw=0.5)
ax.set(xlabel=r"$\phi_1$ [deg]", ylabel="dm (stream - cluster) [mag]", xlim=(-3, 31), ylim=(-1.4, 0.6), title="dm detections (4-deg bins; open: rejected) and smooth lines")
ax.legend(fontsize=8, loc="lower left"); fig.savefig(ROOT/"plots/stream_dm_lines.png", dpi=100); print("plots/stream_dm_lines.png")
