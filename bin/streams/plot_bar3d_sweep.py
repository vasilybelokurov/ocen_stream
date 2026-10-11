#!/usr/bin/env python3
"""Plots of the 3D bar sweep (bin/streams/bar3d_sweep.py): lnL - best in the (angle, Omega_b) plane for each bar size, and the
profile (max over the other two) along each parameter. Input results/plot_data/bar3d_sweep/w*.jsonl.
Output plots/bar3d_sweep_slices.png, plots/bar3d_sweep_profiles.png. Usage: python bin/streams/plot_bar3d_sweep.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
R = [json.loads(l) for f in sorted((ROOT/"results/plot_data/bar3d_sweep").glob("w*.jsonl")) for l in f.read_text().splitlines()]
an = np.array([r["an"] for r in R]); om = np.array([r["om"] for r in R]); sz = np.array([r["size"] for r in R]); L = np.array([r["lnL"] for r in R])
top = L.max(); ib = L.argmax(); lab = f"n = {R[0]['nrel']} epochs (1x = 8000), seed {R[0]['seed']}, amp {R[0]['am']}, d 5.6, catalogue PMs"
print(f"{len(R)} points; best: angle {an[ib]:g}, Omega {om[ib]:g}, size {sz[ib]:g}, lnL {top:.1f}")
ANG = np.unique(an); OM = np.unique(om); SZ = np.unique(sz)

fig, ax = plt.subplots(1, len(SZ), figsize=(3.4*len(SZ), 4.2), sharey=True, constrained_layout=True)
for a, s in zip(np.atleast_1d(ax), SZ):
    Z = np.full((len(OM), len(ANG)), np.nan)
    for i in np.where(sz == s)[0]:
        Z[np.searchsorted(OM, om[i]), np.searchsorted(ANG, an[i])] = L[i]-top
    im = a.pcolormesh(np.r_[ANG-1, ANG[-1]+1], np.r_[OM-0.5, OM[-1]+0.5], Z, cmap="viridis", vmin=-1500, vmax=0)
    j = np.where(sz == s)[0][np.argmax(L[sz == s])]
    a.plot(an[j], om[j], "r*", ms=12); a.set_title(f"size x{s:g}: best {L[j]-top:+.0f}", fontsize=9)
    a.axvspan(25, 29, color="w", alpha=0.15); a.set_xlabel("bar angle [deg]")
np.atleast_1d(ax)[0].set_ylabel(r"$\Omega_b$ [km/s/kpc]")
fig.colorbar(im, ax=ax, label="lnL - best"); fig.suptitle(f"3D bar sweep: lnL in (angle, Omega_b) per bar size; red star = best per size; "
                                                          f"white band = angle 27+-2 (prior); blank = beyond corotation. {lab}", fontsize=9)
fig.savefig(ROOT/"plots/bar3d_sweep_slices.png", dpi=90)

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
for a, (x, g, nm) in zip(ax, ((an, ANG, "bar angle [deg]"), (om, OM, r"$\Omega_b$ [km/s/kpc]"), (sz, SZ, "bar size"))):
    a.plot(g, [L[x == v].max()-top for v in g], "o-"); a.set_xlabel(nm); a.set_ylabel("max lnL - best (other two free)"); a.grid(alpha=.3)
ax[0].axvspan(25, 29, color="orange", alpha=0.2)
fig.suptitle(f"Profiles of the 3D bar sweep ({lab}); seed noise at 1x ~50-250", fontsize=9)
fig.savefig(ROOT/"plots/bar3d_sweep_profiles.png", dpi=90)
print("plots/bar3d_sweep_slices.png plots/bar3d_sweep_profiles.png")
