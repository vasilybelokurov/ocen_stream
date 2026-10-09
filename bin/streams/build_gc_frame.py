#!/usr/bin/env python3
"""Publishable stream frame for stream 54: great circle (pole fitted to the ordered ridge points of the smooth path, origin at
omega Cen (l, b) = (309.10, 14.97)) + lowest-degree polynomial phi2(phi1) within 1 deg of every ridge point. Members projected to
(phi1, dphi2); single-valuedness checked as for the spline path (one dphi2 peak per equal-number phi1 bin).
Usage: python bin/streams/build_gc_frame.py [--chord]  (--chord: great circle through omega Cen and the far end of the ridge)
      python bin/streams/build_gc_frame.py -> results/plot_data/stream54_gc_frame.json, plots/stream54_gc_frame.png
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"bin/streams")); sys.path.insert(0, str(ROOT/"src"))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from spray_bar_grid2 import load_data
from ocen_dm.streams.path import fit_gc_frame, chord_gc_frame, project_gc, gc_rotation, to_stream

d = load_data(); l, b = d["l"], d["b"]
P = json.loads((ROOT/"results/plot_data/stream54_path.json").read_text()); ridge = np.array(P["ridge"])
F = chord_gc_frame(ridge, (309.10-360., 14.97), tuple(ridge[-1])) if "--chord" in sys.argv else fit_gc_frame(ridge, (309.10-360., 14.97))
(ROOT/f"results/plot_data/stream54_gc_frame{'_chord' if '--chord' in sys.argv else ''}.json").write_text(json.dumps(F, indent=1))
print(f"pole (l, b) = ({F['pole'][0]:.3f}, {F['pole'][1]:.3f}) deg; origin {F['origin']}; polynomial degree {F['degree']}, "
      f"coeffs (high->low) {np.round(F['coeffs'], 6).tolist()}; max ridge deviation {F['max_dev']:.2f} deg; ridge phi1 {np.round(F['phi1_range'],1)}")
f1, df2 = project_gc(l, b, F)
Rm = gc_rotation(F["pole"], F["origin"]); rf1, rf2 = to_stream(ridge[:, 0], ridge[:, 1], Rm)
print("ridge phi1 sequence (should increase monotonically):", np.round(rf1, 1).tolist())
print(f"members: phi1 {f1.min():.1f}..{f1.max():.1f}; |dphi2| 5/50/95% {np.percentile(np.abs(df2), [5, 50, 95]).round(2)}")


def peaks(v, h=0.4, frac=0.15):
    g = np.linspace(v.min()-1, v.max()+1, 400); dens = np.exp(-0.5*((g[:, None]-v[None, :])/h)**2).sum(1)
    return [round(float(g[i]), 2) for i in range(1, len(g)-1) if dens[i] > dens[i-1] and dens[i] >= dens[i+1] and dens[i] > frac*dens.max()]


qs = np.percentile(f1, np.linspace(0, 100, 17))
for lo, hi in zip(qs[:-1], qs[1:]):
    k = (f1 >= lo) & (f1 <= hi)
    print(f"  phi1 {lo:6.1f}..{hi:6.1f} (b {np.median(b[k]):5.1f}, l {np.median(l[k]):6.1f}) N {k.sum():4d}: dphi2 peaks {peaks(df2[k])}")
fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
f2 = df2+np.polyval(F["coeffs"], f1)
ax[0].scatter(f1, f2, s=2, c="0.4"); g = np.linspace(f1.min(), f1.max(), 200); ax[0].plot(g, np.polyval(F["coeffs"], g), "r-", lw=2)
ax[0].plot(rf1, rf2, "co", ms=5); ax[0].set_xlabel("phi1 [deg]"); ax[0].set_ylabel("phi2 [deg]"); ax[0].set_title(f"stream frame; red: degree-{F['degree']} track")
ax[1].scatter(f1, df2, s=2, c="0.4"); ax[1].set_xlabel("phi1"); ax[1].set_ylabel("dphi2 = phi2 - track"); ax[1].set_ylim(-6, 6)
sc = ax[2].scatter(l, b, c=f1, s=3, cmap="plasma"); ax[2].set_xlim(-25, -70); ax[2].set_ylim(10, 45); ax[2].set_xlabel("l"); ax[2].set_ylabel("b")
fig.colorbar(sc, ax=ax[2], label="phi1")
for a in ax:
    a.grid(alpha=0.3)
fig.tight_layout(); fn = f"plots/stream54_gc_frame{'_chord' if '--chord' in sys.argv else ''}.png"; fig.savefig(ROOT/fn, dpi=75); print(fn)
