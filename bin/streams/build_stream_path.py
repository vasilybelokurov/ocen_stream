#!/usr/bin/env python3
"""Build the smooth stream-54 path (src/ocen_dm/streams/path.py), project the members, and check single-valuedness: in
equal-number bins of the along-stream coordinate s, the across-stream distribution x should have one peak (1D KDE, h = 0.4 deg;
secondary peaks >= 15% of the main one reported).
Usage: python bin/streams/build_stream_path.py -> results/plot_data/stream54_path.json, plots/stream54_path.png
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"bin/streams")); sys.path.insert(0, str(ROOT/"src"))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from spray_bar_grid2 import load_data
from ocen_dm.streams.path import build_path, project, save, gnomonic

d = load_data(); l, b = d["l"], d["b"]
track = json.loads((ROOT/"results/plot_data/stream54_track.json").read_text())["all"]
P = build_path(l, b, track); save(P, ROOT/"results/plot_data/stream54_path.json")
s, x = project(l, b, P)
print(f"path: smoothing {P['smoothing']:.2f}, max deviation from ridge points {P['max_dev']:.2f} deg, length {P['arc'][-1]:.1f} deg; "
      f"{len(P['ridge'])} ridge points")
print(f"members: s {s.min():.1f}..{s.max():.1f} deg; |x| 5/50/95% {np.percentile(np.abs(x), [5, 50, 95]).round(2)}")


def peaks(v, h=0.4, frac=0.15):
    g = np.linspace(v.min()-1, v.max()+1, 400); dens = np.exp(-0.5*((g[:, None]-v[None, :])/h)**2).sum(1)
    return [round(float(g[i]), 2) for i in range(1, len(g)-1) if dens[i] > dens[i-1] and dens[i] >= dens[i+1] and dens[i] > frac*dens.max()]


qs = np.percentile(s, np.linspace(0, 100, 17))
print("equal-number s bins: N, x peaks")
for lo, hi in zip(qs[:-1], qs[1:]):
    k = (s >= lo) & (s <= hi)
    print(f"  s {lo:5.1f}..{hi:5.1f} (b {np.median(b[k]):5.1f}, l {np.median(l[k]):6.1f}) N {k.sum():4d}: x peaks {peaks(x[k])}")
fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
cl, cb = None, None
# back-project the curve for plotting: invert gnomonic numerically via a dense l-b grid lookup
gl, gb = np.meshgrid(np.linspace(-75, -20, 600), np.linspace(5, 50, 500)); gx, gy = gnomonic(gl.ravel(), gb.ravel())
from scipy.spatial import cKDTree
_, ii = cKDTree(np.column_stack((gx, gy))).query(np.column_stack((P["cx"], P["cy"])))
cl, cb = gl.ravel()[ii], gb.ravel()[ii]
sc = ax[0].scatter(l, b, c=s, s=3, cmap="plasma"); ax[0].plot(cl, cb, "k-", lw=1.5); rp = np.array(P["ridge"]); ax[0].plot(rp[:, 0], rp[:, 1], "co", ms=5)
ax[0].set_xlim(-25, -70); ax[0].set_ylim(10, 45); ax[0].set_xlabel("l"); ax[0].set_ylabel("b"); fig.colorbar(sc, ax=ax[0], label="s [deg]")
ax[0].set_title("members coloured by s; black: path; cyan: ridge points")
ax[1].scatter(s, x, s=2, c="0.4"); ax[1].set_xlabel("s [deg] (along stream)"); ax[1].set_ylabel("x [deg] (across)"); ax[1].set_ylim(-6, 6)
ax[2].scatter(s, b, s=2, c="0.4"); ax[2].set_xlabel("s [deg]"); ax[2].set_ylabel("b [deg]")
for a in ax:
    a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(ROOT/"plots/stream54_path.png", dpi=75); print("plots/stream54_path.png")
