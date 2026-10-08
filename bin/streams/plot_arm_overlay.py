#!/usr/bin/env python3
"""Overlay of the northern-arm debris (A, B, C; prescribed potential, today; b > 5, -80 < l < -15) on Ibata+2024 stream 54
in pmra-b and pmdec-b (zoom b 10-45). Model PMs are not convolved with Gaia errors.
Usage: python bin/streams/plot_arm_overlay.py plots/streams_arm_overlay_pm_b.png
"""
import sys, os; from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"bin/streams")); sys.path.insert(0, str(ROOT/"src"))
import numpy as np, astropy.coordinates as coord, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from astropy.table import Table
from plot_debris_vs_ibata import observables, MODELS
from ocen_dm.streams.analysis import load, run_core, run_particles
from ocen_dm.streams.restricted import OCEN_TODAY, bound_set
w = lambda x: np.where(x > 180, x-360, x)
t = Table.read(os.path.expanduser("$OCEN_CATALOGUES/streamfinder_ibata2024_dr3.fits")); t = t[np.asarray(t["Stream"]) == 54]
g = coord.SkyCoord(np.asarray(t["RAdeg"], float), np.asarray(t["DEdeg"], float), unit="deg").galactic
fig, ax = plt.subplots(2, 3, figsize=(16, 10))
for j, (key, lab, col) in enumerate(MODELS):
    R = ROOT/"results/streams/prescribed"/key; m, s = run_particles(R); _, xv = load(R, name="snap_today.npz"); c = OCEN_TODAY.copy()
    bnd, _ = bound_set(xv, c, m, start=np.linalg.norm(xv[:, :3]-c[:3], axis=1) < 0.5, core=run_core(R)); o = observables(xv[~bnd])
    k = (o["b"] > 5) & (w(o["l"]) < -15) & (w(o["l"]) > -80)
    for i, q, qq in ((0, "pmra", "pmRA"), (1, "pmdec", "pmDE")):
        a = ax[i, j]
        a.plot(g.b.deg, np.asarray(t[qq], float), ".", color="0.6", ms=2, label="Ibata+2024 54")
        a.plot(o["b"][k], o[q][k], ".", color=col, ms=4, label=lab)
        a.set_xlim(10, 45); a.set_ylim(-20, 0); a.set_xlabel("b [deg]"); a.set_ylabel(q); a.grid(alpha=.3); a.legend(loc="lower left")
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=80)
