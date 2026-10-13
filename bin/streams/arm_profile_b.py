#!/usr/bin/env python3
"""Northern arm (b > 15, -75 < l < -15), 2.5-deg b bins: pmra ridge (histogram mode, 0.5 mas/yr bins, then median within
+-1.5 mas/yr) and the fraction of stars more than 3 mas/yr below the mode, for the prescribed-potential debris (A, B, C) and
Ibata+2024 stream 54. Medians are NOT used: in the models they are dragged by a broad low-pmra spray.
Usage: python bin/streams/arm_profile_b.py   (prints tables; seconds)
"""
import sys, os; from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"bin/streams")); sys.path.insert(0, str(ROOT/"src"))
import numpy as np, astropy.coordinates as coord
from astropy.table import Table
from plot_debris_vs_ibata import observables, MODELS
from ocen_dm.streams.analysis import load, run_core, run_particles
from ocen_dm.streams.restricted import OCEN_TODAY, bound_set
w = lambda x: np.where(x > 180, x-360, x)
E = np.arange(15, 46, 2.5)
def rows(name, l, b, pm):
    a = (b > 15) & (w(l) < -15) & (w(l) > -75)
    print(name)
    for lo, hi in zip(E[:-1], E[1:]):
        k = a & (b >= lo) & (b < hi)
        if k.sum() >= 5:
            x = pm[k]; h, e = np.histogram(x, bins=np.arange(-25, 3, 0.5)); mode = e[h.argmax()]+0.25; r = x[np.abs(x-mode) < 1.5]; print(f"  {lo:4.1f}-{hi:4.1f} N={k.sum():5d} ridge {np.median(r):6.2f} (n={len(r)})  frac pmra<mode-3: {np.mean(x < mode-3):.2f}")
for key, lab, _ in MODELS:
    R = ROOT/"results/streams/prescribed"/key; m, s = run_particles(R); _, xv = load(R, name="snap_today.npz"); c = OCEN_TODAY.copy()
    bnd, _ = bound_set(xv, c, m, start=np.linalg.norm(xv[:, :3]-c[:3], axis=1) < 0.5, core=run_core(R)); o = observables(xv[~bnd])
    rows(lab+"", o["l"], o["b"], o["pmra"])
t = Table.read(os.path.expandvars("$OCEN_CATALOGUES/streamfinder_ibata2024_dr3.fits")); t = t[np.asarray(t["Stream"]) == 54]
g = coord.SkyCoord(np.asarray(t["RAdeg"], float), np.asarray(t["DEdeg"], float), unit="deg").galactic
rows("Ibata 54", g.l.deg, g.b.deg, np.asarray(t["pmRA"], float))
