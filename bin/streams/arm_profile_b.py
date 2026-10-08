#!/usr/bin/env python3
"""Northern arm (b > 15, -75 < l < -15) in 5-deg b bins: median l, d, pmra, pmdec, v_los of the prescribed-potential debris
(A, B, C) vs Ibata+2024 stream 54 (median parallax + 0.017 mas zero point; e_plx = 1.2533 x 1.4826 MAD / sqrt N).
Usage: python bin/streams/arm_profile_b.py   (prints tables; seconds)
"""
import sys, os; from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"src"))
import numpy as np, astropy.coordinates as coord
from astropy.table import Table
from plot_debris_vs_ibata import observables, sep_deg, MODELS
from ocen_dm.streams.analysis import load, run_core, run_particles
from ocen_dm.streams.restricted import OCEN_TODAY, bound_set
edges = np.arange(15, 50, 5)
def wrap(x): return np.where(x > 180, x-360, x)
def table(name, b, l, cols):
    arm = (b > 15) & (wrap(l) < -15) & (wrap(l) > -75)
    print(f"\n{name}: N arm = {arm.sum()}")
    print("  b-bin     N   " + "  ".join(f"{k:>7s}" for k in cols))
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = arm & (b >= lo) & (b < hi)
        if k.sum() < 5: continue
        print(f"  {lo:2d}-{hi:2d} {k.sum():5d}   " + "  ".join(f"{f(k):7.2f}" for f in cols.values()))
for key, lab, _ in MODELS:
    R = ROOT/"results/streams/prescribed"/key
    m, s = run_particles(R); _, xv = load(R, name="snap_today.npz"); c = OCEN_TODAY.copy()
    near = np.linalg.norm(xv[:, :3]-c[:3], axis=1) < 0.5
    bnd, _ = bound_set(xv, c, m, start=near, core=run_core(R)); o = observables(xv[~bnd])
    table(lab, o["b"], o["l"], {"l": lambda k: np.median(wrap(o["l"][k])), "d[kpc]": lambda k: np.median(o["dist"][k]),
          "pmra": lambda k: np.median(o["pmra"][k]), "pmdec": lambda k: np.median(o["pmdec"][k]), "vlos": lambda k: np.median(o["vlos"][k])})
t = Table.read(os.path.expanduser("$OCEN_CATALOGUES/streamfinder_ibata2024_dr3.fits")); t = t[np.asarray(t["Stream"]) == 54]
g = coord.SkyCoord(np.asarray(t["RAdeg"], float), np.asarray(t["DEdeg"], float), unit="deg").galactic
plx = np.asarray(t["plx"], float) + 0.017
def plxd(k):
    p = plx[k]; md = np.median(p); e = 1.2533*np.std(p)/np.sqrt(len(p)); return 1/md
def plxe(k):
    p = plx[k]; return 1.2533*1.4826*np.median(np.abs(p-np.median(p)))/np.sqrt(len(p))
table("Ibata+2024 stream 54", g.b.deg, g.l.deg, {"l": lambda k: np.median(wrap(g.l.deg[k])), "plx": lambda k: np.median(plx[k]),
      "e_plx": plxe, "d[kpc]": plxd, "pmra": lambda k: np.median(np.asarray(t["pmRA"], float)[k]), "pmdec": lambda k: np.median(np.asarray(t["pmDE"], float)[k])})
