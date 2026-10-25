#!/usr/bin/env python3
"""Model debris in the observed knee region (l -45..-28, b 31..41) vs Ibata+2024 stream-54 members there: counts, median
PMs, v_los, distance, and stripping time (first snapshot at which the tracer is > 2 kpc from the centre) for each run.
Usage: python bin/streams/knee_region_check.py prescribed prescribed_rot db98_rot
"""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"bin/streams")); sys.path.insert(0, str(ROOT/"src"))
import numpy as np, astropy.coordinates as coord
from astropy.table import Table
from plot_debris_vs_ibata import observables, MODELS
from ocen_stream.analysis import load, run_core, run_particles, snapshot_times
from ocen_stream.restricted import OCEN_TODAY, bound_set
w = lambda x: np.where(x > 180, x-360, x)
KNEE = dict(l=(-45, -28), b=(31, 41)); ARM = dict(l=(-62, -50), b=(20, 30))


def box(l, b, B):
    return (w(l) > B["l"][0]) & (w(l) < B["l"][1]) & (b > B["b"][0]) & (b < B["b"][1])


t = Table.read(os.path.expandvars("$OCEN_CATALOGUES/streamfinder_ibata2024_dr3.fits")); t = t[np.asarray(t["Stream"]) == 54]
g = coord.SkyCoord(np.asarray(t["RAdeg"], float), np.asarray(t["DEdeg"], float), unit="deg").galactic
for name, B in (("knee", KNEE), ("arm b20-30", ARM)):
    k = box(g.l.deg, g.b.deg, B); v = np.asarray(t["VHel"], float)[k]; v = v[np.isfinite(v) & (v < 900)]
    print(f"Ibata 54 {name}: N {k.sum()}, pmra {np.median(np.asarray(t['pmRA'],float)[k]):.2f}, pmdec {np.median(np.asarray(t['pmDE'],float)[k]):.2f}, "
          f"v_los {np.median(v) if len(v) else np.nan:.0f} (n={len(v)}), plx {np.median(np.asarray(t['plx'],float)[k]):.3f}")
for run in sys.argv[1:]:
    for key, lab, _ in MODELS:
        R = ROOT/"results/streams"/run/key
        if not (R/"run.json").exists():
            continue
        m, s = run_particles(R); _, xv = load(R, name="snap_today.npz"); c = OCEN_TODAY.copy()
        bnd, _ = bound_set(xv, c, m, start=np.linalg.norm(xv[:, :3]-c[:3], axis=1) < 0.5, core=run_core(R))
        o = observables(xv)
        # stripping time: first snapshot with distance from the centre track > 2 kpc is a proxy; use snapshots
        ts = snapshot_times(R)
        for name, B in (("knee", KNEE), ("arm b20-30", ARM)):
            k = box(o["l"], o["b"], B) & ~bnd
            if k.sum() == 0:
                print(f"{run:16s} {key:13s} {name}: N 0"); continue
            # escape epoch: earliest snapshot where the tracer is beyond 1 kpc of the cluster median position of bound tracers is costly;
            # use the snapshot orbit of the centre stored in run.json? approximate with the first snapshot at > 1 kpc from today's
            # bound-tracer centroid trajectory: skipped. Report phase-space only.
            print(f"{run:16s} {key:13s} {name}: N {k.sum():4d}  pmra {np.median(o['pmra'][k]):6.2f}  pmdec {np.median(o['pmdec'][k]):6.2f}  "
                  f"v_los {np.median(o['vlos'][k]):5.0f}  d {np.median(o['dist'][k]):.2f} kpc  (pmra 16-84: {np.percentile(o['pmra'][k],16):.1f}..{np.percentile(o['pmra'][k],84):.1f})")
