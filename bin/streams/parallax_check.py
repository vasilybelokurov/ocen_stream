#!/usr/bin/env python3
"""Parallax sanity check (no CMD distances): members' error-weighted mean Gaia DR3 parallax per stretch of stream 54 vs the models'.
Members: Ibata+2024 stream 54 (load_data cuts); parallax, parallax_error from gaia_dr3.gaia_source (WSDB local_join on source_id;
cached in results/streams/stream54_parallax.npz). Zero-point: not corrected per star; treated as a band: corrected = observed - ZP
with ZP = -0.017 mas (quasar median, Lindegren+2021, arXiv:2012.01742) and a band ZP in [-0.04, 0] (band width our assumption).
Model: for each member, the median model parallax (1/d) of footprint debris (|dphi2| < 6, age < 700, chi > 0) within |dphi1| < 1 deg
of the member; the segment value is then averaged with the members' own weights 1/sigma^2 (like-with-like sampling).
Segments in chord phi1: < 4, 4-17, 17-23, > 23. Usage: python bin/streams/parallax_check.py
"""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"bin/streams"))
import numpy as np, astropy.coordinates as coord
from astropy.table import Table
from ocen_dm.streams.path import project_gc

w = lambda x: np.where(x > 180, x-360, x)
GC = json.loads((ROOT/"results/plot_data/stream54_gc_frame_chord.json").read_text())
t = Table.read(os.path.expandvars("$OCEN_CATALOGUES/streamfinder_ibata2024_dr3.fits")); t = t[np.asarray(t["Stream"]) == 54]
g = coord.SkyCoord(np.asarray(t["RAdeg"], float), np.asarray(t["DEdeg"], float), unit="deg").galactic; l, b = w(g.l.deg), g.b.deg
keep = (b > 15) & (l < -20) & (l > -75); sid = np.asarray(t["Gaia"], np.int64)[keep]; u, x = project_gc(l[keep], b[keep], GC)
cache = ROOT/"results/streams/stream54_parallax.npz"
if cache.exists():
    z = dict(np.load(cache))
else:
    import sqlutilpy as sqlutil
    r = sqlutil.local_join("SELECT m.sid, g.parallax, g.parallax_error, g.phot_g_mean_mag FROM mytab AS m LEFT JOIN gaia_dr3.gaia_source AS g "
                           "ON g.source_id = m.sid ORDER BY m.xid", "mytab", (sid, np.arange(len(sid))), ("sid", "xid"), asDict=True)
    z = {k: np.asarray(v) for k, v in r.items()}; np.savez_compressed(cache, **z)
assert np.all(z["sid"] == sid)
plx, e = z["parallax"].astype(float), z["parallax_error"].astype(float); ok = np.isfinite(plx) & np.isfinite(e) & (e > 0)
print(f"members {len(sid)}, with parallax {ok.sum()}, median parallax_error {np.median(e[ok]):.3f} mas")
MODELS = {"28 deg longer bar (working)": "phase2b/long_om36_am1.2_sz1.15_eta0_pm0.npz", "28 deg baseline": "d3_free_pm/pm_+0_+0.npz",
          "16 deg": "seed_test/om34.5_an16_am1.4_d5.6_n32000_s1.npz"}
SEG = [(-99, 4, "phi1 < 4"), (4, 17, "phi1 4-17"), (17, 23, "phi1 17-23"), (23, 99, "phi1 > 23")]
out = {}
for lo, hi, nm in SEG:
    s = ok & (u >= lo) & (u < hi); wt = 1/e[s]**2; m_obs = np.sum(wt*plx[s])/wt.sum(); err = 1/np.sqrt(wt.sum())
    row = dict(n=int(s.sum()), obs=float(m_obs), err=float(err), zp_central=float(m_obs+0.017), zp_band=[float(m_obs), float(m_obs+0.04)])
    line = f"{nm:11s} n {s.sum():4d}: observed {m_obs:.4f} +- {err:.4f} mas; ZP-corrected {m_obs+0.017:.4f} (band {m_obs:.4f}-{m_obs+0.04:.4f}) |"
    for mn, p in MODELS.items():
        m = dict(np.load(ROOT/"results/streams"/p)); mu, mx = project_gc(m["l"], m["b"], GC); k = (np.abs(mx) < 6) & (m["age"] < 700) & (m["chi"] > 0)
        mu, mp = mu[k], 1/m["d"][k]; o = np.argsort(mu); mu, mp = mu[o], mp[o]
        pred = np.array([np.median(mp[np.searchsorted(mu, ui-1):np.searchsorted(mu, ui+1)]) if np.searchsorted(mu, ui+1)-np.searchsorted(mu, ui-1) > 10 else np.nan
                         for ui in u[s]])
        f = np.isfinite(pred); mod = np.sum(wt[f]*pred[f])/wt[f].sum(); row[mn] = float(mod)
        line += f" {mn.split(' (')[0]}: {mod:.4f}"
    out[nm] = row; print(line)
(ROOT/"results/plot_data/parallax_check.json").write_text(json.dumps(out, indent=1))
