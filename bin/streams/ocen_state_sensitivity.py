#!/usr/bin/env python3
"""Sensitivity of the robust chi-KDE likelihood to omega Cen's present-day observables at a fixed bar (default Omega_b 34.5,
angle 28 deg = Hunter+2024 / Bland-Hawthorn & Gerhard 2016, amplitude 1.2). Baseline = OCEN_TODAY (frames 'baumgardt': d 5.43,
PM (-3.2499, -6.7461), v_los 232.78). One-at-a-time shifts: catalogue +-2 sigma (gc_catalog_full.fits: d 0.05 kpc, PM 0.011
mas/yr, v_los 0.21 km/s) and wider cases (PM +-0.026 mas/yr; d +-0.2 kpc). Same release epochs and seed (common random numbers).
Usage: python bin/streams/ocen_state_sensitivity.py -> results/plot_data/ocen_state_sensitivity.json
"""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"bin/streams")); sys.path.insert(0, str(ROOT/"src"))
import numpy as np
from spray_bar_grid2 import load_data, w
from ocen_dm.streams.restricted import AGAMA_T_MYR, host_potential
from ocen_dm.streams.frames import observables, to_model, OCEN_OBS
from ocen_dm.streams.spray import spray_unwrapped
from ocen_dm.streams.score import score_chi_kde

OM, AN, AM = 34.5, 28., 1.2
d = load_data()
prof = json.loads((ROOT/"results/nbody/A_nodm/model_profiles.json").read_text()); r_kpc, M = np.array(prof["r_pc"])/1e3, np.array(prof["enclosed"]["total"])
T = 1955.58/AGAMA_T_MYR; trel = np.linspace(T-1000/AGAMA_T_MYR, T, 8000)
host = host_potential("x", bar_omega=OM, bar_angle_deg=AN, t_today=T, bar_amp=AM)
base = dict(d=5.43, pmra=OCEN_OBS["pmra"], pmdec=OCEN_OBS["pmdec"], vlos=OCEN_OBS["vlos"])
cases = [("baseline", {})]
for q, s in (("d", 0.05), ("pmra", 0.011), ("pmdec", 0.011), ("vlos", 0.21)):
    cases += [(f"{q} +2sig", {q: 2*s}), (f"{q} -2sig", {q: -2*s})]
cases += [("pmra +0.026", {"pmra": 0.026}), ("pmra -0.026", {"pmra": -0.026}), ("pmdec +0.026", {"pmdec": 0.026}), ("pmdec -0.026", {"pmdec": -0.026}),
          ("d +0.2", {"d": 0.2}), ("d -0.2", {"d": -0.2})]
out = []
for name, dlt in cases:
    p = {k: base[k]+dlt.get(k, 0.) for k in base}
    today = to_model(OCEN_OBS["ra"], OCEN_OBS["dec"], p["d"], p["pmra"], p["pmdec"], p["vlos"])[0]
    t0 = time.time(); sp = spray_unwrapped(host, today, T, r_kpc, M, trel, seed=1)
    tr = sp["arm"] == 1; o = observables(sp["xv"][tr])
    ck = score_chi_kde(d, dict(l=w(o["l"]), b=o["b"], pmra=o["pmra"], pmdec=o["pmdec"], vlos=o["vlos"], chi=sp["chi"][tr], age=sp["t_release_myr_ago"][tr]), robust=True)
    out.append(dict(case=name, shift=dlt, lnL=ck["total"]))
    print(f"{name:14s}: lnL {ck['total']:10.1f}  dlnL vs baseline {ck['total']-out[0]['lnL']:8.1f}  [{time.time()-t0:.0f} s]", flush=True)
    (ROOT/"results/plot_data/ocen_state_sensitivity.json").write_text(json.dumps(out, indent=1))
