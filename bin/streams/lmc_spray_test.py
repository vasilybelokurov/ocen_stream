#!/usr/bin/env python3
"""End-to-end test of the LMC option (host_potential(..., lmc=dict(mass=M)); src/ocen_stream/lmc.py): omega Cen spray
and score at fixed bar models with M_LMC = 0, 1.0, 1.5, 2.0 e11 Msun, everything else identical (bar3d_sweep settings: amp
1.2, d 5.6 kpc, catalogue PMs, baumgardt frame, 8000 epochs over the last 1000 Myr, seed 1, debris < 700 Myr).
Per run: lnL, phi1-segment lnL, footprint medians (pmra, pmdec, vlos, d) per 2-deg phi1 bin, and the omega Cen centre-orbit
offset from the no-LMC orbit (max over the last 1 Gyr). Appends JSON lines to results/plot_data/lmc_spray_test.jsonl.
Usage: PYTHONPATH=src python bin/streams/lmc_spray_test.py --an 16 --om 34 --size 1.0 --mass 1.5e11   (1 core, ~30 s)"""
import argparse, json, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from grid4_common import ROOT, T, make_spray, score, project_gc, GC, du, OCEN_OBS
from ocen_stream.restricted import agama_kpc, host_potential, AGAMA_T_MYR
from ocen_stream.frames import to_model

ap = argparse.ArgumentParser()
ap.add_argument("--an", type=float, default=16.); ap.add_argument("--om", type=float, default=34.); ap.add_argument("--size", type=float, default=1.0)
ap.add_argument("--mass", type=float, default=0.); ap.add_argument("--amp", type=float, default=1.2); ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
CAT = (OCEN_OBS["pmra"], OCEN_OBS["pmdec"]); SEG = [(-99, 4), (4, 17), (17, 99)]
edges = np.arange(np.floor(du.min()), np.ceil(du.max())+2, 2.); cen = 0.5*(edges[1:]+edges[:-1])
hk = dict(bar_size=a.size, **(dict(lmc=dict(mass=a.mass)) if a.mass > 0 else {}))
t0 = time.time()
m = make_spray(a.om, a.an, a.amp, 5.6, nrel=8000, seed=a.seed, pm=CAT, host_kw=hk, max_age_myr=700.)
s = score(m); mu, mx = project_gc(m["l"], m["b"], GC); k = (np.abs(mx) < 6) & (m["age"] < 700) & (m["chi"] > 0)
ib = np.digitize(mu[k], edges)-1
med = {q: [float(np.median(m[q][k][ib == j])) if (ib == j).sum() >= 10 else None for j in range(len(cen))] for q in ("pmra", "pmdec", "vlos", "d")}
# omega Cen centre orbit over the last 1 Gyr, with and without the LMC
agama = agama_kpc(); today = to_model(OCEN_OBS["ra"], OCEN_OBS["dec"], 5.6, *CAT, OCEN_OBS["vlos"])[0]
orbs = [agama.orbit(potential=host_potential("x", bar_omega=a.om, bar_angle_deg=a.an, t_today=T, bar_amp=a.amp, **h), ic=today,
                    timestart=T, time=-1000/AGAMA_T_MYR, trajsize=1001, accuracy=1e-12)[1] for h in (dict(bar_size=a.size), hk)]
dx = np.linalg.norm(orbs[1][:, :3]-orbs[0][:, :3], axis=1); dv = np.linalg.norm(orbs[1][:, 3:]-orbs[0][:, 3:], axis=1)
r = dict(an=a.an, om=a.om, size=a.size, amp=a.amp, mass=a.mass, seed=a.seed, lnL=float(s["total"]),
         seg=[float(s["per_star"][(du >= x0) & (du < x1)].sum()) for x0, x1 in SEG], med=med, phi1=cen.tolist(),
         orbit_dx_max_kpc=float(dx.max()), orbit_dv_max_kms=float(dv.max()), orbit_dx_250=float(dx[:251].max()), orbit_dv_250=float(dv[:251].max()))
with open(ROOT/"results/plot_data/lmc_spray_test.jsonl", "a") as fh:
    fh.write(json.dumps(r)+"\n")
print(f"an {a.an:g} om {a.om:g} size {a.size:g} M_LMC {a.mass:.1e}: lnL {r['lnL']:.1f}; centre orbit offset max (1 Gyr / 250 Myr): "
      f"{dx.max()*1e3:.0f} / {dx[:251].max()*1e3:.0f} pc, {dv.max():.2f} / {dv[:251].max():.2f} km/s  [{time.time()-t0:.0f} s]", flush=True)
