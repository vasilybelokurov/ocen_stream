#!/usr/bin/env python3
"""Cheapest LMC refit: bar fixed (default 16 deg, Omega_b 34, size 1.0, amp 1.2 = bar3d_sweep best), free omega Cen distance and
PMs, with or without the LMC (host_potential(..., lmc=dict(mass=M)), src/ocen_dm/streams/lmc.py). Question: can omega Cen's
present-day d / PMs absorb the LMC-induced shift of the far stream?
Objective: score() of grid4_common (score_conditional, chord frame, debris < 700 Myr) + priors of fit_bar_ocen.py:
  d ~ N(5.43, 0.15) kpc, pmra/pmdec ~ N(catalogue, 0.027) mas/yr; bounds d 5.0-5.9, PMs catalogue +- 0.15.
Spray: 8000 epochs over the last 1000 Myr, seed 1 (common random numbers between the LMC-on and -off fits).
Nelder-Mead from (5.6 kpc, catalogue PMs). Every evaluation logged to results/plot_data/fit_lmc_ocen_M<mass>.json.
Usage: OMP_NUM_THREADS=1 PYTHONPATH=src python bin/streams/fit_lmc_ocen.py --mass 1.5e11 --maxfev 40   (~30 s per evaluation)"""
import argparse, json, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ocen_dm.cluster import OCEN_DISTANCE_KPC
import numpy as np
from scipy.optimize import minimize
from grid4_common import ROOT, make_spray, score, OCEN_OBS

ap = argparse.ArgumentParser()
ap.add_argument("--mass", type=float, default=0.); ap.add_argument("--maxfev", type=int, default=40)
ap.add_argument("--an", type=float, default=16.); ap.add_argument("--om", type=float, default=34.)
ap.add_argument("--size", type=float, default=1.0); ap.add_argument("--amp", type=float, default=1.2); ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
NAMES = ("d", "pmra", "pmdec")
PRIOR = dict(d=(OCEN_DISTANCE_KPC, 0.15), pmra=(OCEN_OBS["pmra"], 0.027), pmdec=(OCEN_OBS["pmdec"], 0.027))
BOUNDS = dict(d=(5.0, 5.9), pmra=(OCEN_OBS["pmra"]-0.15, OCEN_OBS["pmra"]+0.15), pmdec=(OCEN_OBS["pmdec"]-0.15, OCEN_OBS["pmdec"]+0.15))
SCALE = np.array([0.05, 0.02, 0.02])     # simplex step scale (as fit_bar_ocen.py)
hk = dict(bar_size=a.size, **(dict(lmc=dict(mass=a.mass)) if a.mass > 0 else {}))
outp = ROOT/f"results/plot_data/fit_lmc_ocen_M{a.mass:.1e}_an{a.an:g}_om{a.om:g}_s{a.seed}.json"
log = []


def neg_post(y):
    p = dict(zip(NAMES, y*SCALE))
    if any(not (BOUNDS[k][0] <= p[k] <= BOUNDS[k][1]) for k in NAMES):
        return 1e9
    t0 = time.time()
    m = make_spray(a.om, a.an, a.amp, p["d"], nrel=8000, seed=a.seed, pm=(p["pmra"], p["pmdec"]), host_kw=hk, max_age_myr=700.)
    lnl = float(score(m)["total"]); lp = sum(-0.5*((p[k]-mu)/s)**2 for k, (mu, s) in PRIOR.items())
    log.append(dict(**{k: float(v) for k, v in p.items()}, lnL=lnl, lnprior=lp, lnpost=lnl+lp, seconds=time.time()-t0))
    print(f"M {a.mass:.1e} d {p['d']:.3f} pmra {p['pmra']:.4f} pmdec {p['pmdec']:.4f}: lnL {lnl:.1f} lnprior {lp:.2f} "
          f"lnpost {lnl+lp:.1f} [{time.time()-t0:.0f} s]", flush=True)
    outp.write_text(json.dumps(dict(args=vars(a), prior=PRIOR, bounds=BOUNDS, log=log), indent=1))
    return -(lnl+lp)


x0 = np.array([5.6, OCEN_OBS["pmra"], OCEN_OBS["pmdec"]])/SCALE
r = minimize(neg_post, x0, method="Nelder-Mead", options=dict(maxfev=a.maxfev, initial_simplex=np.vstack((x0, x0+np.eye(3)*2.))))
best = max(log, key=lambda e: e["lnpost"])
print("best:", json.dumps(best), flush=True)
