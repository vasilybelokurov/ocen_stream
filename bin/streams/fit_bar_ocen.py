#!/usr/bin/env python3
"""Fit at fixed bar angle (28 deg; Hunter+2024 / Bland-Hawthorn & Gerhard 2016): free Omega_b, bar amplitude and omega Cen's
present-day distance and PMs; v_los fixed (insensitive). Objective: robust chi-KDE lnL (score_chi_kde) + log-priors:
  amplitude: bounds [0.8, 1.2], Gaussian N(1.0, 0.1) (user: avoid very strong bars)
  distance:  N(5.43, 0.15) kpc (wider than the catalogue 0.05: published distances span ~5.2-5.6)
  pmra, pmdec: N(catalogue-frame values, 0.027 mas/yr) (0.011 statistical + 0.025 Gaia systematic floor, Vasiliev & Baumgardt 2021)
  Omega_b: bounds [33, 37] km/s/kpc, flat.
Spray: model A mass profile, 1.96 Gyr, release over the last 1000 Myr at 8000 fixed epochs, seed 1 (common random numbers).
Nelder-Mead (scipy) from two starts; out-of-bounds -> -1e9. Every evaluation logged to results/plot_data/fit_bar_ocen.json.
Usage: python bin/streams/fit_bar_ocen.py --maxfev 45
"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"bin/streams")); sys.path.insert(0, str(ROOT/"src"))
import numpy as np
from scipy.optimize import minimize
from spray_bar_grid2 import load_data, w
from ocen_dm.streams.restricted import AGAMA_T_MYR, host_potential
from ocen_dm.streams.frames import observables, to_model, OCEN_OBS
from ocen_dm.streams.spray import spray_unwrapped
from ocen_dm.streams.score import score_chi_kde

ANGLE = 28.
PRIOR = dict(amp=(1.0, 0.1), d=(5.43, 0.15), pmra=(OCEN_OBS["pmra"], 0.027), pmdec=(OCEN_OBS["pmdec"], 0.027))
BOUNDS = dict(omega=(33., 37.), amp=(0.8, 1.2), d=(5.0, 5.9), pmra=(OCEN_OBS["pmra"]-0.15, OCEN_OBS["pmra"]+0.15),
              pmdec=(OCEN_OBS["pmdec"]-0.15, OCEN_OBS["pmdec"]+0.15))
NAMES = ("omega", "amp", "d", "pmra", "pmdec")
SCALE = np.array([1.0, 0.1, 0.05, 0.02, 0.02])     # simplex step scale


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--maxfev", type=int, default=45); a = ap.parse_args()
    d = load_data()
    prof = json.loads((ROOT/"results/nbody/A_nodm/model_profiles.json").read_text()); r_kpc, M = np.array(prof["r_pc"])/1e3, np.array(prof["enclosed"]["total"])
    T = 1955.58/AGAMA_T_MYR; trel = np.linspace(T-1000/AGAMA_T_MYR, T, 8000)
    log = []; outp = ROOT/"results/plot_data/fit_bar_ocen.json"

    def neg_post(x, start):
        p = dict(zip(NAMES, x))
        if any(not (BOUNDS[k][0] <= p[k] <= BOUNDS[k][1]) for k in NAMES):
            return 1e9
        t0 = time.time()
        host = host_potential("x", bar_omega=p["omega"], bar_angle_deg=ANGLE, t_today=T, bar_amp=p["amp"])
        today = to_model(OCEN_OBS["ra"], OCEN_OBS["dec"], p["d"], p["pmra"], p["pmdec"], OCEN_OBS["vlos"])[0]
        sp = spray_unwrapped(host, today, T, r_kpc, M, trel, seed=1)
        tr = sp["arm"] == 1; o = observables(sp["xv"][tr])
        lnl = score_chi_kde(d, dict(l=w(o["l"]), b=o["b"], pmra=o["pmra"], pmdec=o["pmdec"], vlos=o["vlos"], chi=sp["chi"][tr],
                                    age=sp["t_release_myr_ago"][tr]), robust=True)["total"]
        lp = sum(-0.5*((p[k]-m)/s)**2 for k, (m, s) in PRIOR.items())
        log.append(dict(start=start, **{k: float(v) for k, v in p.items()}, lnL=lnl, lnprior=lp, lnpost=lnl+lp, seconds=time.time()-t0))
        print(f"[{start}] Om {p['omega']:6.2f} amp {p['amp']:5.3f} d {p['d']:5.3f} pmra {p['pmra']:8.4f} pmdec {p['pmdec']:8.4f}: "
              f"lnL {lnl:10.1f} lnprior {lp:7.2f} lnpost {lnl+lp:10.1f} [{time.time()-t0:.0f} s]", flush=True)
        outp.write_text(json.dumps(dict(angle=ANGLE, prior=PRIOR, bounds=BOUNDS, log=log), indent=1))
        return -(lnl+lp)

    starts = [np.array([34.5, 1.0, 5.43, OCEN_OBS["pmra"], OCEN_OBS["pmdec"]]),
              np.array([35.5, 1.1, 5.53, OCEN_OBS["pmra"], OCEN_OBS["pmdec"]-0.02])]
    for i, x0 in enumerate(starts):
        simplex = np.vstack([x0]+[x0+SCALE*np.eye(5)[j] for j in range(5)])
        r = minimize(neg_post, x0, args=(i,), method="Nelder-Mead",
                     options=dict(initial_simplex=simplex, maxfev=a.maxfev, xatol=1e-3, fatol=20.))
        print(f"start {i}: best {dict(zip(NAMES, np.round(r.x, 4)))} lnpost {-r.fun:.1f}", flush=True)


if __name__ == "__main__":
    main()
