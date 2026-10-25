#!/usr/bin/env python3
"""Fit at fixed bar angle (28 deg; Hunter+2024 / Bland-Hawthorn & Gerhard 2016): free Omega_b, bar amplitude and omega Cen's
present-day distance and PMs; v_los fixed (insensitive). Objective: robust chi-KDE lnL (score_chi_kde) + log-priors:
  amplitude: bounds [0.8, 1.2], Gaussian N(1.0, 0.1) (user: avoid very strong bars)
  distance:  N(5.43, 0.15) kpc (wider than the catalogue 0.05: published distances span ~5.2-5.6)
  pmra, pmdec: N(catalogue-frame values, 0.027 mas/yr) (0.011 statistical + 0.025 Gaia systematic floor, Vasiliev & Baumgardt 2021)
  Omega_b: bounds [33, 37] km/s/kpc, flat.
Spray: model A mass profile, 1.96 Gyr, release over the last 1000 Myr at 8000 fixed epochs, seed 1 (common random numbers).
Nelder-Mead (scipy) from two starts; out-of-bounds -> -1e9. Every evaluation logged to results/plot_data/fit_bar_ocen.json.
Usage: python bin/streams/fit_bar_ocen.py --maxfev 45 [--free-angle] [--conditional]
--conditional: score_conditional (density-free along the stream; chord great-circle frame); starts at the free-angle best with angle 24.14, 28, 22.
--free-angle: bar angle free with prior N(27, 2) deg, bounds 21-33 (user decision 2026-10-09); 3 starts (angle 28, 24, 32).
"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"bin/streams")); sys.path.insert(0, str(ROOT/"src"))
import numpy as np
from scipy.optimize import minimize
from spray_bar_grid2 import load_data, w
from ocen_stream.cluster import OCEN_DISTANCE_KPC
from ocen_stream.restricted import AGAMA_T_MYR, host_potential
from ocen_stream.frames import observables, to_model, OCEN_OBS
from ocen_stream.spray import spray_unwrapped
from ocen_stream.score import score_chi_kde, score_conditional
from ocen_stream.path import project_gc

ANGLE = 28.
PRIOR = dict(amp=(1.0, 0.1), d=(OCEN_DISTANCE_KPC, 0.15), pmra=(OCEN_OBS["pmra"], 0.027), pmdec=(OCEN_OBS["pmdec"], 0.027))
BOUNDS = dict(omega=(33., 37.), amp=(0.8, 1.2), d=(5.0, 5.9), pmra=(OCEN_OBS["pmra"]-0.15, OCEN_OBS["pmra"]+0.15),
              pmdec=(OCEN_OBS["pmdec"]-0.15, OCEN_OBS["pmdec"]+0.15))
NAMES = ("omega", "amp", "d", "pmra", "pmdec")
SCALE = np.array([1.0, 0.1, 0.05, 0.02, 0.02])     # simplex step scale
FREE_ANGLE = "--free-angle" in sys.argv
COND = "--conditional" in sys.argv     # score_conditional, variant A (chord great-circle frame), user decision 2026-10-09
if FREE_ANGLE:   # user decision 2026-10-09: soft prior N(27, 2) deg (Wegg & Gerhard 2013; Bland-Hawthorn & Gerhard 2016), bounds 21-33
    PRIOR["angle"] = (27., 2.); BOUNDS["angle"] = (21., 33.); NAMES = NAMES+("angle",); SCALE = np.append(SCALE, 2.0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--maxfev", type=int, default=45); ap.add_argument("--free-angle", action="store_true")
    ap.add_argument("--conditional", action="store_true")
    a = ap.parse_args()
    d = load_data()
    prof = json.loads((ROOT/"results/nbody/A_nodm/model_profiles.json").read_text()); r_kpc, M = np.array(prof["r_pc"])/1e3, np.array(prof["enclosed"]["total"])
    T = 1955.58/AGAMA_T_MYR; trel = np.linspace(T-1000/AGAMA_T_MYR, T, 8000)
    log = []; outp = ROOT/f"results/plot_data/fit_bar_ocen{'_freeangle' if FREE_ANGLE else ''}{'_cond' if COND else ''}.json"
    if COND:
        GC = json.loads((ROOT/"results/plot_data/stream54_gc_frame_chord.json").read_text())
        nd = len(d["l"]); cov = np.zeros((nd, 2, 2)); cov[:, 0, 0] = d["e_pmra"]**2; cov[:, 1, 1] = d["e_pmdec"]**2
        cov[:, 0, 1] = cov[:, 1, 0] = d["rho"]*d["e_pmra"]*d["e_pmdec"]
        du, dx = project_gc(d["l"], d["b"], GC); dW = np.column_stack((dx, d["pmra"], d["pmdec"]))

    def neg_post(x, start):
        p = dict(zip(NAMES, x))
        if any(not (BOUNDS[k][0] <= p[k] <= BOUNDS[k][1]) for k in NAMES):
            return 1e9
        t0 = time.time()
        host = host_potential("x", bar_omega=p["omega"], bar_angle_deg=p.get("angle", ANGLE), t_today=T, bar_amp=p["amp"])
        today = to_model(OCEN_OBS["ra"], OCEN_OBS["dec"], p["d"], p["pmra"], p["pmdec"], OCEN_OBS["vlos"])[0]
        sp = spray_unwrapped(host, today, T, r_kpc, M, trel, seed=1)
        tr = sp["arm"] == 1; o = observables(sp["xv"][tr])
        if COND:
            mu_, mx_ = project_gc(w(o["l"]), o["b"], GC)
            lnl = score_conditional(du, dW, d["v"], d["e_v"], cov, mu_, np.column_stack((mx_, o["pmra"], o["pmdec"])), o["vlos"],
                                    sp["chi"][tr], sp["t_release_myr_ago"][tr], u_window=(du.min(), du.max()))["total"]
        else:
            lnl = score_chi_kde(d, dict(l=w(o["l"]), b=o["b"], pmra=o["pmra"], pmdec=o["pmdec"], vlos=o["vlos"], chi=sp["chi"][tr],
                                        age=sp["t_release_myr_ago"][tr]), robust=True)["total"]
        lp = sum(-0.5*((p[k]-m)/s)**2 for k, (m, s) in PRIOR.items())
        log.append(dict(start=start, **{k: float(v) for k, v in p.items()}, lnL=lnl, lnprior=lp, lnpost=lnl+lp, seconds=time.time()-t0))
        print(f"[{start}] ang {p.get('angle', ANGLE):5.2f} Om {p['omega']:6.2f} amp {p['amp']:5.3f} d {p['d']:5.3f} pmra {p['pmra']:8.4f} pmdec {p['pmdec']:8.4f}: "
              f"lnL {lnl:10.1f} lnprior {lp:7.2f} lnpost {lnl+lp:10.1f} [{time.time()-t0:.0f} s]", flush=True)
        outp.write_text(json.dumps(dict(angle=ANGLE, prior=PRIOR, bounds=BOUNDS, log=log), indent=1))
        return -(lnl+lp)

    if FREE_ANGLE:   # from the fixed-angle best fit, and with the angle started at 24 and 32
        b = [35.04, 1.178, 5.508, -3.2367, -6.7700]
        starts = [np.array(b+[24.14]), np.array(b+[28.]), np.array(b+[22.])] if COND else [np.array(b+[28.]), np.array(b+[24.]), np.array(b+[32.])]
    else:
        starts = [np.array([34.5, 1.0, OCEN_DISTANCE_KPC, OCEN_OBS["pmra"], OCEN_OBS["pmdec"]]),
                  np.array([35.5, 1.1, 5.53, OCEN_OBS["pmra"], OCEN_OBS["pmdec"]-0.02])]
    for i, x0 in enumerate(starts):
        n = len(x0); simplex = np.vstack([x0]+[x0+SCALE*np.eye(n)[j] for j in range(n)])
        r = minimize(neg_post, x0, args=(i,), method="Nelder-Mead",
                     options=dict(initial_simplex=simplex, maxfev=a.maxfev, xatol=1e-3, fatol=20.))
        print(f"start {i}: best {dict(zip(NAMES, np.round(r.x, 4)))} lnpost {-r.fun:.1f}", flush=True)


if __name__ == "__main__":
    main()
