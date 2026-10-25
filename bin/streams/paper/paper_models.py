#!/usr/bin/env python3
"""Models for the stream-54 paper: one consistent set, recomputed with the current code and reference values.

Stages (outputs in results/streams/paper/, results/plot_data/paper_models.json, results/plot_data/paper_nullfit_start{0,1}.json):
  nullfit   Hunter+2024 axisymmetric host (bar amplitude 0, exactly the axisymmetric model) with omega Cen's distance and
            both proper motions refitted under the conditional likelihood (score_conditional, chord frame), with the bounds
            of bin/streams/refine_bar_profiles.py (d 5.25-5.95 kpc, PMs within 3 sigma = 0.081 mas/yr of the catalogue).
            Host shape fixed. 8000 release epochs, seed 1, two Nelder-Mead starts (catalogue state; barred-16 state).
  display   the fitted null, the null at the catalogue state, and the refined barred candidates of 2026-10-13 at 28 and
            16 deg (results/plot_data/bar_profile_refinement_20261013/fit_an{16,28}.json): 32000 epochs, seeds 4, 5, 6;
            the seed-4 sprays and per-star ln L are saved for the figures.
  variants  axisymmetric host-shape variants at the fitted null state: halo flattening q 0.8 / 1.2, disc mass x0.8 / x1.2,
            v_c(R0) 220 / 240 km/s (halo renormalised), McMillan (2017); 32000 epochs, seed 4.
ln L values are comparable only at equal release-epoch number; they are not calibrated (members treated as independent).
Usage: python bin/streams/paper/paper_models.py nullfit|display|variants [--start 0|1]
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"bin/streams")); sys.path.insert(0, str(ROOT/"src"))
import numpy as np
from scipy.optimize import minimize

from grid4_common import GC, M, T, d, make_spray, r_kpc, score, w
from ocen_stream.cluster import MU_SYS, MU_SYS_ERR, OCEN_DEC, OCEN_DISTANCE_ERR_KPC, OCEN_DISTANCE_KPC, OCEN_RA, OCEN_VSYS_KMS
from ocen_stream.frames import observables, to_model
from ocen_stream.restricted import AGAMA_T_MYR, host_potential
from ocen_stream.spray import spray_unwrapped

OUT = ROOT/"results/streams/paper"; OUT.mkdir(parents=True, exist_ok=True)
JS = ROOT/"results/plot_data/paper_models.json"
PM_SIG = 0.027                                             # per-component PM measurement width used by the refinements
BOUNDS = np.array([[5.25, 5.95], [MU_SYS[0]-3*PM_SIG, MU_SYS[0]+3*PM_SIG], [MU_SYS[1]-3*PM_SIG, MU_SYS[1]+3*PM_SIG]])
NULL = dict(om=35., an=28., am=0.)                         # bar amplitude 0: exactly the axisymmetric Hunter host


def load():
    return json.loads(JS.read_text()) if JS.exists() else {}


def save(rec):
    tmp = JS.with_suffix(".tmp"); tmp.write_text(json.dumps(rec, indent=1)); os.replace(tmp, JS)


def measurement(dist, pmra, pmdec):
    return -0.5*(((dist-OCEN_DISTANCE_KPC)/OCEN_DISTANCE_ERR_KPC)**2+((pmra-MU_SYS[0])/PM_SIG)**2+((pmdec-MU_SYS[1])/PM_SIG)**2)


def refined(angle):
    b = json.loads((ROOT/f"results/plot_data/bar_profile_refinement_20261013/fit_an{angle}.json").read_text())["best"]
    return dict(om=b["omega"], an=float(angle), am=b["amp"], size=b["size"], dist=b["dist"], pmra=b["pmra"], pmdec=b["pmdec"])


def spray(p, nrel, seed, host_kw=None):
    kw = dict(host_kw or {})
    if p.get("size", 1.0) != 1.0:
        kw["bar_size"] = p["size"]
    return make_spray(p["om"], p["an"], p["am"], p["dist"], nrel=nrel, seed=seed, pm=(p["pmra"], p["pmdec"]),
                      host_kw=kw or None, max_age_myr=700.)


def spray_host(host, p, nrel, seed):
    """make_spray with an explicit host (McMillan 2017); same release, cut and projection."""
    today = to_model(OCEN_RA, OCEN_DEC, p["dist"], p["pmra"], p["pmdec"], OCEN_VSYS_KMS)[0]
    trel = np.linspace(T-1000./AGAMA_T_MYR, T, nrel)
    sp = spray_unwrapped(host, today, T, r_kpc, M, trel, seed=seed, arm_only=1, max_age_myr=700., with_psi=False)
    tr = sp["arm"] == 1; o = observables(sp["xv"][tr])
    return dict(l=w(o["l"]), b=o["b"], pmra=o["pmra"], pmdec=o["pmdec"], vlos=o["vlos"], d=o["dist"], chi=sp["chi"][tr],
                age=sp["t_release_myr_ago"][tr], age_cut_myr=np.float64(700.))


def nullfit_path(start):
    return ROOT/f"results/plot_data/paper_nullfit_start{start}.json"


def nullfit(start):
    path = nullfit_path(start)
    x0 = np.array([OCEN_DISTANCE_KPC, *MU_SYS]) if start == 0 else np.array([5.6157, -3.2477, -6.7871])
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    log = []

    def f(z):
        x = lo+(hi-lo)*np.clip(z, 0, 1)
        t0 = time.time()
        p = dict(NULL, dist=x[0], pmra=x[1], pmdec=x[2])
        lnl = score(spray(p, 8000, 1))["total"]
        log.append(dict(dist=x[0], pmra=x[1], pmdec=x[2], lnL=lnl, lnmeas=measurement(*x), s=time.time()-t0))
        print(f"[{start}] d {x[0]:.4f} pmra {x[1]:.4f} pmdec {x[2]:.4f}  lnL {lnl:.1f}", flush=True)
        tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(dict(log=log, best=max(log, key=lambda r: r["lnL"])), indent=1))
        os.replace(tmp, path)
        return -lnl
    minimize(f, (x0-lo)/(hi-lo), method="Nelder-Mead", options=dict(maxfev=80, xatol=1e-3, fatol=2.0, initial_simplex=None))


def best_null():
    cands = [json.loads(nullfit_path(i).read_text())["best"] for i in (0, 1) if nullfit_path(i).exists()]
    b = max(cands, key=lambda r: r["lnL"])
    return dict(NULL, dist=b["dist"], pmra=b["pmra"], pmdec=b["pmdec"])


def display():
    rec = load()
    models = dict(null_fit=best_null(), null_catalogue=dict(NULL, dist=OCEN_DISTANCE_KPC, pmra=MU_SYS[0], pmdec=MU_SYS[1]),
                  bar28=refined(28), bar16=refined(16))
    out = rec.get("display", {})
    for name, p in models.items():
        r = out.get(name, dict(params=p, lnL={}))
        for seed in (4, 5, 6):
            if str(seed) in r["lnL"]:
                continue
            m = spray(p, 32000, seed); s = score(m)
            r["lnL"][str(seed)] = s["total"]
            if seed == 4:
                np.savez_compressed(OUT/f"{name}_seed4.npz", **m)
                np.save(OUT/f"{name}_seed4_perstar.npy", s["per_star"])
            r["lnmeas"] = measurement(p["dist"], p["pmra"], p["pmdec"])
            out[name] = r; rec["display"] = out; save(rec)
            print(name, seed, f"{s['total']:.1f}", flush=True)


def variants():
    rec = load(); p = best_null(); out = rec.get("variants", {})
    cases = dict(halo_q0p8=dict(axi_variant=dict(halo_q=0.8)), halo_q1p2=dict(axi_variant=dict(halo_q=1.2)),
                 disc_x0p8=dict(axi_variant=dict(disc_scale=0.8)), disc_x1p2=dict(axi_variant=dict(disc_scale=1.2)),
                 vc220=dict(axi_variant=dict(vc_target=220.)), vc240=dict(axi_variant=dict(vc_target=240.)),
                 mcmillan17="McMillan17")
    for name, kw in cases.items():
        if name in out:
            continue
        m = spray_host(host_potential("McMillan17"), p, 32000, 4) if kw == "McMillan17" else spray(p, 32000, 4, host_kw=kw)
        s = score(m)
        np.savez_compressed(OUT/f"variant_{name}_seed4.npz", **m)
        out[name] = dict(params=p, host=kw if isinstance(kw, str) else kw["axi_variant"], lnL=s["total"])
        rec["variants"] = out; save(rec)
        print(name, f"{s['total']:.1f}", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "nullfit":
        nullfit(int(sys.argv[sys.argv.index("--start")+1]) if "--start" in sys.argv else 0)
    elif stage == "display":
        display()
    elif stage == "variants":
        variants()
    else:
        raise SystemExit(__doc__)
