#!/usr/bin/env python3
"""Converged bar + cluster fits on a deterministic stream-track objective (2026-10-16; review plan step 2).

Objective (minimised): chi2_track + prior terms + missing-bin penalty, with
- chi2_track: tracks m = (Delta phi2, mu_alpha*, mu_delta) in 2-deg phi1 bins, defined exactly as in
  bin/streams/track_sensitivity.py; C diagonal = bootstrap variance of the observed bin medians + the model seed variance
  (per component, the mean over the three points of track_sensitivity; fixed for all fits);
- priors (user, 2026-10-16): bar amplitude A ~ N(1.0, 0.1); cluster PMs ~ N(MU_SYS, MU_SYS_ERR); distance per case:
  A = fixed at the measured value, B = N(5.43, 0.05) prior, C = free (bounds only); bar angle fixed at 27 deg or free
  (bounds only);
- missing bins (no model prediction where data exist): PENALTY each (declared; models that do not reach the outer turn
  must not gain by dropping bins);
- bar length bounded by corotation (refine_bar_profiles.corotation_function, R_bar0 4.25 kpc), as in the refinement.
Determinism: one seed (4) and 8000 release epochs for every evaluation (common random numbers). Optimiser: bounded
COBYQA on scaled coordinates. Segment chi2 (phi1 < 12, 12-18, 18-24, 24-28) is recorded at every evaluation so that
near-cluster and outer-turn trade-offs stay visible. Afterwards (``rescore``): the best point of each run at 32000
epochs, held-out seeds 5 and 6, with both the track chi2 and the original star-by-star score.

Usage: OMP_NUM_THREADS=1 PYTHONPATH=src python bin/streams/fit_track_objective.py fit --case 27B --start short
       (cases 27A, 27B, 27C, freeB; starts short, long); `launch` starts all 8 runs as background processes.
Output: results/plot_data/track_fits_20261016/<case>_<start>.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"bin/streams/paper"), str(ROOT/"src")]
import numpy as np
from scipy.optimize import minimize

from track_sensitivity import EDGES, SEGMENTS, observed, tracks, vec
from ocen_dm.cluster import MU_SYS, MU_SYS_ERR, OCEN_DISTANCE_ERR_KPC, OCEN_DISTANCE_KPC

OUTDIR = ROOT/"results/plot_data/track_fits_20261016"
SENS = ROOT/"results/plot_data/track_sensitivity_20261016.json"
NAMES = ("om", "am", "size", "dist", "pmra", "pmdec", "an")
LOW = dict(om=30., am=0.6, size=0.8, dist=5.25, pmra=MU_SYS[0]-4*MU_SYS_ERR, pmdec=MU_SYS[1]-4*MU_SYS_ERR, an=10.)
HIGH = dict(om=40., am=2.2, size=1.5, dist=5.95, pmra=MU_SYS[0]+4*MU_SYS_ERR, pmdec=MU_SYS[1]+4*MU_SYS_ERR, an=35.)
AMP_PRIOR = (1.0, 0.1)
PENALTY = 100.
SEED, NREL = 4, 8000
CASES = {"27A": dict(an=27., dist="fixed"), "27B": dict(an=27., dist="prior"), "27C": dict(an=27., dist="free"),
         "freeB": dict(an=None, dist="prior")}


def starts(start):
    rec = json.loads(SENS.read_text())["analysis"]["points"]
    return dict(rec["bar28s" if start == "short" else "bar28l"]["params"])


def weights():
    an = json.loads(SENS.read_text())["analysis"]
    sd = np.nanmean([np.array(o["seed_sd"], float) for o in an["points"].values()], axis=0)
    obs, err = observed(np.random.default_rng(7))
    return vec(obs), np.asarray(vec(err))**2+sd**2


def free_names(case):
    c = CASES[case]
    return [q for q in NAMES if not (q == "an" and c["an"] is not None) and not (q == "dist" and c["dist"] == "fixed")]


def prior_terms(p, case):
    t = ((p["am"]-AMP_PRIOR[0])/AMP_PRIOR[1])**2+((p["pmra"]-MU_SYS[0])/MU_SYS_ERR)**2+((p["pmdec"]-MU_SYS[1])/MU_SYS_ERR)**2
    if CASES[case]["dist"] == "prior":
        t += ((p["dist"]-OCEN_DISTANCE_KPC)/OCEN_DISTANCE_ERR_KPC)**2
    return float(t)


def track_chi2(tr, m_obs, var):
    m = vec(tr)
    r = m_obs-m
    have = np.isfinite(m_obs) & np.isfinite(var)
    good = have & np.isfinite(m)
    centre = np.tile(0.5*(EDGES[:-1]+EDGES[1:]), 3)
    comp = np.repeat(["dphi2", "pmra", "pmdec"], len(EDGES)-1)
    c2 = np.where(good, r**2/np.where(good, var, 1.), 0.)
    seg = {f"{a}-{b}": float(c2[(centre >= a) & (centre < b)].sum()) for a, b in SEGMENTS}
    return float(c2.sum()), int((have & ~good).sum()), seg, {q: float(c2[comp == q].sum()) for q in ("dphi2", "pmra", "pmdec")}


def fit(args):
    from refine_bar_profiles import corotation_function
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR/f"{args.case}_{args.start}.json"
    if out.exists():
        raise FileExistsError(out)
    m_obs, var = weights()
    rcr = corotation_function()
    p0 = starts(args.start)
    c = CASES[args.case]
    if c["an"] is not None:
        p0["an"] = c["an"]
    if c["dist"] == "fixed":
        p0["dist"] = OCEN_DISTANCE_KPC
    free = free_names(args.case)
    lo = np.array([LOW[q] for q in free]); hi = np.array([HIGH[q] for q in free])
    x0 = np.clip(np.array([p0[q] for q in free]), lo, hi)
    log = []

    def objective(y):
        x = lo+(hi-lo)*np.asarray(y)
        p = dict(p0, **dict(zip(free, map(float, x))))
        ratio = rcr(p["om"])/(4.25*p["size"])
        if ratio < 1:
            return 1e5*(1+(1-ratio))
        t0 = time.time()
        tr = tracks(p, SEED)
        chi2, missing, seg, comp = track_chi2(tr, m_obs, var)
        pr = prior_terms(p, args.case)
        f = chi2+pr+PENALTY*missing
        log.append(dict(params=p, objective=f, chi2_track=chi2, prior=pr, missing=missing, segments=seg, components=comp,
                        seconds=time.time()-t0))
        best = min(log, key=lambda r: r["objective"])
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(case=args.case, start=args.start, case_def=c, free=free, bounds=[lo.tolist(), hi.tolist()],
                                       amp_prior=AMP_PRIOR, penalty=PENALTY, seed=SEED, nrel=NREL, best=best, log=log), indent=1))
        os.replace(tmp, out)
        print(f"{len(log):4d} f {f:8.1f} chi2 {chi2:7.1f} prior {pr:5.1f} miss {missing} | best {best['objective']:.1f} | "
              + " ".join(f"{q}={p[q]:.3f}" for q in free), flush=True)
        return f

    res = minimize(objective, (x0-lo)/(hi-lo), method="COBYQA", bounds=[(0., 1.)]*len(free),
                   options=dict(maxfev=args.maxfev, initial_tr_radius=0.15, final_tr_radius=2e-3))
    rec = json.loads(out.read_text())
    rec["optimizer"] = dict(success=bool(res.success), message=str(res.message), nfev=int(res.nfev))
    out.write_text(json.dumps(rec, indent=1))
    print("DONE", rec["optimizer"], flush=True)


def launch(args):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **{k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
           "PYTHONPATH": str(ROOT/"src")}
    for case in CASES:
        for start in ("short", "long"):
            log = OUTDIR/f"{case}_{start}.log"
            cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "fit", "--case", case, "--start", start,
                   "--maxfev", str(args.maxfev)]
            with log.open("x") as fh:
                pr = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                      start_new_session=True)
            print(case, start, "PID", pr.pid, flush=True)


def rescore(args):
    from grid4_common import score
    from paper_models import spray
    from multiprocessing import Pool
    m_obs, var = weights()
    jobs = []
    for f in sorted(OUTDIR.glob("*.json")):
        if f.name.startswith("rescore"):
            continue
        r = json.loads(f.read_text())
        for seed in (5, 6):
            jobs.append((f.stem, r["best"]["params"], seed))
    with Pool(args.workers) as pool:
        res = pool.map(_rescore_one, jobs)
    out = {}
    for (name, p, seed), (tr, lnl) in zip(jobs, res):
        chi2, missing, seg, comp = track_chi2(tr, m_obs, var)
        out.setdefault(name, dict(params=p, seeds={}))["seeds"][str(seed)] = dict(chi2_track=chi2, missing=missing,
                                                                                    segments=seg, lnL_star=lnl)
    (OUTDIR/"rescore.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, {s: (round(x["chi2_track"]), x["missing"], round(x["lnL_star"])) for s, x in v["seeds"].items()})


def _rescore_one(job):
    from grid4_common import score
    from paper_models import spray
    import track_sensitivity as ts
    name, p, seed = job
    old = ts.NREL
    ts.NREL = 32000
    tr = ts.tracks(p, seed)
    ts.NREL = old
    m = spray(p, 32000, seed)
    return tr, float(score(m)["total"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("fit", "launch", "rescore"))
    ap.add_argument("--case", choices=sorted(CASES))
    ap.add_argument("--start", choices=("short", "long"))
    ap.add_argument("--maxfev", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    dict(fit=fit, launch=launch, rescore=rescore)[args.action](args)


if __name__ == "__main__":
    main()
