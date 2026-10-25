#!/usr/bin/env python3
"""Can the existing bar + cluster parameters remove the stream track residual? Local track Jacobians (2026-10-16).

Observable tracks m = (Delta phi2, mu_alpha*, mu_delta) in 2-deg phi1 bins (-2..28):
- PMs as in paper Fig. 3: a member's model PM is the median PM of model particles within 0.5 deg of its (phi1, dphi2)
  (fig3_residuals.predict; observed PMs never select particles); the bin value is the median over members; the observed
  value is the median of the same members' PMs.
- Delta phi2: median dphi2 of model particles (|dphi2| < 4, >= 20 particles) vs the median of members (|dx| < 4).
Particles: trailing arm, chi > 0, released < 700 Myr ago (as the likelihood).

Points (parameters from saved fits): bar28s = refined 28 deg (short bar, S 0.87); bar28l = 28 deg long-bar grid
start (Omega 36, A 1.4, S 1.3, d 5.6, catalogue PMs); bar16 = refined 16 deg. At each point, central differences
in theta = (Omega_b, A, S, D, pmra, pmdec, phi_b) with common random numbers (the same seed for theta, theta +- h),
8000 release epochs, seeds 4-6; J is the seed mean.

Analysis: r = m_obs - m_mod(theta0); C = C_data (bootstrap of the bin medians, diagonal) + C_seed (variance of
m_mod over the three seeds, diagonal). Weighted least squares dtheta = (J^T C^-1 J + P)^-1 J^T C^-1 r, with P from the
measurements of D (5.43 +- 0.05 kpc) and the cluster PMs (+- 0.027 mas/yr) acting on theta0 + dtheta; no prior on the
bar parameters. Reported: chi2 = r^T C^-1 r before and after (overall and per segment phi1 < 12, 12-18, 18-24,
24-28), the fraction of chi2 removed, dtheta; with and without phi_b. Linearised: dtheta outside the step range is a
direction, not a prediction.

Check of the linearisation (``verify``): sprays at theta0 + dtheta (measurement-prior solution; angle fixed and angle
free), seeds 4-6, scored with the same C; chi2 compared with the linear prediction.

Usage: OMP_NUM_THREADS=1 PYTHONPATH=src python bin/streams/track_sensitivity.py run --workers 8
       PYTHONPATH=src python bin/streams/track_sensitivity.py analyse
       OMP_NUM_THREADS=1 PYTHONPATH=src python bin/streams/track_sensitivity.py verify --workers 8
Output: results/plot_data/track_sensitivity_20261016.json (tracks per spray, analysis), plots/track_sensitivity_20261016.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"bin/streams")); sys.path.insert(0, str(ROOT/"bin/streams/paper")); sys.path.insert(0, str(ROOT/"src"))
import numpy as np

OUT = ROOT/"results/plot_data/track_sensitivity_20261016.json"
EDGES = np.arange(-2., 28.1, 2.)
NREL, SEEDS = 8000, (4, 5, 6)
NAMES = ("om", "am", "size", "dist", "pmra", "pmdec", "an")
STEPS = dict(om=0.5, am=0.1, size=0.05, dist=0.05, pmra=0.027, pmdec=0.027, an=2.0)
SEGMENTS = [(-2, 12), (12, 18), (18, 24), (24, 28)]


def points():
    import json as _j
    from paper_models import refined
    from grid4_common import OCEN_OBS
    p28, p16 = refined(28), refined(16)
    long28 = dict(om=36., an=28., am=1.4, size=1.3, dist=5.6, pmra=OCEN_OBS["pmra"], pmdec=OCEN_OBS["pmdec"])
    return dict(bar28s=p28, bar28l=long28, bar16=p16)


def tracks(p, seed):
    from fig3_residuals import predict
    from grid4_common import GC, d, du, dx, project_gc
    from paper_models import spray
    m = spray(p, NREL, seed)
    k = (m["chi"] > 0) & (m["age"] < 700.)
    mu, mx = project_gc(m["l"][k], m["b"][k], GC)
    mm = dict(phi1=mu, dphi2=mx, pmra=m["pmra"][k], pmdec=m["pmdec"][k])
    pred = predict(mm)
    out = dict(dphi2=[], pmra=[], pmdec=[], n_pred=[])
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        km = (mu >= a) & (mu < b) & (np.abs(mx) < 4)
        out["dphi2"].append(float(np.median(mx[km])) if km.sum() >= 20 else np.nan)
        ks = (du >= a) & (du < b)
        for q in ("pmra", "pmdec"):
            v = pred[q][ks]
            out[q].append(float(np.nanmedian(v)) if np.isfinite(v).sum() >= 15 else np.nan)
        out["n_pred"].append(int(np.isfinite(pred["pmra"][ks]).sum()))
    return out


def task(args):
    name, label, p, seed = args
    t0 = time.time()
    tr = tracks(p, seed)
    return dict(point=name, label=label, seed=seed, params=p, tracks=tr, seconds=time.time()-t0)


def observed(rng):
    from fig3_residuals import predict  # noqa: F401  (same member set convention)
    from grid4_common import d, du, dx
    obs, err = dict(dphi2=[], pmra=[], pmdec=[]), dict(dphi2=[], pmra=[], pmdec=[])
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        k = (du >= a) & (du < b)
        for q, v in (("dphi2", dx[k & (np.abs(dx) < 4)]), ("pmra", d["pmra"][k]), ("pmdec", d["pmdec"][k])):
            v = v[np.isfinite(v)]
            obs[q].append(float(np.median(v)) if v.size >= 15 else np.nan)
            err[q].append(float(np.std(np.median(rng.choice(v, (400, v.size)), axis=1))) if v.size >= 15 else np.nan)
    return obs, err


def run(args):
    pts = points()
    jobs = []
    for name, p in pts.items():
        for seed in SEEDS:
            jobs.append((name, "base", p, seed))
            for q in NAMES:
                for sgn in (+1, -1):
                    jobs.append((name, f"{q}{'+' if sgn > 0 else '-'}", dict(p, **{q: p[q]+sgn*STEPS[q]}), seed))
    if args.only_time:
        r = task(jobs[0]); print(f"one spray: {r['seconds']:.1f} s"); return
    done = json.loads(OUT.read_text())["sprays"] if OUT.exists() else []
    have = {(r["point"], r["label"], r["seed"]) for r in done}
    todo = [j for j in jobs if (j[0], j[1], j[3]) not in have]
    print(f"{len(jobs)} sprays, {len(todo)} to run on {args.workers} workers", flush=True)
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(task, todo), 1):
            done.append(r)
            tmp = OUT.with_suffix(".tmp"); tmp.write_text(json.dumps(dict(sprays=done), indent=1)); os.replace(tmp, OUT)
            print(f"{i}/{len(todo)} {r['point']} {r['label']} seed {r['seed']} {r['seconds']:.0f}s", flush=True)


def vec(tr):
    return np.concatenate([np.asarray(tr[q], float) for q in ("dphi2", "pmra", "pmdec")])


def analyse(args):
    rec = json.loads(OUT.read_text())
    sprays = rec["sprays"]
    obs, err = observed(np.random.default_rng(7))
    m_obs, e_obs = vec(obs), vec(err)
    nb = len(EDGES)-1
    centre = np.tile(0.5*(EDGES[:-1]+EDGES[1:]), 3)
    comp = np.repeat(["dphi2", "pmra", "pmdec"], nb)
    from ocen_stream.cluster import MU_SYS, MU_SYS_ERR, OCEN_DISTANCE_ERR_KPC, OCEN_DISTANCE_KPC
    meas = dict(dist=(OCEN_DISTANCE_KPC, OCEN_DISTANCE_ERR_KPC), pmra=(MU_SYS[0], MU_SYS_ERR), pmdec=(MU_SYS[1], MU_SYS_ERR))
    result = {}
    for name in dict.fromkeys(r["point"] for r in sprays):
        S = {(r["label"], r["seed"]): vec(r["tracks"]) for r in sprays if r["point"] == name}
        p0 = next(r["params"] for r in sprays if r["point"] == name and r["label"] == "base")
        base = np.array([S[("base", s)] for s in SEEDS])
        m0 = np.nanmean(base, axis=0); s_seed = np.nanstd(base, axis=0, ddof=1)
        J = np.column_stack([np.nanmean([(S[(f"{q}+", s)]-S[(f"{q}-", s)])/(2*STEPS[q]) for s in SEEDS], axis=0) for q in NAMES])
        r = m_obs-m0
        var = e_obs**2+s_seed**2
        ok = np.isfinite(r) & np.isfinite(var) & np.all(np.isfinite(J), axis=1) & (var > 0)
        w = 1/var[ok]
        out = dict(params=p0, n_components=int(ok.sum()), residual=r.tolist(), sigma=np.sqrt(var).tolist(), jacobian=J.tolist(),
                   track_model=m0.tolist(), seed_sd=s_seed.tolist())
        for use_angle in (False, True):
            cols = [i for i, q in enumerate(NAMES) if use_angle or q != "an"]
            Jk = J[ok][:, cols]
            A = Jk.T @ (w[:, None]*Jk); bvec = Jk.T @ (w*r[ok])
            P = np.zeros((len(cols), len(cols))); pc = np.zeros(len(cols))
            for j, ci in enumerate(cols):
                q = NAMES[ci]
                if q in meas:
                    mu_, sg = meas[q]
                    P[j, j] = 1/sg**2; pc[j] = (mu_-p0[q])/sg**2
            dth = np.linalg.solve(A+P, bvec+pc)
            dth_free = np.linalg.lstsq(Jk*np.sqrt(w)[:, None], r[ok]*np.sqrt(w), rcond=None)[0]
            after = r[ok]-Jk @ dth
            after_free = r[ok]-Jk @ dth_free
            seg = {}
            for a, b in SEGMENTS:
                ks = (centre[ok] >= a) & (centre[ok] < b)
                seg[f"{a}-{b}"] = dict(n=int(ks.sum()), chi2_before=float(np.sum(w[ks]*r[ok][ks]**2)),
                                       chi2_after_prior=float(np.sum(w[ks]*after[ks]**2)),
                                       chi2_after_free=float(np.sum(w[ks]*after_free[ks]**2)))
            bycomp = {c: dict(chi2_before=float(np.sum((w*r[ok]**2)[comp[ok] == c])),
                              chi2_after_prior=float(np.sum((w*after**2)[comp[ok] == c])),
                              chi2_after_free=float(np.sum((w*after_free**2)[comp[ok] == c]))) for c in ("dphi2", "pmra", "pmdec")}
            sv = np.linalg.svd(Jk*np.sqrt(w)[:, None], compute_uv=False)
            out["with_angle" if use_angle else "fixed_angle"] = dict(
                parameters=[NAMES[c] for c in cols], dtheta_prior=dth.tolist(), dtheta_free=dth_free.tolist(),
                chi2_before=float(np.sum(w*r[ok]**2)), chi2_after_prior=float(np.sum(w*after**2)),
                chi2_after_free=float(np.sum(w*after_free**2)), segments=seg, components=bycomp,
                singular_values=sv.tolist(), model_after_prior=(m0[ok]+Jk @ dth).tolist(), ok_index=np.flatnonzero(ok).tolist())
        result[name] = out
    rec["analysis"] = dict(observed=m_obs.tolist(), observed_err=e_obs.tolist(), edges=EDGES.tolist(), points=result)
    OUT.write_text(json.dumps(rec, indent=1))
    plot(rec)
    for name, o in result.items():
        for key in ("fixed_angle", "with_angle"):
            a = o[key]
            print(f"{name} {key}: chi2 {a['chi2_before']:.0f} -> {a['chi2_after_prior']:.0f} (with measurement priors), "
                  f"{a['chi2_after_free']:.0f} (free); dtheta(prior) " +
                  ", ".join(f"{q} {v:+.3f}" for q, v in zip(a["parameters"], a["dtheta_prior"])))
            print("   segments:", {k: (round(v["chi2_before"]), round(v["chi2_after_prior"]), round(v["chi2_after_free"])) for k, v in a["segments"].items()})
            print("   components:", {k: (round(v["chi2_before"]), round(v["chi2_after_prior"]), round(v["chi2_after_free"])) for k, v in a["components"].items()})


def verify(args):
    rec = json.loads(OUT.read_text())
    an = rec["analysis"]
    jobs = []
    for name, o in an["points"].items():
        for key in ("fixed_angle", "with_angle"):
            p = dict(o["params"])
            for q, v in zip(o[key]["parameters"], o[key]["dtheta_prior"]):
                p[q] = p[q]+v
            for seed in SEEDS:
                jobs.append((name, key, p, seed))
    print(f"{len(jobs)} verification sprays on {args.workers} workers", flush=True)
    with Pool(args.workers) as pool:
        res = list(pool.imap_unordered(task, jobs))
    m_obs = np.array(an["observed"]); out = {}
    for name, o in an["points"].items():
        ok = np.array(o["fixed_angle"]["ok_index"]); sig = np.array(o["sigma"])
        w = 1/sig[ok]**2
        for key in ("fixed_angle", "with_angle"):
            rows = [r for r in res if r["point"] == name and r["label"] == key]
            mods = np.array([vec(r["tracks"]) for r in rows])
            m = np.nanmean(mods, axis=0)
            rr = (m_obs-m)[ok]
            good = np.isfinite(rr)
            chi2 = float(np.sum((w*rr**2)[good]))
            centre = np.tile(0.5*(EDGES[:-1]+EDGES[1:]), 3)[ok]
            seg = {f"{a}-{b}": float(np.sum((w*rr**2)[good & (centre >= a) & (centre < b)])) for a, b in SEGMENTS}
            out[f"{name}/{key}"] = dict(params=rows[0]["params"], chi2_actual=chi2, n_used=int(good.sum()), n_expected=int(len(ok)),
                                         chi2_linear=o[key]["chi2_after_prior"], chi2_before=o[key]["chi2_before"], segments=seg,
                                         track=m.tolist())
            print(f"{name} {key}: chi2 before {o[key]['chi2_before']:.0f}, linear prediction {o[key]['chi2_after_prior']:.0f}, "
                  f"actual {chi2:.0f} ({good.sum()}/{len(ok)} comps); segments {dict((k, round(v)) for k, v in seg.items())}; "
                  + ", ".join(f"{q} {rows[0]['params'][q]:.3f}" for q in NAMES), flush=True)
    rec["verify"] = out
    OUT.write_text(json.dumps(rec, indent=1))


def plot(rec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    an = rec["analysis"]
    nb = len(EDGES)-1
    c = 0.5*(EDGES[:-1]+EDGES[1:])
    obs, err = np.array(an["observed"]), np.array(an["observed_err"])
    fig, ax = plt.subplots(3, 3, figsize=(13, 8.5), sharex=True)
    labels = [r"$\Delta\phi_2$ [deg]", r"$\mu_{\alpha*}$ [mas/yr]", r"$\mu_\delta$ [mas/yr]"]
    for j, (name, o) in enumerate(an["points"].items()):
        m0 = np.array(o["track_model"]); sd = np.array(o["seed_sd"])
        for key, col, ls in (("fixed_angle", "C3", "-"), ("with_angle", "C2", "--")):
            mod = np.full(3*nb, np.nan); mod[o[key]["ok_index"]] = o[key]["model_after_prior"]
            for i in range(3):
                ax[i, j].plot(c, (mod-m0)[i*nb:(i+1)*nb] + 0, ls, color=col, lw=1.2,
                              label=("best linear shift, angle fixed" if key == "fixed_angle" else "best linear shift, angle free") if i == 0 else None)
        for i in range(3):
            sl = slice(i*nb, (i+1)*nb)
            ax[i, j].errorbar(c, (obs-m0)[sl], np.sqrt(err[sl]**2+sd[sl]**2), fmt="ko", ms=3, lw=0.8, label="data − model" if i == 0 else None)
            ax[i, j].axhline(0, color="0.5", lw=0.5)
            if j == 0:
                ax[i, j].set_ylabel("Δ " + labels[i])
        p = o["params"]
        a = o["fixed_angle"]; b = o["with_angle"]
        ax[0, j].set_title(f"{name}: {p['an']:.0f}°, Ω {p['om']:.2f}, A {p['am']:.2f}, S {p['size']:.2f}, d {p['dist']:.2f}\n"
                           f"χ² {a['chi2_before']:.0f} → {a['chi2_after_prior']:.0f} (angle fixed), {b['chi2_after_prior']:.0f} (angle free)",
                           fontsize=8)
        ax[2, j].set_xlabel(r"$\phi_1$ [deg]")
    ax[0, 0].legend(fontsize=7)
    fig.suptitle("Track residual (data − model) and the best linearised parameter shift (measurement priors on d and PMs)", fontsize=10)
    fig.savefig(ROOT/"plots/track_sensitivity_20261016.png", dpi=130, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("run", "analyse", "verify"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only-time", action="store_true")
    args = ap.parse_args()
    dict(run=run, analyse=analyse, verify=verify)[args.action](args)


if __name__ == "__main__":
    main()
