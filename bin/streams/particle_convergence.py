#!/usr/bin/env python3
"""How many release epochs does the stream likelihood need? Seed scatter and bias against particle number (2026-10-17).

Models: A = refined 28 deg; B = refined 16 deg; C = best 27B long-bar track fit; E = A with an axisymmetric host variant
(halo flattening q 0.9, halo normalised to v_c(R0) = 230 km/s), which also times the variant build. Release epochs
NREL = 4k..128k, seeds 1-5. Score: grid4_common.score (normalised background, 2026-10-17); tracks as in
track_sensitivity.py. Stored per run: total and per-star ln L, tracks, timings; full sprays only for model A, seed 1.
Usage: OMP_NUM_THREADS=1 PYTHONPATH=src python bin/streams/particle_convergence.py run --workers 10
       PYTHONPATH=src python bin/streams/particle_convergence.py plot
Output: results/plot_data/particle_convergence_20261017.json, results/streams/particle_convergence_20261017/ (A seed-1 sprays),
plots/particle_convergence_20261017_{noise,images}.png
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
sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"bin/streams/paper"), str(ROOT/"src")]
import numpy as np

VERSION = os.environ.get("CONV_VERSION", "")
OUT = ROOT/f"results/plot_data/particle_convergence_20261017{VERSION}.json"
SPRAYS = ROOT/"results/streams/particle_convergence_20261017"
NRELS = (4000, 8000, 16000, 32000, 64000, 128000)
SEEDS = (1, 2, 3, 4, 5)
PAIRS = (("A", "B"), ("A", "C"), ("A", "E"))


def models():
    pa = json.loads((ROOT/"results/plot_data/paper_models.json").read_text())["display"]
    A = dict(pa["bar28"]["params"]); B = dict(pa["bar16"]["params"])
    C = json.loads((ROOT/"results/plot_data/track_fits_20261016/27B_long.json").read_text())["best"]["params"]
    return dict(A=(A, None), B=(B, None), C=(C, None), E=(A, dict(axi_variant=dict(halo_q=0.9, vc_target=230.))))


def task(job):
    name, p, host_kw, nrel, seed = job
    from grid4_common import score
    from paper_models import spray
    from track_sensitivity import EDGES
    from fig3_residuals import predict
    from grid4_common import GC, du, project_gc
    t0 = time.time()
    m = spray(p, nrel, seed, host_kw=host_kw)
    t1 = time.time()
    s = score(m, legacy_bins=True)
    t2 = time.time()
    extra = {}
    if VERSION:
        for lab, kw in (("pooled", dict(pooled=True)), ("chi200", dict(chi_edges=np.arange(0., 205., 5.)))):
            ta = time.time(); e = score(m, **kw)
            extra[lab] = dict(total=float(e["total"]), per_star=np.asarray(e["per_star"], np.float32).tolist(), t=time.time()-ta)
    k = (m["chi"] > 0) & (m["age"] < 700.)
    mu, mx = project_gc(m["l"][k], m["b"][k], GC)
    mm = dict(phi1=mu, dphi2=mx, pmra=m["pmra"][k], pmdec=m["pmdec"][k])
    pred = predict(mm)
    tr = dict(dphi2=[], pmra=[], pmdec=[])
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        km = (mu >= a) & (mu < b) & (np.abs(mx) < 4)
        tr["dphi2"].append(float(np.median(mx[km])) if km.sum() >= 20 else np.nan)
        ks = (du >= a) & (du < b)
        for q in ("pmra", "pmdec"):
            v = pred[q][ks]
            tr[q].append(float(np.nanmedian(v)) if np.isfinite(v).sum() >= 15 else np.nan)
    if name == "A" and seed == 1 and not VERSION:
        SPRAYS.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(SPRAYS/f"A_n{nrel}_s1.npz", **{q: m[q] for q in ("l", "b", "pmra", "pmdec", "vlos", "d", "chi", "age")})
    return dict(model=name, nrel=nrel, seed=seed, total=float(s["total"]), per_star=np.asarray(s["per_star"], np.float32).tolist(),
                K=int(s["K"]), frac_bg=float(s["frac_bg"]), n_particles=int(k.sum()), tracks=tr,
                t_spray=t1-t0, t_score=t2-t1, scores=extra)


def run(args):
    ms = models()
    done = json.loads(OUT.read_text())["runs"] if OUT.exists() else []
    have = {(r["model"], r["nrel"], r["seed"]) for r in done}
    jobs = [(n, p, kw, nrel, s) for nrel in NRELS[::-1] for n, (p, kw) in ms.items() for s in SEEDS if (n, nrel, s) not in have]
    print(f"{len(jobs)} runs on {args.workers} workers", flush=True)
    with Pool(args.workers, maxtasksperchild=4) as pool:
        for i, r in enumerate(pool.imap_unordered(task, jobs), 1):
            done.append(r)
            tmp = OUT.with_suffix(".tmp"); tmp.write_text(json.dumps(dict(models={k: dict(params=v[0], host_kw=v[1]) for k, v in ms.items()},
                                                                          runs=done))); os.replace(tmp, OUT)
            print(f"{i}/{len(jobs)} {r['model']} n{r['nrel']} s{r['seed']} lnL {r['total']:.1f} spray {r['t_spray']:.0f}s score {r['t_score']:.0f}s",
                  flush=True)


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from grid4_common import GC, d, du, dx, project_gc
    from track_sensitivity import EDGES
    rec = json.loads(OUT.read_text())
    R = {(r["model"], r["nrel"], r["seed"]): r for r in rec["runs"]}
    names = sorted({r["model"] for r in rec["runs"]})
    N = np.array([n for n in NRELS if all((m, n, s) in R for m in names for s in SEEDS)])
    out = dict(nrel=N.tolist(), models={}, pairs={})
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))
    for j, mname in enumerate(names):
        tot = np.array([[R[(mname, n, s)]["total"] for s in SEEDS] for n in N])
        sd = tot.std(axis=1, ddof=1); mean = tot.mean(axis=1)
        ax[0].plot(N, sd, "o-", color=f"C{j}", label=f"{mname}: sd of ln L over 5 seeds")
        ax[1].plot(N, mean-mean[-1], "o-", color=f"C{j}", label=f"{mname}")
        out["models"][mname] = dict(mean=mean.tolist(), sd=sd.tolist(),
                                    t_spray=[float(np.mean([R[(mname, n, s)]["t_spray"] for s in SEEDS])) for n in N],
                                    t_score=[float(np.mean([R[(mname, n, s)]["t_score"] for s in SEEDS])) for n in N])
    for j, (a, b) in enumerate(PAIRS):
        if a not in names or b not in names:
            continue
        dl = np.array([[R[(a, n, s)]["total"]-R[(b, n, s)]["total"] for s in SEEDS] for n in N])
        sd = dl.std(axis=1, ddof=1); mean = dl.mean(axis=1)
        ax[0].plot(N, sd, "s--", color=f"C{j+5}", label=f"{a}−{b}: sd of Δln L")
        ax[2].errorbar(N*(1+0.04*j), mean, sd, fmt="s-", color=f"C{j+5}", capsize=2, label=f"{a}−{b}")
        out["pairs"][f"{a}-{b}"] = dict(mean=mean.tolist(), sd=sd.tolist())
    ref = out["models"][names[0]]["sd"][0]*np.sqrt(N[0]/N)
    ax[0].plot(N, ref, "k:", lw=1, label=r"$\propto N^{-1/2}$")
    ax[0].axhline(1, color="0.5", lw=0.6)
    ax[0].set(xscale="log", yscale="log", xlabel="release epochs", ylabel="seed scatter")
    ax[0].legend(fontsize=6.5)
    ax[1].set(xscale="log", xlabel="release epochs", ylabel=r"mean ln L − mean at the largest N  (bias)"); ax[1].legend(fontsize=7)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[2].set(xscale="log", xlabel="release epochs", ylabel=r"Δln L (mean ± sd over seeds)"); ax[2].legend(fontsize=7)
    # track scatter (model A), all components
    trs = {n: np.array([np.concatenate([R[("A", n, s)]["tracks"][q] for q in ("dphi2", "pmra", "pmdec")]) for s in SEEDS]) for n in N}
    nb = len(EDGES)-1; c = 0.5*(EDGES[:-1]+EDGES[1:])
    for k_, q in enumerate(("dphi2", "pmra", "pmdec")):
        s_ = [np.nanmedian(np.nanstd(trs[n][:, k_*nb:(k_+1)*nb], axis=0, ddof=1)) for n in N]
        ax[3].plot(N, s_, "o-", label={"dphi2": r"$\Delta\phi_2$ [deg]", "pmra": r"$\mu_{\alpha*}$ [mas/yr]", "pmdec": r"$\mu_\delta$ [mas/yr]"}[q])
    ax[3].set(xscale="log", yscale="log", xlabel="release epochs", ylabel="median seed scatter of track bins (model A)"); ax[3].legend(fontsize=7)
    for a, t in zip(ax, "abcd"):
        a.text(.03, .94, t, transform=a.transAxes, fontweight="bold", va="top")
    fig.savefig(ROOT/"plots/particle_convergence_20261017_noise.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    # images
    fig, ax = plt.subplots(2, len(N), figsize=(3.2*len(N), 6.2), sharex=True, sharey="row")
    xe = np.linspace(-3, 28, 125)
    for j, n in enumerate(N):
        m = dict(np.load(SPRAYS/f"A_n{n}_s1.npz")); k = (m["chi"] > 0) & (m["age"] < 700.)
        mu, mx = project_gc(m["l"][k], m["b"][k], GC)
        for i, (y, yl, lim, dv) in enumerate(((mx, r"$\Delta\phi_2$", (-6, 6), dx), (m["pmdec"][k], r"$\mu_\delta$", (-14, -4), d["pmdec"]))):
            h, _, _ = np.histogram2d(mu, y, (xe, np.linspace(*lim, 90)))
            ax[i, j].pcolormesh(xe, np.linspace(*lim, 90), h.T, cmap="Greys", norm=LogNorm(vmin=1, vmax=max(h.max(), 2)), rasterized=True)
            ax[i, j].scatter(du, dv, s=0.5, lw=0, color="#d95f02", alpha=0.5, rasterized=True)
            ax[i, j].set_ylim(*lim)
            if j == 0:
                ax[i, j].set_ylabel(yl)
        lnl = [R[("A", n, s)]["total"] for s in SEEDS]
        ax[0, j].set_title(f"{n//1000}k epochs: {k.sum()} particles\nln L {np.mean(lnl):.0f} ± {np.std(lnl, ddof=1):.0f}", fontsize=8)
        ax[1, j].set_xlabel(r"$\phi_1$ [deg]")
    fig.savefig(ROOT/"plots/particle_convergence_20261017_images.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    rec["summary"] = out
    OUT.write_text(json.dumps(rec))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("run", "plot"))
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    dict(run=run, plot=plot)[args.action](args)


if __name__ == "__main__":
    main()
