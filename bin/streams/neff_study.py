#!/usr/bin/env python3
"""Effective number of model particles behind each member, and the seed noise it predicts (2026-10-17).

Models A (refined 28 deg) and B (refined 16 deg); release epochs 32k, 128k, 256k, 512k; seeds 1-5; pooled score
(grid4_common.score(pooled=True), normalised background) with return_neff: per member the Kish effective number of
particles behind its foreground density. Stored per run: total, per-star ln L and n_eff (float32), timings.
Plots (plot): (a) members coloured by median n_eff (model A, largest N) in (phi1, Delta phi2) and (b) (phi1, mu_delta);
(c) seed sd of ln L(A) - ln L(B) against the median n_eff of the members (all, and phi1 18-26 deg), with the independent-
star expectation sqrt(sum var) and N^-1/2; (d) per-star seed variance of ln L(A) against 1/n_eff (shot-noise test).
Usage: OMP_NUM_THREADS=1 PYTHONPATH=src python bin/streams/neff_study.py run --workers 10
       PYTHONPATH=src python bin/streams/neff_study.py plot
Output: results/plot_data/neff_study_20261017.json, plots/neff_study_20261017.png
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

OUT = ROOT/"results/plot_data/neff_study_20261017.json"
NRELS = (32000, 128000, 256000, 512000)
SEEDS = (1, 2, 3, 4, 5)
TURN = (18., 26.)


def models():
    pa = json.loads((ROOT/"results/plot_data/paper_models.json").read_text())["display"]
    return dict(A=dict(pa["bar28"]["params"]), B=dict(pa["bar16"]["params"]))


def task(job):
    name, p, nrel, seed = job
    from grid4_common import score
    from paper_models import spray
    t0 = time.time(); m = spray(p, nrel, seed); t1 = time.time()
    s = score(m, pooled=True, return_neff=True); t2 = time.time()
    k = (m["chi"] > 0) & (m["age"] < 700.)
    return dict(model=name, nrel=nrel, seed=seed, total=float(s["total"]), n_particles=int(k.sum()),
                per_star=np.asarray(s["per_star"], np.float32).tolist(), n_eff=np.asarray(s["n_eff"], np.float32).tolist(),
                t_spray=t1-t0, t_score=t2-t1)


def run(args):
    ms = models()
    done = json.loads(OUT.read_text())["runs"] if OUT.exists() else []
    have = {(r["model"], r["nrel"], r["seed"]) for r in done}
    jobs = [(n, p, nrel, s) for nrel in NRELS[::-1] for n, p in ms.items() for s in SEEDS if (n, nrel, s) not in have]
    print(f"{len(jobs)} runs on {args.workers} workers", flush=True)
    with Pool(args.workers, maxtasksperchild=2) as pool:
        for i, r in enumerate(pool.imap_unordered(task, jobs), 1):
            done.append(r)
            tmp = OUT.with_suffix(".tmp"); tmp.write_text(json.dumps(dict(models=ms, runs=done))); os.replace(tmp, OUT)
            print(f"{i}/{len(jobs)} {r['model']} n{r['nrel']} s{r['seed']} lnL {r['total']:.1f} median n_eff "
                  f"{np.median(r['n_eff']):.1f} spray {r['t_spray']:.0f}s score {r['t_score']:.0f}s", flush=True)


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from grid4_common import d, du, dx
    rec = json.loads(OUT.read_text())
    R = {(r["model"], r["nrel"], r["seed"]): r for r in rec["runs"]}
    N = [n for n in NRELS if all((m, n, s) in R for m in "AB" for s in SEEDS)]
    turn = (du >= TURN[0]) & (du < TURN[1])
    out = dict(nrel=N, rows=[])
    fig, ax = plt.subplots(1, 4, figsize=(23, 4.9), gridspec_kw=dict(wspace=0.45))
    ne = np.median([R[("A", N[-1], s)]["n_eff"] for s in SEEDS], axis=0)
    o = np.argsort(-ne)
    for a, y, lab in ((ax[0], dx, r"$\Delta\phi_2$ [deg]"), (ax[1], d["pmdec"], r"$\mu_\delta$ [mas/yr]")):
        sc = a.scatter(du[o], y[o], c=np.maximum(ne[o], 0.1), s=3, lw=0, cmap="viridis", norm=LogNorm(1, np.percentile(ne, 99)))
        a.set(xlabel=r"$\phi_1$ [deg]", ylabel=lab, xlim=(-6, 30))
    ax[0].set_ylim(-9, 22); ax[1].set_ylim(-15, -4)
    cb = fig.colorbar(sc, ax=ax[1], fraction=0.05, pad=0.02); cb.set_label(f"n_eff per member (A, {N[-1]//1000}k)")
    for lab, sel, mk in (("all members", np.ones_like(turn), "o"), (f"turn, φ1 {TURN[0]:.0f}–{TURN[1]:.0f}°", turn, "s")):
        xs, act, ind = [], [], []
        for n in N:
            D = np.array([np.asarray(R[("A", n, s)]["per_star"])-np.asarray(R[("B", n, s)]["per_star"]) for s in SEEDS])[:, sel]
            act.append(D.sum(1).std(ddof=1)); ind.append(np.sqrt(D.var(0, ddof=1).sum()))
            xs.append(np.median(np.array([R[("A", n, s)]["n_eff"] for s in SEEDS])[:, sel]))
            out["rows"].append(dict(subset=lab, nrel=n, median_neff=xs[-1], sd_actual=act[-1], sd_independent=ind[-1],
                                    dlnL_mean=float(D.sum(1).mean())))
        ax[2].plot(xs, act, mk+"-", label=f"{lab}: actual")
        ax[2].plot(xs, ind, mk+"--", label=f"{lab}: if stars independent")
        for x_, n in zip(xs, N):
            ax[2].annotate(f"{n//1000}k", (x_, act[N.index(n)]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    xx = np.array(ax[2].get_xlim())
    ax[2].set(xscale="log", yscale="log", xlabel="median n_eff of the members (model A)", ylabel="seed sd of Δln L (A − B)")
    ax[2].legend(fontsize=6.5)
    # shot-noise test: per-star seed variance vs 1/n_eff, model A, each N
    for j, n in enumerate(N):
        P = np.array([R[("A", n, s)]["per_star"] for s in SEEDS]); E = np.median([R[("A", n, s)]["n_eff"] for s in SEEDS], axis=0)
        v = P.var(0, ddof=1)
        bins = np.geomspace(0.3, max(E.max(), 1), 25); c = np.sqrt(bins[:-1]*bins[1:]); med = []
        for a_, b_ in zip(bins[:-1], bins[1:]):
            k_ = (E >= a_) & (E < b_)
            med.append(np.median(v[k_]) if k_.sum() >= 10 else np.nan)
        ax[3].plot(c, med, "o-", ms=3, color=f"C{j}", label=f"{n//1000}k: median per-star seed variance")
    g = np.geomspace(0.3, 3e3, 50)
    ax[3].plot(g, 1/g, "k:", label="1/n_eff (shot noise)")
    ax[3].set(xscale="log", yscale="log", xlabel="n_eff of the member", ylabel="seed variance of the member's ln L (A)")
    ax[3].legend(fontsize=6.5)
    for a, t in zip(ax, "abcd"):
        a.text(.03, .95, t, transform=a.transAxes, fontweight="bold", va="top")
    fig.savefig(ROOT/"plots/neff_study_20261017.png", dpi=130, bbox_inches="tight")
    rec["summary"] = out; OUT.write_text(json.dumps(rec))
    for r in out["rows"]:
        print(r)
    for n in N:
        E = np.array([R[("A", n, s)]["n_eff"] for s in SEEDS])
        print(n, "A n_eff median/p10 all:", np.median(E).round(1), np.percentile(np.median(E, 0), 10).round(2),
              "turn:", np.median(E[:, turn]).round(1), "frac members n_eff<5:", (np.median(E, 0) < 5).mean().round(2),
              "particles", int(np.mean([R[("A", n, s)]["n_particles"] for s in SEEDS])))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("run", "plot"))
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    dict(run=run, plot=plot)[args.action](args)


if __name__ == "__main__":
    main()
