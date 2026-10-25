#!/usr/bin/env python3
"""Cheap emulator benchmarks on existing scores (Codex 2026-10-18, docs/reviews/codex_emulator_bias_2026-10-18.md).

Datasets: real data (rescore_real_20261018) and the wide-box mocks (rescore_mockwide_T<t>c<c>_20261018). For each:
  designs  : 'full' (all stored runs) vs 'sobol' (training design + replicates only, i.e. no refinement rounds);
  trends   : constant (current), linear, quadratic (ridge) mean + GP residual;
  metrics  : random 5-fold CV and BLOCKED CV (5 folds of angle x Omega cells, 5 x 5 cells), each with RMSE / z sd / 95 %
             coverage over the held-out top-20 % points; for mocks: emulator at the truth (z), emulator max (L-BFGS from
             the 20 best measured points) and its angle error;
  oracle   : (mocks) refit with the truth-check observation added; does the max move to the truth?
Noise: the replicate-fitted noise surface (emulator_design.noise_var) refitted inside each fold.
Output: results/plot_data/emulator_benchmark_20261018.json and a printed table.
Usage: OMP_NUM_THREADS=2 PYTHONPATH=src python bin/streams/emulator_benchmark.py --workers 5
"""
import argparse, json, sys, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
warnings.filterwarnings("ignore")
HELD = "validation (held out)"
DATASETS = ["real"]+[f"T{t}c{c}" for t in (1, 2, 3) for c in (0, 1)]


def rows_of(ds, design):
    p = ROOT/("results/streams/rescore_real_20261018" if ds == "real" else f"results/streams/rescore_mockwide_{ds}_20261018")/"rescore.json"
    R = [r for r in json.loads(p.read_text())["rows"] if r["purpose"] != HELD]
    if design == "sobol":
        R = [r for r in R if r["purpose"] in ("training", "replicate (stochastic kriging)")]
    return R


def grouped(R):
    from ocen_stream.emulator import group_replicates
    return group_replicates(np.array([r["unit"] for r in R]), np.array([r["total_jk"] for r in R]), np.array([r["var_jk"] for r in R]),
                            np.array([r["key"] for r in R]))


def cv(g, folds, trend):
    import emulator_design as ED
    from ocen_stream.emulator import MeanEmulator
    n = len(g["ybar"]); mu, sp, nv = np.zeros(n), np.zeros(n), np.zeros(n)
    for f in folds:
        tr = np.ones(n, bool); tr[f] = False
        s2 = ED.noise_var(g, tr); nv[f] = s2[f]/g["r"][f]
        em = MeanEmulator(n_restarts=2, trend=trend).fit(g["X"][tr], g["ybar"][tr], s2[tr]/g["r"][tr]); mu[f], sp[f] = em.predict(g["X"][f])
    z = (g["ybar"]-mu)/np.sqrt(sp**2+nv); top = g["ybar"] > np.percentile(g["ybar"], 80)
    return dict(rmse_top=float(np.sqrt(np.mean((g["ybar"][top]-mu[top])**2))), bias_top=float(np.mean(g["ybar"][top]-mu[top])),
                z_sd_top=float(np.std(z[top])), cover95_top=float(np.mean(np.abs(z[top]) < 1.96)), rmse_all=float(np.sqrt(np.mean((g["ybar"]-mu)**2))))


def task(args):
    ds, design, trend = args
    import emulator_design as ED
    import emulator_report as ER
    from mock_recovery import to_unit
    from ocen_stream.emulator import MeanEmulator
    from scipy.optimize import minimize
    g = grouped(rows_of(ds, design)); n = len(g["ybar"])
    rnd = np.array_split(np.random.default_rng(0).permutation(n), 5)
    ia = np.clip((g["X"][:, 0]*5).astype(int), 0, 4); io = np.clip((g["X"][:, 1]*5).astype(int), 0, 4); cell = ia*5+io
    perm = np.random.default_rng(1).permutation(25); blk = [np.flatnonzero(np.isin(cell, perm[k::5])) for k in range(5)]
    out = dict(dataset=ds, design=design, trend=trend, n_points=n, cv_random=cv(g, rnd, trend), cv_blocked=cv(g, blk, trend))
    s2 = ED.noise_var(g); em = MeanEmulator(n_restarts=3, trend=trend).fit(g["X"], g["ybar"], s2/g["r"])

    def emax(em, X0):
        best = None
        for x0 in X0:
            r = minimize(lambda x: -em.predict(x[None], return_std=False)[0], x0, method="L-BFGS-B", bounds=[(0, 1)]*9)
            best = r if best is None or r.fun < best.fun else best
        return best
    b = emax(em, g["X"][np.argsort(-g["ybar"])[:20]]); pb = ED.unit_to_params(b.x)
    out["max"] = dict(mu=float(-b.fun), params=pb)
    if ds != "real":
        rec = json.loads((ROOT/f"results/streams/rescore_mockwide_{ds}_20261018/recovery.json").read_text())
        truth = rec["truth"]; ut = np.array(rec["truth_unit"]); tcheck = rec["truth_score"]["check_spray_jk"]
        m, s = em.predict(ut[None])
        out["truth"] = dict(check=tcheck, mu=float(m[0]), sd=float(s[0]), z=float((tcheck-m[0])/np.hypot(s[0], rec["truth_score"]["check_spray_sd_jk"])))
        out["max"]["angle_error"] = pb["angle"]-truth["angle"]; out["max"]["truth_minus_max"] = tcheck-float(-b.fun)
        # oracle: add the truth-check observation
        X2 = np.vstack((g["X"], ut)); y2 = np.append(g["ybar"], tcheck); n2 = np.append(s2/g["r"], rec["truth_score"]["check_spray_sd_jk"]**2)
        em2 = MeanEmulator(n_restarts=3, trend=trend).fit(X2, y2, n2); b2 = emax(em2, X2[np.argsort(-y2)[:20]])
        out["oracle"] = dict(angle_error=ED.unit_to_params(b2.x)["angle"]-truth["angle"], max_mu=float(-b2.fun))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=5); a = ap.parse_args()
    ds = [d for d in DATASETS if d == "real" or (ROOT/f"results/streams/rescore_mockwide_{d}_20261018/recovery.json").exists()]
    jobs = [(d, des, tr) for d in ds for des in ("full", "sobol") for tr in ("constant", "linear", "quadratic")]
    with Pool(a.workers) as pool:
        res = pool.map(task, jobs)
    (ROOT/"results/plot_data/emulator_benchmark_20261018.json").write_text(json.dumps(res, indent=1))
    print(f"{'data':6s} {'design':6s} {'trend':9s} | blocked CV top20: rmse  bias  z_sd cov95 | random top20 rmse | truth z | max angle err | oracle angle err")
    for r in res:
        b, c = r["cv_blocked"], r["cv_random"]; t = r.get("truth", {}); m = r["max"]; o = r.get("oracle", {})
        print(f"{r['dataset']:6s} {r['design']:6s} {r['trend']:9s} | {b['rmse_top']:7.0f} {b['bias_top']:+6.0f} {b['z_sd_top']:5.2f} {b['cover95_top']:5.2f} |"
              f" {c['rmse_top']:6.0f} | {t.get('z', np.nan):+6.2f} | {m.get('angle_error', np.nan):+6.1f} | {o.get('angle_error', np.nan):+6.1f}")


if __name__ == "__main__":
    main()
