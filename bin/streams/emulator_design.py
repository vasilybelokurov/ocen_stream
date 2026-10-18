#!/usr/bin/env python3
"""Emulator fit and the next design batch for the 9-D stream likelihood (2026-10-18).

Data: training_9d_20261017 (replicate 0 of every design point, 256k epochs, jackknife-corrected; the calibration seeds
r >= 1 share the release-epoch grid of r = 0 and are left out) + training_9d_reps_20261018 (independent replicates with a
random epoch-grid start) + any later batches given with --batch. Replicates of a point are averaged; the noise surface
(emulator.NoiseModel) is fitted to their sample variances with the jackknife variance as covariate; the mean GP
(emulator.MeanEmulator) uses s2/r as fixed per-point noise. Validation: 5-fold over points.
Design (action 'design'): 4096 scrambled-Sobol candidates; batch of N points by greedy maximum of mu + kappa sd, with
the predictive sd updated after each pick (kriging believer: fixed kernel, the pick added at its predicted mean), so the
batch spreads; plus 4 seeds at the candidate with the highest mu, and 2 replicates at the 8 batch points with the highest
mu. Writes OUTDIR/jobs.json for stream_batch.py (independent seeds, random epoch-grid starts).
Usage: OMP_NUM_THREADS=2 PYTHONPATH=src python bin/streams/emulator_design.py validate [--batch DIR ...]
       OMP_NUM_THREADS=2 PYTHONPATH=src python bin/streams/emulator_design.py design --outdir DIR [--n 150] [--batch DIR ...]
"""
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
from ocen_dm.streams.emulator import group_replicates, NoiseModel, MeanEmulator, kfold_validate   # noqa: E402

warnings.filterwarnings("ignore", module="sklearn")
TRAIN = ROOT/"results/streams/training_9d_20261017"
REPS = ROOT/"results/streams/training_9d_reps_20261018"
NAMES = ("angle", "omega", "amp", "size", "dist", "pmra", "pmdec", "vc", "halo_q")
MU_SYS, MU_ERR, RBAR = (-3.250, -6.746), 0.022, 4.25


def unit_to_params(u):
    """Same map as training_set.design (checked against design.json in main())."""
    from training_set import r_cr
    p = dict(angle=10.+25.*u[0], omega=30.+10.*u[1], amp=0.6+1.4*u[2], dist=5.25+0.7*u[4],
             pmra=MU_SYS[0]+MU_ERR*(8.*u[5]-4.), pmdec=MU_SYS[1]+MU_ERR*(8.*u[6]-4.), vc=220.+20.*u[7], halo_q=0.8+0.4*u[8])
    smax = min(1.5, r_cr(p["omega"], p["vc"], p["halo_q"])/RBAR)
    p["size"] = 0.8+u[3]*(smax-0.8)
    return {k: float(p[k]) for k in NAMES}


def load(batches):
    """Rows (key, unit, y_jk, var_jk) from replicate 0 of the training set and every run of the replicate/refinement batches."""
    D = json.loads((TRAIN/"design.json").read_text())["points"]
    rows = []
    for f in TRAIN.glob("runs/*_r0_n256k_jk.json"):
        j = json.loads(f.read_text()); rows.append((f"t{j['point']}", D[j["point"]]["unit"], j["total_jk"], j["var_jk"]))
    for b in [REPS]+[Path(x) if Path(x).is_absolute() else ROOT/x for x in batches]:
        for f in (b/"runs").glob("*.json"):
            if f.name.endswith(".error.json"):
                continue
            j = json.loads(f.read_text())
            key, unit = (f"t{j['point']}", D[j["point"]]["unit"]) if "unit" not in j else (j["key"], j["unit"])
            rows.append((key, unit, j["total_jk"], j["var_jk"]))
    key = np.array([r[0] for r in rows]); X = np.array([r[1] for r in rows], float)
    return X, np.array([r[2] for r in rows]), np.array([r[3] for r in rows]), key


def fit(batches, n_restarts=5):
    X, y, v, key = load(batches)
    g = group_replicates(X, y, v, key)
    nrep = int(np.sum(g["r"] >= 2))
    if nrep >= 10:
        nm = NoiseModel().fit(g["X"], g["S2"], g["r"], g["vjk_mean"]); s2 = nm.predict(g["X"], g["vjk_mean"])
        info = dict(noise="replicate-fitted", n_replicated=nrep, coef=nm.coef.tolist(), resid_sd=nm.resid_sd)
    else:
        nm, s2 = None, g["vjk_mean"]; info = dict(noise="jackknife (too few replicates)", n_replicated=nrep)
    em = MeanEmulator(n_restarts=n_restarts).fit(g["X"], g["ybar"], s2/g["r"])
    return g, s2, nm, em, info


def validate(a):
    g, s2, nm, em, info = fit(a.batch)
    val = kfold_validate(g["X"], g["ybar"], s2/g["r"], K=5, n_restarts=3)
    top = g["ybar"] > np.percentile(g["ybar"], 80)
    out = dict(info, n_points=len(g["ybar"]), n_runs=int(g["r"].sum()), kernel=str(em.gp.kernel_), rmse=val["rmse"],
               z_sd=val["z_sd"], cover68=val["cover68"], cover95=val["cover95"],
               rmse_top20=float(np.sqrt(np.mean((g["ybar"][top]-val["mu"][top])**2))), z_sd_top20=float(np.std(val["z"][top])),
               median_pred_sd=float(np.median(val["sd_pred"])), median_noise_sd=float(np.median(np.sqrt(s2))))
    print(json.dumps(out, indent=1))
    return out


def design(a):
    from scipy.stats import qmc
    from sklearn.gaussian_process import GaussianProcessRegressor
    g, s2, nm, em, info = fit(a.batch)
    C = qmc.Sobol(9, scramble=True, seed=a.seed).random_base2(12)
    mu, sd = em.predict(C)
    Xb, yb, nb = g["X"].copy(), (g["ybar"]-em.ymu)/em.ysd, (s2/g["r"])/em.ysd**2
    pick = []
    for k in range(a.n):
        i = int(np.argmax(mu+a.kappa*sd)); pick.append(i)
        Xb = np.vstack((Xb, C[i])); yb = np.append(yb, (mu[i]-em.ymu)/em.ysd); nb = np.append(nb, np.median(nb))
        gp = GaussianProcessRegressor(em.gp.kernel_, alpha=nb, optimizer=None).fit(Xb, yb)
        sd = em.ysd*gp.predict(C, return_std=True)[1]
    best = int(np.argmax(mu))
    jobs = []
    for k, i in enumerate(pick):
        jobs.append(dict(key=f"d{a.seed}_{i}", unit=C[i].tolist(), purpose=f"refine (ucb, rank {k})", mu=float(mu[i])))
    top = sorted(pick, key=lambda i: -mu[i])[:8]
    jobs += [dict(key=f"d{a.seed}_{i}", unit=C[i].tolist(), purpose="refine replicate", mu=float(mu[i])) for i in top for _ in range(2)]
    jobs += [dict(key=f"d{a.seed}_{best}", unit=C[best].tolist(), purpose="predicted best, check", mu=float(mu[best]),
                  pred_sd=float(em.predict(C[best:best+1])[1][0])) for _ in range(4)]
    out = []
    for n, j in enumerate(jobs):
        seed = 300000+n
        out.append(dict(j, run_id=f"{j['key']}_{n:03d}", point=-1, replicate=n, seed=seed, nrel=256000,
                        epoch_offset=float(np.random.default_rng([seed, 777]).uniform()), params=unit_to_params(np.array(j["unit"]))))
    od = Path(a.outdir) if Path(a.outdir).is_absolute() else ROOT/a.outdir
    od.mkdir(parents=True, exist_ok=True)
    if (od/"jobs.json").exists():
        raise SystemExit("jobs.json exists; refusing to overwrite")
    (od/"jobs.json").write_text(json.dumps(out, indent=1))
    (od/"design_info.json").write_text(json.dumps(dict(info, kappa=a.kappa, seed=a.seed, n=a.n, best_unit=C[best].tolist(),
                                                       best_params=unit_to_params(C[best]), best_mu=float(mu[best]),
                                                       kernel=str(em.gp.kernel_)), indent=1))
    print(f"{len(out)} jobs ({a.n} ucb, 16 replicates, 4 at predicted best mu {mu[best]:.0f}) -> {od/'jobs.json'}")
    print("predicted best:", {k: round(v, 3) for k, v in unit_to_params(C[best]).items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("validate", "design", "checkmap"))
    ap.add_argument("--batch", nargs="*", default=[]); ap.add_argument("--outdir"); ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--kappa", type=float, default=2.0); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.action == "checkmap":
        D = json.loads((TRAIN/"design.json").read_text())["points"]
        err = max(abs(unit_to_params(np.array(p["unit"]))[k]-p["params"][k]) for p in D[:40] for k in NAMES)
        print(f"max |param difference| over 40 design points: {err:.2e}"); return
    dict(validate=validate, design=design)[a.action](a)


if __name__ == "__main__":
    main()
