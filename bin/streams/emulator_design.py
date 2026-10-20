#!/usr/bin/env python3
"""Emulator fit and the next design batch for the 9-D stream likelihood (2026-10-18).

Data: training_9d_20261017 (replicate 0 of every design point, 256k epochs, jackknife-corrected; the calibration seeds
r >= 1 share the release-epoch grid of r = 0 and are left out) + training_9d_reps_20261018 (independent replicates with a
random epoch-grid start) + any later batches given with --batch. Replicates of a point are averaged; the noise surface
(emulator.NoiseModel) is fitted to their sample variances with the jackknife variance as covariate; the mean GP
(emulator.MeanEmulator) uses s2/r as fixed per-point noise. Validation: 5-fold over points.
Design (action 'design'): 4096 scrambled-Sobol candidates + 2000 local ones (Gaussian, sd 0.05 in the unit cube, around
the 10 best measured points; added 2026-10-18 after round 1); batch of N points by greedy maximum of mu + kappa sd, with
the predictive sd updated after each pick (kriging believer: fixed kernel, the pick added at its predicted mean), so the
batch spreads; plus 4 seeds at the candidate with the highest mu, and 2 replicates at the 8 batch points with the highest
mu; plus --n-explore pure-exploration (max sd) points and --n-valid random held-out validation points
(never used in fitting). Batch-specific seeds; distinct candidates asserted (Codex review 2026-10-18). Writes OUTDIR/jobs.json for stream_batch.py (independent seeds, random epoch-grid starts).
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


# Design ranges (lo, hi) of dist [kpc] and vc [km/s], and the frame whose R0 normalises vc. "v1": the original design
# (training_set.design); "step2" (2026-10-20, agreed with the user): distance band 5.33-5.53 kpc (+-2 sigma of 5.43 +- 0.05),
# vc 220-250 searched under the VC_PRIOR_KMS prior, gravity21 frame.
DESIGNS = {"v1": dict(dist=(5.25, 5.95), vc=(220., 240.), frame="baumgardt"),
           "step2": dict(dist=(5.33, 5.53), vc=(220., 250.), frame="gravity21")}


def _r0(design):
    from ocen_dm.streams.frames import frame_r0
    return None if DESIGNS[design]["frame"] == "baumgardt" else frame_r0(DESIGNS[design]["frame"])


def unit_to_params(u, design="v1"):
    """Same map as training_set.design for design "v1" (checked against design.json in main())."""
    from training_set import r_cr
    D = DESIGNS[design]; (d0, d1), (v0, v1) = D["dist"], D["vc"]
    p = dict(angle=10.+25.*u[0], omega=30.+10.*u[1], amp=0.6+1.4*u[2], dist=d0+(d1-d0)*u[4],
             pmra=MU_SYS[0]+MU_ERR*(8.*u[5]-4.), pmdec=MU_SYS[1]+MU_ERR*(8.*u[6]-4.), vc=v0+(v1-v0)*u[7], halo_q=0.8+0.4*u[8])
    smax = min(1.5, r_cr(p["omega"], p["vc"], p["halo_q"], _r0(design))/RBAR)
    p["size"] = 0.8+u[3]*(smax-0.8)
    return {k: float(p[k]) for k in NAMES}


def params_to_unit(p, design="v1"):
    """Inverse of unit_to_params (clipped to [0, 1]); used to start a search in one design from points of another."""
    from training_set import r_cr
    D = DESIGNS[design]; (d0, d1), (v0, v1) = D["dist"], D["vc"]
    u = np.array([(p["angle"]-10.)/25., (p["omega"]-30.)/10., (p["amp"]-0.6)/1.4, 0., (p["dist"]-d0)/(d1-d0),
                  (p["pmra"]-MU_SYS[0])/MU_ERR/8.+0.5, (p["pmdec"]-MU_SYS[1])/MU_ERR/8.+0.5, (p["vc"]-v0)/(v1-v0), (p["halo_q"]-0.8)/0.4])
    u = np.clip(u, 0., 1.); q = unit_to_params(u, design)                     # size fraction in the target design's range
    smax = min(1.5, r_cr(q["omega"], q["vc"], q["halo_q"], _r0(design))/RBAR)
    u[3] = np.clip((p["size"]-0.8)/(smax-0.8), 0., 1.)
    return u


def load(batches, purposes=None, exclude_purposes=("validation (held out)",)):
    """Rows (key, unit, y_jk, var_jk, purpose) from replicate 0 of the training set and every completed run of the replicate
    and refinement batches. Checks (Codex 2026-10-18): one manifest per batch, no duplicated simulation identity
    (params, seed, epoch_offset, nrel), consistent coordinates per key, nrel 256k, chi 0-200 jackknife present.
    Runs whose purpose is in exclude_purposes are left out (held-out validation points)."""
    D = json.loads((TRAIN/"design.json").read_text())["points"]
    rows, ident = [], set()

    def add(key, unit, j, purpose):
        idt = (json.dumps(j["params"], sort_keys=True), j["seed"], j.get("epoch_offset"), j["nrel"])
        if idt in ident:
            raise SystemExit(f"duplicate simulation identity {key} seed {j['seed']}")
        ident.add(idt)
        if j["nrel"] != 256000:
            raise SystemExit(f"{key}: nrel {j['nrel']} != 256000")
        rows.append((key, unit, j["total_jk"], j["var_jk"], purpose))
    for f in TRAIN.glob("runs/*_r0_n256k_jk.json"):
        j = json.loads(f.read_text()); add(f"t{j['point']}", D[j["point"]]["unit"], j, "training")
    dirs = list(dict.fromkeys([REPS]+[Path(x) if Path(x).is_absolute() else ROOT/x for x in batches]))
    for b in dirs:
        if not (b/"runs").is_dir():
            raise SystemExit(f"batch directory {b} has no runs/ (missing or mistyped)")
        mids = set()
        for f in (b/"runs").glob("*.json"):
            if f.name.endswith(".error.json"):
                continue
            j = json.loads(f.read_text()); mids.add(j["manifest_sha256"])
            purpose = j.get("purpose", "")
            if purpose in exclude_purposes or (purposes is not None and purpose not in purposes):
                continue
            key, unit = (f"t{j['point']}", D[j["point"]]["unit"]) if "unit" not in j else (j["key"], j["unit"])
            add(key, unit, j, purpose)
        if len(mids) > 1:
            raise SystemExit(f"{b}: runs from {len(mids)} manifests")
    key = np.array([r[0] for r in rows]); X = np.array([r[1] for r in rows], float)
    for k in set(key):                       # one coordinate per key
        if np.ptp(X[key == k], axis=0).max() > 1e-12:
            raise SystemExit(f"key {k}: inconsistent coordinates")
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


def noise_var(g, train=None):
    """s2 at all points of g from a noise model fitted on the points `train` (bool mask) only."""
    train = np.ones(len(g["ybar"]), bool) if train is None else train
    ok = train & (g["r"] >= 2)
    if ok.sum() >= 10:
        nm = NoiseModel().fit(g["X"][train], g["S2"][train], g["r"][train], g["vjk_mean"][train])
        return nm.predict(g["X"], g["vjk_mean"])
    return g["vjk_mean"]


def cv(g, K=5, seed=0, n_restarts=3):
    """K-fold over points with the noise model refitted inside each fold (Codex 2026-10-18)."""
    n = len(g["ybar"]); idx = np.random.default_rng(seed).permutation(n)
    mu, sp, nv = np.zeros(n), np.zeros(n), np.zeros(n)
    for f in np.array_split(idx, K):
        tr = np.ones(n, bool); tr[f] = False
        s2 = noise_var(g, tr); nv[f] = s2[f]/g["r"][f]
        em = MeanEmulator(n_restarts=n_restarts).fit(g["X"][tr], g["ybar"][tr], s2[tr]/g["r"][tr])
        mu[f], sp[f] = em.predict(g["X"][f])
    z = (g["ybar"]-mu)/np.sqrt(sp**2+nv)
    return mu, sp, z


def summarise(y, mu, sp, z, sel):
    return dict(n=int(sel.sum()), rmse=float(np.sqrt(np.mean((y[sel]-mu[sel])**2))), bias=float(np.mean(y[sel]-mu[sel])),
                z_sd=float(np.std(z[sel])), cover68=float(np.mean(np.abs(z[sel]) < 1.)), cover95=float(np.mean(np.abs(z[sel]) < 1.96)),
                median_pred_sd=float(np.median(sp[sel])))


def validate(a):
    g, s2, nm, em, info = fit(a.batch)
    mu, sp, z = cv(g)
    y = g["ybar"]; top = y > np.percentile(y, 80)
    out = dict(info, n_points=len(y), n_runs=int(g["r"].sum()), kernel=str(em.gp.kernel_),
               cv_all=summarise(y, mu, sp, z, np.ones(len(y), bool)), cv_top20=summarise(y, mu, sp, z, top),
               median_noise_sd=float(np.median(np.sqrt(s2))))
    # independent held-out validation points (never used in fitting)
    X, yv, vv, kv = load(a.batch, purposes=("validation (held out)",), exclude_purposes=())
    if len(yv):
        hv = np.array([not k.startswith("t") for k in kv]) & np.isin(kv, [k for k in kv if not k.startswith("t")])
        X, yv, vv = X[hv], yv[hv], vv[hv]
        if len(yv):
            m, s = em.predict(X); zz = (yv-m)/np.sqrt(s**2+vv)
            out["heldout"] = summarise(yv, m, s, zz, np.ones(len(yv), bool))
    print(json.dumps(out, indent=1))
    return out


def completion_gate(batches):
    """Every batch: no error records except jobs refused after an intentional stop (documented in the journal)."""
    for b in [REPS]+[Path(x) if Path(x).is_absolute() else ROOT/x for x in batches]:
        errs = [json.loads(f.read_text()).get("error", "") for f in (b/"runs").glob("*.error.json")]
        bad = [e for e in errs if "source files changed" not in e]
        if bad:
            raise SystemExit(f"{b}: {len(bad)} failed runs, e.g. {bad[0][:200]}")


def design(a):
    from scipy.stats import qmc
    completion_gate(a.batch)
    D = json.loads((TRAIN/"design.json").read_text())["points"]
    assert max(abs(unit_to_params(np.array(p["unit"]))[k]-p["params"][k]) for p in D[:20] for k in NAMES) < 1e-9
    from sklearn.gaussian_process import GaussianProcessRegressor
    g, s2, nm, em, info = fit(a.batch)
    C = qmc.Sobol(9, scramble=True, seed=a.seed).random_base2(12)
    rng = np.random.default_rng(500+a.seed)                     # local candidates around the 10 best measured points
    top = g["X"][np.argsort(-g["ybar"])[:10]]
    C = np.vstack((C, np.clip(np.repeat(top, 200, axis=0)+rng.normal(0, 0.05, (2000, 9)), 0, 1)))
    mu, sd = em.predict(C)
    Xb, yb, nb = g["X"].copy(), (g["ybar"]-em.ymu)/em.ysd, (s2/g["r"])/em.ysd**2
    pick = []
    sd0 = sd.copy()
    for k in range(a.n):
        acq = mu+a.kappa*sd; acq[pick] = -np.inf                    # distinct candidates (Codex 2026-10-18)
        i = int(np.argmax(acq)); pick.append(i)
        Xb = np.vstack((Xb, C[i])); yb = np.append(yb, (mu[i]-em.ymu)/em.ysd); nb = np.append(nb, np.median(nb))
        gp = GaussianProcessRegressor(em.gp.kernel_, alpha=nb, optimizer=None).fit(Xb, yb)
        sd = em.ysd*gp.predict(C, return_std=True)[1]
    sdx = sd.copy(); explore = []                                  # global discovery: largest remaining sd
    for k in range(a.n_explore):
        s_ = sdx.copy(); s_[pick+explore] = -np.inf; i = int(np.argmax(s_)); explore.append(i)
        Xb = np.vstack((Xb, C[i])); yb = np.append(yb, (mu[i]-em.ymu)/em.ysd); nb = np.append(nb, np.median(nb))
        sdx = em.ysd*GaussianProcessRegressor(em.gp.kernel_, alpha=nb, optimizer=None).fit(Xb, yb).predict(C, return_std=True)[1]
    rest = np.setdiff1d(np.arange(len(C)), pick+explore)
    valid = list(np.random.default_rng(1000+a.seed).choice(rest, a.n_valid, replace=False))
    allpts = pick+explore+valid
    assert len(set(allpts)) == len(allpts), "duplicate candidates"
    best = int(np.argmax(mu))
    jobs = []
    for k, i in enumerate(pick):
        jobs.append(dict(key=f"d{a.seed}_{i}", unit=C[i].tolist(), purpose="refine (ucb)", mu=float(mu[i]), sd=float(sd0[i])))
    jobs += [dict(key=f"d{a.seed}_{i}", unit=C[i].tolist(), purpose="explore (max sd)", mu=float(mu[i]), sd=float(sd0[i])) for i in explore]
    jobs += [dict(key=f"d{a.seed}_{i}", unit=C[i].tolist(), purpose="validation (held out)", mu=float(mu[i]), sd=float(sd0[i])) for i in valid]
    top = sorted(pick, key=lambda i: -mu[i])[:a.n_reptop]
    jobs += [dict(key=f"d{a.seed}_{i}", unit=C[i].tolist(), purpose="refine replicate", mu=float(mu[i])) for i in top for _ in range(2)]
    jobs += [dict(key=f"d{a.seed}_{best}", unit=C[best].tolist(), purpose="predicted best, check", mu=float(mu[best]),
                  pred_sd=float(em.predict(C[best:best+1])[1][0])) for _ in range(4)]
    out = []
    for n, j in enumerate(jobs):
        seed = 300000+10000*a.seed+n                                 # batch-specific streams
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
    print(f"{len(out)} jobs ({a.n} ucb, {a.n_explore} explore, {a.n_valid} held-out validation, {2*a.n_reptop} replicates, "
          f"4 at predicted best mu {mu[best]:.0f}) -> {od/'jobs.json'}")
    print("predicted best:", {k: round(v, 3) for k, v in unit_to_params(C[best]).items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("validate", "design", "checkmap"))
    ap.add_argument("--batch", nargs="*", default=[]); ap.add_argument("--outdir"); ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--kappa", type=float, default=2.0); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-explore", type=int, default=6); ap.add_argument("--n-valid", type=int, default=6)
    ap.add_argument("--n-reptop", type=int, default=4)
    a = ap.parse_args()
    if a.action == "checkmap":
        D = json.loads((TRAIN/"design.json").read_text())["points"]
        err = max(abs(unit_to_params(np.array(p["unit"]))[k]-p["params"][k]) for p in D[:40] for k in NAMES)
        print(f"max |param difference| over 40 design points: {err:.2e}"); return
    dict(validate=validate, design=design)[a.action](a)


if __name__ == "__main__":
    main()
