#!/usr/bin/env python3
"""Emulator recovery test on a mock catalogue (2026-10-18).

Input: OUTDIR/rescore.json from rescore_batch.py for the mock (all stored emulator sprays scored against it) and the mock's
truth (mocks_20261018/T<t>/mock_c<c>.json). Steps (same pipeline as the real data):
1. emulator fit: replicate means, noise surface from the replicate variances (jackknife covariate), mean GP; held-out
   validation points excluded from the fit;
2. 5-fold CV with the noise model refitted per fold; held-out points;
3. emulator maximum (L-BFGS-B from the 20 best runs) and optimised 1-D profiles (per bin: L-BFGS-B over the other 8
   coordinates from the bin's 3 best candidates);
4. truth: emulator mean +- sd at the truth's unit coordinates vs the jackknife score of the independent truth-check spray
   (same parameters, other seed) against the mock; the generating spray's own score for reference.
Output: OUTDIR/recovery.json, plots/mock_recovery_<tag>.png
Usage: OMP_NUM_THREADS=4 PYTHONPATH=src python bin/streams/mock_recovery.py --mock results/streams/mocks_20261018/T1/mock_c0.npz \
           --rescore DIR --tag T1c0
"""
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
warnings.filterwarnings("ignore")
import emulator_design as ED      # noqa: E402
import emulator_report as ER      # noqa: E402
from ocen_dm.streams.emulator import group_replicates, MeanEmulator   # noqa: E402

HELD = "validation (held out)"


def to_unit(p, smax):
    u = np.array([(p["angle"]-10)/25, (p["omega"]-30)/10, (p["amp"]-0.6)/1.4, 0., (p["dist"]-5.25)/0.7,
                  (p["pmra"]+3.250)/(8*0.022)+0.5, (p["pmdec"]+6.746)/(8*0.022)+0.5, (p["vc"]-220)/20, (p["halo_q"]-0.8)/0.4])
    sm = float(smax(np.array([[p["omega"], p["vc"], p["halo_q"]]]))[0]); u[3] = (p["size"]-0.8)/(sm-0.8)
    return u


def score_spray(path, cat, bounds):
    from grid4_common import win, project_gc, GC
    BG_BOUNDS, V_BOUNDS = [tuple(map(tuple, bounds[0])), tuple(bounds[1])]
    from ocen_dm.streams.score import score_conditional
    m = dict(np.load(path)); mu, mx = project_gc(m["l"], m["b"], GC)
    return score_conditional(cat["du"], cat["dw"], cat["dv"], cat["dev"], cat["cov"], mu, np.column_stack((mx, m["pmra"], m["pmdec"])),
                             m["vlos"], m["chi"], m["age"], u_window=win, w_bounds=BG_BOUNDS, v_bounds=V_BOUNDS, chi_edges=np.arange(0., 205., 5.),
                             cutoff=7., jackknife_groups=10, chunk=50)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mock", required=True); ap.add_argument("--rescore", required=True)
    ap.add_argument("--tag", required=True); a = ap.parse_args()
    mock = Path(a.mock) if Path(a.mock).is_absolute() else ROOT/a.mock; rd = Path(a.rescore) if Path(a.rescore).is_absolute() else ROOT/a.rescore
    info = json.loads(mock.with_suffix(".json").read_text()); truth = info["params"]
    R = json.loads((rd/"rescore.json").read_text())
    rows = [r for r in R["rows"] if r["purpose"] != HELD]; held = [r for r in R["rows"] if r["purpose"] == HELD]
    X = np.array([r["unit"] for r in rows]); y = np.array([r["total_jk"] for r in rows]); v = np.array([r["var_jk"] for r in rows])
    key = np.array([r["key"] for r in rows])
    g = group_replicates(X, y, v, key)
    s2 = ED.noise_var(g); em = MeanEmulator(n_restarts=5).fit(g["X"], g["ybar"], s2/g["r"])
    mu_cv, sp_cv, z_cv = ED.cv(g)
    top = g["ybar"] > np.percentile(g["ybar"], 80)
    out = dict(tag=a.tag, truth=truth, n_points=len(g["ybar"]), n_runs=int(g["r"].sum()), kernel=str(em.gp.kernel_),
               cv_all=ED.summarise(g["ybar"], mu_cv, sp_cv, z_cv, np.ones(len(g["ybar"]), bool)),
               cv_top20=ED.summarise(g["ybar"], mu_cv, sp_cv, z_cv, top))
    if held:
        Xh = np.array([r["unit"] for r in held]); yh = np.array([r["total_jk"] for r in held]); vh = np.array([r["var_jk"] for r in held])
        m, s = em.predict(Xh); out["heldout"] = ED.summarise(yh, m, s, (yh-m)/np.sqrt(s**2+vh), np.ones(len(yh), bool))
    # truth
    smax = ER.smax_interp(); ut = to_unit(truth, smax)
    mt, st = em.predict(ut[None]); cat = dict(np.load(mock))
    bd = R["manifest"]["bounds"]
    chk = score_spray(mock.parent/"check_spray.npz", cat, bd); gen = score_spray(mock.parent/"truth_spray.npz", cat, bd)
    out["truth_unit"] = ut.tolist()
    out["truth_score"] = dict(check_spray_jk=chk["total_jk"], check_spray_sd_jk=float(np.sqrt(chk["var_jk"])), generating_spray_jk=gen["total_jk"],
                              emulator_mu=float(mt[0]), emulator_sd=float(st[0]),
                              z=float((chk["total_jk"]-mt[0])/np.hypot(st[0], np.sqrt(chk["var_jk"]))))
    out["best_measured"] = dict(total_jk=float(y.max()), params=rows[int(np.argmax(y))]["params"])
    # emulator maximum
    from scipy.optimize import minimize
    best = None
    for x0 in list(g["X"][np.argsort(-g["ybar"])[:20]])+[ut]:
        r = minimize(lambda x: -em.predict(x[None], return_std=False)[0], x0, method="L-BFGS-B", bounds=[(0, 1)]*9)
        best = r if best is None or r.fun < best.fun else best
    mb, sb = em.predict(best.x[None]); pb = ED.unit_to_params(best.x)
    out["emulator_max"] = dict(params=pb, mu=float(mb[0]), sd=float(sb[0]), unit=best.x.tolist(),
                               truth_minus_max=float(out["truth_score"]["check_spray_jk"]-mb[0]),
                               param_error={k: float(pb[k]-truth[k]) for k in ED.NAMES})
    # 1-D profiles with the truth
    from scipy.stats import qmc
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rng = np.random.default_rng(5); C = qmc.Sobol(9, scramble=True, seed=21).random_base2(15)
    C = np.vstack((C, np.clip(np.repeat(np.vstack((g["X"][np.argsort(-g["ybar"])[:10]], ut)), 300, 0)+rng.normal(0, 0.05, (3300, 9)), 0, 1)))
    mu, sd = em.predict(C); P = ER.physical(C, smax); ref = max(mu.max(), float(mb[0]))
    fig, ax = plt.subplots(3, 3, figsize=(11, 9), constrained_layout=True); out["profiles"] = {}
    for k, axx in zip(ED.NAMES, ax.ravel()):
        e = np.linspace(P[k].min(), P[k].max(), 21); c = 0.5*(e[1:]+e[:-1]); pm, ps = [], []
        j = ED.NAMES.index(k)
        for x0, x1 in zip(e[:-1], e[1:]):
            # optimised profile (2026-10-18): maximise mu over the other 8 coordinates, coordinate j fixed at the bin centre (in
            # unit space), from the 3 best candidates of the bin; size uses its unit coordinate
            s = (P[k] >= x0) & (P[k] <= x1); cand = np.flatnonzero(s)[np.argsort(-mu[s])[:3]]
            uc = np.median(C[s, j]); bestv, bests = -np.inf, 0.
            for i in cand:
                free = [q for q in range(9) if q != j]
                def negmu(z, i=i):
                    x = C[i].copy(); x[free] = z; x[j] = uc
                    return -em.predict(x[None], return_std=False)[0]
                r = minimize(negmu, C[i][free], method="L-BFGS-B", bounds=[(0, 1)]*8, options=dict(maxiter=60))
                if -r.fun > bestv:
                    x = C[i].copy(); x[free] = r.x; x[j] = uc; bestv = -r.fun; bests = em.predict(x[None])[1][0]
            pm.append(bestv-ref); ps.append(bests)
        pm, ps = np.array(pm), np.array(ps); out["profiles"][k] = dict(x=c.tolist(), dlnL=pm.tolist(), sd=ps.tolist())
        axx.fill_between(c, pm-2*ps, pm+2*ps, color="0.85"); axx.plot(c, pm, "k-"); axx.axvline(truth[k], color="C3", lw=1.2)
        axx.axvline(pb[k], color="C0", lw=0.8, ls="--"); axx.set(xlabel=k, ylim=(-4000, 600))
    fig.suptitle(f"mock {a.tag}: emulator profiles (max over the rest, rel. to the max; 2 sd band); red: truth; blue dashed: emulator max",
                 fontsize=9)
    fig.savefig(ROOT/f"plots/mock_recovery_{a.tag}.png", dpi=120); plt.close(fig)
    (rd/"recovery.json").write_text(json.dumps(out, indent=1))
    ts = out["truth_score"]; ex = out["emulator_max"]
    print(f"{a.tag}: CV all rmse {out['cv_all']['rmse']:.0f} z_sd {out['cv_all']['z_sd']:.2f} cover95 {out['cv_all']['cover95']:.2f}; "
          f"top20 rmse {out['cv_top20']['rmse']:.0f} bias {out['cv_top20']['bias']:+.0f}; held-out rmse {out.get('heldout', {}).get('rmse', np.nan):.0f}")
    print(f"  truth: check-spray jk {ts['check_spray_jk']:.0f} (generating {ts['generating_spray_jk']:.0f}); emulator at truth "
          f"{ts['emulator_mu']:.0f} +- {ts['emulator_sd']:.0f} (z {ts['z']:+.2f}); best training run {out['best_measured']['total_jk']:.0f}")
    print(f"  emulator max {ex['mu']:.0f} +- {ex['sd']:.0f}; truth - max {ex['truth_minus_max']:+.0f}; param errors " +
          ", ".join(f"{k} {v:+.2f}" for k, v in ex["param_error"].items()))


if __name__ == "__main__":
    main()
