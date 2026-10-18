#!/usr/bin/env python3
"""What the emulator says about the 9 parameters (2026-10-18). Profiles, not posteriors (no tempering chosen yet).

1. Profiles: for each parameter, the maximum over the other eight of the emulator mean mu, in bins of that parameter
   (2^16 scrambled-Sobol candidates; size in kpc-free units = bar size factor), with the emulator sd at the maximising
   candidate. 2-D profiles for (angle, Omega), (Omega, distance), (amplitude, size).
2. Emulator maximum (L-BFGS-B on mu from the 20 best measured points) and check of the predicted best point: measured jackknife totals of its seeds vs the emulator prediction (fitted without them).
3. Tracks of the best measured run vs the members: Delta phi2 median and the PM residuals (model PM = median of particles
   within 0.5 deg of the member's (phi1, Delta phi2), as paper fig3_residuals) in 2-deg phi1 bins; refined 16 deg paper
   model (32k, seed 4) for comparison.
Output: plots/emulator_profiles_<tag>.png, plots/emulator_tracks_<tag>.png, results/plot_data/emulator_report_<tag>.json
Usage: OMP_NUM_THREADS=2 PYTHONPATH=src python bin/streams/emulator_report.py --tag round1 [--batch DIR ...]
"""
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
warnings.filterwarnings("ignore")
import emulator_design as ED   # noqa: E402

LAB = dict(angle="bar angle [deg]", omega=r"$\Omega_b$ [km s$^{-1}$ kpc$^{-1}$]", amp="bar amplitude", size="bar size",
           dist=r"$\omega$ Cen distance [kpc]", pmra=r"$\mu_{\alpha*}$ [mas yr$^{-1}$]", pmdec=r"$\mu_\delta$ [mas yr$^{-1}$]",
           vc=r"$v_c(R_0)$ [km s$^{-1}$]", halo_q="halo flattening q")


def smax_interp():
    """S_max(omega, vc, q) = min(1.5, R_CR/4.25) on a grid (corotation of the point's own axisymmetric host)."""
    from scipy.interpolate import RegularGridInterpolator
    from training_set import r_cr
    om, vc, q = np.linspace(30, 40, 11), np.linspace(220, 240, 5), np.linspace(0.8, 1.2, 5)
    S = np.array([[[min(1.5, r_cr(a, b, c)/4.25) for c in q] for b in vc] for a in om])
    return RegularGridInterpolator((om, vc, q), S)


def physical(U, smax):
    P = dict(angle=10+25*U[:, 0], omega=30+10*U[:, 1], amp=0.6+1.4*U[:, 2], dist=5.25+0.7*U[:, 4],
             pmra=-3.250+0.022*(8*U[:, 5]-4), pmdec=-6.746+0.022*(8*U[:, 6]-4), vc=220+20*U[:, 7], halo_q=0.8+0.4*U[:, 8])
    P["size"] = 0.8+U[:, 3]*(smax(np.column_stack((P["omega"], P["vc"], P["halo_q"])))-0.8)
    return P


def turn_tracks(m):
    from scipy.spatial import cKDTree
    from grid4_common import d, du, dx, project_gc, GC
    mu, mx = project_gc(m["l"], m["b"], GC); k = (m["chi"] > 0) & (m["age"] < 700.)
    tree = cKDTree(np.column_stack((mu[k], mx[k]))); nb = tree.query_ball_point(np.column_stack((du, dx)), 0.5)
    pred = {q: np.array([np.median(m[q][k][i]) if len(i) >= 5 else np.nan for i in nb]) for q in ("pmra", "pmdec")}
    edges = np.arange(-2., 28.1, 2.); c = 0.5*(edges[1:]+edges[:-1]); out = dict(phi1=c.tolist())
    for q in ("pmra", "pmdec"):
        r = d[q]-pred[q]
        out[f"d_{q}"] = [float(np.nanmedian(r[(du >= a) & (du < b)])) if np.isfinite(r[(du >= a) & (du < b)]).sum() >= 15 else None
                         for a, b in zip(edges[:-1], edges[1:])]
    km = np.abs(mx[k]) < 4
    out["dphi2_model"] = [float(np.median(mx[k][km][(mu[k][km] >= a) & (mu[k][km] < b)])) if ((mu[k][km] >= a) & (mu[k][km] < b)).sum() >= 20 else None
                          for a, b in zip(edges[:-1], edges[1:])]
    out["dphi2_data"] = [float(np.median(dx[(du >= a) & (du < b)])) if ((du >= a) & (du < b)).sum() >= 15 else None for a, b in zip(edges[:-1], edges[1:])]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--batch", nargs="*", default=[])
    a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from scipy.stats import qmc
    g, s2, nm, em, info = ED.fit(a.batch)
    smax = smax_interp()
    U = qmc.Sobol(9, scramble=True, seed=7).random_base2(16); mu, sd = em.predict(U); P = physical(U, smax)
    rep = dict(tag=a.tag, info=info, kernel=str(em.gp.kernel_), n_points=len(g["ybar"]), n_runs=int(g["r"].sum()),
               best_observed=float(g["ybar"].max()), profiles={}, profiles2d={})
    fig, ax = plt.subplots(3, 3, figsize=(11, 9), constrained_layout=True)
    for n, (k, axx) in enumerate(zip(ED.NAMES, ax.ravel())):
        lo, hi = P[k].min(), P[k].max(); e = np.linspace(lo, hi, 21); c = 0.5*(e[1:]+e[:-1]); pm, ps = [], []
        for x0, x1 in zip(e[:-1], e[1:]):
            s = (P[k] >= x0) & (P[k] <= x1); i = np.flatnonzero(s)[np.argmax(mu[s])]; pm.append(mu[i]); ps.append(sd[i])
        pm, ps = np.array(pm), np.array(ps)
        rep["profiles"][k] = dict(x=c.tolist(), mu_max=pm.tolist(), sd_at_max=ps.tolist())
        axx.fill_between(c, pm-2*ps, pm+2*ps, color="0.85"); axx.plot(c, pm, "k-")
        axx.axhline(g["ybar"].max(), color="C3", lw=0.7, ls="--")
        axx.set(xlabel=LAB[k], ylabel=r"max $\ln L$ (emulator)" if n % 3 == 0 else None)
    fig.suptitle(f"Emulator profiles ({a.tag}; {len(g['ybar'])} points, {int(g['r'].sum())} runs; band = 2 sd at the maximum; "
                 f"red: best measured)", fontsize=9)
    fig.savefig(ROOT/f"plots/emulator_profiles_{a.tag}.png", dpi=130); plt.close(fig)
    for kx, ky in (("angle", "omega"), ("omega", "dist"), ("amp", "size")):
        ex, ey = np.linspace(P[kx].min(), P[kx].max(), 13), np.linspace(P[ky].min(), P[ky].max(), 13)
        ix, iy = np.clip(np.digitize(P[kx], ex)-1, 0, 11), np.clip(np.digitize(P[ky], ey)-1, 0, 11)
        M = np.full((12, 12), np.nan)
        for i in range(12):
            for j in range(12):
                s = (ix == i) & (iy == j)
                if s.any():
                    M[j, i] = mu[s].max()
        rep["profiles2d"][f"{kx}_{ky}"] = dict(x_edges=ex.tolist(), y_edges=ey.tolist(), mu_max=np.where(np.isfinite(M), M, None).tolist())
    # emulator maximum: L-BFGS-B on mu from the 20 best measured points
    from scipy.optimize import minimize
    best = None
    for x0 in g["X"][np.argsort(-g["ybar"])[:20]]:
        r = minimize(lambda x: -em.predict(x[None], return_std=False)[0], x0, method="L-BFGS-B", bounds=[(0, 1)]*9)
        if best is None or r.fun < best.fun:
            best = r
    mb, sb = em.predict(best.x[None]); pb = ED.unit_to_params(best.x)
    rep["emulator_max"] = dict(unit=best.x.tolist(), params=pb, mu=float(mb[0]), sd=float(sb[0]))
    # predicted-best check
    od = [Path(x) if Path(x).is_absolute() else ROOT/x for x in a.batch]
    chk = []
    for b in od:
        for f in (b/"runs").glob("*.json"):
            if f.name.endswith(".error.json"):
                continue
            j = json.loads(f.read_text())
            if j.get("purpose") == "predicted best, check":
                chk.append(j)
    if chk:
        m_, s_ = em.predict(np.array([chk[0]["unit"]]))
        y = np.array([j["total_jk"] for j in chk])
        rep["best_check"] = dict(params=chk[0]["params"], predicted_at_design=chk[0]["mu"], predicted_now=float(m_[0]),
                                 sd_now=float(s_[0]), measured=y.tolist(), measured_mean=float(y.mean()),
                                 measured_sd=float(y.std(ddof=1)) if len(y) > 1 else None)
    # tracks of the best measured run
    rows = []
    for b in [ED.TRAIN]+od+[ED.REPS]:
        for f in (b/"runs").glob("*.json"):
            if f.name.endswith(".error.json") or f.name.endswith("_jk.json"):
                continue
            j = json.loads(f.read_text())
            if j.get("nrel") == 256000 and "spray_sha256" in j:
                jk = f.with_name(f.stem+"_jk.json")
                tot = j.get("total_jk", json.loads(jk.read_text())["total_jk"] if jk.exists() else None)
                if tot is not None:
                    rows.append((tot, f, j["params"]))
    tot, f, par = max(rows, key=lambda r: r[0])
    m = dict(np.load(f.with_suffix(".npz"))); tb = turn_tracks(m)
    ref = turn_tracks(dict(np.load(ROOT/"results/streams/paper/bar16_seed4.npz")))
    rep["best_run"] = dict(file=str(f.relative_to(ROOT)), total_jk=float(tot), params=par, tracks=tb, paper_bar16_tracks=ref)
    fig, ax = plt.subplots(3, 1, figsize=(6, 7.5), sharex=True, constrained_layout=True); c = np.array(tb["phi1"])
    f2 = lambda v: np.array([np.nan if x is None else x for x in v])
    ax[0].plot(c, f2(tb["dphi2_data"]), "ko", ms=3, label="members"); ax[0].plot(c, f2(tb["dphi2_model"]), "C0-", label="best run")
    ax[0].plot(c, f2(ref["dphi2_model"]), "C1--", label="paper 16°")
    for i, q in enumerate(("pmra", "pmdec")):
        ax[i+1].plot(c, f2(tb[f"d_{q}"]), "C0o-", ms=3); ax[i+1].plot(c, f2(ref[f"d_{q}"]), "C1s--", ms=3); ax[i+1].axhline(0, color="k", lw=0.5)
    ax[0].set_ylabel(r"$\Delta\phi_2$ median [deg]"); ax[1].set_ylabel(r"data $-$ model $\mu_{\alpha*}$"); ax[2].set_ylabel(r"data $-$ model $\mu_\delta$")
    ax[2].set_xlabel(r"$\phi_1$ [deg]"); ax[0].legend(fontsize=7)
    fig.suptitle(f"best measured run (jackknife ln L {tot:.0f})", fontsize=9)
    fig.savefig(ROOT/f"plots/emulator_tracks_{a.tag}.png", dpi=130); plt.close(fig)
    (ROOT/f"results/plot_data/emulator_report_{a.tag}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(dict(best_observed=rep["best_observed"], best_check=rep.get("best_check"), best_run_params=par,
                          best_run_lnL=tot, emulator_max=rep["emulator_max"]), indent=1))


if __name__ == "__main__":
    main()
