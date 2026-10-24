#!/usr/bin/env python3
"""Data-model comparison of particle-spray models with the single-component knot vector (2026-10-24).

Data vector (no refits):
  sky  -- local fits (stream_sky_local.py --strip 0): track c and width s per 3-deg phi1 window, Hessian errors (mock-checked),
          window phi1 6-9 dropped (mock recovery failed);
  PMs  -- accepted spline fits (stream_spline_fit.json keys pmra, pmdec): track values AT THE SPLINE KNOTS with their full
          covariance (evaluating the spline at more points than knots gives a singular covariance).
Model, measured like the data:
  particles: chi > 0, age < age_cut_myr (file), projected into the chord frame; each particle gets the PM errors of a random data
  star of the same phi1 degree (for the masks only); sky: both frozen PM masks; PMs: frozen sky window + the other PM mask.
  sky  -- "Asimov" local fit: expected counts lam_true = fixed band background + model stream density (particle histogram,
          scaled to the data's stream count in the window) x area map; fitted with the SAME extended likelihood, fixed slope and
          width bounds as the data (stream_sky_local.nll), maximising the expected log-likelihood (no Poisson noise);
  PMs  -- truncated-Gaussian ML fit (|v| < W of the data fit) with a linear trend within +-1.5 deg of each knot, value at the knot.
  errors: sd over NBOOT particle bootstraps.
Score: chi2 = sum over sky c, s (independent windows) + PM tracks ((d - m)^T (C_d + C_m)^-1 (d - m), per PM); no free nuisance.
Usage: PYTHONPATH=src python bin/streams/stream_model_compare.py FILE.npz [FILE.npz ...] [--tag NAME]
       -> results/plot_data/stream_model_compare{_tag}.json, plots/stream_model_compare{_tag}.png (+ _maps.png)
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
FILES = [a for a in sys.argv[1:] if a.endswith(".npz")]
TAG = ("_"+sys.argv[sys.argv.index("--tag")+1]) if "--tag" in sys.argv else ""
sys.argv = [sys.argv[0], "--strip", "0"]                                    # stream_sky_local: no branch cut (its argv set-up)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.ndimage import gaussian_filter
import stream_sky_local as SL
from stream_spline_fit import basis
from ocen_dm.streams.path import project_gc

NBOOT, DROP = 8, ((6., 9.),)
J = lambda n: json.loads((ROOT/f"results/plot_data/{n}.json").read_text())


def data_vector():
    L = J("stream_sky_local_nostrip"); F = J("stream_spline_fit"); win = [w for w in L["windows"] if tuple(w["phi1"]) not in DROP]
    pk = np.array([0.5*sum(w["phi1"]) for w in win]); out = dict(pk=pk, win=win, sky_c=np.array([w["c"] for w in win]), sky_ce=np.array([w["c_err"] for w in win]),
                                                                   sky_s=np.array([w["s"] for w in win]), sky_se=np.array([w["s_err"] for w in win]))
    for q in ("pmra", "pmdec"):
        f = F[q]; n = len(f["fit"]["m"]["knots"]); out[q+"_k"] = np.array(f["fit"]["m"]["knots"])
        out[q] = np.array(f["fit"]["m"]["values"]); out[q+"_C"] = np.array(f["cov"])[:n, :n]; out[q+"_W"] = f["window"]
    return out


class Obs:
    """data-side ingredients for observing a model: frames, frozen masks, background, area map, PM errors."""
    def __init__(s):
        s.GC = J("stream54_gc_frame_chord"); s.R = J("stream_masks"); s.G = np.array(s.R["grid"]); s.d = SL.SF.load("dphi2")
        from stream_spline_plot import catalogue
        S = catalogue(); u = S["phi1"]; win = np.abs(S["dphi2"]-np.interp(u, s.G, s.R["sky_window"]["centre"])) < np.interp(u, s.G, s.R["sky_window"]["half_width"])
        s.err_u, s.err = u[win], np.column_stack((S["pmra_error"][win], S["pmdec_error"][win]))
        s.L = J("stream_sky_local_nostrip"); s.m1, s.m2 = SL.SF.sky_tracks()

    def ev(s, p, u, k="c"):
        return np.interp(u, s.G, s.R["tracks"][p][k])

    def particles(s, f, rng):
        z = np.load(f); ac = float(z["age_cut_myr"]) if "age_cut_myr" in z.files else 700.; k = (z["chi"] > 0) & (z["age"] < ac)
        u, x = project_gc(z["l"][k], z["b"][k], s.GC); P = dict(u=u, x=x, pmra=z["pmra"][k], pmdec=z["pmdec"][k])
        j = np.searchsorted(np.sort(s.err_u), u); o = np.argsort(s.err_u)                 # errors of a data star at a similar phi1
        j = np.clip(j+rng.integers(-50, 50, u.size), 0, len(o)-1); e = s.err[o[j]]
        for i, p in enumerate(("pmra", "pmdec")):
            obs = P[p]+e[:, i]*rng.standard_normal(u.size)
            P["m_"+p] = np.abs(obs-s.ev(p, u)) < 2*np.sqrt(s.ev(p, u, "s")**2+e[:, i]**2)
        P["v"] = x-np.interp(u, s.G, s.R["sky_window"]["centre"]); P["inwin"] = np.abs(P["v"]) < np.interp(u, s.G, s.R["sky_window"]["half_width"])
        return P

    def sky(s, P, idx):
        """Asimov local fits of the model's sky density, per data window."""
        d = s.d; out = []; k = P["m_pmra"][idx] & P["m_pmdec"][idx]; u, v = P["u"][idx][k], P["v"][idx][k]
        for wd in s.L["windows"]:
            a0, a1 = wd["phi1"]; w = SL.window(d, a0, a1, s.m2, s.m1); ev_ = np.r_[w["vg"]-0.5*(w["vg"][1]-w["vg"][0]), w["vg"][-1]+0.5*(w["vg"][1]-w["vg"][0])]
            eu = np.r_[w["ug"]-0.05, w["ug"][-1]+0.05]; H = gaussian_filter(np.histogram2d(u, v, (eu, ev_))[0], (1., 2.))
            jg = np.clip(np.digitize(w["ug"], w["Eu"])-1, 0, len(w["Eu"])-2); lb = np.clip(w["cf"][jg]@np.vander(w["vg"], 3).T, 0, None)/w["cell"]*w["Ag"]
            dA = w["du"]*(w["vg"][1]-w["vg"][0]); Sd = H*w["Ag"]; ntot = Sd.sum()
            if ntot < 20:
                out.append(dict(c=np.nan, s=np.nan, n=int(H.sum()))); continue
            lam_true = lb+Sd/ntot*wd["r"]*(a1-a0)/dA                                        # scaled to the data's stream count
            def enll(p):
                c, s_, r = SL.unpack(p); mg = c+w["g"]*(w["ug"]-w["pk"])
                lf = lb+r*np.exp(-0.5*((w["vg"][None, :]-mg[:, None])/s_)**2)/(s_*np.sqrt(2*np.pi))*w["Ag"]
                return -np.sum(lam_true*np.log(np.maximum(lf, 1e-30))-lf)*dA
            best = None
            for xs0 in (-1.5, 0., 1.5):
                r_ = minimize(enll, [wd["c"], xs0, np.log(wd["r"])], method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-7, maxiter=6000))
                best = r_ if best is None or r_.fun < best.fun else best
            c, s_, r = SL.unpack(best.x); out.append(dict(c=c, s=s_, n=int(H.sum())))
        return out

    def pm(s, P, idx, D, W):
        out = {}
        for q, oth in (("pmra", "pmdec"), ("pmdec", "pmra")):
            vq = P[q][idx]-s.ev(q, P["u"][idx]); k = P["inwin"][idx] & P["m_"+oth][idx] & (np.abs(vq) < W[q]); u, vq = P["u"][idx][k], vq[k]; vals = []
            for c0 in D[q+"_k"]:
                kk = np.abs(u-c0) < 1.5; x, y = u[kk]-c0, vq[kk]
                if kk.sum() < 20:
                    vals.append(np.nan); continue
                def nl(p):
                    m = p[0]+p[1]*x; sg = np.exp(p[2]); Z = ndtr((W[q]-m)/sg)-ndtr((-W[q]-m)/sg)
                    return -np.sum(-0.5*((y-m)/sg)**2-np.log(sg)-np.log(np.maximum(Z, 1e-12)))
                vals.append(minimize(nl, [np.median(y), 0., np.log(max(np.std(y), 0.1))], method="Nelder-Mead").x[0])
            out[q] = np.array(vals)
        return out


def measure(O, D, f, rng):
    P = O.particles(f, rng); n = P["u"].size; W = {q: D[q+"_W"] for q in ("pmra", "pmdec")}; res = []
    for b in range(NBOOT+1):
        idx = np.arange(n) if b == 0 else rng.integers(0, n, n)
        sk = O.sky(P, idx); keep = [i for i, w in enumerate(O.L["windows"]) if tuple(w["phi1"]) not in DROP]
        pm = O.pm(P, idx, D, W); res.append(np.r_[[sk[i]["c"] for i in keep], [sk[i]["s"] for i in keep], pm["pmra"], pm["pmdec"]])
        if b == 0:
            nwin = [sk[i]["n"] for i in keep]
    R = np.array(res); m0, sd = R[0], np.nanstd(R[1:], 0); k = len(D["pk"]); a = 2*k+len(D["pmra_k"])
    return dict(file=str(Path(f).relative_to(ROOT)) if str(f).startswith(str(ROOT)) else f, n_particles=int(n), n_win=nwin,
                sky_c=m0[:k], sky_s=m0[k:2*k], pmra=m0[2*k:a], pmdec=m0[a:], sky_ce=sd[:k], sky_se=sd[k:2*k], pmra_e=sd[2*k:a], pmdec_e=sd[a:], P=P)


def chi2(D, M):
    out = {}
    for q, e in (("sky_c", "sky_ce"), ("sky_s", "sky_se")):
        r = (D[q]-M[q])**2/(D[e]**2+M[e]**2); out[q] = float(np.nansum(r)) + (1e3*np.isnan(r).sum())    # missing window: 1e3 penalty
    for q in ("pmra", "pmdec"):
        r = D[q]-M[q]; ok = np.isfinite(r); C = D[q+"_C"][np.ix_(ok, ok)]+np.diag(M[q+"_e"][ok]**2)
        out[q] = float(r[ok]@np.linalg.solve(C, r[ok]))+1e3*(~ok).sum()
    out["total"] = sum(out.values()); return out


def main():
    rng = np.random.default_rng(3); D = data_vector(); O = Obs(); Ms = []
    for f in FILES:
        M = measure(O, D, f, rng); M["chi2"] = chi2(D, M); Ms.append(M); name = Path(f).stem
        print(f"{name}: N {M['n_particles']}; chi2 sky track {M['chi2']['sky_c']:.1f}, sky width {M['chi2']['sky_s']:.1f}, pmra {M['chi2']['pmra']:.1f}, "
              f"pmdec {M['chi2']['pmdec']:.1f}; total {M['chi2']['total']:.1f} (n data {2*len(D['pk'])+len(D['pmra_k'])+len(D['pmdec_k'])})", flush=True)
    json.dump(dict(pk=D["pk"].tolist(), models=[{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in M.items() if k != "P"} for M in Ms]),
              open(ROOT/f"results/plot_data/stream_model_compare{TAG}.json", "w"), indent=1)
    plot(D, O, Ms)


def plot(D, O, Ms):
    pk = D["pk"]; fig, ax = plt.subplots(2, 2, figsize=(15, 8.5), constrained_layout=True); ax = ax.ravel(); g = np.linspace(-2, 28, 301)
    rows = (("sky_c", "sky_ce", r"sky track v [deg]"), ("sky_s", "sky_se", "sky width [deg]"), ("pmra", None, r"$\mu_{\alpha*}$ - frozen track [mas/yr]"), ("pmdec", None, r"$\mu_\delta$ - frozen track [mas/yr]"))
    for a, (q, e, lab) in zip(ax, rows):
        x = D[q+"_k"] if q.startswith("pm") else pk; de = D[e] if e else np.sqrt(np.diag(D[q+"_C"])); a.errorbar(x, D[q], de, fmt="ko", capsize=3, ms=5, label="data", zorder=5)
        for i, M in enumerate(Ms):
            me = M[e] if e else M[q+"_e"]; a.errorbar(x+0.15*(i+1), M[q], me, fmt="s-", ms=3, lw=1, color=f"C{i}", capsize=2,
                                                       label=f"{Path(M['file']).stem}: chi2 {M['chi2'][q.replace('pmra', 'pmra')]:.0f}")
        a.set(xlabel=r"$\phi_1$ (sky: window centre; PM: spline knot) [deg]", ylabel=lab); a.legend(fontsize=7); a.axhline(0, color="k", lw=0.3)
    fig.suptitle("Data (local sky fits, PM splines) vs models measured the same way; chi2 per observable; sky 7 windows (phi1 6-9 dropped), PMs at the spline knots")
    fig.savefig(ROOT/f"plots/stream_model_compare{TAG}.png", dpi=85); print(f"plots/stream_model_compare{TAG}.png")
    # sky maps: data - background | each model's observed density (scaled to the data's stream count per window) | residual
    d = O.d; E1, Ey = np.arange(0, 24.01, 0.5), np.arange(-6, 6.01, 0.25); H = np.histogram2d(d["u"], d["v"], (E1, Ey))[0]
    jr = np.clip(np.digitize(0.5*(E1[1:]+E1[:-1]), d["area"]["Eu"])-1, 0, len(d["area"]["Eu"])-2); cy = 0.5*(Ey[1:]+Ey[:-1])
    cell = (d["area"]["Eu"][1]-d["area"]["Eu"][0])*(d["area"]["Ey"][1]-d["area"]["Ey"][0]); jy = np.clip(np.digitize(cy, d["area"]["Ey"])-1, 0, len(d["area"]["Ey"])-2)
    A = d["area"]["A"][jr][:, jy]; B = np.clip(d["bg_coef"][jr]@np.vander(cy, 3).T, 0, None)/cell*A*0.5*0.25; sb = np.sqrt(np.maximum(B, 1)); X = H-B
    fig, ax = plt.subplots(len(Ms), 3, figsize=(19, 3.6*len(Ms)), sharex=True, sharey=True, constrained_layout=True, squeeze=False)
    for i, M in enumerate(Ms):
        P = M["P"]; k = P["m_pmra"] & P["m_pmdec"]; Hm = np.histogram2d(P["u"][k], P["v"][k], (E1, Ey))[0]*A; Sm = np.zeros_like(Hm)
        for wd in O.L["windows"]:
            jj = (0.5*(E1[1:]+E1[:-1]) >= wd["phi1"][0]) & (0.5*(E1[1:]+E1[:-1]) < wd["phi1"][1]); tot = Hm[jj].sum()
            Sm[jj] = Hm[jj]/tot*wd["r"]*(wd["phi1"][1]-wd["phi1"][0]) if tot > 0 else 0.
        for a, im, t in zip(ax[i], (X/sb, Sm/sb, (X-Sm)/sb), ("data - background", f"model {Path(M['file']).stem} (scaled per window)", "residual")):
            a.imshow(gaussian_filter(im, 1.).T, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3, extent=(0, 24, -6, 6))
            a.plot(pk, D["sky_c"], "ko", ms=3); a.plot(pk, M["sky_c"], "s", color="C1", ms=3); a.set(title=t+f"  (total chi2 {M['chi2']['total']:.0f})" if t == "residual" else t)
        ax[i, 0].set_ylabel("v [deg]")
    for a in ax[-1]:
        a.set_xlabel(r"$\phi_1$ [deg]")
    fig.savefig(ROOT/f"plots/stream_model_compare{TAG}_maps.png", dpi=80); print(f"plots/stream_model_compare{TAG}_maps.png")


if __name__ == "__main__":
    main()
