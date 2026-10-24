#!/usr/bin/env python3
"""Upper sky branch vs main stream: proper motions and distance modulus in the same phi1 bins (2026-10-24).

Windows (y = dphi2 - sky-window centre of stream_masks.json): main |y - m(phi1)| < HW, branch |y - m2(phi1)| < HW, null
|y - (m(phi1) - 2.5)| < HW; m, m2 = the dphi2_sky_final spline tracks (stream_spline_fit.json). Bins phi1 12-16, 16-20, 20-22.
1. PMs: CMD-pass stars (gaia_cmd_mask.cmd_pass, k = 2) in a wide PM box |v_ra| < 4, |v_de| < 3 around the frozen main PM tracks
   (no k = 2 PM masks: they are centred on the main stream). Unbinned 2-D mixture (1 - f) B(v) + f N(v; mu, diag(s^2) + e_i^2);
   B = smoothed 2-D histogram of the same stars in the off-sky bands 6 <= |y| < 12. Errors from a finite-difference Hessian.
2. dm: all stars (no CMD mask) in each window with a PM selection around that window's own fitted PM centre (|dv_ra| < 1.5,
   |dv_de| < 1.0); Hess diagram vs per-pixel background (quadratic in y from 1-deg bands at 6 <= |y| < 14, averaged over the
   window); Poisson cluster-template profile likelihood in dm (stream_dm_fit.profile / domain), Delta lnL = 0.5 interval.
Outputs: results/plot_data/stream_branch_compare.json, plots/stream_branch_compare.png.
Usage: PYTHONPATH=src python bin/streams/stream_branch_compare.py
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from stream_spline_fit import basis
import stream_dm_fit as DMF

BINS = [(12., 16.), (16., 20.), (20., 22.)]
HW = 0.8; SFL = 0.05;   # intrinsic PM width floor [mas/yr] (keeps the Hessian finite when a width collapses)
WRA, WDE = 4.0, 3.0; OFF = (6., 12.); KEY = "dphi2_sky_final"


def tracks():
    R = json.loads((ROOT/"results/plot_data/stream_masks.json").read_text()); F = json.loads((ROOT/"results/plot_data/stream_spline_fit.json").read_text())[KEY]["fit"]
    G = np.array(R["grid"]); sc = lambda u: np.interp(u, G, R["sky_window"]["centre"])
    m = lambda u: basis(F["m"]["knots"], u)@np.array(F["m"]["values"]); m2 = lambda u: basis(F["m2"]["knots"], u)@np.array(F["m2"]["values"])
    pm = {q: (lambda u, q=q: np.interp(u, G, R["tracks"][q]["c"])) for q in ("pmra", "pmdec")}
    return R, sc, m, m2, pm


def hessian(f, p, h=1e-3):
    n = len(p); H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            pp = [np.array(p, float) for _ in range(4)]
            pp[0][i] += h; pp[0][j] += h; pp[1][i] += h; pp[1][j] -= h; pp[2][i] -= h; pp[2][j] += h; pp[3][i] -= h; pp[3][j] -= h
            H[i, j] = H[j, i] = (f(pp[0])-f(pp[1])-f(pp[2])+f(pp[3]))/(4*h*h)
    return H


def fit_pm2(v, e, vb):
    """2-D mixture on |v_ra| < WRA, |v_de| < WDE; v, e: (n, 2); vb: background-band stars (m, 2)."""
    ex, ey = np.linspace(-WRA, WRA, 33), np.linspace(-WDE, WDE, 25)
    hb = gaussian_filter(np.histogram2d(vb[:, 0], vb[:, 1], (ex, ey))[0]+0.5, 1.5); hb /= hb.sum()*(ex[1]-ex[0])*(ey[1]-ey[0])
    ix = np.clip(np.digitize(v[:, 0], ex)-1, 0, len(ex)-2); iy = np.clip(np.digitize(v[:, 1], ey)-1, 0, len(ey)-2); pB = hb[ix, iy]
    def nll(p):
        mx, my, lsx, lsy, lf = p; f = 1/(1+np.exp(-lf)); sx2, sy2 = np.exp(2*lsx)+SFL**2+e[:, 0]**2, np.exp(2*lsy)+SFL**2+e[:, 1]**2
        # truncation of the Gaussian at the box edges neglected (centres within 1.5 mas/yr of 0, widths < 0.6)
        g = np.exp(-0.5*((v[:, 0]-mx)**2/sx2+(v[:, 1]-my)**2/sy2))/(2*np.pi*np.sqrt(sx2*sy2))
        return -np.sum(np.log((1-f)*pB+f*g+1e-300))
    best = None
    for mx0 in (-1., 0., 1.):
        for my0 in (-0.5, 0., 0.5):
            r = minimize(nll, (mx0, my0, np.log(0.3), np.log(0.2), -2.), method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-6, maxiter=8000))
            if best is None or r.fun < best.fun:
                best = r
    p = best.x; H = hessian(nll, p); C = np.linalg.pinv(H); err = np.sqrt(np.clip(np.diag(C), 0, None))
    L0 = -np.sum(np.log(pB)); f = 1/(1+np.exp(-p[4]))
    sw = np.sqrt(np.exp(2*p[2:4])+SFL**2)
    return dict(mu_ra=p[0], mu_ra_err=err[0], mu_de=p[1], mu_de_err=err[1], s_ra=sw[0], s_de=sw[1], f=f, N=f*len(v), n=len(v), sig=float(np.sqrt(max(2*(L0-best.fun), 0))),
                rho=float(C[0, 1]/max(err[0]*err[1], 1e-12)))


def hess_window(S, sel, y, yc, a0, a1):
    """on-window Hess diagram and background for the window |y - yc| < HW (yc per star)."""
    EC, EG = DMF.EC, DMF.EG; u = S["phi1"]; k = sel & (u >= a0) & (u < a1); on = k & (np.abs(y-yc) < HW)
    Hn = np.histogram2d(S["c0"][on], S["g0"][on], (EC, EG))[0]
    Eb = np.arange(-14, 14.01, 1.); cb = 0.5*(Eb[1:]+Eb[:-1]); ob = np.abs(cb) >= 6; ks = k & (np.abs(y) < 14)
    H3 = np.histogramdd(np.column_stack((y[ks], S["c0"][ks], S["g0"][ks])), (Eb, EC, EG))[0]; ob &= H3.sum((1, 2)) > 0
    coef = np.tensordot(np.linalg.pinv(np.vander(cb[ob], 3)), H3[ob], 1)
    ycs = yc[on] if on.any() else np.array([0.]); yy = np.concatenate([np.linspace(c-HW, c+HW, 9) for c in ycs[::max(len(ycs)//50, 1)]])
    B = np.clip(np.tensordot(np.vander(yy, 3), coef, 1), 0, None).mean(0)*(2*HW)
    return Hn, gaussian_filter(B, 1.5)


def main():
    R, sc, m, m2, pm = tracks()
    # --- 1. PMs (CMD-pass catalogue)
    from gaia_cmd_mask import cmd_pass
    from gaia_footprint_plots import gc_masked
    raw = dict(np.load(ROOT/"results/streams/gaia_footprint_dr3.npz")); lg = np.where(raw["l"] > 180, raw["l"]-360, raw["l"]).astype(float)
    keep = cmd_pass(2.0) & ~gc_masked(lg, raw["b"].astype(float)); C = {k: v[keep].astype(float) for k, v in raw.items() if v.dtype.kind == "f"}
    u = C["phi1"]; y = C["dphi2"]-sc(u); vra, vde = C["pmra"]-pm["pmra"](u), C["pmdec"]-pm["pmdec"](u)
    box = (np.abs(vra) < WRA) & (np.abs(vde) < WDE)
    wins = dict(main=m, branch=m2, null=lambda x: m(x)-2.5); out = dict(bins=[], hw=HW, key=KEY)
    for a0, a1 in BINS:
        kb = box & (u >= a0) & (u < a1); bg = kb & (np.abs(y) >= OFF[0]) & (np.abs(y) < OFF[1]); row = dict(phi1=(a0, a1))
        for w, fn in wins.items():
            s = kb & (np.abs(y-fn(u)) < HW)
            row[w] = fit_pm2(np.column_stack((vra[s], vde[s])), np.column_stack((C["pmra_error"][s], C["pmdec_error"][s])), np.column_stack((vra[bg], vde[bg])))
        out["bins"].append(row)
    # --- 2. dm (no CMD mask; PM selection around each window's own PM centre)
    S, clus = DMF.load(); uS = S["phi1"]; yS = S["dphi2"]-sc(uS); vraS, vdeS = S["pmra"]-pm["pmra"](uS), S["pmdec"]-pm["pmdec"](uS)
    HessSum = {}
    for row in out["bins"]:
        a0, a1 = row["phi1"]; c = 0.5*(a0+a1); dm0 = DMF.dm_ref(c); dom = DMF.domain(dm0)["all"]
        for w, fn in wins.items():
            p = row["main" if w == "null" else w]
            sel = (np.abs(vraS-p["mu_ra"]) < 1.5) & (np.abs(vdeS-p["mu_de"]) < 1.0)
            Hn, B = hess_window(S, sel, yS, fn(uS), a0, a1); r = DMF.profile(Hn, B, clus, dm0, dom)
            row[w] |= dict(dm=r["dm"], dm_lo=r["lo"], dm_hi=r["hi"], dm_sig=r["sig"], dm_edge=r["edge"], dm_curve=(r["dms"], r["L"]), n_cmd=float(Hn.sum()), b_cmd=float(B.sum()))
            HessSum[w] = HessSum.get(w, 0)+(Hn-B)
    # --- report
    print(f"windows |y - track| < {HW} deg; PM offsets relative to the frozen main PM tracks [mas/yr]; dm Delta lnL = 0.5 interval")
    for row in out["bins"]:
        a0, a1 = row["phi1"]
        for w in wins:
            r = row[w]
            print(f"phi1 {a0:.0f}-{a1:.0f} {w:6s}: N {r['N']:5.0f}/{r['n']:5d} sig {r['sig']:4.1f} | mu_ra {r['mu_ra']:+.2f} +- {r['mu_ra_err']:.2f} (s {r['s_ra']:.2f}) "
                  f"mu_de {r['mu_de']:+.2f} +- {r['mu_de_err']:.2f} (s {r['s_de']:.2f}) | dm {r['dm']:+.2f} [{r['dm_lo']:+.2f}, {r['dm_hi']:+.2f}] sig {r['dm_sig']:.1f}{' EDGE' if r['dm_edge'] else ''}")
        b, mn = row["branch"], row["main"]; e = lambda q: np.hypot(b[q+"_err"], mn[q+"_err"])
        ddm, edm = b["dm"]-mn["dm"], np.hypot(0.5*(b["dm_hi"]-b["dm_lo"]), 0.5*(mn["dm_hi"]-mn["dm_lo"]))
        row["diff"] = dict(d_mu_ra=b["mu_ra"]-mn["mu_ra"], e_mu_ra=e("mu_ra"), d_mu_de=b["mu_de"]-mn["mu_de"], e_mu_de=e("mu_de"), d_dm=ddm, e_dm=edm)
        print(f"   branch - main: d mu_ra {row['diff']['d_mu_ra']:+.2f} +- {e('mu_ra'):.2f} ({row['diff']['d_mu_ra']/e('mu_ra'):+.1f} s), "
              f"d mu_de {row['diff']['d_mu_de']:+.2f} +- {e('mu_de'):.2f} ({row['diff']['d_mu_de']/e('mu_de'):+.1f} s), d dm {ddm:+.2f} +- {edm:.2f} ({ddm/max(edm, 1e-3):+.1f} s)")
    json.dump(out, open(ROOT/"results/plot_data/stream_branch_compare.json", "w"), indent=1, default=float)
    # --- figure: PM offsets, dm per bin; summed background-subtracted Hess diagrams
    fig, ax = plt.subplots(1, 5, figsize=(24, 4.8), constrained_layout=True); cs = dict(main="k", branch="C3", null="0.6")
    xc = np.array([0.5*sum(r["phi1"]) for r in out["bins"]])
    for i, (q, lab) in enumerate((("mu_ra", r"$\mu_{\alpha*}$ - main track [mas/yr]"), ("mu_de", r"$\mu_\delta$ - main track [mas/yr]"))):
        for j, w in enumerate(("main", "branch")):
            ax[i].errorbar(xc+0.2*j, [r[w][q] for r in out["bins"]], [r[w][q+"_err"] for r in out["bins"]], fmt="o", color=cs[w], label=w)
        ax[i].axhline(0, color="k", lw=0.5, ls=":"); ax[i].set(xlabel=r"$\phi_1$ [deg]", ylabel=lab); ax[i].legend()
    for j, w in enumerate(("main", "branch")):
        ax[2].errorbar(xc+0.2*j, [r[w]["dm"] for r in out["bins"]], [[r[w]["dm"]-r[w]["dm_lo"] for r in out["bins"]], [r[w]["dm_hi"]-r[w]["dm"] for r in out["bins"]]], fmt="o", color=cs[w], label=w)
    ax[2].set(xlabel=r"$\phi_1$ [deg]", ylabel="dm relative to omega Cen [mag]"); ax[2].legend()
    for a, w in zip(ax[3:], ("main", "branch")):
        a.imshow(gaussian_filter(HessSum[w], 1.).T, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-6, vmax=6, extent=(DMF.EC[0], DMF.EC[-1], DMF.EG[0], DMF.EG[-1]))
        a.invert_yaxis(); a.set(xlabel=r"$(BP-RP)_0$", ylabel=r"$G_0$", title=f"{w}: N - B, phi1 12-22 (own PM window)")
    fig.suptitle(f"Upper sky branch vs main stream (windows +-{HW} deg on the {KEY} tracks)")
    fig.savefig(ROOT/"plots/stream_branch_compare.png", dpi=90); print("plots/stream_branch_compare.png")


if __name__ == "__main__":
    main()
