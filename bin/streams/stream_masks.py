#!/usr/bin/env python3
"""Data-driven mask borders (centre and intrinsic width vs phi1) for the omega Cen stream in Gaia DR3 (2026-10-23).

Procedure (docs/reviews/masks_fable_codex_20261023.md): stars = footprint (G < 20, parallax cut, GC masks) inside the isochrone CMD
mask (bin/streams/gaia_cmd_mask.cmd_pass, k = 2). Alternating 1-D unbinned fits per phi1 bin, for q in (pmdec, pmra*, dphi2):
  - select by the OTHER two masks at k_iter = 2.5 (per-star width sqrt(s_int^2 + e_i^2));
  - PMs: residual v = q - track_q(phi1) of stars in the sky window; background shape = histogram of the residuals of the same
    selection in sky bands 4-12 deg off the track (upper-branch window excluded); model (1 - f) B(v) + f N(v; mu + beta (phi1 - c),
    s^2 + e_i^2) on |v| < 4 mas/yr;
  - sky: residual v = dphi2 - track(phi1) of stars passing both PM masks, |v| < 8 deg; model f1 N(mu1, s1) [+ f2 N(mu2, s2) for the
    upper branch at phi1 14-21] + (1 - f1 - f2) quadratic background (normalised on the window); phi1 >= 4 only (footprint edge);
  - widths floored (PM 0.15 mas/yr, sky 0.3 deg; a fit collapsing onto the floor is a noise spike and is not used); significance from the likelihood ratio against f = 0.
Guard rails (a first version ran away near the cluster): bins at phi1 >= 2 only; accepted at >= 4 sigma, width between 1.05 x floor
and the cap (1.5 mas/yr, 1.5 deg) and offset < 1.5 x the current width; damped updates (half the measured offset / log width).
Smoothing (accepted bins): centre offsets by a weighted least-squares cubic spline
(knots every 6 deg), tapered to 0 over 3 deg outside the significant range (previous track kept there); log width by a weighted
quadratic in phi1 over the significant range (previous width kept outside). A GCV smoothing spline under-smoothed (oscillating widths). 4 iterations or until
all centres move by < 0.1 of their errors.
Initial tracks: the user-marked sequences (gaia_xsel_marked.MARK); initial widths 0.5 mas/yr, 1 deg.
Outputs: results/plot_data/stream_masks.json (tracks and widths on a phi1 grid, per-bin fit tables, iteration history),
plots/stream_masks_fits.png (per-bin centres/widths with errors per iteration), plots/stream_masks_xsel.png (cross-selection with the
final masks at k = 2, background-subtracted, with STREAMFINDER members in a separate column).
Usage: PYTHONPATH=src python bin/streams/stream_masks.py
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import make_smoothing_spline
from scipy.ndimage import gaussian_filter

GRID = np.arange(-6, 32.01, 0.5)
BINS = [(c-1, c+1) for c in range(3, 30, 2)]          # phi1 >= 2 (footprint edge and the omega Cen mask below)
K_ITER, K_FINAL, NIT = 2.5, 2.0, 6
CAP = dict(pmra=1.5, pmdec=1.5, dphi2=1.5)              # width caps
DAMP, SIG_MIN = 0.5, 4.0                                # damped track updates; bins accepted at >= 4 sigma
FLOOR = dict(pmra=0.15, pmdec=0.15, dphi2=0.3)   # zero-width spikes on noise (null test) -> floors; floor-collapsed bins not used


def load_data():
    import gaia_xsel_marked as GX
    from gaia_cross_selection import KNOTS, load
    from gaia_cmd_mask import cmd_pass
    from gaia_footprint_plots import gc_masked
    raw = np.load(ROOT/"results/streams/gaia_footprint_dr3.npz"); lg = np.where(raw["l"] > 180, raw["l"]-360, raw["l"])
    cp = cmd_pass(2.0)[~gc_masked(lg.astype(float), raw["b"].astype(float))]
    D = load(); D = {k: v[cp] for k, v in D.items()}
    K = np.array(KNOTS); ta = GX.track(GX.MARK["pmra"], lambda q: np.interp(q, K[:, 0], K[:, 1])); td = GX.track(GX.MARK["pmdec"], lambda q: np.interp(q, K[:, 0], K[:, 2]))
    ml = np.array(GX.MARK["lower"]); tl = GX.track(GX.MARK["lower"], lambda q: np.where(q < ml[0, 0], 0., ml[-1, 1])+0.*q)
    up = np.array(GX.MARK["upper"])
    T = dict(pmra=dict(c=ta(GRID), s=np.full(GRID.size, 0.5)), pmdec=dict(c=td(GRID), s=np.full(GRID.size, 0.5)),
             dphi2=dict(c=tl(GRID), s=np.full(GRID.size, 1.0)))
    return D, T, up


def ev(T, q, u, key="c"):
    return np.interp(u, GRID, T[q][key])


def other_masks(D, T, q, k):
    u = D["phi1"]; m = np.ones(len(u), bool)
    for p, col, err in (("pmra", "pmra", "pmra_error"), ("pmdec", "pmdec", "pmdec_error"), ("dphi2", "dphi2", None)):
        if p == q:
            continue
        e2 = D[err]**2 if err else 0.
        m &= np.abs(D[col]-ev(T, p, u)) < k*np.sqrt(ev(T, p, u, "s")**2+e2)
    return m


def fit_pm(v, e, du, vb, W=4.0):
    """Unbinned: (1 - f) B + f N(mu + beta du, s^2 + e^2) on |v| < W; B = smoothed histogram of the background residuals vb."""
    k, kb = np.abs(v) < W, np.abs(vb) < W; v, e, du, vb = v[k], e[k], du[k], vb[kb]
    if len(v) < 30 or len(vb) < 30:
        return None
    hb = np.histogram(vb, np.linspace(-W, W, 41))[0].astype(float)+0.5; hb = gaussian_filter(hb, 1.); xc = np.linspace(-W, W, 41); xc = 0.5*(xc[1:]+xc[:-1])
    pB = np.interp(v, xc, hb/(hb.sum()*(xc[1]-xc[0])))
    def nll(p):
        mu, be, ls, lf = p; s2 = np.exp(2*ls)+e**2; f = 1/(1+np.exp(-lf))
        return -np.sum(np.log((1-f)*pB+f*np.exp(-0.5*(v-mu-be*du)**2/s2)/np.sqrt(2*np.pi*s2)+1e-300))
    best = None
    for mu0 in (0., -0.5, 0.5):
        r = minimize(nll, (mu0, 0., np.log(0.4), -2.), method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-5, maxiter=5000))
        if best is None or r.fun < best.fun:
            best = r
    p = best.x.copy(); p[2] = max(p[2], np.log(FLOOR["pmdec"])); best = type("R", (), dict(fun=nll(p)))
    L0 = -np.sum(np.log(pB+1e-300)); dL = L0-best.fun
    # errors: numerical Hessian
    f = 1/(1+np.exp(-p[3]))
    return dict(mu=p[0], mu_err=err1d(nll, p, 0), beta=p[1], s=np.exp(p[2]), s_err=np.exp(p[2])*err1d(nll, p, 2), f=f, N=f*len(v),
                sig=np.sqrt(max(2*dL, 0)), n=len(v))


def fit_sky(v, du, two, mu2_0, W=8.0):
    k = np.abs(v) < W; v, du = v[k], du[k]
    if len(v) < 50:
        return None
    def bg(p):                                                    # quadratic background, normalised on [-W, W]
        a1, a2 = p; norm = 2*W+a2*2*W**3/3
        return np.clip(1+a1*v+a2*v**2, 1e-6, None)/norm
    def comp(mu, ls):
        s = np.exp(ls); return np.exp(-0.5*((v-mu)/s)**2)/(np.sqrt(2*np.pi)*s)
    def nll(p):
        if two:
            mu1, ls1, lf1, mu2, ls2, lf2, be, a1, a2 = p; f1, f2 = np.exp(lf1)/(1+np.exp(lf1)+np.exp(lf2)), np.exp(lf2)/(1+np.exp(lf1)+np.exp(lf2))
            return -np.sum(np.log(f1*np.exp(-0.5*((v-mu1-be*du)/np.exp(ls1))**2)/(np.sqrt(2*np.pi)*np.exp(ls1))+f2*comp(mu2, ls2)+(1-f1-f2)*bg((a1, a2))+1e-300))
        mu1, ls1, lf1, be, a1, a2 = p; f1 = 1/(1+np.exp(-lf1))
        return -np.sum(np.log(f1*np.exp(-0.5*((v-mu1-be*du)/np.exp(ls1))**2)/(np.sqrt(2*np.pi)*np.exp(ls1))+(1-f1)*bg((a1, a2))+1e-300))
    x0 = (0., np.log(0.8), -2., mu2_0, np.log(0.8), -3., 0., 0., 0.) if two else (0., np.log(0.8), -2., 0., 0., 0.)
    r = minimize(nll, x0, method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-5, maxiter=20000))
    p = r.x.copy(); p[1] = max(p[1], np.log(FLOOR["dphi2"])); r = type("R", (), dict(fun=nll(p)))
    r0 = minimize(lambda a: -np.sum(np.log(bg(a)+1e-300)), (0., 0.), method="Nelder-Mead"); dL = r0.fun-r.fun
    f1 = (np.exp(p[2])/(1+np.exp(p[2])+np.exp(p[5]))) if two else 1/(1+np.exp(-p[2]))
    out = dict(mu=p[0], mu_err=err1d(nll, p, 0, 0.05), s=np.exp(p[1]), s_err=np.exp(p[1])*err1d(nll, p, 1, 0.05), f=f1, N=f1*len(v), sig=np.sqrt(max(2*dL, 0)), n=len(v))
    if two:
        out.update(mu2=p[3], s2=np.exp(p[4]), f2=np.exp(p[5])/(1+np.exp(p[2])+np.exp(p[5])))
    return out


def err1d(f, p, i, h=0.02):
    """1-parameter curvature error (others fixed at the best fit); nan if not positive."""
    try:
        q = np.array(p, float); f0 = f(q); a = q.copy(); a[i] += h; b = q.copy(); b[i] -= h; d2 = (f(a)-2*f0+f(b))/h**2
        return float(1/np.sqrt(d2)) if np.isfinite(d2) and d2 > 0 else np.nan
    except Exception:
        return np.nan


def hess(f, p, h=1e-3):
    n = len(p); H = np.zeros((n, n)); f0 = f(p)
    for i in range(n):
        for j in range(i, n):
            pp = np.array(p, float); pm_ = pp.copy(); mp = pp.copy(); mm = pp.copy()
            pp[i] += h; pp[j] += h; pm_[i] += h; pm_[j] -= h; mp[i] -= h; mp[j] += h; mm[i] -= h; mm[j] -= h
            H[i, j] = H[j, i] = (f(pp)-f(pm_)-f(mp)+f(mm))/(4*h*h)
    return H


def smooth_offset(cs, y, err, ok, taper=3.0):
    """Track update: weighted least-squares cubic spline (knots every 6 deg) through the significant bins' offsets; outside their range
    the offset tapers linearly to 0 over `taper` deg (the previous track is kept there)."""
    from scipy.interpolate import LSQUnivariateSpline
    if ok.sum() < 5:
        return None
    x, yy, w = cs[ok], y[ok], 1/np.maximum(err[ok], 1e-3); lo, hi = x.min(), x.max()
    kn = np.arange(lo+6, hi-5.9, 6.); sp = LSQUnivariateSpline(x, yy, kn, w=w, k=3) if len(kn) else np.poly1d(np.polyfit(x, yy, min(3, len(x)-1), w=w))
    def f(g):
        g = np.asarray(g, float); v = sp(np.clip(g, lo, hi))
        t = np.clip(1-np.maximum(lo-g, g-hi).clip(0)/taper, 0, 1); return v*t
    return f, (float(lo), float(hi))


def smooth_logwidth(cs, logs, err, ok, old, deg=2):
    """Width update: weighted quadratic in phi1 for log s over the significant range; outside it the previous width is kept."""
    if ok.sum() < 5:
        return None
    x = cs[ok]; p = np.polyfit(x, logs[ok], deg, w=1/np.maximum(err[ok], 0.05)); lo, hi = x.min(), x.max()
    def f(g, taper=3.0):                                           # blend to the previous width over `taper` deg outside the range
        g = np.asarray(g, float); new = np.exp(np.polyval(p, np.clip(g, lo, hi))); t = np.clip(1-np.maximum(lo-g, g-hi).clip(0)/taper, 0, 1)
        return np.exp(t*np.log(new)+(1-t)*np.log(old))
    return f


def main():
    D, T, up = load_data(); u, x = D["phi1"], D["dphi2"]
    print(f"{len(u)} stars in the CMD mask", flush=True)
    tu = lambda p: np.interp(p, up[:, 0], up[:, 1], left=np.nan, right=np.nan)
    hist, tables = [], {}
    for it in range(NIT):
        moved = []
        for q in ("pmdec", "pmra", "dphi2"):
            if q == "dphi2":
                sel = other_masks(D, T, q, K_ITER)
            else:                                                     # PM fits: other-PM mask only; the sky window/off bands are applied below
                sel = other_masks(D, {**T, "dphi2": dict(c=T["dphi2"]["c"], s=np.full(GRID.size, 1e6))}, q, K_ITER)
            rows = []
            for a0, a1 in BINS:
                c = 0.5*(a0+a1); kb = sel & (u >= a0) & (u < a1)
                if q in ("pmra", "pmdec"):
                    dsky = x-ev(T, "dphi2", u); w_on = np.abs(dsky) < K_ITER*ev(T, "dphi2", u, "s")
                    offb = (np.abs(dsky) > 4) & (np.abs(dsky) < 12) & ~(np.abs(x-tu(u)) < 2)
                    v = D[q]-ev(T, q, u); e = D[q+"_error"]
                    r = fit_pm(v[kb & w_on], e[kb & w_on], u[kb & w_on]-c, v[kb & offb])
                else:
                    if c < 4:
                        continue
                    v = x-ev(T, q, u); two = 14 <= c <= 21
                    r = fit_sky(v[kb], u[kb]-c, two, float(np.nanmedian(tu(np.array([c]))-ev(T, q, np.array([c])))) if two else 0.)
                if r is not None:
                    r.update(c=c); rows.append(r)
            cs = np.array([r["c"] for r in rows]); mu = np.array([r["mu"] for r in rows]); me = np.array([r["mu_err"] for r in rows])
            ss = np.array([r["s"] for r in rows]); se = np.array([max(r["s_err"], 1e-3) for r in rows]); sig = np.array([r["sig"] for r in rows])
            cur_s = np.array([ev(T, q, np.array([c_]), "s")[0] for c_ in cs])
            ok = (sig >= SIG_MIN) & np.isfinite(me) & (me > 0) & np.isfinite(se) & (ss > 1.05*FLOOR[q]) & (ss < CAP[q]) & (np.abs(mu) < 1.5*np.maximum(cur_s, 0.5))
            sm = smooth_offset(cs, mu, me, ok); sw = smooth_logwidth(cs, np.log(ss), se/np.maximum(ss, 1e-3), ok, T[q]["s"])
            if sm is not None:
                T[q]["c"] = T[q]["c"]+DAMP*sm[0](GRID); moved.append(np.max(np.abs(mu[ok])/np.maximum(me[ok], 1e-3)))
            if sw is not None:
                T[q]["s"] = np.clip(np.exp(DAMP*np.log(np.maximum(sw(GRID), 1e-3))+(1-DAMP)*np.log(T[q]["s"])), FLOOR[q], CAP[q])
            T[q]["range"] = sm[1] if sm is not None else None
            tables[q] = rows
            print(f"iter {it} {q:5s}: " + " ".join(f"[{r['c']:.0f}: d{r['mu']:+.2f}+-{r['mu_err']:.2f} s{r['s']:.2f} {r['sig']:.0f}s]" for r in rows), flush=True)
        hist.append({q: dict(c=T[q]["c"].tolist(), s=T[q]["s"].tolist()) for q in T})
        if moved and max(moved) < 0.1:
            break
    out = dict(grid=GRID.tolist(), tracks={q: dict(c=T[q]["c"].tolist(), s=T[q]["s"].tolist(), range=T[q]["range"]) for q in T},
               tables={q: [{k: float(v) for k, v in r.items()} for r in rows] for q, rows in tables.items()}, history=hist,
               k_iter=K_ITER, k_final=K_FINAL, note="mask: |q - c(phi1)| < k sqrt(s(phi1)^2 + e_i^2); outside 'range' the track is held at the end value")
    (ROOT/"results/plot_data/stream_masks.json").write_text(json.dumps(out, indent=1, default=float))
    # --- figure 1: fits
    fig, ax = plt.subplots(2, 3, figsize=(18, 8), constrained_layout=True)
    for j, q in enumerate(("dphi2", "pmra", "pmdec")):
        rows = tables[q]; cs = np.array([r["c"] for r in rows])
        cen = np.array([r["mu"] for r in rows])+0.; sig = np.array([r["sig"] for r in rows])
        for ih, h in enumerate(hist):
            ax[0, j].plot(GRID, h[q]["c"], "-", color=plt.cm.viridis(ih/max(len(hist)-1, 1)), lw=1, label=f"iteration {ih}")
            ax[1, j].plot(GRID, h[q]["s"], "-", color=plt.cm.viridis(ih/max(len(hist)-1, 1)), lw=1)
        ax[0, j].errorbar(cs, ev(T, q, cs)+cen*0, yerr=[r["mu_err"] for r in rows], fmt="none", ecolor="0.5")
        ax[1, j].errorbar(cs, [r["s"] for r in rows], yerr=[r["s_err"] for r in rows], fmt="o", color="k", ms=3)
        for cc, sg in zip(cs, sig):
            ax[1, j].text(cc, 0.02, f"{sg:.0f}", ha="center", fontsize=7, transform=ax[1, j].get_xaxis_transform())
        ax[0, j].set(title=f"{q}: track per iteration", xlabel=r"$\phi_1$"); ax[1, j].set(title=f"{q}: intrinsic width (bins; numbers: significance)", xlabel=r"$\phi_1$")
    ax[0, 0].legend(fontsize=7); fig.savefig(ROOT/"plots/stream_masks_fits.png", dpi=95); plt.close(fig)
    # --- figure 2: cross-selection with the final masks
    from grid4_common import du, dx, d
    E1 = np.linspace(-8, 32, 61); c1 = 0.5*(E1[1:]+E1[:-1]); ph = GRID
    fig, ax = plt.subplots(3, 2, figsize=(16, 12), constrained_layout=True, sharex=True)
    rows_ = (("dphi2", x, np.arange(-15, 15.01, 0.25), r"$\Delta\phi_2$ [deg]", None), ("pmra", D["pmra"], np.arange(-28, 4.01, 0.25), r"$\mu_{\alpha*}$ [mas/yr]", "pmra_error"),
             ("pmdec", D["pmdec"], np.arange(-18, 2.01, 0.25), r"$\mu_\delta$ [mas/yr]", "pmdec_error"))
    for i, (q, v, Ey, yl, err) in enumerate(rows_):
        sel = other_masks(D, T, q, K_FINAL); H = np.histogram2d(u[sel], v[sel], (E1, Ey))[0]
        if q == "dphi2":
            cy = 0.5*(Ey[1:]+Ey[:-1]); B = np.zeros_like(H)
            for jj, cc in enumerate(c1):
                off = (np.abs(cy-ev(T, q, np.array([cc]))[0]) > 4) & (np.abs(cy-ev(T, q, np.array([cc]))[0]) < 12)
                if off.sum() > 10:
                    B[jj] = np.clip(np.polyval(np.polyfit(cy[off], H[jj, off], 2), cy), 0, None)
        else:
            dsky = x-ev(T, "dphi2", u); onS = np.abs(dsky) < K_FINAL*ev(T, "dphi2", u, "s"); offS = (np.abs(dsky) > 4) & (np.abs(dsky) < 12) & ~(np.abs(x-tu(u)) < 2)
            sel2 = other_masks(D, T, q, K_FINAL) | True
            selp = other_masks(D, {**T, "dphi2": dict(c=T["dphi2"]["c"], s=np.full(GRID.size, 1e6))}, q, K_FINAL)
            H = np.histogram2d(u[selp & onS], v[selp & onS], (E1, Ey))[0]; Hf = np.histogram2d(u[selp & offS], v[selp & offS], (E1, Ey))[0]
            nn = np.histogram(u[onS], E1)[0]; nf = np.histogram(u[offS], E1)[0]; B = Hf*np.divide(nn, nf, out=np.zeros(len(c1)), where=nf > 0)[:, None]
        e = (E1[0], E1[-1], Ey[0], Ey[-1])
        ax[i, 0].imshow(gaussian_filter((H-B)/np.sqrt(np.maximum(B, 1)), 1.).T, origin="lower", extent=e, aspect="auto", cmap="RdBu_r", vmin=-4, vmax=4)
        wt = np.median(D[err]) if err else 0.
        for a_ in ax[i]:
            a_.fill_between(ph, ev(T, q, ph)-K_FINAL*np.sqrt(ev(T, q, ph, "s")**2+wt**2), ev(T, q, ph)+K_FINAL*np.sqrt(ev(T, q, ph, "s")**2+wt**2),
                            color="C0", alpha=0.15 if a_ is ax[i, 0] else 0.3, lw=0)
            a_.set_ylabel(yl); a_.set_ylim(Ey[0], Ey[-1])
        rows = tables[q]; ax[i, 1].errorbar([r["c"] for r in rows], [ev(T, q, np.array([r["c"]]))[0] for r in rows], fmt="none")
        ax[i, 1].plot(du, {"dphi2": dx, "pmra": d["pmra"], "pmdec": d["pmdec"]}[q], ".", color="C2", ms=1.5, alpha=0.5)
        rg = T[q]["range"]
        if rg:
            for a_ in ax[i]:
                a_.axvspan(E1[0], rg[0], color="0.85", alpha=0.4, lw=0); a_.axvspan(rg[1], E1[-1], color="0.85", alpha=0.4, lw=0)
        ax[i, 0].set_title(f"{yl.split(' [')[0]}: CMD + other two data-driven masks (k = {K_FINAL:g}), background-subtracted", fontsize=10)
        ax[i, 1].set_title("final mask (median-error width) and STREAMFINDER members; grey: track held (no >= 3 sigma bins)", fontsize=9)
    for a_ in ax[-1]:
        a_.set_xlabel(r"$\phi_1$ [deg]")
    fig.suptitle("Data-driven masks (alternating unbinned fits, CMD-filtered Gaia DR3)", fontsize=11)
    fig.savefig(ROOT/"plots/stream_masks_xsel.png", dpi=100); plt.close(fig)
    print("plots/stream_masks_fits.png, plots/stream_masks_xsel.png")


if __name__ == "__main__":
    main()
