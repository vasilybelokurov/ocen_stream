#!/usr/bin/env python3
"""Data-driven mask borders (centre and intrinsic width vs phi1) for the omega Cen stream in Gaia DR3 (2026-10-23).

Procedure (docs/reviews/masks_fable_codex_20261023.md): stars = footprint (G < 20, parallax cut, GC masks) inside the isochrone CMD
mask (bin/streams/gaia_cmd_mask.cmd_pass, k = 2). Alternating 1-D unbinned fits per phi1 bin, for q in (pmdec, pmra*, dphi2):
  - select by the OTHER two masks at k_iter = 2.5 (per-star width sqrt(s_int^2 + e_i^2));
  - PMs: residual v = q - track_q(phi1) of stars in the sky window |dphi2| < 2 (fixed, user 2026-10-23: the on-sky signal is too
    complex for a fitted border; |dphi2 - centre| < 3 adopted 2026-10-23); background shape = residuals of the same selection at 4 < |dphi2 - centre| < 14; the window is centred on the
    starting (user-marked) sky track, ~0 at phi1 < 20 rising to +3 at 28; model (1 - f) B(v) + f N(v; mu + beta (phi1 - c),
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
Empirical (completeness) borders: after convergence, per bin the central interval holding 95 per cent of the background-subtracted
excess in a wide window (completeness_borders), 3-bin running mean, linearly interpolated in phi1 (bins >= 3 sigma);
tracks are anchored within 1.5 / 1.0 mas/yr and 2 deg of the starting (marked) tracks (with the wide |dphi2| < 5 window the pmra
iteration otherwise ran onto field structure at phi1 5-17); drawn in orange and stored
as lo95 / hi95 offsets from the track (the Gaussian core underestimates the borders where the stream has wings).
Displayed PM panels: background per (phi1 column, PM pixel) = quadratic in (dphi2 - window centre) fitted to 1-deg bands at
5-14 deg on both sides and integrated over the +-3 deg window (no global or total-count scaling); where bands exist on one side only (b > 8 edge near
the cluster) a constant from that side is used (a quadratic extrapolated badly there).
Figure plots/stream_masks_xsel.png: left: sky row = excess normalised per phi1 column by the integral of the positive excess in
that column; PM rows = (N - B)/sqrt(B); right: masks and members.
Outputs: results/plot_data/stream_masks.json (tracks and widths on a phi1 grid, per-bin fit tables, iteration history),
plots/stream_masks_fits.png (per-bin centres/widths with errors per iteration), plots/stream_masks_xsel.png (cross-selection with the
final masks at k = 2, background-subtracted, with STREAMFINDER members in a separate column).
Sky mask (user 2026-10-23): fitted per 4-deg bin (overlapping, step 2) at phi1 0-24 (fit_sky_binned: Gaussian on the background-subtracted dphi2 profile,
quadratic background from |v| > 4, upper-branch window excluded); the PM fits use the window centre +- clip(2 s, 1, 3) deg (3 deg
where not fitted). Dashed lines in the figure.
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
BINS = [(c-1, c+1) for c in range(-3, 25, 2)]+[(23, 27), (25, 29), (27, 31)]   # 2-deg bins from phi1 -4; 4-deg bins at the far end
# (user: signal to phi1 27-28; 2-deg bins there collapsed onto the width floor); sky fits phi1 >= 14 only (footprint edge)
K_ITER, K_FINAL, NIT = 2.5, 2.0, 6
CAP = dict(pmra=1.5, pmdec=1.5, dphi2=2.5)              # width caps
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


SKY_HALF, OFF_SKY = 3.0, (5.0, 14.0)    # user: sky selection for the other panels = |dphi2| < 2 (was 5; on-sky signal too complex for a
                                         # fitted border); PM backgrounds from 6 < |dphi2| < 14


def SKYC(u):
    """centre of the sky window: the fitted sky track (starts at the user-marked lower-branch track)."""
    return np.interp(u, GRID, SKY0)


def SKYH(u):
    """half-width of the sky window: 2 x fitted width (user 2026-10-23: fit the sky mask too), clipped to 1-3 deg; 3 deg where not fitted."""
    return np.interp(u, GRID, SKYW)


def other_masks(D, T, q, k, sky=True):
    u = D["phi1"]; m = np.ones(len(u), bool)
    for p, col, err in (("pmra", "pmra", "pmra_error"), ("pmdec", "pmdec", "pmdec_error"), ("dphi2", "dphi2", None)):
        if p == q:
            continue
        if p == "dphi2":
            if sky:
                m &= np.abs(D["dphi2"]-SKYC(u)) < SKYH(u)
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
    # starts: peak of the histogram minus a linear background within |v| < 3, and 0 (the single-start fit often stalled at f -> 0)
    E_ = np.linspace(-W, W, int(2*W/0.25)+1); c_ = 0.5*(E_[1:]+E_[:-1]); h_ = np.histogram(v, E_)[0].astype(float)
    wg = np.abs(c_) > 4; lin = np.polyval(np.polyfit(c_[wg], h_[wg], 1), c_); pk = c_[np.argmax(np.where(np.abs(c_) < 3, gaussian_filter(h_-lin, 2), -np.inf))]
    r = None
    for m0 in (float(pk), 0.):
        for ls0 in (np.log(0.6), np.log(1.2)):
            x0 = (m0, ls0, -2., mu2_0, np.log(0.8), -3., 0., 0., 0.) if two else (m0, ls0, -2., 0., 0., 0.)
            rr = minimize(nll, x0, method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-5, maxiter=20000, adaptive=True))
            if r is None or rr.fun < r.fun:
                r = rr
    p = r.x.copy(); p[1] = max(p[1], np.log(FLOOR["dphi2"])); r = type("R", (), dict(fun=nll(p)))
    r0 = minimize(lambda a: -np.sum(np.log(bg(a)+1e-300)), (0., 0.), method="Nelder-Mead"); dL = r0.fun-r.fun
    f1 = (np.exp(p[2])/(1+np.exp(p[2])+np.exp(p[5]))) if two else 1/(1+np.exp(-p[2]))
    out = dict(mu=p[0], mu_err=err1d(nll, p, 0, 0.05), s=np.exp(p[1]), s_err=np.exp(p[1])*err1d(nll, p, 1, 0.05), f=f1, N=f1*len(v), sig=np.sqrt(max(2*dL, 0)), n=len(v))
    if two:
        out.update(mu2=p[3], s2=np.exp(p[4]), f2=np.exp(p[5])/(1+np.exp(p[2])+np.exp(p[5])))
    return out


def fit_sky_binned(v, excl=None, W=10.0, db=0.25):
    """Sky profile in a bin: histogram of v = dphi2 - track over +-W; background = quadratic in v fitted to |v| > 4 (both sides; a
    constant if one side is empty -- footprint edge); excess fitted by a Gaussian (Poisson-weighted least squares, var = N + B);
    bins in `excl` (upper-branch window) left out of both fits. Returns dict like fit_pm (mu, mu_err, s, s_err, N, sig)."""
    from scipy.optimize import curve_fit
    E = np.arange(-W, W+1e-9, db); c = 0.5*(E[1:]+E[:-1]); h = np.histogram(v, E)[0].astype(float)
    use = np.ones(len(c), bool) if excl is None else ~excl(c)
    wing = use & (np.abs(c) > 4) & (h > 0); lo, hi = wing & (c < 0), wing & (c > 0)
    if wing.sum() < 8:
        return None
    dg = 2 if (lo.sum() >= 4 and hi.sum() >= 4) else 0
    B = np.clip(np.polyval(np.polyfit(c[wing], h[wing], dg), c), 0, None); ex = h-B; var = np.maximum(h+B, 1.)
    k = use & (np.abs(c) < 6)
    g = lambda x, A, m, sg: A*np.exp(-0.5*((x-m)/sg)**2)
    try:
        p, cov = curve_fit(g, c[k], ex[k], p0=(max(ex[k].max(), 1.), float(c[k][np.argmax(gaussian_filter(ex[k], 2))]), 1.0),
                           sigma=np.sqrt(var[k]), absolute_sigma=True, bounds=((0, -5, 0.2), (np.inf, 5, 4)), maxfev=20000)
    except Exception:
        return None
    A, m, sg = p; N = A*sg*np.sqrt(2*np.pi)/db; w2 = np.abs(c-m) < 2*sg
    sig = (ex[w2].sum())/np.sqrt(max(B[w2].sum(), 1.))
    return dict(mu=float(m), mu_err=float(np.sqrt(max(cov[1, 1], 0))), s=float(sg), s_err=float(np.sqrt(max(cov[2, 2], 0))), N=float(N),
                sig=float(sig), n=int(len(v)), f=float(N/max(len(v), 1)))


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
    kn = list(np.arange(lo+6, hi-5.9, 6.)); sp = None
    while sp is None:                                             # drop knots until the Schoenberg-Whitney conditions hold (gaps in accepted bins)
        try:
            sp = LSQUnivariateSpline(x, yy, kn, w=w, k=3) if kn else np.poly1d(np.polyfit(x, yy, min(3, len(x)-1), w=w))
        except ValueError:
            kn = kn[1:-1] if len(kn) > 1 else []
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


def completeness_borders(D, T, tu, TAB, frac=0.95):
    """Empirical borders per 2-deg bin: background-subtracted residual distribution in a wide window (pmdec +-4, pmra +-6 mas/yr,
    dphi2 +-8 deg), the same other-dimension selection as the fits (k_iter); background: PMs = off-sky stars scaled on the outer
    wings (|v| > W - 2), sky = quadratic fit to 4 < |v| < 8 with the upper-branch window excluded. Border = central interval holding
    `frac` of the excess: F(w) = excess within +-w of the fitted centre / excess within +-4 sigma_tot (no clipping, monotonised),
w95 = first w with F >= frac (clipping the excess to positive values made noise look like wings). Sky: phi1 >= 14 only."""
    u, x = D["phi1"], D["dphi2"]; out = {}; Wd = dict(pmdec=4., pmra=6., dphi2=8.)
    for q in ("dphi2", "pmra", "pmdec"):
        W = Wd[q]; E = np.linspace(-W, W, int(2*W/0.1)+1); cc = 0.5*(E[1:]+E[:-1]); rows = []
        sel = other_masks(D, T, q, K_ITER) if q == "dphi2" else other_masks(D, T, q, K_ITER, sky=False)
        for a0, a1 in BINS:
            c = 0.5*(a0+a1); kb = sel & (u >= a0) & (u < a1)
            vu = np.nan
            if q == "dphi2":
                if c < 0 or c > 24:
                    continue
                v = x[kb]-ev(T, q, u[kb]); h = np.histogram(v, E)[0].astype(float)
                vu = tu(np.array([c]))[0]-ev(T, q, np.array([c]))[0]; wing = (np.abs(cc) > 4) & ~(np.abs(cc-vu) < 2 if np.isfinite(vu) else False)
                B = np.clip(np.polyval(np.polyfit(cc[wing], h[wing], 2), cc), 0, None)
            else:
                dsky = x-ev(T, "dphi2", u); on = kb & (np.abs(x-SKYC(u)) < SKYH(u)); off = kb & (np.abs(x-SKYC(u)) > OFF_SKY[0]) & (np.abs(x-SKYC(u)) < OFF_SKY[1])
                h = np.histogram((D[q]-ev(T, q, u))[on], E)[0].astype(float); hf = np.histogram((D[q]-ev(T, q, u))[off], E)[0].astype(float)
                wing = np.abs(cc) > W-2; B = hf*h[wing].sum()/max(hf[wing].sum(), 1)
            fit = [r for r in TAB[q] if abs(r["c"]-c) < 0.1]
            if not fit:
                continue
            mu = fit[0]["mu"]; e_med = float(np.median(D[q+"_error"][kb])) if q != "dphi2" else 0.; st = np.hypot(fit[0]["s"], e_med)
            Wm = min(max(4*st, 1.5), W-0.5); ex = h-B; d_ = np.abs(cc-mu); inW = d_ < Wm
            if q == "dphi2" and np.isfinite(vu):
                ex = np.where(np.abs(cc-vu) < 1.25, 0., ex)               # the upper branch has its own window
            N = ex[inW].sum(); sig = N/np.sqrt(max(B[inW].sum(), 1))
            if N < 20 or sig < 3 or sig > 50:                            # > 50 sigma: failed background fit
                continue
            ws = np.arange(0.1, Wm+1e-9, 0.05); F = np.array([ex[d_ < w_].sum() for w_ in ws])/N; F = np.maximum.accumulate(F)
            w_emp = float(ws[np.argmax(F >= frac)]) if (F >= frac).any() else float(Wm)
            # wings tested: use the empirical width only if the excess beyond +-2 sigma_tot is significant (>= 2 sigma); otherwise
            # the Gaussian 1.96 sigma_tot (noise in the background-subtracted wings otherwise inflates w95, e.g. pmdec at phi1 < 6)
            wo = inW & (d_ > 2*st); ex_out = ex[wo].sum(); e_out = np.sqrt(h[wo].sum()+B[wo].sum()+1)
            wing = ex_out >= 2*e_out; w95 = w_emp if wing else float(1.96*st)
            rows.append(dict(c=c, lo=float(mu-w95), mid=float(mu), hi=float(mu+w95), N=float(N), sig=float(sig), w95=w95, w_emp=w_emp, s_gauss=float(st),
                             wing=bool(wing), wing_sig=float(ex_out/e_out)))
        out[q] = rows
    return out


def main():
    D, T, up = load_data(); u, x = D["phi1"], D["dphi2"]; global SKY0, SKYW; SKY0 = T["dphi2"]["c"].copy(); SKYW = np.full(GRID.size, SKY_HALF)
    print(f"{len(u)} stars in the CMD mask", flush=True)
    tu = lambda p: np.interp(p, up[:, 0], up[:, 1], left=np.nan, right=np.nan)
    hist, tables = [], {}; T0 = {q: T[q]["c"].copy() for q in T}; ANCHOR = dict(pmra=1.5, pmdec=1.0, dphi2=2.0)   # max departure from the start
    for it in range(NIT):
        moved = []
        for q in ("pmdec", "pmra", "dphi2"):
            if q == "dphi2":
                sel = other_masks(D, T, q, K_ITER)
            else:                                                     # PM fits: other-PM mask only; the sky window/off bands are applied below
                sel = other_masks(D, T, q, K_ITER, sky=False)
            rows = []
            for a0, a1 in BINS:
                c = 0.5*(a0+a1); kb = sel & (u >= a0) & (u < a1)
                if q in ("pmra", "pmdec"):
                    w_on = np.abs(x-SKYC(u)) < SKYH(u)
                    offb = (np.abs(x-SKYC(u)) > OFF_SKY[0]) & (np.abs(x-SKYC(u)) < OFF_SKY[1])
                    v = D[q]-ev(T, q, u); e = D[q+"_error"]
                    r = fit_pm(v[kb & w_on], e[kb & w_on], u[kb & w_on]-c, v[kb & offb])
                else:
                    if c < 0 or c > 24:                                       # beyond 24 the sky fits latch onto noise (marked track kept)
                        continue
                    kb = sel & (np.abs(u-c) < 2.0)                            # sky: 4-deg bins (overlapping), 2-deg bins too noisy
                    v = x-ev(T, q, u); vu = tu(np.array([c]))[0]-ev(T, q, np.array([c]))[0]
                    excl = (lambda cc, vu=vu: np.abs(cc-vu) < 1.25) if np.isfinite(vu) else None
                    r = fit_sky_binned(v[kb], excl)
                if r is not None:
                    r.update(c=c); rows.append(r)
            cs = np.array([r["c"] for r in rows]); mu = np.array([r["mu"] for r in rows]); me = np.array([r["mu_err"] for r in rows])
            ss = np.array([r["s"] for r in rows]); se = np.array([max(r["s_err"], 1e-3) for r in rows]); sig = np.array([r["sig"] for r in rows])
            cur_s = np.array([ev(T, q, np.array([c_]), "s")[0] for c_ in cs])
            ok = (sig >= SIG_MIN) & np.isfinite(me) & (me > 0) & np.isfinite(se) & (ss > 1.05*FLOOR[q]) & (ss < CAP[q]) & (np.abs(mu) < 1.5*np.maximum(cur_s, 0.5))
            sm = smooth_offset(cs, mu, me, ok); sw = smooth_logwidth(cs, np.log(ss), se/np.maximum(ss, 1e-3), ok, T[q]["s"])
            if sm is not None:
                T[q]["c"] = np.clip(T[q]["c"]+DAMP*sm[0](GRID), T0[q]-ANCHOR[q], T0[q]+ANCHOR[q]); moved.append(np.max(np.abs(mu[ok])/np.maximum(me[ok], 1e-3)))
            if sw is not None:
                T[q]["s"] = np.clip(np.exp(DAMP*np.log(np.maximum(sw(GRID), 1e-3))+(1-DAMP)*np.log(T[q]["s"])), FLOOR[q], CAP[q])
            T[q]["range"] = sm[1] if sm is not None else None
            tables[q] = rows
            if q == "dphi2":                                          # update the sky window used by the PM fits
                SKY0[:] = T[q]["c"]; fitted = np.zeros(GRID.size, bool)
                if sm is not None:
                    fitted = (GRID >= sm[1][0]-1) & (GRID <= sm[1][1]+1)
                SKYW[:] = np.where(fitted, np.clip(2*T[q]["s"], 1.0, 3.0), SKY_HALF)
            print(f"iter {it} {q:5s}: " + " ".join(f"[{r['c']:.0f}: d{r['mu']:+.2f}+-{r['mu_err']:.2f} s{r['s']:.2f} {r['sig']:.0f}s]" for r in rows), flush=True)
        hist.append({q: dict(c=T[q]["c"].tolist(), s=T[q]["s"].tolist()) for q in T})
        if moved and max(moved) < 0.1:
            break
    CB = completeness_borders(D, T, tu, tables)
    for q, rows in CB.items():
        cs = np.array([r["c"] for r in rows]); sig = np.array([r["sig"] for r in rows]); okb = sig >= 3
        if np.sum(sig >= 5) >= 6:                                     # enough strong bins: weak bins (noisy widths) are interpolated over
            okb = sig >= 5
        mid = np.array([r["mid"] for r in rows]); lo = np.array([r["lo"] for r in rows]); hi = np.array([r["hi"] for r in rows])
        T[q]["c_emp"], T[q]["lo_emp"], T[q]["hi_emp"] = T[q]["c"].copy(), np.full(GRID.size, np.nan), np.full(GRID.size, np.nan)
        if okb.sum() >= 4:
            xs, l_, h_ = cs[okb], lo[okb], hi[okb]
            sm3 = lambda y: np.convolve(np.pad(y, 1, mode="edge"), np.ones(3)/3, mode="valid") if len(y) >= 3 else y
            l_, h_ = sm3(l_), sm3(h_); a_, b_ = xs.min(), xs.max(); g = np.clip(GRID, a_, b_)
            T[q]["lo_emp"] = np.where((GRID >= a_-1) & (GRID <= b_+2), np.interp(g, xs, l_), np.nan); T[q]["hi_emp"] = np.where((GRID >= a_-1) & (GRID <= b_+2), np.interp(g, xs, h_), np.nan)
        print(f"{q} 95% borders: " + " ".join(f"[{r['c']:.0f}: {r['lo']:+.2f}..{r['hi']:+.2f} N{r['N']:.0f} {r['sig']:.0f}s {'W' if r['wing'] else 'g'}{r['wing_sig']:.1f}]" for r in rows), flush=True)
    out_sky = dict(centre=SKY0.tolist(), half_width=SKYW.tolist())
    out = dict(sky_window=out_sky, grid=GRID.tolist(), tracks={q: dict(c=T[q]["c"].tolist(), s=T[q]["s"].tolist(), range=T[q]["range"],
                                                  lo95=np.nan_to_num(T[q]["lo_emp"], nan=-99).tolist(), hi95=np.nan_to_num(T[q]["hi_emp"], nan=99).tolist()) for q in T},
               borders95=CB,
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
    rows_ = (("dphi2", x, np.arange(-10, 10.01, 0.25), r"$\Delta\phi_2$ [deg]", None), ("pmra", D["pmra"], np.arange(-28, 4.01, 0.25), r"$\mu_{\alpha*}$ [mas/yr]", "pmra_error"),
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
            dsky = x-ev(T, "dphi2", u); onS = np.abs(x-SKYC(u)) < SKYH(u); offS = (np.abs(x-SKYC(u)) > OFF_SKY[0]) & (np.abs(x-SKYC(u)) < OFF_SKY[1])
            sel2 = other_masks(D, T, q, K_FINAL) | True
            selp = other_masks(D, T, q, K_FINAL, sky=False)
            H = np.histogram2d(u[selp & onS], v[selp & onS], (E1, Ey))[0]
            # background per (phi1 column, PM pixel): quadratic in y = dphi2 - centre fitted to 1-deg bands at 5 <= |y| < 14 (both sides),
            # integrated over the window |y| < 3; bands outside the footprint (empty in every column/pixel) are dropped per column
            yv = x-SKYC(u); Eb = np.arange(-14, 14.01, 1.); cb = 0.5*(Eb[1:]+Eb[:-1]); offb = np.abs(cb) >= OFF_SKY[0]
            H3 = np.histogramdd(np.column_stack((u[selp], yv[selp], v[selp])), (E1, Eb, Ey))[0]
            yy = np.linspace(-SKY_HALF, SKY_HALF, 61); B = np.zeros_like(H)
            for jj in range(len(c1)):
                ob = offb & (H3[jj].sum(1) > 0)                              # bands with stars in this column
                if ob.sum() < 6:
                    continue
                both = (ob & (cb < 0)).sum() >= 3 and (ob & (cb > 0)).sum() >= 3
                dg = 3 if both else 1                                       # one-sided (footprint edge): constant, not extrapolated
                Pm = np.linalg.pinv(np.vander(cb[ob], dg)); coef = Pm @ H3[jj, ob, :]          # (dg, n_pm) per unit band (1 deg)
                B[jj] = np.clip(np.vander(yy, dg) @ coef, 0, None).mean(0)*(2*SKY_HALF)
        e = (E1[0], E1[-1], Ey[0], Ey[-1])
        if q == "dphi2":       # sky: excess per column normalised by the column's integral of POSITIVE excess (shows the ridge best)
            Xs = gaussian_filter(H-B, 1.); pos = np.clip(Xs, 0, None).sum(1, keepdims=True)
            ax[i, 0].imshow(np.divide(Xs, pos, out=np.zeros_like(Xs), where=pos > 0).T, origin="lower", extent=e, aspect="auto", cmap="RdBu_r", vmin=-0.08, vmax=0.08)
        else:
            ax[i, 0].imshow(gaussian_filter((H-B)/np.sqrt(np.maximum(B, 1)), 1.).T, origin="lower", extent=e, aspect="auto", cmap="RdBu_r", vmin=-4, vmax=4)
        wt = np.median(D[err]) if err else 0.
        if q == "dphi2":
            for a_ in ax[i]:
                a_.plot(ph, SKYC(ph)+SKYH(ph), "k--", lw=0.8); a_.plot(ph, SKYC(ph)-SKYH(ph), "k--", lw=0.8)
        for a_ in ax[i]:
            a_.fill_between(ph, ev(T, q, ph)-K_FINAL*np.sqrt(ev(T, q, ph, "s")**2+wt**2), ev(T, q, ph)+K_FINAL*np.sqrt(ev(T, q, ph, "s")**2+wt**2),
                            color="C0", alpha=0.15 if a_ is ax[i, 0] else 0.3, lw=0)
            a_.set_ylabel(yl); a_.set_ylim(Ey[0], Ey[-1])
        lo_e, hi_e = ev(T, q, ph)+T[q]["lo_emp"], ev(T, q, ph)+T[q]["hi_emp"]
        for a_ in ax[i]:
            a_.plot(ph, lo_e, "C1-", lw=1.2); a_.plot(ph, hi_e, "C1-", lw=1.2)
        cbr = CB[q]; ax[i, 1].errorbar([r["c"] for r in cbr], [ev(T, q, np.array([r["c"]]))[0]+0.5*(r["lo"]+r["hi"]) for r in cbr],
                                       yerr=[0.5*(r["hi"]-r["lo"]) for r in cbr], fmt="o", color="C1", ms=3, capsize=2)
        rows = tables[q]; ax[i, 1].errorbar([r["c"] for r in rows], [ev(T, q, np.array([r["c"]]))[0] for r in rows], fmt="none")
        ax[i, 1].plot(du, {"dphi2": dx, "pmra": d["pmra"], "pmdec": d["pmdec"]}[q], ".", color="C2", ms=1.5, alpha=0.5)
        rg = T[q]["range"]
        if rg:
            for a_ in ax[i]:
                a_.axvspan(E1[0], rg[0], color="0.85", alpha=0.4, lw=0); a_.axvspan(rg[1], E1[-1], color="0.85", alpha=0.4, lw=0)
        ax[i, 0].set_title(f"{yl.split(' [')[0]}: CMD + other masks (k = {K_FINAL:g}), " + ("excess / column integral of positive excess" if q == "dphi2" else "(N - B)/sqrt(B)"), fontsize=10)
        ax[i, 1].set_title("blue: Gaussian-core mask (k = 2, median error); orange: 95% empirical border (points: per bin); green: members", fontsize=9)
    for a_ in ax[-1]:
        a_.set_xlabel(r"$\phi_1$ [deg]")
    fig.suptitle("Data-driven masks (alternating unbinned fits, CMD-filtered Gaia DR3)", fontsize=11)
    fig.savefig(ROOT/"plots/stream_masks_xsel.png", dpi=100); plt.close(fig)
    print("plots/stream_masks_fits.png, plots/stream_masks_xsel.png")


if __name__ == "__main__":
    main()
