"""Fable review 2026-10-28 (docs/reviews/pm_windows_fable_codex_20261028.md). Usage: PYTHONPATH=src python bin/streams/reviews/pm_windows_check_20261028.py
Per 3-deg window along phi1 -2..28: stars, spline stream fraction, expected stream stars, curvature of the residual spline m over
the window, and a quick local truncated-Gaussian + empirical background fit (track c, slope, width, fraction) with Hessian errors."""
import sys, json, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[3]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
sys.argv = [sys.argv[0]]
import stream_spline_fit as SF
from scipy.optimize import minimize
from scipy.special import ndtr
F = json.loads((ROOT/"results/plot_data/stream_spline_fit.json").read_text())
EDGES = np.arange(-2., 28.01, 3.)   # -2..1, 1..4, ..., 25..28
def local_fit(u, v, e, lb, W, c0):
    x = u-c0
    def nll(p):
        m = p[0]+p[1]*x; s = np.sqrt(np.exp(2*p[2])+0.05**2); sig = np.sqrt(s**2+e**2); f = 1/(1+np.exp(-p[3]))
        lg = -0.5*((v-m)/sig)**2-np.log(sig)-0.5*np.log(2*np.pi); Z = ndtr((W-m)/sig)-ndtr((-W-m)/sig)
        return -np.sum(np.logaddexp(np.log(f)+lg-np.log(np.maximum(Z, 1e-12)), np.log(1-f)+lb))
    best = None
    for m0 in (np.median(v), 0.):
        for ls0 in (np.log(0.2), np.log(0.6), np.log(1.5)):
            r = minimize(nll, [m0, 0., ls0, -1.], method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-6, fatol=1e-7))
            r = minimize(nll, r.x, method="BFGS")
            if best is None or r.fun < best.fun: best = r
    p = best.x; h = 1e-3; n = 4; H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.eye(n)[i]*h, np.eye(n)[j]*h
            H[i, j] = H[j, i] = (nll(p+ei+ej)-nll(p+ei-ej)-nll(p-ei+ej)+nll(p-ei-ej))/(4*h*h)
    C = np.linalg.pinv(H); return p, np.sqrt(np.clip(np.diag(C), 0, None)), C
for q in ("pmra", "pmdec"):
    t0 = time.time(); d = SF.load(q); f = F[q]; g = np.array(f["grid"]); u, v, e, lb = d["u"], d["v"], d["e"], np.log(np.maximum(d["b"], 1e-12)); W = d["W"]
    fs = np.interp(u, g, f["f"]); ms = np.interp(u, g, f["m"]); ss = np.interp(u, g, f["s"]); sig = np.sqrt(ss**2+e**2)
    # per-star stream posterior from the spline model
    Z = ndtr((W-ms)/sig)-ndtr((-W-ms)/sig); lg = -0.5*((v-ms)/sig)**2-np.log(sig)-0.5*np.log(2*np.pi)-np.log(Z)
    post = fs*np.exp(lg)/(fs*np.exp(lg)+(1-fs)*np.exp(lb))
    print(f"\n{q}: {u.size} stars, W {W}, load {time.time()-t0:.0f} s; spline m knots {np.round(f['fit']['m']['knots'],2).tolist()}")
    print(" window      N   <f>  Nstr(post) Nstr(f) | m_spline(centre) curv(max|m-lin| over win) | local: c +- err  slope  width  f   | spline sd at nearest knot")
    km = np.array(f["fit"]["m"]["knots"]); sd = np.array(f["fit"]["m"]["err"])
    for a0, a1 in zip(EDGES[:-1], EDGES[1:]):
        k = (u >= a0) & (u < a1); c0 = 0.5*(a0+a1); gg = np.linspace(a0, a1, 31); mg = np.interp(gg, g, f["m"]); lin = np.polyfit(gg, mg, 1); curv = np.max(np.abs(mg-np.polyval(lin, gg)))
        p, pe, C = local_fit(u[k], v[k], e[k], lb[k], W, c0); jn = np.argmin(np.abs(km-c0))
        print(f" {a0:5.1f}..{a1:5.1f} {k.sum():5d} {fs[k].mean():5.3f} {post[k].sum():8.0f} {fs[k].sum():8.0f} | {np.interp(c0, g, f['m']):+6.3f}   {curv:6.3f}               | {p[0]:+6.3f} +- {pe[0]:5.3f} {p[1]:+6.3f} {np.sqrt(np.exp(2*p[2])+0.05**2):5.2f} {1/(1+np.exp(-p[3])):5.3f} | knot {km[jn]:5.1f} sd {sd[jn]:.3f}")
