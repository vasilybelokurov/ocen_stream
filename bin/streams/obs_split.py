"""Per-observable conditional scores (dphi2 alone, PM pair alone, v_los alone) with the score_conditional conventions; moved from
d1_segments.py so it can be imported without running D1."""
import numpy as np
from grid4_common import project_gc, GC, du, dx, d, cov, win

EPS, HU = 0.05, 0.5; g_u = 1/(win[1]-win[0]); hasv = np.isfinite(d["v"])


def single_scores(m):
    """Per-star conditional log-scores for x, pm, v separately (v: nan for stars without v_los)."""
    mu, mx = project_gc(m["l"], m["b"], GC); k = (m["chi"] > 0) & (m["age"] < 700)
    edges = np.arange(0., m["chi"][k].max()+5., 5.)
    bins = [np.where(k & (m["chi"] >= e) & (m["chi"] < e+5.))[0] for e in edges]; bins = [b for b in bins if len(b) >= 20]; K = len(bins)
    idx = np.concatenate(bins); wt = np.concatenate([np.full(len(b), 1./(K*len(b))) for b in bins])
    U, X, PA, PD, V = mu[idx], mx[idx], m["pmra"][idx], m["pmdec"][idx], m["vlos"][idx]
    sx2 = 0.2**2+cov[:, 0, 0]; sy2 = 0.2**2+cov[:, 1, 1]; cxy = cov[:, 0, 1]; det = sx2*sy2-cxy**2; sv = np.sqrt(25.+np.nan_to_num(d["e_v"])**2)
    n = len(du); fu, fx, fp, fv = (np.zeros(n) for _ in range(4))
    for i0 in range(0, n, 300):
        s = slice(i0, min(n, i0+300))
        gu = np.exp(-0.5*((du[s, None]-U)/HU)**2)/(np.sqrt(2*np.pi)*HU)*wt
        fu[s] = gu.sum(1)
        fx[s] = (gu*np.exp(-0.5*((dx[s, None]-X)/0.5)**2)/(np.sqrt(2*np.pi)*0.5)).sum(1)
        a, b = d["pmra"][s, None]-PA, d["pmdec"][s, None]-PD
        q = (sy2[s, None]*a**2-2*cxy[s, None]*a*b+sx2[s, None]*b**2)/det[s, None]
        fp[s] = (gu*np.exp(-0.5*q)/(2*np.pi*np.sqrt(det[s]))[:, None]).sum(1)
        fv[s] = (gu*np.exp(-0.5*((np.nan_to_num(d["v"][s])[:, None]-V)/sv[s, None])**2)/(np.sqrt(2*np.pi)*sv[s, None])).sum(1)
    den = (1-EPS)*fu+EPS*g_u
    L = lambda f, box: np.log(((1-EPS)*f+EPS*g_u/box)/den)
    return dict(x=L(fx, 12.), pm=L(fp, 900.), v=np.where(hasv, L(fv, 400.), np.nan))
