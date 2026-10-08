"""Stream-track estimator shared by data and models (fitting campaign, Stage 0; docs/STREAM_FIT_CAMPAIGN_PLAN.md).

Northern (trailing) arm of omega Cen, parametrised by Galactic latitude b (single-valued along the observed arm). In each
b bin and for each observable q in (l, pmra, pmdec) independently, a 1D mixture is fitted by maximum likelihood:
  p(q_i) = f N(q_i | mu, s^2 + e_i^2) + (1 - f) U(q_i | window),
with e_i the per-star measurement error (0 for model particles), s the intrinsic width, and the uniform background over the
bin's fitting window (median +- WIN[q]). Uncertainties by block bootstrap: resample 0.5-deg sub-blocks in b within the bin.
For model particles the same estimator is used; their error of the mean is the bootstrap error.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

EDGES = np.arange(15., 43.01, 2.)          # deg in b
WIN = dict(l=8., pmra=6., pmdec=4.)        # half-width of the fitting window around the bin median


def _fit1d(q, e, lo, hi, init):
    """ML fit of f*N(mu, s^2+e^2) + (1-f)*U(lo, hi); returns (mu, s, f)."""
    w = hi-lo

    def nll(p):
        mu, ls, lf = p
        s2 = np.exp(2*ls)+e**2; f = 1/(1+np.exp(-lf))
        like = f*np.exp(-0.5*(q-mu)**2/s2)/np.sqrt(2*np.pi*s2) + (1-f)/w
        return -np.sum(np.log(like+1e-300))
    r = minimize(nll, init, method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-6, maxiter=4000))
    mu, ls, lf = r.x
    return mu, np.exp(ls), 1/(1+np.exp(-lf))


def bin_fit(b, q, e, lo, hi):
    k = (q > lo) & (q < hi)
    if k.sum() < 6:
        return np.nan, np.nan, np.nan
    med = np.median(q[k]); mad = 1.4826*np.median(np.abs(q[k]-med))
    return _fit1d(q[k], e[k], lo, hi, [med, np.log(max(mad, 1e-3)), 2.])


def measure(l, b, pmra, pmdec, e_pmra=None, e_pmdec=None, nboot=100, seed=1, edges=EDGES):
    """Track of (l, pmra, pmdec) vs b. Returns dict with b centres, n per bin and for each q: mu, sig_mu (bootstrap), width."""
    rng = np.random.default_rng(seed)
    n = len(b)
    e = dict(l=np.zeros(n), pmra=np.zeros(n) if e_pmra is None else e_pmra, pmdec=np.zeros(n) if e_pmdec is None else e_pmdec)
    vals = dict(l=l, pmra=pmra, pmdec=pmdec)
    out = dict(b=0.5*(edges[1:]+edges[:-1]), n=np.histogram(b, edges)[0])
    for q in vals:
        mu, sm, wd, fr = (np.full(len(edges)-1, np.nan) for _ in range(4))
        for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            k = np.where((b >= lo) & (b < hi))[0]
            if len(k) < 8:
                continue
            x, ee = vals[q][k], e[q][k]; med = np.median(x)
            win = (med-WIN[q], med+WIN[q])
            mu[j], wd[j], fr[j] = bin_fit(b[k], x, ee, *win)
            blocks = np.floor((b[k]-lo)/0.5).astype(int); ub = np.unique(blocks)
            bs = []
            for _ in range(nboot):
                pick = np.concatenate([np.where(blocks == u)[0] for u in rng.choice(ub, len(ub))])
                bs.append(bin_fit(b[k][pick], x[pick], ee[pick], *win)[0])
            sm[j] = np.nanstd(bs)
        out[q] = dict(mu=mu, sig_mu=sm, width=wd, frac=fr)
    return out
