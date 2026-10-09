"""Per-star, sky-conditional likelihood of stream members given model particles (fitting campaign; design reviewed with Codex,
docs/codex_track_objective_2026-10-09.md; cf. Erkal+2019 per-star likelihood).

For each member i at (l_i, b_i) and model trailing-arm particles j:
  sky:   p(l_i | b_i)         = sum_j K(b_i-b_j; h_b) N(l_i | l_j, h_l^2) / sum_j K(b_i-b_j; h_b)
  PM:    p(mu_i | l_i, b_i)   = sum_j K_sky(i,j) N(mu_i | mu_j, C_i + H) / sum_j K_sky(i,j),  K_sky Gaussian in (l, b), width h_s
  v_los: p(v_i | l_i, b_i)    = same with variance e_i^2 + h_v^2 (29 stars)
each mixed with a uniform background: L = (1 - eps) p_model + eps p_bg. The along-stream density is not used (no selection
function). Fixed estimator choices (set before scoring): h_b = h_l = h_s = 1 deg, H = (0.3 mas/yr)^2 I, h_v = 5 km/s, eps = 0.05,
background boxes: l in 55 deg, PMs in 30 x 30 mas/yr, v_los in 400 km/s. If no model particle lies within 3 h_b in latitude,
p_model = 0 (background only).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

H_B = H_L = H_S = 1.0
H_PM = 0.3
H_V = 5.0
EPS = 0.05
BG = dict(l=1/55., pm=1/900., v=1/400.)


def _gauss(x, s):
    return np.exp(-0.5*(x/s)**2)/(np.sqrt(2*np.pi)*s)


def score(data, model):
    """data: dict l, b, pmra, pmdec, e_pmra, e_pmdec, rho (pm correlation), v (nan if absent), e_v.
    model: dict l, b, pmra, pmdec, vlos (trailing-arm particles). Returns dict with total and per-term log-likelihoods."""
    ml, mb = model["l"], model["b"]
    tree_b = cKDTree(mb[:, None]); tree_s = cKDTree(np.column_stack((ml, mb)))
    n = len(data["l"]); ll_sky = np.zeros(n); ll_pm = np.zeros(n); ll_v = []
    nb = tree_b.query_ball_point(data["b"][:, None], 3*H_B)
    ns = tree_s.query_ball_point(np.column_stack((data["l"], data["b"])), 3*H_S)
    for i in range(n):
        j = np.asarray(nb[i], int)
        if len(j):
            wb = _gauss(data["b"][i]-mb[j], H_B)
            p = np.sum(wb*_gauss(data["l"][i]-ml[j], H_L))/np.sum(wb)
        else:
            p = 0.
        ll_sky[i] = np.log((1-EPS)*p+EPS*BG["l"])
        k = np.asarray(ns[i], int)
        if len(k):
            ws = _gauss(data["l"][i]-ml[k], H_S)*_gauss(data["b"][i]-mb[k], H_S)
            sx2 = data["e_pmra"][i]**2+H_PM**2; sy2 = data["e_pmdec"][i]**2+H_PM**2
            cxy = data["rho"][i]*data["e_pmra"][i]*data["e_pmdec"][i]; det = sx2*sy2-cxy**2
            dx = data["pmra"][i]-model["pmra"][k]; dy = data["pmdec"][i]-model["pmdec"][k]
            q = (sy2*dx**2-2*cxy*dx*dy+sx2*dy**2)/det
            pm = np.sum(ws*np.exp(-0.5*q)/(2*np.pi*np.sqrt(det)))/np.sum(ws)
            if np.isfinite(data["v"][i]):
                sv = np.sqrt(data["e_v"][i]**2+H_V**2)
                pv = np.sum(ws*_gauss(data["v"][i]-model["vlos"][k], sv))/np.sum(ws)
        else:
            pm = 0.; pv = 0.
        ll_pm[i] = np.log((1-EPS)*pm+EPS*BG["pm"])
        if np.isfinite(data["v"][i]):
            ll_v.append(np.log((1-EPS)*pv+EPS*BG["v"]))
    ll_v = np.array(ll_v)
    return dict(total=float(ll_sky.sum()+ll_pm.sum()+ll_v.sum()), sky=float(ll_sky.sum()), pm=float(ll_pm.sum()), vlos=float(ll_v.sum()),
                n_bg_sky=int(np.sum(ll_sky <= np.log(EPS*BG["l"])+1e-9)), n_bg_pm=int(np.sum(ll_pm <= np.log(EPS*BG["pm"])+1e-9)))


def overshoot_fraction(model_l, model_b, age_myr, data_l, data_b, age_max=450., tube=2.0, bmin=12.):
    """Fraction of young (< age_max) trailing debris with b > bmin that lies more than `tube` deg (sky, l-b plane) from every
    member: debris where the observed arm has none (diagnostic only; not in the likelihood)."""
    k = (age_myr < age_max) & (model_b > bmin)
    if not k.any():
        return np.nan
    d, _ = cKDTree(np.column_stack((data_l, data_b))).query(np.column_stack((model_l[k], model_b[k])))
    return float(np.mean(d > tube))


def chi_track(chi, obs, nq=20):
    """Model track ordered by the Gibbons phase chi (> 0, trailing): medians of each observable in nq chi-quantile bins."""
    qs = np.percentile(chi, np.linspace(0, 100, nq+1)); out = {k: [] for k in ("chi",)+tuple(obs)}
    for lo, hi in zip(qs[:-1], qs[1:]):
        s = (chi >= lo) & (chi < hi)
        out["chi"].append(float(np.median(chi[s])))
        for k, v in obs.items():
            out[k].append(float(np.median(v[s])))
    return out
