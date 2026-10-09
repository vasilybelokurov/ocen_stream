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


def chi_ridge(chi, l, b, extra, nbins=40, h=1.0, rmatch=1.0, min_rel_density=0.05, nmax_kde=3000, max_jump=6.0):
    """Ridge of a stream along the Gibbons phase chi (> 0): in each of nbins equal-number chi bins, the (l, b) point of maximum
    2D Gaussian-kernel density (bandwidth h deg, evaluated at the bin's particles), and medians of the `extra` observables
    (dict name -> array) over the bin's particles within rmatch deg of that point. The ridge is truncated at the first bin
    whose peak density (particles per deg^2, from the kernel sum) falls below min_rel_density x the maximum over bins, or at the
    second consecutive point more than max_jump deg (sky) from the last kept point; a single outlying bin (chi jumps at
    pericentres) is skipped, two in a row mean the stream has left the coherent arm.
    Returns dict of arrays: chi, l, b, density and each extra observable."""
    from scipy.spatial import cKDTree
    k = chi > 0
    chi, l, b = chi[k], l[k], b[k]; ex = {n: v[k] for n, v in extra.items()}
    qs = np.percentile(chi, np.linspace(0, 100, nbins+1))
    rows = []
    for lo, hi in zip(qs[:-1], qs[1:]):
        s = np.where((chi >= lo) & (chi < hi))[0]
        if len(s) < 10:
            continue
        pts = np.column_stack((l[s], b[s]))
        cand = pts if len(s) <= nmax_kde else pts[np.linspace(0, len(s)-1, nmax_kde).astype(int)]
        tree = cKDTree(pts)
        dens = np.array([np.sum(np.exp(-0.5*np.sum((pts[j]-c)**2, axis=1)/h**2))
                         for c, j in zip(cand, tree.query_ball_point(cand, 3*h))])/(2*np.pi*h**2)
        c = cand[np.argmax(dens)]
        m = np.sum((pts-c)**2, axis=1) < rmatch**2
        rows.append(dict(chi=float(np.median(chi[s])), l=float(c[0]), b=float(c[1]), density=float(dens.max()),
                         **{n: float(np.median(v[s][m])) for n, v in ex.items()}))
    dmax = max(r["density"] for r in rows)
    out = []
    miss = 0
    for r in rows:
        if r["density"] < min_rel_density*dmax:
            break
        if out and np.hypot(r["l"]-out[-1]["l"], r["b"]-out[-1]["b"]) > max_jump:
            miss += 1
            if miss == 2:
                break
            continue
        miss = 0
        out.append(r)
    return {key: np.array([r[key] for r in out]) for key in out[0]}


def age_ridge(age, l, b, extra, dage=15., age_max=450., h=1.0, rmatch=1.0, nmin=15, centre=(-50.9, 15.0), r_excl=1.5, track=4.0):
    """Display ridge of the trailing arm: in release-age bins of width dage [Myr] up to age_max, the (l, b) point of maximum 2D
    Gaussian-kernel density (bandwidth h deg) and medians of `extra` observables within rmatch deg of it. Release age is monotonic
    by construction; used for plotting/diagnosis only (not in the likelihood). Bins with < nmin particles are skipped.
    Particles within r_excl deg of the cluster (still bound / recaptured) are excluded. Tracking: the peak maximises
    density x exp(-|c - previous ridge point|^2 / (2 track^2)), starting from the cluster, so branches of debris released at
    the same time do not make the ridge jump."""
    from scipy.spatial import cKDTree
    rows = []; prev = np.array(centre, float)
    keep = np.hypot(l-centre[0], b-centre[1]) > r_excl
    for a0 in np.arange(0., age_max, dage):
        s = np.where(keep & (age >= a0) & (age < a0+dage))[0]
        if len(s) < nmin:
            continue
        pts = np.column_stack((l[s], b[s])); tree = cKDTree(pts)
        dens = np.array([np.sum(np.exp(-0.5*np.sum((pts[j]-c)**2, axis=1)/h**2)) for c, j in zip(pts, tree.query_ball_point(pts, 3*h))])
        c = pts[np.argmax(dens*np.exp(-0.5*np.sum((pts-prev)**2, axis=1)/track**2))]; prev = c
        m = np.sum((pts-c)**2, axis=1) < rmatch**2
        rows.append(dict(age=a0+dage/2, l=float(c[0]), b=float(c[1]), n=int(len(s)), **{n: float(np.median(v[s][m])) for n, v in extra.items()}))
    return {key: np.array([r[key] for r in rows]) for key in rows[0]}


def score_chi_kde(data, model, dchi=5., chi_max=None, nmin=20, age_max=700., floors=(0.5, 0.5, 0.2, 0.2), eps=0.05,
                  bg=1/(55.*30.*900.), chunk=400):
    """Dillamore+2022-style likelihood with the Gibbons phase chi as a latent along-stream coordinate.
    model: dict l, b, pmra, pmdec, vlos, chi, age (trailing arm). Bins of width dchi in chi (chi > 0, age < age_max, >= nmin
    particles). In each bin k a Gaussian KDE in x = (l, b, pmra, pmdec) with diagonal bandwidth H_k = max(Scott x std, floors)
    (Scott factor n^(-1/(d+4)), d = 4); for each member the kernel covariance is H_k^2 + diag(0, 0, C_pm) (Gaia PM covariance
    incl. correlation). Members with v_los get a 5th dimension (bandwidth max(Scott x std, 5 km/s) plus e_v^2).
    L_i = (1 - eps) sum_k (1/K) KDE_k(x_i) + eps * bg  (flat weights in chi: along-stream density not used).
    Returns dict: total lnL, K, per-bin mean likelihood share (which chi bins explain the data)."""
    k = (model["chi"] > 0) & (model["age"] < age_max)
    chi = model["chi"][k]; X = np.column_stack([model[q][k] for q in ("l", "b", "pmra", "pmdec")]); V = model["vlos"][k]
    hi = chi.max() if chi_max is None else chi_max
    edges = np.arange(0., hi+dchi, dchi)
    bins = []
    for e0, e1 in zip(edges[:-1], edges[1:]):
        s = (chi >= e0) & (chi < e1)
        if s.sum() >= nmin:
            n = s.sum(); f = n**(-1/8.)
            h = np.maximum(f*X[s].std(axis=0), floors); hv = max(f**(8/9.)*V[s].std(), 5.)
            bins.append((X[s], V[s], h, hv, 0.5*(e0+e1)))
    K = len(bins)
    D = np.column_stack([data[q] for q in ("l", "b", "pmra", "pmdec")]); n = len(D)
    hasv = np.isfinite(data["v"])
    like = np.zeros((n, K))
    for kk, (Xb, Vb, h, hv, _) in enumerate(bins):
        for i0 in range(0, n, chunk):
            sl = slice(i0, min(n, i0+chunk)); d = D[sl]
            # sky part (diagonal)
            q = ((d[:, None, 0]-Xb[None, :, 0])/h[0])**2+((d[:, None, 1]-Xb[None, :, 1])/h[1])**2
            norm = 1/(2*np.pi*h[0]*h[1])
            # PM part with Gaia covariance added
            sx2 = h[2]**2+data["e_pmra"][sl]**2; sy2 = h[3]**2+data["e_pmdec"][sl]**2
            cxy = data["rho"][sl]*data["e_pmra"][sl]*data["e_pmdec"][sl]; det = sx2*sy2-cxy**2
            dx = d[:, None, 2]-Xb[None, :, 2]; dy = d[:, None, 3]-Xb[None, :, 3]
            qp = (sy2[:, None]*dx**2-2*cxy[:, None]*dx*dy+sx2[:, None]*dy**2)/det[:, None]
            val = np.exp(-0.5*(q+qp))*norm/(2*np.pi*np.sqrt(det))[:, None]
            hv_i = np.where(hasv[sl], np.sqrt(hv**2+np.nan_to_num(data["e_v"][sl])**2), 1.)
            vfac = np.where(hasv[sl][:, None], np.exp(-0.5*((np.nan_to_num(data["v"][sl])[:, None]-Vb[None, :])/hv_i[:, None])**2)/(np.sqrt(2*np.pi)*hv_i[:, None]), 1.)
            like[sl, kk] = np.mean(val*vfac, axis=1)
    bgv = np.where(hasv, bg/400., bg)
    Li = (1-eps)*like.mean(axis=1)+eps*bgv
    share = (like/np.maximum(like.sum(axis=1, keepdims=True), 1e-300)).mean(axis=0)
    return dict(total=float(np.log(Li).sum()), K=K, chi_centres=[b[4] for b in bins], share=share.tolist(),
                n_bg_dominated=int(np.sum((1-eps)*like.mean(axis=1) < eps*bgv)))
