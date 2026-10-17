"""Small-particle-number bias of the stream score from a single spray (2026-10-17): delete-group jackknife (recommended)
and the second-order delta method.

Toy test (i.i.d. particles, 300 members, chi bins 0-200; reference = 6 x 500k particles; 60 samples per N), bias of the
total ln L: N = 1000 raw -10.9, delta -5.0, jackknife(G = 10) -1.1 (+-1.0); N = 4000 -2.8 / -0.6 / -0.2 (+-0.4);
N = 16000 -1.00 / -0.39 / -0.37 (+-0.16). The delta method misses ~half the bias where members rest on few particles;
the jackknife removes it with no extra noise. Groups are interleaved in particle order (= release-epoch order of the
spray), as needed for deterministic epochs (Codex 2026-10-17). Both still need validation against seed replicates.

Delta method:

For member i the score is ln L_i = ln A_i - ln B_i with
    A_i = (1 - eps) f_i + eps g_u g_w,   f_i = (1/K) sum_k f_ik,   f_ik = mean_{j in bin k} G_ij   (joint kernel)
    B_i = (1 - eps) h_i + eps g_u,       h_i = (1/K) sum_k h_ik,   h_ik = mean_{j in bin k} H_ij   (u kernel)
(score_conditional with chi_edges: K fixed bins, empty bins kept with zero density). For unbiased A, B the leading bias of
ln A - ln B is -Var(A)/(2A^2) + Var(B)/(2B^2) (Codex review 2026-10-17, docs/reviews/codex_unbiased_likelihood_2026-10-17.md),
so the corrected score is ln L_i + 0.5 [Var(A_i)/A_i^2 - Var(B_i)/B_i^2].
Variances are estimated within the run, treating the particles of a bin as i.i.d.:
    Var(f_ik) = (mean G^2 - f_ik^2)/n_k = f_ik^2/n_eff,ik - f_ik^2/n_k,   Var(h_ik) = (mean H^2 - h_ik^2)/n_k,
    Var(A_i) = (1 - eps)^2 / K^2 sum_k Var(f_ik), same for B.
Assumptions to validate against seed replicates (training-set calibration runs): epochs are deterministic, not i.i.d.; the
random bin occupancy n_k is ignored; the expansion is second order and poor at n_eff of a few.
The joint kernel sums come from score_conditional itself (one pooled call per bin with return_neff and return_components),
so the estimator is not re-implemented; only the 1-D u kernel's second moment is computed here.
"""
import numpy as np

from .score import score_conditional

__all__ = ["chi_bin_bias", "jackknife_total"]


def chi_bin_bias(du, dw, dv, dev, dcov, mu, mw, mv, chi, age, chi_edges, h=(0.5, 0.5, 0.2, 0.2), eps=0.05, age_max=700.,
                 chunk=50, **kw):
    """Per-member bias correction and within-run variances for score_conditional(..., chi_edges=chi_edges).

    Parameters are those of score_conditional (same meaning, same defaults); kw passes the rest (hv, u_window, w_bounds,
    v_bounds, ...). Returns dict(per_star, per_star_corrected, correction (N,), var_lnA, var_lnB (N,), total,
    total_corrected, n_k (K,)). per_star reproduces score_conditional(..., chi_edges=chi_edges)["per_star"].
    """
    du = np.asarray(du, float); n = len(du)
    ce = np.asarray(chi_edges, float); K = len(ce)-1
    sel = (chi > 0) & (age < age_max)
    f = np.zeros(n); hsum = np.zeros(n); vf = np.zeros(n); vh = np.zeros(n); nk = np.zeros(K, int)
    hu = h[0]
    common = None
    for k, (a, b) in enumerate(zip(ce[:-1], ce[1:])):
        idx = np.where(sel & (chi >= a) & (chi < b))[0]; nk[k] = len(idx)
        if len(idx) == 0:
            continue
        r = score_conditional(du, dw, dv, dev, dcov, mu[idx], mw[idx], mv[idx], chi[idx], age[idx], h=h, eps=eps, age_max=age_max,
                              pooled=True, nmin=1, return_components=True, return_neff=True, chunk=chunk, **kw)
        fk, hk, ne = r["foreground_joint"], r["foreground_u"], r["n_eff"]
        common = r
        f += fk/K; hsum += hk/K
        with np.errstate(divide="ignore", invalid="ignore"):
            vf += np.where(ne > 0, fk**2/ne - fk**2/len(idx), 0.)/K**2
        U = mu[idx]
        m2 = np.zeros(n)
        for i0 in range(0, n, chunk):
            sl = slice(i0, min(n, i0+chunk))
            gu = np.exp(-0.5*((du[sl, None]-U[None, :])/hu)**2)/(np.sqrt(2*np.pi)*hu)
            m2[sl] = np.mean(gu**2, axis=1)
        vh += (m2 - hk**2)/len(idx)/K**2
    if common is None:
        raise ValueError("no particles in any chi bin")
    gu_, gw_ = common["background_u"], common["background_w"]
    A = (1-eps)*f+eps*gu_*gw_; B = (1-eps)*hsum+eps*gu_
    vA, vB = (1-eps)**2*vf, (1-eps)**2*vh
    per = np.log(A/B); corr = 0.5*(vA/A**2 - vB/B**2)
    return dict(per_star=per, per_star_corrected=per+corr, correction=corr, var_lnA=vA/A**2, var_lnB=vB/B**2,
                total=float(per.sum()), total_corrected=float((per+corr).sum()), n_k=nk)


def jackknife_total(score_fn, model, G=10):
    """Delete-group jackknife bias-corrected total score: G l - (G - 1) mean_g l_(-g).

    score_fn(model_dict) -> dict with "total" (e.g. lambda m: grid4_common.score(m, chi_edges=E)); model: spray dict of
    equal-length arrays (non-array entries are passed through). Group g = particles with index % G == g (interleaved in
    release order). Returns dict(total, total_corrected, bias_estimate, totals_leave_out (G,)).
    """
    n = len(model["chi"])
    full = score_fn(model)["total"]
    idx = np.arange(n)
    lo = []
    for g in range(G):
        keep = idx % G != g
        sub = {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n else v) for k, v in model.items()}
        lo.append(score_fn(sub)["total"])
    lo = np.asarray(lo)
    corr = G*full-(G-1)*lo.mean()
    return dict(total=float(full), total_corrected=float(corr), bias_estimate=float(full-corr), totals_leave_out=lo)
