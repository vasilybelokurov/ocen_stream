#!/usr/bin/env python3
"""Distance-modulus offset dm(phi1) of the omega Cen stream from a binned Poisson CMD likelihood (2026-10-23).

Plan: docs/reviews/dm_iteration_fable_codex_20261023.md. Per 4-deg phi1 bin:
  stars  = footprint (G < 20, parallax cut, GC masks), dereddened (SFD x 0.86), inside both data-driven PM masks (k = 2,
           results/plot_data/stream_masks[_TAG].json); NO CMD cut;
  data   = Hess diagram of the sky window (|dphi2 - centre| < half-width), pixels 0.03 mag (colour) x 0.1 mag (G0);
  B      = per-pixel background: quadratic in (dphi2 - centre) fitted to 1-deg bands at 5-14 deg (constant where one-sided),
           integrated over the window, smoothed by 1.5 pixels in the CMD (the field CMD is smooth; reduces its noise);
  S(dm)  = stream template: omega Cen annulus stars (20-40', |dPM| < 0.7 mas/yr, ruwe < 1.4) minus the PM-control annulus
           (1.5 < |dPM| < 3, scaled by the PM-area ratio 0.49/6.75), shifted by dm in G0, Hess on the same pixels, smoothed by
           1 pixel, normalised to unit sum over the domain;
  model  = Poisson(B + A S(dm)), A >= 0 free (profiled); dm on a 0.02-mag grid in [dm0 - 0.5, dm0 + 0.5] (dm0 = FROZEN reference
           line DM_REF_SLOPE x phi1, independent of the iterated relation).
  domain = FIXED per bin in observed quantities: 0.3 < c0 < 1.4, 15.5 < G0 < min(19.6, 19.4 + dm0 - 0.5) -- the faint edge keeps
           the template inside the calibrated cluster range (template-frame G0 <= 19.4) for every trial dm.
Reported: dm (profile maximum), 68% interval (Delta lnL = 0.5), amplitude significance sqrt(2 [lnL(best) - lnL(A = 0)]); the same
for the MS only (G0 > 17.8 + dm0) and the TO/SGB only (G0 <= 17.8 + dm0); nulls: PM masks shifted by -6 and +6 mas/yr in pmra (+3 still overlapped the
stream wings at phi1 18-26, where the pmra mask is +-2.5-3 wide); the larger null significance is used.
A bin is accepted if its significance >= 3 and the null's < 2. Relation: weighted linear fit of accepted bins (no cap).
--inject: half of the cluster stars build the template, the other half (N per realisation) are shifted by dm_true and added to the
null field of one bin; recovered dm bias and interval coverage.
Usage: PYTHONPATH=src python bin/streams/stream_dm_fit.py [--masks orbitstart] [--inject]
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize_scalar

TAG = sys.argv[sys.argv.index("--masks")+1] if "--masks" in sys.argv else ""
BINS = [(a, a+4) for a in range(-2, 30, 4)]
EC = np.arange(0.3, 1.4001, 0.03); EG = np.arange(15.5, 19.6001, 0.1)
DGRID = 0.02; HALF = 0.5; G_SPLIT = 17.8; DM_REF_SLOPE = -0.0266   # FROZEN reference line (Fable 2026-10-23, anchored template fit):
# sets each bin's dm grid centre, faint domain edge and MS/TO split, so the measurement does not move with the iterated relation
ERR_INFL = 2.0                                                          # injection: Delta lnL = 0.5 intervals ~2x too narrow
dm_ref = lambda p: float(DM_REF_SLOPE*np.clip(p, 0., 24.))
OC, MU = (201.69683, -47.47957), (-3.250, -6.746)


def load():
    from gaia_footprint_plots import gc_masked
    raw = dict(np.load(ROOT/"results/streams/gaia_footprint_dr3.npz")); z0 = np.load(ROOT/"results/streams/gaia_footprint_dered.npz")
    assert np.array_equal(z0["source_id"], raw["source_id"]), "dereddened cache not aligned"
    c0, g0 = z0["c0"].astype(float), z0["g0"].astype(float); ra, dec = raw["ra"].astype(float), raw["dec"].astype(float)
    sep = np.degrees(np.arccos(np.clip(np.sin(np.radians(dec))*np.sin(np.radians(OC[1]))+np.cos(np.radians(dec))*np.cos(np.radians(OC[1]))
                                       *np.cos(np.radians(ra-OC[0])), -1, 1)))*60
    dpm = np.hypot(raw["pmra"]-MU[0], raw["pmdec"]-MU[1]); ann = (sep > 20) & (sep < 40) & (raw["ruwe"] < 1.4) & np.isfinite(c0)
    clus = dict(C=(c0[ann & (dpm < 0.7)], g0[ann & (dpm < 0.7)]), F=(c0[ann & (dpm > 1.5) & (dpm < 3)], g0[ann & (dpm > 1.5) & (dpm < 3)]))
    lg = np.where(raw["l"] > 180, raw["l"]-360, raw["l"]).astype(float); keep = ~gc_masked(lg, raw["b"].astype(float)) & np.isfinite(c0)
    S = {k: (v.astype(float) if v.dtype.kind == "f" else v)[keep] for k, v in raw.items()}; S["c0"], S["g0"] = c0[keep], g0[keep]
    return S, clus


def pm_mask(S, R, shift=0.):
    G = np.array(R["grid"]); u = S["phi1"]; m = np.ones(u.size, bool)
    for q in ("pmra", "pmdec"):
        c = np.interp(u, G, R["tracks"][q]["c"])+(shift if q == "pmra" else 0.); s = np.interp(u, G, R["tracks"][q]["s"])
        m &= np.abs(S[q]-c) < 2*np.sqrt(s**2+S[q+"_error"]**2)
    return m


def hess_on_bg(S, R, sel, a0, a1, EC=EC, EG=EG, smooth=1.5):
    """on-window Hess diagram and per-pixel background (interpolated across dphi2 from the off-sky bands)."""
    G = np.array(R["grid"]); u = S["phi1"]; yv = S["dphi2"]-np.interp(u, G, R["sky_window"]["centre"]); hw = np.interp(u, G, R["sky_window"]["half_width"])
    k = sel & (u >= a0) & (u < a1); on = k & (np.abs(yv) < hw); h = float(np.median(hw[on])) if on.any() else 3.
    Hn = np.histogram2d(S["c0"][on], S["g0"][on], (EC, EG))[0]
    Eb = np.arange(-14, 14.01, 1.); cb = 0.5*(Eb[1:]+Eb[:-1]); offb = np.abs(cb) >= 5; ks = k & (np.abs(yv) < 14)
    H3 = np.histogramdd(np.column_stack((yv[ks], S["c0"][ks], S["g0"][ks])), (Eb, EC, EG))[0]
    ob = offb & (H3.sum((1, 2)) > 0); dg = 3 if ((ob & (cb < 0)).sum() >= 3 and (ob & (cb > 0)).sum() >= 3) else 1
    coef = np.tensordot(np.linalg.pinv(np.vander(cb[ob], dg)), H3[ob], 1); yy = np.linspace(-h, h, 41)
    B = np.clip(np.tensordot(np.vander(yy, dg), coef, 1), 0, None).mean(0)*(2*h)
    return Hn, gaussian_filter(B, smooth), h


def ms_centring(S, R, sel, a0, a1, M, nboot=200, seed=5):
    """Old estimator with the new masks/background: dm at which the background-subtracted mean MS colour residual (G0 < 19.6,
    template-frame G0 >= 17.8, |r| < 0.15) crosses zero. Background per pixel on a fine grid (0.01 x 0.05 mag); error from
    Poisson resampling of the on-window counts."""
    ec, eg = np.arange(0.3, 1.4001, 0.01), np.arange(15.5, 19.6001, 0.05); Hn, B, _ = hess_on_bg(S, R, sel, a0, a1, ec, eg, smooth=3.)
    cc, gg = np.meshgrid(0.5*(ec[1:]+ec[:-1]), 0.5*(eg[1:]+eg[:-1]), indexing="ij"); cc, gg = cc.ravel(), gg.ravel()
    dms = np.arange(-1.2, 0.41, 0.02)
    def mean_r(H):
        out = []
        for d in dms:
            Gt = gg-d; r = cc-(M.ms_fn(Gt)+np.polyval(M.P["dc_ms"], np.minimum(Gt, 19.6))); w = (Gt >= 17.8) & (gg < 19.6) & (np.abs(r) < 0.15)
            ex = (H.ravel()-B.ravel())*w; out.append(np.sum(ex*r)/max(np.sum(ex), 1.))
        out = np.array(out); j = np.where(np.diff(np.sign(out)) != 0)[0]
        if len(j) == 0:
            return np.nan
        j = j[0]; return float(dms[j]-out[j]*(dms[j+1]-dms[j])/(out[j+1]-out[j]))
    best = mean_r(Hn); rng = np.random.default_rng(seed); bs = np.array([mean_r(rng.poisson(Hn)) for _ in range(nboot)])
    return dict(dm=best, err=float(np.nanstd(bs)), fail=float(np.mean(~np.isfinite(bs))))


def template(clus, dm, idx=None):
    cC, gC = clus["C"]; cF, gF = clus["F"]; frac = 1.
    if idx is not None:                                           # subset of the cluster stars: field correction scaled to the subset
        frac = len(idx)/len(cC); cC, gC = cC[idx], gC[idx]
    T = np.histogram2d(cC, gC+dm, (EC, EG))[0]-(0.49/6.75)*frac*np.histogram2d(cF, gF+dm, (EC, EG))[0]
    return gaussian_filter(np.clip(T, 0, None), 1.)


def profile(Hn, B, clus, dm0, dom, idx=None):
    """lnL(dm) profiled over A >= 0 on the domain mask `dom` (pixels)."""
    dms = np.arange(dm0-HALF, dm0+HALF+1e-9, DGRID); n, b = Hn[dom], np.maximum(B[dom], 1e-3); L0 = np.sum(n*np.log(b)-b); out = []
    for d in dms:
        T = template(clus, d, idx)[dom]; T = T/max(T.sum(), 1e-9)
        f = lambda A: -np.sum(n*np.log(b+A*T)-(b+A*T))
        r = minimize_scalar(f, bounds=(0, max(4*n.sum(), 10)), method="bounded"); out.append((-r.fun, r.x))
    L = np.array([o[0] for o in out]); A = np.array([o[1] for o in out]); j = int(np.argmax(L))
    ok = L >= L[j]-0.5; lo, hi = dms[ok].min(), dms[ok].max()
    return dict(dm=float(dms[j]), lo=float(lo), hi=float(hi), A=float(A[j]), sig=float(np.sqrt(max(2*(L[j]-L0), 0))), edge=bool(j in (0, len(dms)-1)),
                dms=dms.tolist(), L=(L-L[j]).tolist())


def domain(dm0):
    gc = 0.5*(EG[1:]+EG[:-1]); gmax = min(19.6, 19.4+dm0-HALF)
    base = np.broadcast_to((gc < gmax)[None, :], (len(EC)-1, len(EG)-1))
    return dict(all=base, ms=base & (gc > G_SPLIT+dm0)[None, :], tosgb=base & (gc <= G_SPLIT+dm0)[None, :])


def main():
    S, clus = load(); R = json.loads((ROOT/f"results/plot_data/stream_masks{'_' + TAG if TAG else ''}.json").read_text())
    from gaia_cmd_mask import dm_of
    P = json.loads((ROOT/"results/plot_data/gaia_cmd_mask.json").read_text()); coef0 = P["dm_relation"]
    dm_rel = lambda p: float(dm_of(P, p))                                 # relation the CMD mask currently uses (for reference only)
    print(f"cluster template: {len(clus['C'][0])} stars, PM control {len(clus['F'][0])}", flush=True)
    from gaia_cmd_mask import CMDMask
    global MSK; MSK = CMDMask(P)
    sel, nulls = pm_mask(S, R), [pm_mask(S, R, shift=sh) for sh in (-6., 6.)]; seln = nulls[1]
    if "--inject" in sys.argv:
        return inject(S, R, clus, nulls)
    rows = []; fig, ax = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=True)
    for a, (a0, a1) in zip(ax.ravel(), BINS):
        c = 0.5*(a0+a1); dm0 = dm_ref(c); dmc = dm_rel(c); D = domain(dm0); Hn, B, h = hess_on_bg(S, R, sel, a0, a1)
        res = {k: profile(Hn, B, clus, dm0, m) for k, m in D.items()}
        nul = max((profile(*hess_on_bg(S, R, sn, a0, a1)[:2], clus, dm0, D["all"]) for sn in nulls), key=lambda r_: r_["sig"])
        acc = res["all"]["sig"] >= 3 and nul["sig"] < 2 and not res["all"]["edge"]
        mc = ms_centring(S, R, sel, a0, a1, MSK)
        rows.append(dict(phi1=c, dm0=dm0, dm_current=dmc, accepted=acc, null_sig=nul["sig"], null_dm=nul["dm"], msc_dm=mc["dm"], msc_err=mc["err"], msc_fail=mc["fail"], **{f"{k}_{q}": v[q] for k, v in res.items() for q in ("dm", "lo", "hi", "A", "sig", "edge")}))
        r = res["all"]; print(f"phi1 {a0}-{a1}: dm {r['dm']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] A {r['A']:.0f} sig {r['sig']:.1f}{' EDGE' if r['edge'] else ''} | "
                              f"MS {res['ms']['dm']:+.2f} ({res['ms']['sig']:.1f}) TO/SGB {res['tosgb']['dm']:+.2f} ({res['tosgb']['sig']:.1f}) | "
                              f"null sig {nul['sig']:.1f} dm {nul['dm']:+.2f} | MS centring {mc['dm']:+.2f} +- {mc['err']:.2f} | current {dmc:+.2f} -> {'ACCEPT' if acc else 'reject'}", flush=True)
        for k, col in (("all", "k"), ("ms", "C0"), ("tosgb", "C3")):
            a.plot(res[k]["dms"], res[k]["L"], "-", color=col, label=f"{k}: {res[k]['dm']:+.2f} ({res[k]['sig']:.1f}$\\sigma$)")
        a.plot(nul["dms"], nul["L"], ":", color="0.5", label=f"null: {nul['sig']:.1f}$\\sigma$"); a.axvline(dmc, color="C2", lw=0.8, ls="--", label=f"current relation {dmc:+.2f}")
        a.set(ylim=(-15, 1), xlabel="dm", ylabel=r"$\Delta\ln L$", title=f"$\\phi_1$ {a0}-{a1}" + (" (accepted)" if acc else " (rejected)")); a.legend(fontsize=7)
    X = np.array([r["phi1"] for r in rows if r["accepted"]]); Y = np.array([r["all_dm"] for r in rows if r["accepted"]])
    E = np.array([max(0.5*(r["all_hi"]-r["all_lo"]), DGRID) for r in rows if r["accepted"]])
    if len(X) < 3:
        raise SystemExit(f"only {len(X)} accepted dm bins: no relation (refusing to keep the old one silently)")
    cf, cov = np.polyfit(X, Y, 1, w=1/E, cov="unscaled"); chi2 = float(np.sum(((Y-np.polyval(cf, X))/E)**2))
    print(f"free linear (accepted bins {X.tolist()}): dm = {cf[1]:+.3f} {cf[0]:+.4f} phi1; chi2 {chi2:.1f} for {len(X)-2} dof", flush=True)
    Ei = ERR_INFL*E; wt = 1/Ei**2; slope = float(np.sum(wt*X*Y)/np.sum(wt*X*X)); slope_err = float(1/np.sqrt(np.sum(wt*X*X)))
    chi2a = float(np.sum(((Y-slope*X)/Ei)**2))
    anch = dict(form="anchored_linear", slope=slope, slope_err=slope_err, phi1_min=0., phi1_max=float(X.max()), chi2=chi2a, dof=len(X)-1,
                accepted=X.tolist(), err_inflation=ERR_INFL, note="dm = slope x clip(phi1, phi1_min, phi1_max); errors x ERR_INFL")
    cq = np.polyfit(X, Y, 2, w=1/E); chi2q = float(np.sum(((Y-np.polyval(cq, X))/E)**2))
    poly = dict(form="poly", coef=[float(v) for v in cq], phi1_lo=-2., phi1_hi=30., chi2=chi2q, dof=len(X)-3, accepted=X.tolist(),
                note="quadratic through the accepted bins (user 2026-10-23: the mask line must go through the points); phi1 clipped to [-2, 30]")
    print(f"quadratic: dm = {cq[2]:+.3f} {cq[1]:+.5f} phi1 {cq[0]:+.6f} phi1^2; chi2 {chi2q:.1f} for {len(X)-3} dof; at 0/8/16/24/28: "
          + " ".join(f"{np.polyval(cq, p_):+.2f}" for p_ in (0, 8, 16, 24, 28)), flush=True)
    print(f"anchored linear (errors x{ERR_INFL:g}): dm = {slope:+.4f} +- {slope_err:.4f} x clip(phi1, 0, {X.max():g}); chi2 {chi2a:.1f} for {len(X)-1} dof", flush=True)
    fig.suptitle(f"dm profile likelihood per phi1 bin (Poisson CMD fit, cluster template, masks: {'orbit start' if TAG else 'marked start'}); "
                 f"anchored relation dm = {slope:+.4f} phi1 (to phi1 {X.max():g})")
    out = ROOT/f"plots/stream_dm_fit{'_' + TAG if TAG else ''}.png"; fig.savefig(out, dpi=90); plt.close(fig)
    json.dump(dict(bins=rows, relation=dict(coef=[float(cf[0]), float(cf[1])], cov=np.asarray(cov).tolist(), chi2=chi2, accepted=X.tolist()), relation_anchored=anch, relation_poly=poly,
                   masks=f"stream_masks{'_' + TAG if TAG else ''}.json", old_relation=coef0),
              open(ROOT/f"results/plot_data/stream_dm_fit{'_' + TAG if TAG else ''}.json", "w"), indent=1)
    print(out.relative_to(ROOT), flush=True)


def inject(S, R, clus, nulls, nrep=20, N=100):
    """Half the cluster stars -> template, other half (N in the domain) shifted by dm_true and added to the OBSERVED null-field Hess
    diagram (pmra masks +6 mas/yr) of three bins; background = that null field's own interpolated background. Reports bias and
    coverage of the raw and the x ERR_INFL intervals."""
    rng = np.random.default_rng(3); n = len(clus["C"][0]); perm = rng.permutation(n); iT, iI = np.sort(perm[:n//2]), np.sort(perm[n//2:])
    cI, gI = clus["C"][0][iI], clus["C"][1][iI]
    for a0, a1 in ((10, 14), (18, 22), (22, 26)):
        dm0 = dm_ref(0.5*(a0+a1)); D = domain(dm0)["all"]; Hz, Bz, _ = hess_on_bg(S, R, nulls[1], a0, a1)
        for dm_t in (dm0-0.3, dm0, dm0+0.3):
            Hfull = np.histogram2d(cI, gI+dm_t, (EC, EG))[0]; p = (Hfull*D).ravel(); p = p/p.sum(); bias, cov1, cov2 = [], [], []
            for _ in range(nrep):
                # null field resampled (Poisson around the observed counts) so the realisations differ; injected stars multinomial
                r = profile(rng.poisson(Hz)+rng.multinomial(N, p).reshape(Hfull.shape), Bz, clus, dm0, D, idx=iT)
                hw = 0.5*(r["hi"]-r["lo"]); bias.append(r["dm"]-dm_t); cov1.append(abs(r["dm"]-dm_t) <= max(hw, DGRID)); cov2.append(abs(r["dm"]-dm_t) <= ERR_INFL*max(hw, DGRID))
            print(f"inject phi1 {a0}-{a1}, dm_true {dm_t:+.2f}, N {N}: bias {np.mean(bias):+.3f} +- {np.std(bias)/np.sqrt(nrep):.3f}, scatter {np.std(bias):.3f}, "
                  f"coverage raw {np.mean(cov1):.2f}, x{ERR_INFL:g} {np.mean(cov2):.2f}", flush=True)


if __name__ == "__main__":
    main()
