#!/usr/bin/env python3
"""Isochrone-based CMD mask for the omega Cen stream in Gaia DR3, broadened by per-star photometric errors and by the
cluster's intrinsic (population) width from error deconvolution (2026-10-23; plan reviewed by Codex and Fable,
docs/reviews/cmd_mask_fable_codex_20261023.md).

Photometry: dereddened (SFD x 0.86; R_G 2.740, R_BP 3.374, R_RP 2.035), results/streams/gaia_footprint_dered.npz (source_id
alignment with the raw cache asserted). Per-star errors sigma = 1.0857 / flux_over_error (+ floors G 0.0028, BP 0.0028, RP 0.0038;
UNVERIFIED floor values) from results/streams/gaia_footprint_photerr_subset.npz; stream stars without fetched errors use the
median sigma(G) of the random footprint subset (cluster region excluded).
Sequence (cluster frame, G0 at 5.43 kpc), MIST 12.5 Gyr [Fe/H] -1.7 (minimint, Gaia EDR3), colour +0.085, in three segments,
each = isochrone + a smooth offset fitted to omega Cen (MIST's SGB/RGB do not match after the MS calibration):
  MS  (G0 >= 17.6): colour residual r = c - [c_iso(G) + dc_MS(G)]; dc_MS = quadratic fit to running medians at G0 17.6-19.6,
      held constant beyond 19.6 (cluster limit) -- the isochrone carries the shape fainter;
  RGB (G0 <= 16.9): r = c - [c_iso,RGB(G) + dc_RGB] (constant offset);
  TO/SGB (16.6 < G0 < 17.9): empirical ridge of the cluster (running median of G0 in 0.02-mag colour bins), joined to the MS and RGB
      ridges (MIST's SGB is ~0.35 mag too bright after the MS calibration); a star's distance = min over the ridge of the
      error-normalised ellipse distance, colour width = MS intrinsic width at G0 17.7, magnitude width calibrated to 95.4 per cent
      control-subtracted completeness of the cluster at k = 2.
Intrinsic width per bin (0.2 mag in G0 for MS/RGB, 0.04 mag in colour for SGB): maximise sum log[(1 - f) N(r; mu, s_int^2 + e_i^2)
+ f p_F(r)], p_F from a PM-offset control (same radii, 1.5 < |dPM| < 3, scaled on the red CMD region); s_int(G) on the MS fitted
linearly in G0 and extrapolated (capped at 0.12), RGB/SGB: medians.
dm(phi1): refitted with this ridge (per signal bin phi1 14-18, 18-22, 22-26 the shift at which the background-subtracted mean
MS colour residual is zero; bootstrap errors),
weighted linear relation capped at 0; stored in results/plot_data/gaia_cmd_mask.json.
Mask for a star (stream frame: G0_t = G0 - dm(phi1)): inside the segment band |r - mu| < k sqrt(s_int^2 + e^2) (+ sigma_dm in the
magnitude direction on the SGB; on the MS the colour equivalent dc/dG x sigma_dm); HB excluded; G0_t >= 13.5.
Outputs: plots/gaia_cmd_mask.png (cluster CMD with borders; widths; completeness on the cluster; S/N vs k at phi1 18-22; CMD excess
per phi1 bin with the mask), results/plot_data/gaia_cmd_mask.json (segment parameters).
Usage: PYTHONPATH=src python bin/streams/gaia_cmd_mask.py [--k 2]
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
from scipy.special import erf

OC, MU = (201.69683, -47.47957), (-3.250, -6.746)
FLOOR = dict(g=0.0028, bp=0.0028, rp=0.0038)
SIG_DM, COL_SHIFT, G_MS, G_RGB = 0.1, 0.085, 17.6, 16.9
BINS = ((6, 14), (14, 18), (18, 22), (22, 26), (26, 30))


def isochrone_branches():
    import minimint, os
    import ocen_stream  # noqa: F401
    ii = minimint.Interpolator(["Gaia_G_EDR3", "Gaia_BP_EDR3", "Gaia_RP_EDR3"], data_prefix=os.path.join(os.path.expanduser(os.path.expandvars("$OCEN_ISOCHRONES")), "minimint"))
    la = np.log10(12.5e9); m = np.linspace(0.12, ii.getMaxMass(la, -1.7)-1e-4, 20000); r = ii(m, la, -1.7)
    G = r["Gaia_G_EDR3"]+5*np.log10(5430)-5; c = r["Gaia_BP_EDR3"]-r["Gaia_RP_EDR3"]+COL_SHIFT
    ok = np.isfinite(G) & np.isfinite(c); G, c = G[ok], c[ok]; tip = int(np.argmin(G)); G, c = G[:tip+1], c[:tip+1]
    ito = int(np.argmin(np.where(G > 16, c, 9)))                                   # turnoff = bluest MS point
    ms = (np.arange(len(G)) <= ito); post = ~ms
    gm, cm = G[ms], c[ms]; o = np.argsort(gm); ms_fn = lambda g: np.interp(g, gm[o], cm[o])
    # RGB branch: post-turnoff points after the isochrone starts climbing steeply (|dG/dc| > 4) past the subgiant branch
    ip = np.where(post)[0]; dG, dc = np.gradient(G[ip]), np.gradient(c[ip]); steep = (dG < 0) & (np.abs(dG) > 4*np.abs(dc)) & (c[ip] > c[ito]+0.1)
    i_rgb = ip[np.argmax(steep)]; rgb = np.arange(len(G)) >= i_rgb; o2 = np.argsort(G[rgb]); rgb_fn = lambda g: np.interp(g, G[rgb][o2], c[rgb][o2])
    sgb = post & (G >= G_RGB-0.4); o3 = np.argsort(c[sgb]); sgb_fn = lambda cc: np.interp(cc, c[sgb][o3], G[sgb][o3])
    return ms_fn, rgb_fn, sgb_fn, (c[ito], G[ito]), (G, c)


def fit_width(r, e, rF, half):
    """1 Gaussian (intrinsic s, centre mu) convolved with per-star errors + control-shaped contaminant."""
    hb = np.histogram(rF, np.linspace(-half, half, 31))[0]+0.5; xc = np.linspace(-half, half, 31); xc = 0.5*(xc[1:]+xc[:-1])
    pf = np.interp(r, xc, hb/hb.sum()/(xc[1]-xc[0]))
    def nll(p):
        mu, ls, lf = p; s2 = np.exp(2*ls)+e**2; f = 1/(1+np.exp(-lf))
        return -np.sum(np.log((1-f)*np.exp(-0.5*(r-mu)**2/s2)/np.sqrt(2*np.pi*s2)+f*pf+1e-300))
    p = minimize(nll, (np.median(r), np.log(max(1.4826*np.median(np.abs(r-np.median(r))), 1e-3)), -3.), method="Nelder-Mead",
                 options=dict(xatol=1e-5, fatol=1e-5, maxiter=6000)).x
    return float(p[0]), float(np.exp(p[1])), float(1/(1+np.exp(-p[2])))


class CMDMask:
    def __init__(self, P):
        self.P = P; self.ms_fn, self.rgb_fn, self.sgb_fn, self.to, self.iso = isochrone_branches()

    def residuals(self, c, G):
        """segment id (0 MS, 1 SGB, 2 RGB, -1 none), residual (colour or magnitude), using the fitted offsets."""
        P = self.P; seg = np.full(len(c), -1); r = np.full(len(c), np.nan)
        ms = G >= G_MS; rgb = (G <= G_RGB) & (G >= 13.5)
        dms = np.polyval(P["dc_ms"], np.minimum(G, 19.6))
        r[ms] = c[ms]-(self.ms_fn(G[ms])+dms[ms]); seg[ms] = 0
        r[rgb] = c[rgb]-(self.rgb_fn(G[rgb])+P["dc_rgb"]); seg[rgb] = 2
        sgb = (G > 15.9) & (G < G_MS+0.4); seg[sgb & (seg < 0)] = 1                     # TO/SGB window: ellipse distance in z()
        self.in_sgb = sgb
        return seg, r

    def sgb_dist(self, c, G, sc, sg):
        """min over the empirical TO/SGB polyline of the error-normalised ellipse distance."""
        P = self.P; pc, pg = np.array(P["sgb_poly_c"]), np.array(P["sgb_poly_G"])
        Sc = np.sqrt(P["s_c_sgb"]**2+sc[:, None]**2); Sg = np.sqrt(P["s_G_sgb"]**2+sg[:, None]**2+SIG_DM**2)
        return np.sqrt(np.min(((c[:, None]-pc[None, :])/Sc)**2+((G[:, None]-pg[None, :])/Sg)**2, axis=1))

    def width(self, seg, G, c):
        P = self.P; s = np.full(len(G), np.nan); mu = np.zeros(len(G))
        m = seg == 0; s[m] = np.clip(np.polyval(P["s_ms"], G[m]), P["s_ms_min"], 0.12); mu[m] = 0.
        m = seg == 2; s[m] = P["s_rgb"]
        return mu, s

    def z(self, c, G, sc, sg, dm=0.):
        """Normalised residual in the stream frame (G0_t = G - dm); photometric errors at the star's own magnitude."""
        Gt = G-dm; seg, r = self.residuals(c, Gt); mu, s = self.width(seg, Gt, c)
        slope = np.abs(np.gradient(self.ms_fn(np.array([17.6, 21.5]))).mean()/3.9)          # dc/dG on the MS (~0.1)
        e2 = sc**2+np.where(seg == 0, (slope*SIG_DM)**2, 0.)
        zz = np.abs(r-mu)/np.sqrt(s**2+e2); zz[seg < 0] = np.inf
        m = self.in_sgb                                                                   # all stars in the TO/SGB window, any segment
        if m.any():
            zs = self.sgb_dist(c[m], Gt[m], sc[m], sg[m]); zz[m] = np.where(np.isfinite(zz[m]), np.minimum(zz[m], zs), zs)
        return zz


def main():
    k_def = float(sys.argv[sys.argv.index("--k")+1]) if "--k" in sys.argv else 2.0
    raw = dict(np.load(ROOT/"results/streams/gaia_footprint_dr3.npz")); z0 = np.load(ROOT/"results/streams/gaia_footprint_dered.npz")
    assert np.array_equal(z0["source_id"], raw["source_id"]), "dereddened cache not aligned with the raw catalogue"
    g0, c0 = z0["g0"].astype(float), z0["c0"].astype(float); sid = raw["source_id"]; graw = raw["g"].astype(float)
    E = dict(np.load(ROOT/"results/streams/gaia_footprint_photerr_subset.npz")); o = np.argsort(E["sid"])
    pos = np.clip(np.searchsorted(E["sid"][o], sid), 0, len(o)-1); has = E["sid"][o][pos] == sid
    def sig(foe, fl):
        v = np.full(len(sid), np.nan); f = np.asarray(foe, float)[o][pos]; v[has] = np.sqrt((1.0857/f[has])**2+fl**2); return v
    sg, sbp, srp = sig(E["g_foe"], FLOOR["g"]), sig(E["bp_foe"], FLOOR["bp"]), sig(E["rp_foe"], FLOOR["rp"]); sc = np.sqrt(sbp**2+srp**2)
    ra, dec = raw["ra"].astype(float), raw["dec"].astype(float)
    sep = np.degrees(np.arccos(np.clip(np.sin(np.radians(dec))*np.sin(np.radians(OC[1]))+np.cos(np.radians(dec))*np.cos(np.radians(OC[1]))
                                       *np.cos(np.radians(ra-OC[0])), -1, 1)))*60
    rnd = has & (sep > 60) & np.isfinite(sc); Gb = np.arange(10, 20.01, 0.1); gcen = 0.5*(Gb[1:]+Gb[:-1])
    med_sc = np.array([np.nanmedian(sc[rnd & (graw >= a) & (graw < a+0.1)]) for a in Gb[:-1]])
    med_sg = np.array([np.nanmedian(sg[rnd & (graw >= a) & (graw < a+0.1)]) for a in Gb[:-1]])
    sc_all = np.where(np.isfinite(sc), sc, np.interp(graw, gcen, med_sc)); sg_all = np.where(np.isfinite(sg), sg, np.interp(graw, gcen, med_sg))
    dpm = np.hypot(raw["pmra"]-MU[0], raw["pmdec"]-MU[1]); ann = (sep > 20) & (sep < 40) & (raw["ruwe"] < 1.4) & has & np.isfinite(c0)
    C, F = ann & (dpm < 0.7), ann & (dpm > 1.5) & (dpm < 3.0)
    print(f"cluster {C.sum()}, control {F.sum()}", flush=True)
    M = CMDMask(dict(dc_ms=[0., 0., 0.], dc_rgb=0., dG_sgb=0., sgb_c=(0., 0.), s_ms=[0., 0.03], s_ms_min=0.01, s_rgb=0.03, s_sgb=0.1))
    cC, gC = c0[C], g0[C]
    # MS offset: running medians 17.6-19.6
    gq = np.arange(17.6, 19.61, 0.2); med = [np.median(cC[(gC >= a) & (gC < a+0.2)]-M.ms_fn(gC[(gC >= a) & (gC < a+0.2)])) for a in gq[:-1]]
    dc_ms = np.polyfit(gq[:-1]+0.1, med, 2)
    # RGB offset: G0 14-16.9
    mR = (gC >= 14) & (gC <= G_RGB) & (np.abs(cC-M.rgb_fn(gC)) < 0.3); dc_rgb = float(np.median(cC[mR]-M.rgb_fn(gC[mR])))
    # TO/SGB: empirical ridge of the cluster (running median of G0 in colour bins between the MS ridge at G0 17.6 and the RGB ridge at
    # G0 16.9), joined to the MS and RGB ridges; densely resampled
    c_to = float(M.ms_fn(17.6)+np.polyval(dc_ms, 17.6)); c_rb = float(M.rgb_fn(G_RGB)+dc_rgb)
    cb = np.arange(c_to, c_rb+0.001, 0.02); pts = []
    gg_ = np.arange(16.0, 17.9, 0.01)
    for cc in cb:                                  # peak (not median: TO/MS stars dominate the colour bins) of G0 in the SGB window
        m = (np.abs(cC-cc) < 0.015) & (gC > 16.0) & (gC < 17.9)
        if m.sum() > 20:
            kde = np.exp(-0.5*((gg_[:, None]-gC[m][None, :])/0.05)**2).sum(1); pts.append((cc, float(gg_[np.argmax(kde)])))
    gm_ = np.linspace(G_MS+0.3, G_MS, 6); gr_ = np.linspace(G_RGB, G_RGB-0.3, 6)            # ordered: MS (faint -> TO), SGB by colour, RGB (base -> up)
    pts = [p_ for p_ in sorted(pts) if c_to+0.01 < p_[0] < c_rb-0.01]
    P_ = [(float(M.ms_fn(g)+np.polyval(dc_ms, g)), float(g)) for g in gm_]+pts+[(float(M.rgb_fn(g)+dc_rgb), float(g)) for g in gr_]
    pc, pg = np.array(P_).T; d = np.concatenate(([0], np.cumsum(np.hypot(np.diff(pc), 0.1*np.diff(pg))))); tt = np.linspace(0, d[-1], 300)
    M.P.update(dc_ms=list(dc_ms), dc_rgb=dc_rgb, sgb_poly_c=list(np.interp(tt, d, pc)), sgb_poly_G=list(np.interp(tt, d, pg)), s_c_sgb=0.03, s_G_sgb=0.1)
    print(f"offsets: MS dc(G) at 17.6/18.6/19.6 = {np.polyval(dc_ms, 17.6):+.3f}/{np.polyval(dc_ms, 18.6):+.3f}/{np.polyval(dc_ms, 19.6):+.3f}; RGB dc {dc_rgb:+.3f}; "
          f"TO/SGB ridge from {len(pts)} colour bins ({c_to:.2f}-{c_rb:.2f})", flush=True)
    # intrinsic widths
    segC, rC = M.residuals(cC, gC); segF, rF = M.residuals(c0[F], g0[F]); gF = g0[F]; rows = []
    for a in np.arange(17.6, 19.81, 0.2):
        m = (segC == 0) & (gC >= a) & (gC < a+0.2) & (np.abs(rC) < 0.3); mf = (segF == 0) & (gF >= a-0.4) & (gF < a+0.6) & (np.abs(rF) < 0.3)
        if m.sum() > 50:
            mu, s, f = fit_width(rC[m], sc[C][m], rF[mf], 0.3); rows.append(("MS", a+0.1, m.sum(), mu, s, f, float(np.median(sc[C][m])), 1.4826*float(np.median(np.abs(rC[m]-np.median(rC[m]))))))
    for a in np.arange(14.0, 16.9, 0.5):
        m = (segC == 2) & (gC >= a) & (gC < a+0.5) & (np.abs(rC) < 0.25); mf = (segF == 2) & (np.abs(rF) < 0.25)
        if m.sum() > 30:
            mu, s, f = fit_width(rC[m], sc[C][m], rF[mf], 0.25); rows.append(("RGB", a+0.25, m.sum(), mu, s, f, float(np.median(sc[C][m])), 1.4826*float(np.median(np.abs(rC[m]-np.median(rC[m]))))))
    # TO/SGB widths: colour width = MS intrinsic at G0 17.7; magnitude width calibrated so that the control-subtracted completeness of
    # the cluster in G0 16.9-17.6 at k = 2 equals erf(2/sqrt 2) = 0.954
    M.P["s_c_sgb"] = float([r_[4] for r_ in rows if r_[0] == "MS"][0])
    mS = (gC > G_RGB) & (gC < G_MS); mSF = (g0[F] > G_RGB) & (g0[F] < G_MS); wF = np.sum(C & (c0 > 1.25))/max(np.sum(F & (c0 > 1.25)), 1)
    for sGi in np.arange(0.02, 0.41, 0.01):
        M.P["s_G_sgb"] = float(sGi); zc = M.sgb_dist(cC[mS], gC[mS], sc[C][mS], sg[C][mS]); zf = M.sgb_dist(c0[F][mSF], g0[F][mSF], sc[F][mSF], sg[F][mSF])
        frac = (np.sum(zc < 2)-wF*np.sum(zf < 2))/max(mS.sum()-wF*mSF.sum(), 1)
        if frac >= erf(2/np.sqrt(2)):
            break
    rows.append(("SGB", 17.25, int(mS.sum()), 0., float(M.P["s_G_sgb"]), 0., float(np.median(sg[C][mS])), float("nan")))
    print(f"  TO/SGB: colour width {M.P['s_c_sgb']:.3f}, magnitude width {M.P['s_G_sgb']:.2f} (completeness {frac:.3f})", flush=True)
    for r_ in rows:
        print(f"  {r_[0]:3s} G0 {r_[1]:5.2f} N {r_[2]:5d}: mu {r_[3]:+.3f}, s_int {r_[4]:.3f}, f {r_[5]:.2f}, sigma_phot {r_[6]:.3f}, s_obs {r_[7]:.3f}")
    msr = [r_ for r_ in rows if r_[0] == "MS"]; Gm = np.array([r_[1] for r_ in msr]); Sm = np.array([r_[4] for r_ in msr])
    s_ms = np.polyfit(Gm, Sm, 1, w=np.sqrt([r_[2] for r_ in msr]))
    M.P.update(s_ms=list(s_ms), s_ms_min=float(Sm.min()), s_rgb=float(np.median([r_[4] for r_ in rows if r_[0] == "RGB"])))
    np.savez(ROOT/"results/plot_data/gaia_cmd_mask_errors.npz", gcen=gcen, med_sc=med_sc, med_sg=med_sg)
    json.dump(dict(M.P, k=k_def, widths=[dict(seg=r_[0], G0=r_[1], N=r_[2], mu=r_[3], s_int=r_[4], f=r_[5], sigma_phot=r_[6], s_obs=r_[7]) for r_ in rows],
                   floors=FLOOR, col_shift=COL_SHIFT, sig_dm=SIG_DM), open(ROOT/"results/plot_data/gaia_cmd_mask.json", "w"), indent=1, default=float)
    # cluster completeness
    zC = M.z(cC, gC, sc[C], sg[C])
    # stream
    import gaia_xsel_marked as GX
    from gaia_cross_selection import KNOTS
    from gaia_footprint_plots import gc_masked
    coef = json.loads((ROOT/"results/plot_data/gaia_cmd_shift.json").read_text())["dm_relation"]["coef"]; dmf = lambda p: np.minimum(np.polyval(coef, p), 0.)
    lg = np.where(raw["l"] > 180, raw["l"]-360, raw["l"]); keep = ~gc_masked(lg, raw["b"].astype(float))
    u, x, pa, pd = raw["phi1"].astype(float), raw["dphi2"].astype(float), raw["pmra"].astype(float), raw["pmdec"].astype(float)
    K = np.array(KNOTS); ta = GX.track(GX.MARK["pmra"], lambda q: np.interp(q, K[:, 0], K[:, 1])); td = GX.track(GX.MARK["pmdec"], lambda q: np.interp(q, K[:, 0], K[:, 2]))
    ml = np.array(GX.MARK["lower"]); tl = GX.track(GX.MARK["lower"], lambda q: np.where(q < ml[0, 0], 0., ml[-1, 1])+0.*q)
    pm = keep & (np.abs(pa-ta(u)) < 2.0) & (np.abs(pd-td(u)) < 1.5) & np.isfinite(c0)
    on = pm & (np.abs(x-tl(u)) < 1.5); off = pm & (np.abs(x-tl(u)) > 4) & (np.abs(x-tl(u)) < 12); sel = on | off
    ctl = (c0 > 1.25) & (c0 < 2.5) & (g0 < 20)
    # refit dm with this ridge: per signal bin, the dm at which the background-subtracted mean MS colour residual (stars with
    # observed G < 19.6 and template-frame G0 >= 17.8, |r| < 0.15) crosses zero (linear interpolation on a 0.02-mag grid);
    # error from 200 bootstrap resamplings of the on and off stars. (Maximising the excess S/N inside the 2-sigma mask was
    # nearly flat in dm and gave biased shifts.)
    dgrid = np.arange(-1.2, 0.31, 0.02); fits = {}; rng = np.random.default_rng(11)
    def ms_offset(ci, gi, wi, dms_):
        out = []
        for d_ in dms_:
            Gt = gi-d_; m = (Gt >= 17.8) & (gi < 19.6); r = ci[m]-(M.ms_fn(Gt[m])+np.polyval(M.P["dc_ms"], np.minimum(Gt[m], 19.6))); w = wi[m]*(np.abs(r) < 0.15)
            out.append(np.sum(w*r)/max(np.sum(w), 1))
        out = np.array(out); j = np.where(np.diff(np.sign(out)) != 0)[0]
        if len(j) == 0:
            return np.nan
        j = j[0]; return float(dms_[j]-out[j]*(dms_[j+1]-dms_[j])/(out[j+1]-out[j]))
    for a0, a1 in ((14, 18), (18, 22), (22, 26)):
        k2 = (u >= a0) & (u < a1); s_ = np.sum(on & k2 & ctl)/max(np.sum(off & k2 & ctl), 1)
        io, iF = np.where(on & k2)[0], np.where(off & k2)[0]
        ci = np.concatenate((c0[io], c0[iF])); gi = np.concatenate((g0[io], g0[iF])); wi = np.concatenate((np.ones(len(io)), -s_*np.ones(len(iF))))
        best = ms_offset(ci, gi, wi, dgrid); bs = []
        for _ in range(200):
            jo, jf = rng.integers(0, len(io), len(io)), rng.integers(0, len(iF), len(iF))
            bs.append(ms_offset(np.concatenate((c0[io][jo], c0[iF][jf])), np.concatenate((g0[io][jo], g0[iF][jf])), wi, dgrid))
        err = float(np.nanstd(bs)); fits[(a0, a1)] = (best, max(err, 0.02), np.nan, None)
        print(f"  dm (MS colour centring) phi1 {a0}-{a1}: {best:+.2f} +- {err:.2f} (bootstrap; old relation {float(dmf(0.5*(a0+a1))):+.2f})", flush=True)
    X = np.array([0.5*(a+b) for a, b in fits]); Y = np.array([v[0] for v in fits.values()]); W = 1/np.array([v[1] for v in fits.values()])
    coef = list(np.polyfit(X, Y, 1, w=W)); dmf = lambda p: np.minimum(np.polyval(coef, p), 0.)
    print(f"  new relation dm = {coef[1]:+.3f} {coef[0]:+.4f} phi1 (capped at 0): " + ", ".join(f"phi1 {p}: {dmf(p):+.2f} ({5.43*10**(dmf(p)/5):.2f} kpc)" for p in (10, 16, 20, 24, 28)), flush=True)
    zz = np.full(len(u), np.inf); zz[sel] = M.z(c0[sel], g0[sel], sc_all[sel], sg_all[sel], dmf(u[sel])); zz[~np.isfinite(zz)] = np.inf
    json.dump(dict(json.loads((ROOT/"results/plot_data/gaia_cmd_mask.json").read_text()), dm_relation=dict(coef=coef, note="dm = min(polyval(coef, phi1), 0); MS colour centring"),
                   dm_fits={f"{a}-{b}": dict(dm=v[0], err=v[1], sn=v[2]) for (a, b), v in fits.items()}), open(ROOT/"results/plot_data/gaia_cmd_mask.json", "w"), indent=1, default=float)
    def counts(k2, kk):
        s_ = np.sum(on & k2 & ctl)/max(np.sum(off & k2 & ctl), 1); w_ = (zz < kk) if np.isfinite(kk) else np.ones(len(zz), bool)
        Nn = np.sum(on & k2 & w_); Nb = s_*np.sum(off & k2 & w_); return Nn-Nb, Nb, s_
    # --- figure
    fig = plt.figure(figsize=(24, 11), constrained_layout=True); gs = fig.add_gridspec(2, 5)
    cg_, gg2 = np.linspace(-0.2, 2.0, 441), np.linspace(12.0, 21.6, 481); CG, GG = np.meshgrid(cg_, gg2)
    def borders(ax, dm, Gref_shift, k, color, lw=0.9):
        """mask boundary z = k on a fine grid (photometric errors = median at the raw G = G0 + 0.3 of the stream field)."""
        e_c = np.interp(GG.ravel()+0.3, gcen, med_sc); e_g = np.interp(GG.ravel()+0.3, gcen, med_sg)
        zg = M.z(CG.ravel(), GG.ravel(), e_c, e_g, dm).reshape(CG.shape); zg[~np.isfinite(zg)] = 99
        ax.contour(cg_, gg2, zg, levels=[k], colors=color, linewidths=lw)
    a = fig.add_subplot(gs[0, 0]); a.plot(cC, gC, ",", color="k", alpha=0.35); a.plot(M.iso[1], M.iso[0], "C2-", lw=0.8, label="MIST 12.5 Gyr [Fe/H] -1.7 (+0.085)")
    borders(a, 0., 0, k_def, "C3"); a.plot([], [], "C3-", label=f"mask, k = {k_def:g} (cluster frame)")
    a.set(xlim=(0.2, 1.6), ylim=(20.3, 13.5), xlabel=r"(BP-RP)$_0$", ylabel=r"G$_0$", title=f"omega Cen 20-40', |dPM| < 0.7 ({C.sum()} stars)"); a.legend(fontsize=7)
    a = fig.add_subplot(gs[0, 1])
    for seg_, mk in (("MS", "o"), ("RGB", "s"), ("SGB", "^")):
        R_ = [r_ for r_ in rows if r_[0] == seg_]
        a.plot([r_[1] for r_ in R_], [r_[4] for r_ in R_], "k"+mk, label=f"{seg_} intrinsic" + (" (mag)" if seg_ == "SGB" else ""))
        a.plot([r_[1] for r_ in R_], [r_[6] for r_ in R_], "C1"+mk, mfc="none", label=f"{seg_} photometric")
        a.plot([r_[1] for r_ in R_], [r_[7] for r_ in R_], "C2"+mk, mfc="none", label=f"{seg_} observed (MAD)")
    gg = np.linspace(17.6, 21, 50); a.plot(gg, np.clip(np.polyval(M.P["s_ms"], gg), M.P["s_ms_min"], 0.12), "k-", lw=0.8)
    a.set(xlabel=r"G$_0$ (cluster frame)", ylabel="width [mag]", title="intrinsic (deconvolved) vs photometric widths"); a.legend(fontsize=6, ncol=2)
    a = fig.add_subplot(gs[0, 2]); Gq = np.arange(14, 19.81, 0.4)
    a.plot(Gq+0.2, [np.mean(zC[(gC >= q) & (gC < q+0.4)] < k_def) for q in Gq], "ko-"); a.axhline(erf(k_def/np.sqrt(2)), color="C0", ls="--", label=f"Gaussian, k = {k_def:g}")
    a.set(xlabel=r"G$_0$", ylabel="fraction inside the mask", ylim=(0.6, 1.02), title="completeness on the cluster sample (incl. ~1-7% field)"); a.legend(fontsize=7)
    k18 = (u >= 18) & (u < 22); ks = np.arange(0.75, 3.51, 0.25); SN = np.array([counts(k18, k_) for k_ in ks]); sn = SN[:, 0]/np.sqrt(np.maximum(SN[:, 1], 1))
    a = fig.add_subplot(gs[0, 3]); a.plot(ks, sn, "ko-"); a.set(xlabel="k", ylabel="(N - B)/sqrt(B)", title=r"S/N vs k, $\phi_1$ 18-22"); a2 = a.twinx(); a2.plot(ks, SN[:, 0], "C0s--"); a2.set_ylabel("excess", color="C0")
    print("k: S/N, excess, B (phi1 18-22): " + "; ".join(f"{k_:.2f}: {s_:.1f}, {e_:.0f}, {b_:.0f}" for k_, s_, e_, b_ in zip(ks, sn, SN[:, 0], SN[:, 1])), flush=True)
    a = fig.add_subplot(gs[0, 4]); a.axis("off")
    EC, EG = np.arange(-0.2, 2.01, 0.04), np.arange(12, 21.51, 0.1)
    for i, (a0, a1) in enumerate(BINS):
        a = fig.add_subplot(gs[1, i]); k2 = (u >= a0) & (u < a1); ex, B_, s_ = counts(k2, k_def); exA, BA, _ = counts(k2, np.inf)
        Hn = np.histogram2d(c0[on & k2], g0[on & k2], (EC, EG))[0]; Hf = np.histogram2d(c0[off & k2], g0[off & k2], (EC, EG))[0]*s_
        a.imshow(gaussian_filter(Hn-Hf, 1.).T, origin="lower", extent=(EC[0], EC[-1], EG[0], EG[-1]), aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
        dmb = float(dmf(0.5*(a0+a1))); borders(a, dmb, 0, k_def, "k", 0.8)
        a.set(xlim=(-0.2, 2.0), ylim=(21.5, 12), xlabel=r"(BP-RP)$_0$", title=f"$\\phi_1$ {a0}-{a1}, dm {dmb:+.2f}\nexcess in mask {ex:.0f} of {exA:.0f}; S/N {ex/np.sqrt(max(B_, 1)):.1f} (no CMD cut {exA/np.sqrt(max(BA, 1)):.1f})")
        print(f"phi1 {a0}-{a1}: in mask {ex:.0f} (B {B_:.0f}, S/N {ex/np.sqrt(max(B_, 1)):.1f}); no CMD cut {exA:.0f} (B {BA:.0f}, S/N {exA/np.sqrt(max(BA, 1)):.1f})", flush=True)
    fig.suptitle(f"Isochrone-based CMD mask (MIST + omega Cen offsets per segment; width^2 = s_int^2 [deconvolved] + photometric^2 + sigma_dm^2), k = {k_def:g}; "
                 "bottom: CMD excess (sky + PM masks, off-sky normalised on red control)")
    fig.savefig(ROOT/"plots/gaia_cmd_mask.png", dpi=95); plt.close(fig); print("plots/gaia_cmd_mask.png")


if __name__ == "__main__":
    main()


G_BRIGHT = 17.0     # template-frame G0 limit (user 2026-10-23: 16.5 -> 17.0; the SGB added mostly background): brighter stars excluded


def dm_of(P, u):
    """dm(phi1) from gaia_cmd_mask.json dm_relation: form 'anchored_linear' -> slope x clip(phi1, phi1_min, phi1_max) (dm = 0 at
    the cluster, held constant outside the measured range; 2026-10-23 iteration); legacy -> min(polyval(coef, phi1), 0)."""
    R = P["dm_relation"]; u = np.asarray(u, float)
    if R.get("form") == "poly":                                   # free low-degree polynomial on clip(phi1, phi1_lo, phi1_hi)
        return np.polyval(R["coef"], np.clip(u, R["phi1_lo"], R["phi1_hi"]))
    if R.get("form") == "anchored_linear":
        return R["slope"]*np.clip(u, R.get("phi1_min", 0.), R["phi1_max"])
    return np.minimum(np.polyval(R["coef"], u), 0.)


def cmd_pass(k=2.0, g_bright=G_BRIGHT):
    """Boolean over the raw footprint catalogue (stars brighter than template-frame G0 = g_bright, i.e. the RGB, excluded) (results/streams/gaia_footprint_dr3.npz order): inside the CMD mask (k, dm relation
    and segment parameters from results/plot_data/gaia_cmd_mask.json; photometric errors = median sigma(G) of the footprint).
    Cached in results/streams/gaia_cmd_pass_<tag>.npy, keyed by a hash of the json content (was its mtime in integer seconds:
    two writes within a second returned a stale cache)."""
    import hashlib
    txt = (ROOT/"results/plot_data/gaia_cmd_mask.json").read_text(); P = json.loads(txt)
    tag = f"k{k:g}_gb{g_bright:g}_{hashlib.sha1(txt.encode()).hexdigest()[:12]}"
    f = ROOT/f"results/streams/gaia_cmd_pass_{tag}.npy"
    if f.exists():
        return np.load(f)
    raw = np.load(ROOT/"results/streams/gaia_footprint_dr3.npz"); z0 = np.load(ROOT/"results/streams/gaia_footprint_dered.npz")
    assert np.array_equal(z0["source_id"], raw["source_id"])
    g0, c0, graw, u = z0["g0"].astype(float), z0["c0"].astype(float), raw["g"].astype(float), raw["phi1"].astype(float)
    Z = np.load(ROOT/"results/plot_data/gaia_cmd_mask_errors.npz"); sc = np.interp(graw, Z["gcen"], Z["med_sc"]); sg = np.interp(graw, Z["gcen"], Z["med_sg"])
    M = CMDMask(P); dm = dm_of(P, u)
    out = np.zeros(len(u), bool); ok = np.isfinite(c0) & np.isfinite(g0)
    for i in range(0, len(u), 200_000):
        s_ = slice(i, i+200_000); m = ok[s_]; z = np.full(m.size, np.inf)
        z[m] = M.z(c0[s_][m], g0[s_][m], sc[s_][m], sg[s_][m], dm[s_][m]); out[s_] = (z < k) & (g0[s_]-dm[s_] >= g_bright)
    np.save(f, out); return out
