#!/usr/bin/env python3
"""CMD excess of the omega Cen stream in Gaia DR3 and a CMD mask built from omega Cen's own stars (2026-10-23).

Photometry dereddened with SFD x 0.86 (Schlafly & Finkbeiner 2011 recalibration) and Gaia-band coefficients per E(B-V)
R_G 2.740, R_BP 3.374, R_RP 2.035 (Casagrande & VandenBerg 2018; colour dependence ignored -- the same correction is applied to
cluster and stream stars, so only the differential reddening along the stream matters).
Cluster template: stars of the cached footprint (results/streams/gaia_footprint_dr3.npz, before GC masking) at 20-40 arcmin from
omega Cen with |PM - (-3.25, -6.75)| < 0.7 mas/yr, G < 20: 2-D histogram in (BP-RP)_0, G_0 (0.04 x 0.1 mag), smoothed; the CMD
mask is the region above the density level that encloses 95 per cent of the template stars, after smoothing by 1.5 bins; the main
sequence is extended below the cluster's G0 ~ 19.5 limit along a linear ridge (colour vs G0 fitted at 18.5-19.5) with the width at 19.4.
Dereddened photometry is cached in results/streams/gaia_footprint_dered.npz.
Stream excess: stars passing the sky + PM masks of bin/streams/gaia_xsel_marked.py (widened: pmra +-2, pmdec +-1.5 mas/yr,
sky +-1.5 deg about the lower branch) minus the same PM selection in sky bands 4-12 deg off the track on both sides, scaled by
the ratio of on/off counts in a red CMD control region ((BP-RP)_0 1.25-2.5, where the stream has no stars) with the same PM
and sky selection, per phi1 bin (area scaling over-subtracted near phi1 6-14).
Distance shift: per phi1 bin, the magnitude offset dm of the (unextended) template that maximises the Poisson likelihood of the
on-sample given (scaled off-sample + a x shifted template), compared only at G0 < 19.4 + dm where the shifted template is complete;
error from Delta lnL = 0.5. The ADOPTED relation dm(phi1) is a weighted linear fit to the three bins with a clear CMD excess (phi1 14-18, 18-22,
22-26; the user's choice), capped at dm = 0 (the cluster) where the extrapolation would place the stream beyond the cluster
(phi1 < ~11); the masks use the relation, extrapolated elsewhere (noisy bins wander to the grid edges).
Green curve: MIST 12.5 Gyr, [Fe/H] = -1.7 isochrone (minimint, Gaia EDR3 bands) at 5.43 kpc + dm, shifted in colour to the
cluster's ridge at G0 17.6-19.3 (MIST colour systematics).
Figures: plots/gaia_cmd_excess.png (per phi1 bin: excess Hess diagram with the shifted template mask; template panel);
results/plot_data/gaia_cmd_shift.json (dm per bin).
Usage: PYTHONPATH=src python bin/streams/gaia_cmd_excess.py [--dm0]  (--dm0: mask at the cluster's dm in all panels ->
       plots/gaia_cmd_excess_dm0.png; fits and relation unchanged, nothing else written)
"""
import json, os, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, shift as ndshift

EC, EG = np.arange(-0.2, 2.01, 0.04), np.arange(12, 21.51, 0.1)
DM0 = "--dm0" in sys.argv                                     # mask at the cluster's distance (dm = 0) in every panel
GLIM_T = 19.4                                                  # cluster template complete to G0 ~ 19.4 (crowding)
LEVEL = 0.95                                                   # template density level enclosing 95 per cent of the cluster stars
RG, RBP, RRP = 2.740, 3.374, 2.035
OC = (201.69683, -47.47957); MU = (-3.250, -6.746)
BINS = ((6, 14), (14, 18), (18, 22), (22, 26), (26, 30))


def deredden(D):
    from dustmaps.config import config
    config["data_dir"] = os.path.expanduser("~/data/dustmaps")
    from dustmaps.sfd import SFDQuery
    from astropy.coordinates import SkyCoord
    ebv = 0.86*SFDQuery()(SkyCoord(D["ra"].astype(float), D["dec"].astype(float), unit="deg"))
    return D["g"]-RG*ebv, D["bp_rp"]-(RBP-RRP)*ebv, ebv


def template(D, g0, c0):
    sep = np.degrees(np.arccos(np.clip(np.sin(np.radians(D["dec"]))*np.sin(np.radians(OC[1]))+np.cos(np.radians(D["dec"]))*np.cos(np.radians(OC[1]))
                                       *np.cos(np.radians(D["ra"]-OC[0])), -1, 1)))*60
    k = (sep > 20) & (sep < 40) & (np.hypot(D["pmra"]-MU[0], D["pmdec"]-MU[1]) < 0.7) & np.isfinite(c0)
    H = np.histogram2d(c0[k], g0[k], (EC, EG))[0]; Hs = gaussian_filter(H, 1.5)
    s = np.sort(Hs.ravel())[::-1]; cum = np.cumsum(s)/s.sum(); lev = s[np.searchsorted(cum, LEVEL)]
    M = Hs >= lev; Tfit = Hs.copy(); Tfit[:, 0.5*(EG[1:]+EG[:-1]) > GLIM_T] = 0; Tfit /= Tfit.sum()
    # extend the main sequence below the cluster's G0 ~ 19.5 limit (stream stars shifted up to ~1 mag brighter): ridge colour
    # linear in G0 from 18.5-19.5, mask half-width as at 19.3-19.5, profile copied
    cg = 0.5*(EG[1:]+EG[:-1]); cc = 0.5*(EC[1:]+EC[:-1]); r = (cg > 18.5) & (cg < 19.5)
    ridge = np.array([np.sum(cc*Hs[:, j])/np.sum(Hs[:, j]) for j in np.where(r)[0]]); p = np.polyfit(cg[r], ridge, 1)
    j0 = np.searchsorted(cg, 19.4); prof = Hs[:, j0]; m0 = M[:, j0]; c_ref = np.polyval(p, cg[j0])
    for j in range(j0+1, len(cg)):
        sh = (np.polyval(p, cg[j])-c_ref)/(EC[1]-EC[0])
        Hs[:, j] = np.clip(ndshift(prof, sh, order=1, mode="constant"), 0, None); M[:, j] = ndshift(m0.astype(float), sh, order=0, mode="constant") > 0.5
    return Tfit, M, k.sum()


def isochrone(age_gyr=12.5, feh=-1.7, d_kpc=5.43):
    """MIST (minimint, $OCEN_ISOCHRONES/minimint) Gaia EDR3 isochrone up to the RGB tip: (BP-RP, G) at distance d_kpc."""
    import minimint
    import ocen_stream  # noqa: F401  (loads .env)
    ii = minimint.Interpolator(["Gaia_G_EDR3", "Gaia_BP_EDR3", "Gaia_RP_EDR3"], data_prefix=os.path.join(os.path.expanduser(os.path.expandvars("$OCEN_ISOCHRONES")), "minimint"))
    m = np.linspace(0.1, ii.getMaxMass(np.log10(age_gyr*1e9), feh)-1e-4, 3000); r = ii(m, np.log10(age_gyr*1e9), feh)
    dmod = 5*np.log10(d_kpc*1e3)-5; ok = np.isfinite(r["Gaia_G_EDR3"])
    return (r["Gaia_BP_EDR3"]-r["Gaia_RP_EDR3"])[ok], r["Gaia_G_EDR3"][ok]+dmod


def main():
    import gaia_xsel_marked as GX
    from gaia_cross_selection import KNOTS
    from gaia_footprint_plots import gc_masked
    raw = dict(np.load(ROOT/"results/streams/gaia_footprint_dr3.npz")); raw = {k: (v.astype(np.float64) if v.dtype.kind == "f" else v) for k, v in raw.items()}
    cache = ROOT/"results/streams/gaia_footprint_dered.npz"
    if cache.exists():
        z = np.load(cache); g0, c0 = z["g0"], z["c0"]
    else:
        g0, c0, ebv = deredden(raw); np.savez(cache, source_id=raw["source_id"], g0=g0.astype(np.float32), c0=c0.astype(np.float32), ebv=ebv.astype(np.float32))
    g0, c0 = g0.astype(float), c0.astype(float)
    T, Tmask, nT = template(raw, g0, c0); print(f"cluster template: {nT} stars; mask covers {Tmask.sum()} cells", flush=True)
    lg = np.where(raw["l"] > 180, raw["l"]-360, raw["l"]); keep = ~gc_masked(lg, raw["b"])
    D = {k: v[keep] for k, v in raw.items()}; g0, c0 = g0[keep], c0[keep]
    u, x, pa, pd = D["phi1"], D["dphi2"], D["pmra"], D["pmdec"]
    K = np.array(KNOTS); ta = GX.track(GX.MARK["pmra"], lambda p: np.interp(p, K[:, 0], K[:, 1])); td = GX.track(GX.MARK["pmdec"], lambda p: np.interp(p, K[:, 0], K[:, 2]))
    ml = np.array(GX.MARK["lower"]); tl = GX.track(GX.MARK["lower"], lambda p: np.where(p < ml[0, 0], 0., ml[-1, 1])+0.*p)
    pm = (np.abs(pa-ta(u)) < 2.0) & (np.abs(pd-td(u)) < 1.5)
    on = pm & (np.abs(x-tl(u)) < 1.5); off = pm & (np.abs(x-tl(u)) > 4) & (np.abs(x-tl(u)) < 12)
    # footprint area of the on / off sky regions per bin (random points, same GC masks and b > 8)
    from ocen_stream.path import project_gc
    from branch_search_gaia import GC
    rng = np.random.default_rng(2); n = 6_000_000; l = rng.uniform(-110, 10, n); b = np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(-20)), np.sin(np.radians(75)), n)))
    ru, rx = project_gc(l, b, GC); ok = (b > 8) & ~gc_masked(l, b); ron = ok & (np.abs(rx-tl(ru)) < 1.5); roff = ok & (np.abs(rx-tl(ru)) > 4) & (np.abs(rx-tl(ru)) < 12)
    dms = np.arange(-1.5, 0.51, 0.02); out = {}; cg = 0.5*(EG[1:]+EG[:-1])
    fig, ax = plt.subplots(1, len(BINS)+2, figsize=(27, 6), constrained_layout=True)
    ax[0].imshow(np.log10(T+1e-9).T, origin="lower", extent=(EC[0], EC[-1], EG[0], EG[-1]), aspect="auto", cmap="Greys", vmin=-5, vmax=-2)
    ax[0].contour(0.5*(EC[1:]+EC[:-1]), cg, Tmask.T.astype(float), levels=[0.5], colors="C0")
    iso0 = isochrone(); cc_ = 0.5*(EC[1:]+EC[:-1]); r_ = (cg > 17.6) & (cg < 19.3)
    ridge = np.array([np.sum(cc_*T[:, j])/np.sum(T[:, j]) for j in np.where(r_)[0]])
    ms = iso0[1] > 17.0                                                   # main sequence below the turnoff (monotonic in G)
    o = np.argsort(iso0[1][ms]); dcol = float(np.median(ridge-np.interp(cg[r_], iso0[1][ms][o], iso0[0][ms][o])))
    ISO = (iso0[0]+dcol, iso0[1]); print(f"isochrone colour offset (cluster ridge - MIST, G0 17.6-19.3): {dcol:+.3f} mag", flush=True)
    ax[0].plot(*ISO, "-", color="C2", lw=1.2, label=f"MIST 12.5 Gyr, [Fe/H] -1.7, 5.43 kpc\n(colour {dcol:+.2f} to match the cluster)"); ax[0].legend(fontsize=7, loc="lower left")
    ax[0].set_ylim(EG[-1], EG[0])
    ax[0].set_title(f"omega Cen 20-40', PM-selected\n({nT} stars); 95% mask (blue)\nMS extended below G0 19.4", fontsize=9); ax[0].set(xlabel=r"(BP-RP)$_0$", ylabel=r"G$_0$")
    ctl = (c0 > 1.25) & (c0 < 2.5) & (g0 < 20)                          # red control: no stream stars, same PM and sky selection
    hists = {}
    for (a0, a1) in BINS:
        k = (u >= a0) & (u < a1); s = np.sum(on & k & ctl)/max(np.sum(off & k & ctl), 1)
        Hn = np.histogram2d(c0[on & k], g0[on & k], (EC, EG))[0]; Hf = np.histogram2d(c0[off & k], g0[off & k], (EC, EG))[0]*s
        Bs = gaussian_filter(Hf, 1.0); prof = []
        for dm in dms:                                                    # compare only where the shifted template is complete
            Tsh = np.clip(ndshift(T, (0, dm/0.1), order=1, mode="constant"), 0, None); v = (cg < GLIM_T+dm)[None, :] & np.ones_like(Hn, bool)
            n_ex = max(Hn[v].sum()-Bs[v].sum(), 1.); best = -np.inf
            for A in np.linspace(0, 3*n_ex, 80):
                lam = Bs[v]+A*Tsh[v]/max(Tsh[v].sum(), 1e-12)+1e-6; L0 = np.sum(Hn[v]*np.log(Bs[v]+1e-6)-Bs[v]-1e-6)
                best = max(best, np.sum(Hn[v]*np.log(lam)-lam)-L0)
            prof.append(best)
        prof = np.array(prof); i = int(np.argmax(prof)); ok = prof >= prof[i]-0.5
        lo, hi = dms[ok].min(), dms[ok].max(); err = max(0.5*(hi-lo), 0.05)
        out[f"{a0}-{a1}"] = dict(dm=float(dms[i]), err=float(err), dlnL=float(prof[i]), center=0.5*(a0+a1))
        hists[(a0, a1)] = (Hn, Hf)
        print(f"phi1 {a0}-{a1}: on {int(Hn.sum())}, bg {Hf.sum():.0f}, dm {dms[i]:+.2f} +- {err:.2f}, dlnL {prof[i]:.1f}", flush=True)
    # smooth distance relation: dm(phi1) quadratic through the cluster (dm = 0 +- 0.05 at phi1 = 0) and bins with dlnL >= 10
    # adopted relation: weighted linear fit dm = a + b phi1 to the three bins with a clear CMD excess (phi1 14-18, 18-22, 22-26)
    good = [out[k] for k in ("14-18", "18-22", "22-26")]
    X = np.array([v["center"] for v in good]); Y = np.array([v["dm"] for v in good]); E = np.array([v["err"] for v in good])
    coef, cov = np.polyfit(X, Y, 1, w=1/E, cov="unscaled"); dmf = lambda p: np.minimum(np.polyval(coef, p), 0.)   # capped at the cluster's dm
    print(f"dm(phi1) = {coef[1]:+.3f} + {coef[0]:+.4f} phi1  (errors {np.sqrt(cov[1,1]):.3f}, {np.sqrt(cov[0,0]):.4f}); "
          f"d = 5.43 x 10^(dm/5): " + ", ".join(f"phi1 {p}: {5.43*10**(dmf(p)/5):.2f} kpc" for p in (0, 10, 16, 20, 24, 28)), flush=True)
    iso = ISO
    for a_, (a0, a1) in zip(ax[1:-1], BINS):
        Hn, Hf = hists[(a0, a1)]; dm = 0. if DM0 else float(dmf(0.5*(a0+a1))); r = out[f"{a0}-{a1}"]
        a_.imshow(gaussian_filter(Hn-Hf, 1.).T, origin="lower", extent=(EC[0], EC[-1], EG[0], EG[-1]), aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
        a_.contour(0.5*(EC[1:]+EC[:-1]), cg, ndshift(Tmask.astype(float), (0, dm/0.1), order=0, mode="constant").T, levels=[0.5], colors="k", linewidths=1)
        a_.plot(iso[0], iso[1]+dm, "-", color="C2", lw=1.2)
        a_.set_title(f"$\\phi_1$ {a0}-{a1}: on {int(Hn.sum())}, bg {Hf.sum():.0f}\nmask dm = {dm:+.2f} ({'cluster' if DM0 else 'relation'})\nfit {r['dm']:+.2f} $\\pm$ {r['err']:.2f}, $\\Delta$lnL {r['dlnL']:.0f}", fontsize=9)
        a_.set(xlabel=r"(BP-RP)$_0$", ylim=(EG[-1], EG[0]))
    # dm vs phi1: fits, relation, model debris distances
    a_ = ax[-1]; ph = np.linspace(0, 31, 100)
    for v in out.values():
        a_.errorbar(v["center"], v["dm"], yerr=v["err"], fmt="o", color="k" if v["dlnL"] >= 10 else "0.7")
    a_.plot(ph, dmf(ph), "C0-", label="adopted relation: linear fit to the 3 black points (capped at 0)")
    a_.plot(ph, np.polyval(coef, ph), "C0:", lw=0.8)
    a_.axhline(0, color="0.6", lw=0.6)
    from grid4_common import project_gc as pg, GC as GC2
    m = dict(np.load(ROOT/"results/streams/profile_step2_20261020/runs/final_a12.5_3.npz")); mu_, mx_ = pg(m["l"], m["b"], GC2)
    kk = (m["chi"] > 0) & (m["age"] < 700) & (np.abs(mx_) < 2)
    e = np.arange(0, 32, 2.); md = [np.median(m["d"][kk & (mu_ >= a) & (mu_ < a+2)]) for a in e]
    a_.plot(e+1, 5*np.log10(np.array(md)/5.43), "C3--", label="best model debris (12.5 deg), 5 log(d/5.43)")
    a_.set(xlabel=r"$\phi_1$ [deg]", ylabel="dm [mag] (relative to omega Cen)", title="distance shift along the stream", ylim=(-1.6, 0.4)); a_.legend(fontsize=8)
    fig.suptitle("CMD excess (stars in sky + PM masks minus off-sky bands, normalised on a red control region), dereddened; black: omega Cen 95% template at the adopted dm(phi1)" if not DM0 else "CMD excess, dereddened; black: omega Cen 95% template at the cluster's distance (dm = 0) in every panel")
    fig.savefig(ROOT/f"plots/gaia_cmd_excess{'_dm0' if DM0 else ''}.png", dpi=100); plt.close(fig)
    if DM0:
        return
    json.dump(dict(bins=out, dm_relation=dict(coef=list(map(float, coef)), note="dm = polyval(coef, max(phi1, 0))"), template=dict(n=int(nT), level="95 per cent", radius_arcmin=[20, 40], pm_tol=0.7),
                   dereddening="SFD x 0.86; R_G 2.740, R_BP 3.374, R_RP 2.035"), open(ROOT/"results/plot_data/gaia_cmd_shift.json", "w"), indent=1)
    np.savez_compressed(ROOT/"results/plot_data/gaia_cmd_template.npz", T=T, mask=Tmask, ec=EC, eg=EG)
    print("plots/gaia_cmd_excess.png")


if __name__ == "__main__":
    main()
