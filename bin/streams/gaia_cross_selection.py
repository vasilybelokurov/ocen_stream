#!/usr/bin/env python3
"""Cross-selection view of the omega Cen stream in Gaia DR3, in the spirit of Koposov et al. (2019, MNRAS 485, 4726), Fig. 2:
each projection shows the stars selected by the masks of the OTHER projections (2026-10-23).

Data: results/streams/gaia_footprint_dr3.npz (G < 20, parallax - 3 sigma < 0.5 mas), globular clusters masked as in
gaia_footprint_plots.py. Common cuts only: those.
Tracks (natural cubic splines in phi1), as in Koposov+2019 placed by hand on the data, not on the STREAMFINDER members:
  PM track: knots every 4 deg read off the background-subtracted PM maps (gaia_footprint_plots.py), each refined by the
    background-subtracted centroid within +-2 mas/yr when the excess is >= 3 sigma (refine_pm). (An automatic ridge-follower
    was tried first and was unreliable at significance ~2-3 per bin.)
  Sky track: dphi2 = 0 of the chord frame; panel 1 shows where the PM-selected ridge actually lies.
Masks: |pmra - track| < 1.5, |pmdec - track| < 1.5 mas/yr, |dphi2 - ridge| < 2 deg.
Panels vs phi1: (1) dphi2 selected by both PM masks; (2) pmra* selected by the sky and pmdec masks; (3) pmdec selected by the
sky and pmra* masks; counts divided by their median in each phi1 column (left) and the same minus a background from the unselected dimension
(dphi2: quadratic fit to the off-ridge bands 4-12 deg on both sides; PMs: the same PM-selected stars from sky bands 4-12 deg
off the ridge, area-scaled). (4) CMDs of stars passing all three masks minus the same selection shifted 4-12 deg off the ridge
on the sky (area-scaled), in phi1 ranges. Third column of gaia_xsel.png: the masks with the STREAMFINDER members only.
Output: plots/gaia_xsel.png, plots/gaia_xsel_cmd.png, results/plot_data/gaia_xsel_tracks.json
Usage: PYTHONPATH=src python bin/streams/gaia_cross_selection.py [--dphi1 1.0]  (phi1 pixel; default 0.5 deg = 80 pixels;
       other values write plots/gaia_xsel_<N>px.png and skip the CMD figure)
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import CubicSpline
from gaia_footprint_plots import gc_masked

W_PM, W_SKY = 1.5, 2.0
BW = 2.0; C1 = np.arange(-6, 31, BW)+BW/2                        # 2-deg phi1 bin centres


def load():
    D = dict(np.load(ROOT/"results/streams/gaia_footprint_dr3.npz"))
    lg = np.where(D["l"] > 180, D["l"]-360, D["l"]).astype(float); keep = ~gc_masked(lg, D["b"].astype(float))
    return {k: v[keep].astype(np.float64) if v.dtype.kind == "f" else v[keep] for k, v in D.items()}


# hand-placed PM knots (phi1, pmra*, pmdec) read off the background-subtracted maps plots/gaia_fp_pm.png and gaia_fp_vpd.png
KNOTS = [(-4, -4.3, -6.3), (0, -4.4, -6.5), (4, -5.0, -6.9), (8, -5.8, -7.3), (12, -6.8, -7.7), (16, -8.0, -8.4),
         (20, -10.5, -9.6), (24, -13.5, -10.9), (28, -17.0, -12.0), (32, -20.0, -12.8)]


def refine_pm(u, x, pa, pd, win=2.0, smin=3.):
    """Refine each hand-placed knot by the background-subtracted centroid of stars at |dphi2| < 3 within +-win mas/yr in a 4-deg
    phi1 bin (background: the same PM window in the 4-12 deg sky bands on both sides, scaled by the ratio of all stars on/off).
    Keep the refinement only if the excess is >= smin sigma; otherwise keep the hand-placed value."""
    out = {}
    for c, a0, d0 in KNOTS:
        k = np.abs(u-c) < 2; on = k & (np.abs(x) < 3); off = k & (np.abs(x) > 4) & (np.abs(x) < 12)
        r = on.sum()/max(off.sum(), 1)
        w_on = on & (np.abs(pa-a0) < win) & (np.abs(pd-d0) < win); w_off = off & (np.abs(pa-a0) < win) & (np.abs(pd-d0) < win)
        ex = w_on.sum()-r*w_off.sum(); sig = ex/np.sqrt(max(r*w_off.sum(), 1))
        if sig >= smin:
            ma = (pa[w_on].sum()-r*pa[w_off].sum())/ex; md = (pd[w_on].sum()-r*pd[w_off].sum())/ex
            ma, md = float(np.clip(ma, a0-1, a0+1)), float(np.clip(md, d0-1, d0+1))
            out[c] = (ma, md, f"refined, excess {ex:.0f} stars ({sig:.1f} sigma)")
        else:
            out[c] = (a0, d0, f"hand-placed (excess {ex:.0f}, {sig:.1f} sigma)")
    return out


def spline(cs, vals, sig, smin=3.):
    cs, vals, sig = map(np.asarray, (cs, vals, sig)); ok = sig >= smin
    return CubicSpline(cs[ok], vals[ok], bc_type="natural"), ok


def main():
    from grid4_common import du, dx, d
    D = load(); u, x, pa, pd = D["phi1"], D["dphi2"], D["pmra"], D["pmdec"]; print(f"{len(u)} stars", flush=True)
    peaks = {c: (a0, d0, "hand-placed") for c, a0, d0 in KNOTS}   # refine_pm's field-ratio background proved biased near the cluster
    cs = np.array(list(peaks)); sa = CubicSpline(cs, [p[0] for p in peaks.values()], bc_type="natural")
    sd = CubicSpline(cs, [p[1] for p in peaks.values()], bc_type="natural")
    for c, p in peaks.items():
        print(f"phi1 {c:5.1f}: PM knot ({p[0]:6.2f}, {p[1]:6.2f})  {p[2]}")
    pmsel = (np.abs(pa-sa(u)) < W_PM) & (np.abs(pd-sd(u)) < W_PM)
    sx = lambda q: np.zeros_like(np.asarray(q, float))                                  # sky mask around the frame track
    json.dump(dict(pm_knots={f"{c:g}": p for c, p in peaks.items()}, w_pm=W_PM, w_sky=W_SKY, sky_track="dphi2 = 0 (chord frame)"),
              open(ROOT/"results/plot_data/gaia_xsel_tracks.json", "w"), indent=1)
    skysel = np.abs(x-sx(u)) < W_SKY; selA = np.abs(pa-sa(u)) < W_PM; selD = np.abs(pd-sd(u)) < W_PM
    offsky = (np.abs(x-sx(u)) > 4) & (np.abs(x-sx(u)) < 12)
    # area scaling of the off-sky bands (fraction of the 16-deg band inside the footprint vs the 4-deg on band), per phi1 bin
    dp = float(sys.argv[sys.argv.index("--dphi1")+1]) if "--dphi1" in sys.argv else 0.5          # phi1 pixel [deg]
    E1 = np.arange(-8, 32.01, dp); c1 = 0.5*(E1[1:]+E1[:-1]); tag = "" if dp == 0.5 else f"_{len(c1)}px"
    n_on = np.histogram(u[skysel], E1)[0]; n_off = np.histogram(u[offsky], E1)[0]
    ratio = np.divide(n_on, n_off, out=np.zeros(len(c1)), where=n_off > 0)               # field ratio (stream < 1 per cent of all stars)
    fig, ax = plt.subplots(3, 3, figsize=(21, 12.5), constrained_layout=True, sharex=True)
    ph = np.linspace(-6, 31, 300)
    rows = (("dphi2", x, pmsel, np.arange(-15, 15.01, 0.25), r"$\Delta\phi_2$ [deg]", sx, W_SKY, "both PM masks"),
            ("pmra", pa, skysel & selD, np.arange(-28, 4.01, 0.25), r"$\mu_{\alpha*}$ [mas/yr]", sa, W_PM, r"sky + $\mu_\delta$ masks"),
            ("pmdec", pd, skysel & selA, np.arange(-18, 2.01, 0.25), r"$\mu_\delta$ [mas/yr]", sd, W_PM, r"sky + $\mu_{\alpha*}$ masks"))
    for i, (q, v, sel, Ey, yl, tr, w, lab) in enumerate(rows):
        H = np.histogram2d(u[sel], v[sel], (E1, Ey))[0]
        if q == "dphi2":                                                                  # background: linear fit to the off-ridge bands per column
            cy = 0.5*(Ey[1:]+Ey[:-1]); B = np.zeros_like(H)
            for j, c in enumerate(c1):
                off = (np.abs(cy-sx(c)) > 4) & (np.abs(cy-sx(c)) < 12)
                if off.sum() > 10:
                    B[j] = np.clip(np.polyval(np.polyfit(cy[off], H[j, off], 2), cy), 0, None)
        else:
            Hoff = np.histogram2d(u[offsky & (selD if q == "pmra" else selA)], v[offsky & (selD if q == "pmra" else selA)], (E1, Ey))[0]
            B = Hoff*ratio[:, None]
        e = (E1[0], E1[-1], Ey[0], Ey[-1])
        Hs = gaussian_filter(H, 0.6)
        colmed = np.nanmedian(np.where(Hs > 0, Hs, np.nan), axis=1)[:, None]                 # raw counts / column median
        ax[i, 0].imshow((Hs/np.where(colmed > 0, colmed, np.nan)).T, origin="lower", extent=e, aspect="auto", cmap="Greys", vmin=0.5, vmax=3)
        ax[i, 1].imshow(gaussian_filter((H-B)/np.sqrt(np.maximum(B, 1)), 1.).T, origin="lower", extent=e, aspect="auto", cmap="RdBu_r", vmin=-4, vmax=4)
        for a_ in ax[i]:
            a_.fill_between(ph, tr(ph)-w, tr(ph)+w, color="C0", alpha=0.12 if a_ is not ax[i, 2] else 0.25, lw=0)
            a_.set_ylabel(yl); a_.set_ylim(Ey[0], Ey[-1])
        ax[i, 2].plot(du, {"dphi2": dx, "pmra": d["pmra"], "pmdec": d["pmdec"]}[q], ".", color="C2", ms=1.5, alpha=0.6)
        ax[i, 2].set_xlim(E1[0], E1[-1]); ax[i, 2].set_title(f"mask (blue) and STREAMFINDER members (green)", fontsize=10)
        ax[i, 0].set_title(f"{yl.split(' [')[0]}: stars passing the {lab} (counts / column median)", fontsize=10)
        ax[i, 1].set_title("same, minus background from the unselected dimension: (N - B)/sqrt(B)", fontsize=10)
    for a_ in ax[-1]:
        a_.set_xlabel(r"$\phi_1$ [deg]")
    fig.suptitle("Gaia DR3 (G < 20, parallax cut, GC masks): each panel selected by the masks of the other projections (blue bands); "
                 "right column: masks and STREAMFINDER members", fontsize=11)
    fig.savefig(ROOT/f"plots/gaia_xsel{tag}.png", dpi=105); plt.close(fig)
    if tag:
        return
    # CMDs: all three masks, on minus off-sky (field-ratio scaled)
    allsel = pmsel & skysel; offsel = pmsel & offsky; Ec, Eg = np.arange(0, 2.01, 0.05), np.arange(12, 20.01, 0.1)
    rng = ((-2, 6), (6, 14), (14, 20), (20, 28))
    fig, ax = plt.subplots(1, 4, figsize=(17, 5.5), constrained_layout=True, sharey=True)
    for a_, (a0, a1) in zip(ax, rng):
        k = (u >= a0) & (u < a1); s = np.sum(n_on[(c1 >= a0) & (c1 < a1)])/max(np.sum(n_off[(c1 >= a0) & (c1 < a1)]), 1)
        Hn = np.histogram2d(D["bp_rp"][allsel & k], D["g"][allsel & k], (Ec, Eg))[0]
        Hf = np.histogram2d(D["bp_rp"][offsel & k], D["g"][offsel & k], (Ec, Eg))[0]*s
        a_.imshow(gaussian_filter(Hn-Hf, 1).T, origin="lower", extent=(Ec[0], Ec[-1], Eg[0], Eg[-1]), aspect="auto", cmap="RdBu_r",
                  vmin=-3, vmax=3)
        a_.set(xlabel="BP - RP", title=f"$\\phi_1$ {a0}-{a1}: on {int(Hn.sum())}, background {Hf.sum():.0f}")
        a_.invert_yaxis() if a_ is ax[0] else None
    ax[0].set_ylabel("G")
    fig.suptitle("CMD of stars passing all three masks minus the same PM selection 4-12 deg off the sky ridge (scaled); no CMD cut, not dereddened")
    fig.savefig(ROOT/"plots/gaia_xsel_cmd.png", dpi=105); plt.close(fig)
    print("plots/gaia_xsel.png, plots/gaia_xsel_cmd.png")


if __name__ == "__main__":
    main()
