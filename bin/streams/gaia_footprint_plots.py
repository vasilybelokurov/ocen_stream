#!/usr/bin/env python3
"""Visualise the omega Cen stream in Gaia DR3 with minimal selection (2026-10-23).

Data: results/streams/gaia_footprint_dr3.npz (bin/streams/gaia_footprint_query.py: chord frame phi1 -8..32, |dphi2| < 15, b > 8,
G < 20, parallax - 3 sigma < 0.5 mas; no PM, CMD or quality cuts). STREAMFINDER members (Ibata+2024, stream 54, b > 15) as
thin contours or points for comparison.
Globular clusters are masked in data and footprint alike (omega Cen 2 deg, others max(0.25 deg, 10 r_h)).
Figures:
  gaia_fp_sky.png      (a) raw counts in (phi1, dphi2), 0.25-deg pixels; (b) background-subtracted significance on 0.5-deg pixels:
                       B = Gaussian-smoothed counts (sigma 3 deg, edge-corrected), shown as (N - B)/sqrt(B), smoothed by 1 pixel.
  gaia_fp_pm.png       phi1 vs pmra* and pmdec for |dphi2| < 3 ("on") minus a background interpolated per (phi1, PM) pixel
                       by a quadratic in dphi2 fitted to 1-deg bands at 4 < |dphi2| < 15 (no PM cut); counts and (on - off)/sqrt(off).
  gaia_fp_vpd.png      PM plane (pmra*, pmdec) in 4-deg phi1 bins: on band minus the same quadratic background.
  gaia_fp_pmslices.png background-subtracted sky maps (as gaia_fp_sky b) for slices in total proper motion |mu|.
Usage: PYTHONPATH=src python bin/streams/gaia_footprint_plots.py
"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

E1 = np.arange(-8, 32.01, 0.25); E2 = np.arange(-15, 15.01, 0.25)


def gc_masked(l, b):
    """True for positions within a globular-cluster mask: omega Cen 2 deg, other clusters max(0.25 deg, 10 r_h)
    (catalogue $OCEN_CATALOGUES/gc_catalog_full.fits; RH in pc, DIST in kpc)."""
    import os
    import astropy.coordinates as coord
    from astropy.table import Table
    t = Table.read(os.path.expanduser(os.path.expandvars("$OCEN_CATALOGUES/gc_catalog_full.fits")))
    g = coord.SkyCoord(np.asarray(t["RA"], float), np.asarray(t["DEC"], float), unit="deg").galactic
    gl, gb = np.where(g.l.deg > 180, g.l.deg-360, g.l.deg), g.b.deg
    rad = np.maximum(0.25, np.degrees(10*np.asarray(t["RH"], float)/(1e3*np.asarray(t["DIST"], float))))
    rad[np.array([n.strip() for n in t["NAME"]]) == "NGC 5139"] = 2.0
    m = np.zeros(len(l), bool); lr, br = np.radians(l), np.radians(b)
    for L0, B0, r in zip(gl, gb, rad):
        near = (np.abs(b-B0) < r) & (np.abs(((l-L0)+180) % 360-180) < r/max(np.cos(np.radians(B0)), 0.05))
        if near.any():
            i = np.where(near)[0]
            c = np.sin(br[i])*np.sin(np.radians(B0))+np.cos(br[i])*np.cos(np.radians(B0))*np.cos(lr[i]-np.radians(L0))
            m[i[np.degrees(np.arccos(np.clip(c, -1, 1))) < r]] = True
    return m


def area_fraction():
    """Fraction of each (phi1, dphi2) pixel inside the footprint (b > 8), from a fine uniform grid on the sphere."""
    from ocen_dm.streams.path import project_gc
    from branch_search_gaia import GC
    n = 4_000_000; rng = np.random.default_rng(1)
    l = rng.uniform(-110, 10, n); sb = rng.uniform(np.sin(np.radians(-20)), np.sin(np.radians(75)), n); b = np.degrees(np.arcsin(sb))
    u, x = project_gc(l, b, GC); k = (b > 8) & ~gc_masked(l, b)
    H_all = np.histogram2d(u, x, (E1, E2))[0]; H_in = np.histogram2d(u[k], x[k], (E1, E2))[0]
    return np.divide(H_in, H_all, out=np.zeros_like(H_in), where=H_all > 20)


def bg_subtract(H, frac, deg=6, sigma_px=None):
    """Smooth background: log of the density (counts / footprint fraction) modelled as a 2-D Legendre series of total degree
    `deg` in (phi1, dphi2), fitted by weighted least squares (weights = counts) to pixels fully inside the footprint (frac > 0.95).
    Structures narrower than ~40/deg deg along phi1 (~6 deg for deg 6) are left in the residuals; the stream is 1-3 deg wide.
    Returns B (counts) and the significance (N - B)/sqrt(B)."""
    from numpy.polynomial import legendre as Lg
    c1 = np.linspace(-1, 1, H.shape[0]); c2 = np.linspace(-1, 1, H.shape[1]); X1, X2 = np.meshgrid(c1, c2, indexing="ij")
    terms = [(i, j) for i in range(deg+1) for j in range(deg+1-i)]
    A = np.stack([Lg.legval(X1, np.eye(deg+1)[i])*Lg.legval(X2, np.eye(deg+1)[j]) for i, j in terms], -1)
    ok = (frac > 0.95) & (H > 0); y = np.log(H[ok]/frac[ok]); w = np.sqrt(H[ok])
    coef = np.linalg.lstsq(A[ok]*w[:, None], y*w, rcond=None)[0]
    B = np.exp(A @ coef)*frac
    S = np.where(frac > 0.5, (H-B)/np.sqrt(np.maximum(B, 1)), np.nan)
    return B, S


def colsum(H, n):
    """Sum pixels in groups of n along phi1 (0.25 -> 0.25 n deg)."""
    return H[:H.shape[0]//n*n].reshape(-1, n, H.shape[1]).sum(1)


def main():
    from grid4_common import du, dx, d
    D = dict(np.load(ROOT/"results/streams/gaia_footprint_dr3.npz"))
    lg = np.where(D["l"] > 180, D["l"]-360, D["l"]).astype(float); keep = ~gc_masked(lg, D["b"].astype(float))
    print(f"globular-cluster masks remove {np.sum(~keep)} stars", flush=True); D = {k: v[keep] for k, v in D.items()}
    u, x = D["phi1"].astype(float), D["dphi2"].astype(float); pa, pd = D["pmra"].astype(float), D["pmdec"].astype(float)
    print(f"{len(u)} stars", flush=True)
    frac = area_fraction()
    ext = (E1[0], E1[-1], E2[0], E2[-1])
    # --- sky
    H = np.histogram2d(u, x, (E1, E2))[0]; Hc = colsum(H, 2); Fc = colsum(frac, 2)/2
    E1c = E1[::2][:Hc.shape[0]+1]
    Bc, Sc = bg_subtract(Hc, Fc, deg=6)
    fig, ax = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True, sharex=True)
    im = ax[0].imshow(np.log10(np.maximum(H, 1)).T, origin="lower", extent=ext, aspect="auto", cmap="Greys")
    fig.colorbar(im, ax=ax[0], label="log10 N per 0.25 deg pixel")
    im = ax[1].imshow(gaussian_filter(np.nan_to_num(Sc), 1).T, origin="lower", extent=(E1c[0], E1c[-1], E2[0], E2[-1]), aspect="auto",
                      cmap="RdBu_r", vmin=-4, vmax=4)
    fig.colorbar(im, ax=ax[1], label="(N - B)/sqrt(B), smoothed 1 px")
    for a_ in ax:
        a_.plot(du, dx, ",", color="C2", alpha=0.6); a_.set_ylabel(r"$\Delta\phi_2$ [deg]")
    ax[0].set_title("Gaia DR3, G < 20, parallax - 3$\\sigma$ < 0.5 mas (no PM/CMD cut); green: STREAMFINDER members")
    ax[1].set_title("significance after removing a smooth background (degree-6 Legendre series in log density)"); ax[1].set_xlabel(r"$\phi_1$ [deg]")
    fig.savefig(ROOT/"plots/gaia_fp_sky.png", dpi=110); plt.close(fig)
    # --- PM tracks: on minus area-scaled off
    # background in the on band (|dphi2| < 3) per (phi1, PM) pixel: quadratic in dphi2 fitted to 1-deg bands with 4 < |dphi2| < 15
    Eb = np.arange(-15, 15.01, 1.); cb = 0.5*(Eb[1:]+Eb[:-1]); offb = np.abs(cb) > 4; onb = np.abs(cb) < 3
    Ep = {"pmra": np.arange(-28, 6.01, 0.25), "pmdec": np.arange(-18, 4.01, 0.25)}; E1h = np.arange(-8, 32.01, 0.5)
    V = np.vander(cb[offb], 3); P = np.linalg.pinv(V); Von = np.vander(cb[onb], 3)
    c2 = 0.5*(E2[1:]+E2[:-1]); F2 = colsum(frac, 2)/2; fb = np.stack([F2[:, (c2 > lo) & (c2 < lo+1)].mean(1) for lo in Eb[:-1]], 1)   # (phi1, band) area fraction
    fig, ax = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True, sharex=True)
    for j, (q, v, lab) in enumerate((("pmra", pa, r"$\mu_{\alpha*}$"), ("pmdec", pd, r"$\mu_\delta$"))):
        H3 = np.histogramdd(np.column_stack((u, x, v)), (E1h, Eb, Ep[q]))[0]               # (phi1, band, PM)
        Hn = H3/np.maximum(fb, 0.05)[:, :, None]                                          # per unit footprint area
        coef = np.einsum("kb,ibp->ikp", P, Hn[:, offb, :])                                 # quadratic per (phi1, PM)
        Bon = np.clip(np.einsum("ib,bk,ikp->ip", fb[:, onb], Von, coef), 0, None); Non = H3[:, onb, :].sum(1)
        diff = Non-Bon; sig = diff/np.sqrt(np.maximum(Bon, 1))
        e = (E1h[0], E1h[-1], Ep[q][0], Ep[q][-1])
        im = ax[0, j].imshow(gaussian_filter(diff, 0.7).T, origin="lower", extent=e, aspect="auto", cmap="RdBu_r", vmin=-8, vmax=8)
        fig.colorbar(im, ax=ax[0, j], label="on - background (stars per 0.5 deg x 0.25 mas/yr)")
        im = ax[1, j].imshow(gaussian_filter(sig, 0.7).T, origin="lower", extent=e, aspect="auto", cmap="RdBu_r", vmin=-4, vmax=4)
        fig.colorbar(im, ax=ax[1, j], label="(on - background)/sqrt(background)")
        for a_ in ax[:, j]:
            a_.plot(du, d[q], ",", color="C2", alpha=0.5); a_.set_ylabel(f"{lab} [mas/yr]")
        ax[0, j].set_title(f"{lab}: |$\\Delta\\phi_2$| < 3 minus background (quadratic in $\\Delta\\phi_2$ from 4-15 deg bands; no PM cut)", fontsize=9)
        ax[1, j].set_xlabel(r"$\phi_1$ [deg]")
    fig.savefig(ROOT/"plots/gaia_fp_pm.png", dpi=110); plt.close(fig)
    # --- vector-point diagrams: on band minus interpolated background, per 4-deg phi1 bin
    Epa, Epd = np.arange(-26, 4.01, 0.5), np.arange(-16, 2.01, 0.5); bins = [(a, a+4) for a in range(-2, 30, 4)]
    fig, ax = plt.subplots(2, 4, figsize=(17, 8.5), constrained_layout=True, sharex=True, sharey=True)
    for a_, (a0, a1) in zip(ax.ravel(), bins):
        k = (u >= a0) & (u < a1)
        H3 = np.histogramdd(np.column_stack((x[k], pa[k], pd[k])), (Eb, Epa, Epd))[0]       # (band, pmra, pmdec)
        i0, i1 = np.searchsorted(E1h, a0), np.searchsorted(E1h, a1); fbb = fb[i0:i1].mean(0)
        Hn = H3/np.maximum(fbb, 0.05)[:, None, None]
        coef = np.einsum("kb,bpq->kpq", P, Hn[offb]); Bon = np.clip(np.einsum("b,bk,kpq->pq", fbb[onb], Von, coef), 0, None)
        Non = H3[onb].sum(0); sig = (Non-Bon)/np.sqrt(np.maximum(Bon, 1))
        a_.imshow(gaussian_filter(sig, 0.8).T, origin="lower", extent=(Epa[0], Epa[-1], Epd[0], Epd[-1]), aspect="auto", cmap="RdBu_r", vmin=-4, vmax=4)
        km = (du >= a0) & (du < a1) & (np.abs(dx) < 3); a_.plot(d["pmra"][km], d["pmdec"][km], ".", color="C2", ms=1.5, alpha=0.6)
        a_.set_title(f"$\\phi_1$ {a0}-{a1}: {k.sum()} stars, members {km.sum()}", fontsize=9)
    for a_ in ax[-1]:
        a_.set_xlabel(r"$\mu_{\alpha*}$ [mas/yr]")
    for a_ in ax[:, 0]:
        a_.set_ylabel(r"$\mu_\delta$ [mas/yr]")
    fig.suptitle("PM plane, |$\\Delta\\phi_2$| < 3 minus background interpolated from 4-15 deg bands: (on - B)/sqrt(B); green: STREAMFINDER members")
    fig.savefig(ROOT/"plots/gaia_fp_vpd.png", dpi=110); plt.close(fig)
    # --- total-PM slices
    mu = np.hypot(pa, pd); sl = ((0, 4), (4, 6), (6, 8), (8, 10), (10, 13), (13, 17), (17, 22), (22, 30))
    fig, ax = plt.subplots(4, 2, figsize=(16, 13), constrained_layout=True, sharex=True, sharey=True)
    for a_, (m0, m1) in zip(ax.ravel(), sl):
        k = (mu >= m0) & (mu < m1); Hk = colsum(np.histogram2d(u[k], x[k], (E1, E2))[0], 2); _, Sk = bg_subtract(Hk, Fc, deg=6)
        a_.imshow(gaussian_filter(np.nan_to_num(Sk), 1).T, origin="lower", extent=(E1c[0], E1c[-1], E2[0], E2[-1]), aspect="auto",
                  cmap="RdBu_r", vmin=-4, vmax=4)
        mm = np.hypot(d["pmra"], d["pmdec"]); km = (mm >= m0) & (mm < m1)
        a_.plot(du[km], dx[km], ".", color="C2", ms=1, alpha=0.5)
        a_.set_title(f"|$\\mu$| {m0}-{m1} mas/yr: {k.sum()} stars; members in slice (green) {km.sum()}", fontsize=9)
    for a_ in ax[-1]:
        a_.set_xlabel(r"$\phi_1$ [deg]")
    for a_ in ax[:, 0]:
        a_.set_ylabel(r"$\Delta\phi_2$ [deg]")
    fig.suptitle("Significance (N - B)/sqrt(B) in total-PM slices; B = smooth 2-D model (degree-6 Legendre series in log density); 0.5 x 0.25-deg pixels, smoothed 1 px")
    fig.savefig(ROOT/"plots/gaia_fp_pmslices.png", dpi=100); plt.close(fig)
    print("plots/gaia_fp_sky.png, gaia_fp_pm.png, gaia_fp_pmslices.png")


if __name__ == "__main__":
    main()
