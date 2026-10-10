#!/usr/bin/env python3
"""Relative photometric distances along Ibata+2024 stream 54 from CMD shifts (dereddened G, BP-RP from the catalogue).

Reference: stream stars at b = 15-20 deg (next to omega Cen). For each b bin, fit the magnitude shift dm (bin = reference + dm)
maximising sum log p_ref(c - dc, G - dm) / N(dm) (dm and colour shift dc fitted jointly), where p_ref is a Gaussian KDE of the reference CMD and N(dm) = fraction of
reference stars with G + dm < G_lim (G_lim = 20, the catalogue's hard limit) -- the truncation correction. Bootstrap errors
(50 resamples of the bin; docstring corrected 2026-10-10). Distance ratio d/d_ref = 10^(dm/5). CMD window: 0.55 < BP-RP < 1.3, 15.5 < G < 20.
Usage: python bin/streams/stream54_cmd_distance.py  -> plots/stream54_cmd_distance.png, results/plot_data/stream54_cmd_distance.json
"""
import json, os
from pathlib import Path
import numpy as np, astropy.coordinates as coord, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.stats import gaussian_kde as _gkde
from scipy.interpolate import RegularGridInterpolator


def gaussian_kde(data, bw_method):
    """KDE evaluated once on a fine (colour, G) grid, then bilinearly interpolated (fast repeated evaluation)."""
    k = _gkde(data, bw_method=bw_method)
    cg, gg = np.linspace(0.3, 1.6, 261), np.linspace(13.5, 22.5, 451)
    C2, G2 = np.meshgrid(cg, gg, indexing="ij")
    z = k(np.vstack((C2.ravel(), G2.ravel()))).reshape(C2.shape)
    f = RegularGridInterpolator((cg, gg), z, bounds_error=False, fill_value=0.)
    return lambda x: f(np.asarray(x).T)

ROOT = Path(__file__).resolve().parents[2]
GLIM, CW, GW = 20.0, (0.55, 1.3), (15.5, 20.0)
BINS = [(20, 25), (25, 30), (30, 33), (33, 36), (36, 39), (39, 42)]
DM = np.arange(-1.5, 1.01, 0.02)
DC = np.arange(-0.10, 0.101, 0.01)        # colour shift (residual reddening / calibration), fitted jointly


def fit(ref, c, g, kde, joint=True):
    """Best (dm, dc) for stars (c, g) against the reference KDE: bin = reference shifted by dm in G and dc in colour."""
    best = (-np.inf, 0., 0.)
    for dc in (DC if joint else [0.]):
        for dm in DM:
            norm = np.mean(ref[1]+dm < GLIM)
            ll = np.sum(np.log(kde(np.vstack((c-dc, g-dm)))+1e-12)) - len(c)*np.log(norm)
            if ll > best[0]:
                best = (ll, dm, dc)
    return best[1], best[2]


def main():
    t = Table.read(os.path.expanduser("$OCEN_CATALOGUES/streamfinder_ibata2024_dr3.fits")); t = t[np.asarray(t["Stream"]) == 54]
    G = np.asarray(t["Gmag"], float); C = np.asarray(t["(B-R)"], float)
    b = coord.SkyCoord(np.asarray(t["RAdeg"], float), np.asarray(t["DEdeg"], float), unit="deg").galactic.b.deg
    win = (C > CW[0]) & (C < CW[1]) & (G > GW[0]) & (G < GW[1])
    r = win & (b >= 15) & (b < 20)
    ref = np.vstack((C[r], G[r])); kde = gaussian_kde(ref, bw_method=0.15)
    # self-test: reference split in halves must give dm ~ 0
    rng = np.random.default_rng(1); idx = np.where(r)[0]; h = rng.permutation(idx)
    kde_h = gaussian_kde(np.vstack((C[h[:len(h)//2]], G[h[:len(h)//2]])), bw_method=0.15)
    dm_self, dc_self = fit(np.vstack((C[h[:len(h)//2]], G[h[:len(h)//2]])), C[h[len(h)//2:]], G[h[len(h)//2:]], kde_h)
    print(f"reference b 15-20: N = {r.sum()};  split-half self-test dm = {dm_self:+.2f} mag, dc = {dc_self:+.2f} (expect 0)")
    # injection test: shift half the reference brighter by 0.5 mag and drop G > GLIM
    gi = G[h[len(h)//2:]]-0.5; ci = C[h[len(h)//2:]]-0.04; k = (gi < GLIM) & (gi > GW[0])
    dm_inj, dc_inj = fit(np.vstack((C[h[:len(h)//2]], G[h[:len(h)//2]])), ci[k], gi[k], kde_h)
    print(f"injection test: true dm = -0.50, dc = -0.04; recovered {dm_inj:+.2f}, {dc_inj:+.2f}")
    out = dict(ref_bin=[15, 20], n_ref=int(r.sum()), self_test=float(dm_self), injection_recovered=float(dm_inj), bins=[])
    fig, ax = plt.subplots(1, len(BINS)+1, figsize=(4*(len(BINS)+1), 4.5), sharey=True)
    ax[0].plot(C[r], G[r], ".", ms=2, color="0.3"); ax[0].set_title("reference b 15-20")
    for j, (lo, hi) in enumerate(BINS):
        k = win & (b >= lo) & (b < hi)
        dm0, _ = fit(ref, C[k], G[k], kde, joint=False)
        dm, dc = fit(ref, C[k], G[k], kde)
        boots = np.array([fit(ref, C[k][s], G[k][s], kde) for s in (rng.integers(0, k.sum(), k.sum()) for _ in range(50))])
        e, ec = float(np.std(boots[:, 0])), float(np.std(boots[:, 1])); ratio = 10**(dm/5)
        out["bins"].append(dict(b=[lo, hi], n=int(k.sum()), dm=float(dm), dm_err=e, dc=float(dc), dc_err=ec, dm_magonly=float(dm0), d_ratio=float(ratio),
                                d_kpc_if_ref_5p43=float(5.43*ratio)))
        print(f"b {lo}-{hi}: N = {k.sum():4d}  joint dm = {dm:+.2f} +- {e:.2f}, dc = {dc:+.3f} +- {ec:.3f}  (mag-only dm {dm0:+.2f})  d/d_ref = {ratio:.2f}  (d = {5.43*ratio:.2f} kpc if ref at 5.43)")
        a = ax[j+1]; a.plot(C[r]+dc, G[r]+dm, ".", ms=2, color="0.75", label=f"ref shifted dm {dm:+.2f}, dc {dc:+.2f}")
        a.plot(C[k], G[k], ".", ms=3, color="C3", label=f"b {lo}-{hi} (N={k.sum()})"); a.legend(fontsize=7, loc="upper left")
    for a in ax:
        a.set_xlim(*CW); a.set_ylim(GW[1], GW[0]); a.set_xlabel("(BP-RP)_0"); a.grid(alpha=0.3)
    ax[0].set_ylabel("G_0")
    fig.suptitle("Ibata+2024 stream 54: CMD shift relative to the b = 15-20 deg stars", fontsize=11)
    fig.tight_layout(); fig.savefig(ROOT/"plots/stream54_cmd_distance.png", dpi=80)
    (ROOT/"results/plot_data/stream54_cmd_distance.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
