#!/usr/bin/env python3
"""Observable count cube of the omega Cen stream for model comparison (2026-10-24; format: docs/reviews/
observable_format_fable_codex_20261023.md, without CMD/magnitude forward modelling on the model side -- user).

Stars: footprint (G < 20, parallax cut, GC masks) passing the CMD mask (gaia_cmd_mask.cmd_pass, k = 2, current dm relation).
Per 2-deg phi1 column (-2..30): counts N in cells of (y, v_ra, v_de) with y = dphi2 - sky centre, v = pm - track (tracks from
results/plot_data/stream_masks.json, evaluated per star), y in [-6, 6] (0.25 deg), v_ra in [-5, 5] (0.25), v_de in [-3, 3] (0.2).
Background B per cell: counts in 1-deg y bands at 7 <= |y| < 14 (same v cells, smoothed by 1.5 cells in v), quadratic in y
(constant where one side is empty), evaluated at the cell's y and scaled to its width; B_err from the fit covariance.
Outputs: results/plot_data/obs_cube.npz (edges, N, B, B_err, column edges, track/sky tables used), plots/obs_cube.png.
Usage: PYTHONPATH=src python bin/streams/obs_cube.py
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

COLS = np.arange(-2, 30.01, 2.)
EY, ERA, EDE = np.arange(-6, 6.001, 0.25), np.arange(-5, 5.001, 0.25), np.arange(-3, 3.001, 0.2)
EB = np.arange(-14, 14.001, 1.); OFF = (7., 14.)


def main():
    from gaia_cmd_mask import cmd_pass
    from gaia_footprint_plots import gc_masked
    raw = dict(np.load(ROOT/"results/streams/gaia_footprint_dr3.npz")); lg = np.where(raw["l"] > 180, raw["l"]-360, raw["l"]).astype(float)
    keep = cmd_pass(2.0) & ~gc_masked(lg, raw["b"].astype(float))
    u, x = raw["phi1"][keep].astype(float), raw["dphi2"][keep].astype(float)
    R = json.loads((ROOT/"results/plot_data/stream_masks.json").read_text()); G = np.array(R["grid"])
    y = x-np.interp(u, G, R["sky_window"]["centre"])
    vra = raw["pmra"][keep].astype(float)-np.interp(u, G, R["tracks"]["pmra"]["c"]); vde = raw["pmdec"][keep].astype(float)-np.interp(u, G, R["tracks"]["pmdec"]["c"])
    cy, cb = 0.5*(EY[1:]+EY[:-1]), 0.5*(EB[1:]+EB[:-1]); offb = (np.abs(cb) >= OFF[0]) & (np.abs(cb) < OFF[1])
    shp = (len(COLS)-1, len(EY)-1, len(ERA)-1, len(EDE)-1); N, B, BE = np.zeros(shp), np.zeros(shp), np.zeros(shp)
    for k, (a0, a1) in enumerate(zip(COLS[:-1], COLS[1:])):
        m = (u >= a0) & (u < a1) & (np.abs(vra) < 5) & (np.abs(vde) < 3)
        N[k] = np.histogramdd(np.column_stack((y[m], vra[m], vde[m])), (EY, ERA, EDE))[0]
        Hb = np.histogramdd(np.column_stack((y[m], vra[m], vde[m])), (EB, ERA, EDE))[0]
        Hb = np.array([gaussian_filter(h, 1.5) for h in Hb])
        ob = offb & (Hb.sum((1, 2)) > 0); dg = 3 if ((ob & (cb < 0)).sum() >= 3 and (ob & (cb > 0)).sum() >= 3) else 1
        V = np.vander(cb[ob], dg); P = np.linalg.pinv(V); coef = np.tensordot(P, Hb[ob], 1)              # per 1-deg band
        Vy = np.vander(cy, dg); B[k] = np.clip(np.tensordot(Vy, coef, 1), 0, None)*(EY[1]-EY[0])
        res = Hb[ob]-np.tensordot(V, coef, 1); s2 = np.maximum((res**2).sum(0)/max(ob.sum()-dg, 1), 1e-6)  # per-cell residual variance
        BE[k] = np.sqrt(np.einsum("ij,jk,ik->i", Vy, P@P.T, Vy)[:, None, None]*s2[None])*(EY[1]-EY[0])
        print(f"phi1 {a0:+.0f}..{a1:+.0f}: N {N[k].sum():.0f}, B {B[k].sum():.0f}, excess {N[k].sum()-B[k].sum():+.0f}; "
              f"core (|y|<2, |v_ra|<2, |v_de|<1): {N[k][np.abs(cy) < 2][:, 12:28, 10:20].sum()-B[k][np.abs(cy) < 2][:, 12:28, 10:20].sum():+.0f}", flush=True)
    np.savez_compressed(ROOT/"results/plot_data/obs_cube.npz", N=N.astype(np.uint16), B=B.astype(np.float32), B_err=BE.astype(np.float32), phi1_edges=COLS,
                        y_edges=EY, vra_edges=ERA, vde_edges=EDE, grid=G, sky_centre=np.array(R["sky_window"]["centre"]),
                        pmra_track=np.array(R["tracks"]["pmra"]["c"]), pmdec_track=np.array(R["tracks"]["pmdec"]["c"]),
                        note="N ~ Poisson(A_col m + B); y = dphi2 - sky centre, v = pm - track (tracks on grid, linear interpolation)")
    # --- figure: projections of (N - B) / sqrt(B)
    cra, cde = 0.5*(ERA[1:]+ERA[:-1]), 0.5*(EDE[1:]+EDE[:-1]); c1 = 0.5*(COLS[1:]+COLS[:-1])
    sel = dict(y=(slice(None), slice(12, 28), slice(10, 20)), ra=(np.abs(cy) < 2, slice(None), slice(10, 20)), de=(np.abs(cy) < 2, slice(12, 28), slice(None)))
    fig = plt.figure(figsize=(18, 13), constrained_layout=True); gs = fig.add_gridspec(4, 4)
    for i, (key, ax_, lab, ce, sumax) in enumerate((("y", (1, 2), r"$\Delta\phi_2$ - centre [deg]", cy, (2, 3)), ("ra", (0, 2), r"$\mu_{\alpha*}$ - track [mas/yr]", cra, (1, 3)),
                                                     ("de", (0, 1), r"$\mu_\delta$ - track [mas/yr]", cde, (1, 2)))):
        a = fig.add_subplot(gs[i, :]); img = []
        for k in range(len(c1)):
            s = sel[key]; n, b = N[k][s[0]][:, s[1]][:, :, s[2]], B[k][s[0]][:, s[1]][:, :, s[2]]
            ax_sum = {"y": (1, 2), "ra": (0, 2), "de": (0, 1)}[key]; nn, bb = n.sum(ax_sum), b.sum(ax_sum); img.append((nn-bb)/np.sqrt(np.maximum(bb, 1)))
        a.imshow(gaussian_filter(np.array(img), (0, 1)).T, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-4, vmax=4,
                 extent=(COLS[0], COLS[-1], ce[0]-0.5*(ce[1]-ce[0]), ce[-1]+0.5*(ce[1]-ce[0])))
        a.set(ylabel=lab, xlabel=r"$\phi_1$ [deg]", title=f"(N - B)/sqrt(B) per 2-deg column; other coordinates: " +
              {"y": "|v_ra| < 2, |v_de| < 1", "ra": "|y| < 2, |v_de| < 1", "de": "|y| < 2, |v_ra| < 2"}[key]); a.axhline(0, color="k", lw=0.4, ls=":")
    for j, k in enumerate((1, 9, 10, 12)):
        a = fig.add_subplot(gs[3, j]); m = np.abs(cy) < 2; n, b = N[k][m].sum(0), B[k][m].sum(0)
        a.imshow(gaussian_filter((n-b)/np.sqrt(np.maximum(b, 1)), 1.).T, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3, extent=(ERA[0], ERA[-1], EDE[0], EDE[-1]))
        a.set(xlabel=r"$\mu_{\alpha*}$ - track", ylabel=r"$\mu_\delta$ - track", title=f"$\\phi_1$ {COLS[k]:.0f}-{COLS[k+1]:.0f}, |y| < 2: excess {(n-b).sum():.0f} on B {b.sum():.0f}")
    fig.suptitle("Observable count cube (CMD-selected Gaia DR3): background-subtracted projections; coordinates relative to the frozen tracks")
    fig.savefig(ROOT/"plots/obs_cube.png", dpi=90); print("plots/obs_cube.png")


if __name__ == "__main__":
    main()
