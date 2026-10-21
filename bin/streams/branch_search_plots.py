#!/usr/bin/env python3
"""Readable figures for the Gaia DR3 branch search (bin/streams/branch_search_gaia.py; 2026-10-21).

Matched PM filters along phi1 (chord frame), applied to the cached Gaia query (results/streams/branch_search_gaia.npz) and to
the step-2 best models (12.5 and 25 deg) identically:
  track filter : |pmra - mu_T(phi1)| < 1.5, |pmdec - nu_T(phi1)| < 0.8 mas/yr, mu_T/nu_T = members' running median; CMD unshifted;
  branch filter: |pmra - mu_B(phi1)| < 2.5, |pmdec - nu_B(phi1)| < 1.2, mu_B/nu_B = running median of the 25-deg model's branch
                 particles (pmra < members' median - 3); CMD shifted brighter by 0.4 mag;
  control      : the same filter with pmra shifted by +5 (track) / -5 (branch) mas/yr, same sky, same CMD.
Common cuts: G 16-19.8, parallax (+0.017) consistent with [0.15, 0.27] at 2 sigma, members' raw-photometry CMD locus
(width max(0.05, 1.5 sigma)); STREAMFINDER members removed from the branch maps.
Model predictions: particles (chi > 0, age < 700 Myr) through the same PM filters, binned identically, scaled by
(observed track-filter excess) / (model track-filter count) inside |dphi2 - track| < 2.5, phi1 14-22.
Figure 1 (plots/branch_search_maps.png): rows track filter / branch filter; columns data (filter - control, 1x1 deg, light
smoothing), 25-deg prediction, 12.5-deg prediction; members' track and the branch sky region marked.
Figure 2 (plots/branch_search_profiles.png): counts vs dphi2 (relative to the track) through the branch filter and the track
filter, phi1 6-14 and 14-22: data excess with Poisson errors vs both predictions.
Usage: PYTHONPATH=src python bin/streams/branch_search_plots.py
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
import astropy.coordinates as coord
import branch_search_gaia as BS
from scipy.ndimage import gaussian_filter

U0, U1, X0, X1 = 4., 24., -25., 10.


def running(u, v, grid, half=2.):
    return np.array([np.median(v[np.abs(u-g) < half]) if np.sum(np.abs(u-g) < half) >= 10 else np.nan for g in grid])


def main():
    mem = BS.members(); q = dict(np.load(BS.CACHE))
    gq = coord.SkyCoord(q["ra"], q["dec"], unit="deg").galactic; q["u"], q["x"] = BS.project_gc(BS.w(gq.l.deg), gq.b.deg, BS.GC)
    ism = np.isin(q["source_id"], mem["sid"])
    gm, bm = q["g"][ism], q["bp_rp"][ism]; Gb = np.arange(14., 20.25, 0.5)
    mc = np.array([np.nanmedian(bm[(gm >= a) & (gm < a+0.5)]) for a in Gb[:-1]]); ms = np.array([1.4826*np.nanmedian(np.abs(bm[(gm >= a) & (gm < a+0.5)]-c)) for a, c in zip(Gb[:-1], mc)])
    ok = np.isfinite(mc) & np.isfinite(ms); gx, mc, ms = Gb[:-1][ok]+0.25, mc[ok], ms[ok]
    plx = q["parallax"]+BS.PLX_ZP; pplx = (plx-2*q["parallax_error"] < 0.27) & (plx+2*q["parallax_error"] > 0.15)

    def cmd(delta):
        g0 = q["g"]+delta; loc = np.interp(g0, gx, mc); wid = np.maximum(0.05, 1.5*np.interp(g0, gx, ms))
        return (np.abs(q["bp_rp"]-loc) < wid) & (q["g"] > 16) & (q["g"] < 19.8) & pplx
    grid = np.arange(U0, U1+0.01, 0.5)
    trk_x = running(mem["u"], mem["x"], grid); muT = running(mem["u"], mem["pmra"], grid); nuT = running(mem["u"], mem["pmdec"], grid)
    for a in (trk_x, muT, nuT):                                        # fill sparse ends by nearest valid value
        v = np.isfinite(a); a[~v] = np.interp(grid[~v], grid[v], a[v])
    M = {}
    for lab, f in BS.MODELS.items():
        m = dict(np.load(ROOT/f)); uu, xx = BS.project_gc(m["l"], m["b"], BS.GC); k = (m["chi"] > 0) & (m["age"] < 700.)
        M[lab] = dict(u=uu[k], x=xx[k], pmra=m["pmra"][k], pmdec=m["pmdec"][k])
    p = M["25"]; br = (np.abs(p["x"]) < 15) & (p["pmra"] < np.interp(p["u"], grid, muT)-3)
    muB = running(p["u"][br], p["pmra"][br], grid); nuB = running(p["u"][br], p["pmdec"][br], grid)
    for a in (muB, nuB):
        v = np.isfinite(a); a[~v] = np.interp(grid[~v], grid[v], a[v])
    track_at = lambda u: np.interp(u, grid, trk_x)
    filt = {"track": (muT, nuT, 1.5, 0.8, +5.), "branch": (muB, nuB, 2.5, 1.2, -5.)}

    def passes(u, pa, pd, which, shift=0.):
        mu, nu, hw, hd, _ = filt[which]
        return (np.abs(pa-(np.interp(u, grid, mu)+shift)) < hw) & (np.abs(pd-np.interp(u, grid, nu)) < hd) & (u >= U0) & (u < U1)
    eu, ex = np.arange(U0, U1+0.01, 1.), np.arange(X0, X1+0.01, 1.)
    H = lambda u, x, k: np.histogram2d(u[k], x[k], bins=(eu, ex))[0]
    data, model = {}, {lab: {} for lab in M}
    for which, delta in (("track", 0.0), ("branch", 0.4)):
        c = cmd(delta) & (~ism if which == "branch" else True)
        on = passes(q["u"], q["pmra"], q["pmdec"], which); off = passes(q["u"], q["pmra"], q["pmdec"], which, filt[which][4])
        data[which] = (H(q["u"], q["x"], c & on), H(q["u"], q["x"], c & off))
        for lab, p in M.items():
            model[lab][which] = H(p["u"], p["x"], passes(p["u"], p["pmra"], p["pmdec"], which))
    # scale: observed track-filter excess in the track tube, phi1 14-22, over the model's track-filter count there
    uc, xc = 0.5*(eu[1:]+eu[:-1]), 0.5*(ex[1:]+ex[:-1]); UU, XX = np.meshgrid(uc, xc, indexing="ij")
    tube = (np.abs(XX-track_at(UU)) < 2.5) & (UU >= 14) & (UU < 22)
    obs_track = float(np.sum((data["track"][0]-data["track"][1])[tube]))
    scale = {lab: obs_track/float(np.sum(model[lab]["track"][tube])) for lab in M}
    print(f"observed track-filter excess in the tube (phi1 14-22): {obs_track:.0f}; model scale factors: " +
          ", ".join(f"{k} deg {v:.4f}" for k, v in scale.items()))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(17, 9), sharex=True, sharey=True, constrained_layout=True)
    ext = (U0, U1, X0, X1)
    for i, which in enumerate(("track", "branch")):
        D = gaussian_filter(data[which][0]-data[which][1], 0.7)
        vmax = np.percentile(gaussian_filter(data["track"][0]-data["track"][1], 0.7), 99.5)
        panels = [("Gaia DR3: filter - control", D)]+[(f"{lab}-deg model prediction (scaled)", gaussian_filter(model[lab][which]*scale[lab], 0.7)) for lab in ("25", "12.5")]
        for j, (title, img) in enumerate(panels):
            im = ax[i, j].imshow(img.T, origin="lower", extent=ext, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax[i, j].plot(grid, trk_x, "k-", lw=0.8); ax[i, j].plot(grid, trk_x-2.5, "k:", lw=0.6); ax[i, j].plot(grid, trk_x+2.5, "k:", lw=0.6)
            if which == "branch":
                ax[i, j].axhline(-1, color="g", lw=0.8, ls="--"); ax[i, j].axhline(-13, color="g", lw=0.8, ls="--")
            ax[i, j].set_title(f"{which} PM filter: {title}", fontsize=10)
            ax[i, j].set(xlabel=r"$\phi_1$ [deg]" if i == 1 else None, ylabel=r"$\Delta\phi_2$ [deg]" if j == 0 else None)
        fig.colorbar(im, ax=ax[i, :], shrink=0.8, label="stars per 1x1 deg (excess)")
    fig.suptitle("Gaia DR3 independent search: background-subtracted star counts through matched PM filters; models scaled to the observed "
                 "stream (track filter, phi1 14-22). Black: members' track +-2.5 deg; green dashed: branch search region", fontsize=10)
    fig.savefig(ROOT/"plots/branch_search_maps.png", dpi=100); print(ROOT/"plots/branch_search_maps.png")
    # profiles across the stream, relative to the track
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    rel = XX-track_at(UU); eb = np.arange(-24, 10.01, 1.); cb = 0.5*(eb[1:]+eb[:-1]); out = {}
    for j, (a0, a1) in enumerate(((6, 14), (14, 22))):
        rng = (UU >= a0) & (UU < a1)
        for i, which in enumerate(("branch", "track")):
            on, off = data[which]; prof = lambda img: np.histogram(rel[rng], eb, weights=img[rng])[0]
            exc, var = prof(on)-prof(off), prof(on)+prof(off)
            ax[i, j].errorbar(cb, exc, np.sqrt(np.maximum(var, 1)), fmt="ko", ms=3, capsize=2, label="Gaia DR3 excess")
            for lab, col in (("25", "C3"), ("12.5", "C0")):
                ax[i, j].step(cb, prof(model[lab][which]*scale[lab]), where="mid", color=col, lw=1.5, label=f"{lab}-deg model (scaled)")
            ax[i, j].axhline(0, color="0.5", lw=0.5); ax[i, j].axvspan(-2.5, 2.5, color="0.92", zorder=0)
            ax[i, j].set(title=f"{which} PM filter, phi1 {a0}-{a1}", xlabel=r"$\Delta\phi_2$ - track [deg]", ylabel="excess stars per 1 deg")
            ax[i, j].legend(fontsize=8)
            out[f"{which}_{a0}-{a1}"] = dict(dphi2=cb.tolist(), excess=exc.tolist(), err=np.sqrt(np.maximum(var, 1)).tolist(),
                                            pred_25=prof(model["25"][which]*scale["25"]).tolist(), pred_12p5=prof(model["12.5"][which]*scale["12.5"]).tolist())
    fig.suptitle("Profiles across the stream (grey band = members' track +-2.5 deg): data excess vs models scaled to the observed stream", fontsize=10)
    fig.savefig(ROOT/"plots/branch_search_profiles.png", dpi=100); print(ROOT/"plots/branch_search_profiles.png")
    (ROOT/"results/plot_data/branch_search_profiles.json").write_text(json.dumps(dict(scale=scale, obs_track=obs_track, profiles=out), indent=1))


if __name__ == "__main__":
    main()
