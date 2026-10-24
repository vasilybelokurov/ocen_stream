#!/usr/bin/env python3
"""Data | spline-mixture model (stream component) | residual for the cross-selection panels (2026-10-24).

Data: the same selection as the fit in stream_spline_fit.py (CMD pass, other frozen masks k = 2; PMs inside the sky window).
Display background B: PMs -- per (phi1 column, PM pixel) quadratic in dphi2 - centre from 1-deg bands at 5-14 deg, integrated
over the column's sky window (median half-width); sky -- per column quadratic in dphi2 from 4 < |dphi2 - centre| < 12.
Model: expected stream counts per pixel = sum over the fit's stars in the column of f_i x (truncated Gaussian mass in the pixel).
Panels: (N - B)/sqrt(B), S/sqrt(B), (N - B - S)/sqrt(B) (PMs); the sky row is normalised per column by the integral of the
positive data excess (as in stream_masks_xsel). Model and residual only inside the fitted phi1 range and window.
Usage: PYTHONPATH=src python bin/streams/stream_spline_plot.py [--q pmdec] -> plots/stream_spline_<q>.png (or _all)
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.special import ndtr
import torch
from stream_spline_fit import load, Model, CFG

ROWS = dict(dphi2=(np.arange(-10, 10.01, 0.25), r"$\Delta\phi_2$ [deg]", 31), pmra=(np.arange(-28, 4.01, 0.25), r"$\mu_{\alpha*}$ [mas/yr]", 31),
            pmdec=(np.arange(-18, 2.01, 0.25), r"$\mu_\delta$ [mas/yr]", 61))


def catalogue():
    from gaia_cmd_mask import cmd_pass
    from gaia_footprint_plots import gc_masked
    raw = dict(np.load(ROOT/"results/streams/gaia_footprint_dr3.npz")); lg = np.where(raw["l"] > 180, raw["l"]-360, raw["l"]).astype(float)
    keep = cmd_pass(2.0) & ~gc_masked(lg, raw["b"].astype(float)); return {k: v[keep].astype(float) for k, v in raw.items() if v.dtype.kind == "f"}


def panel(q, S, R, F, axs):
    G = np.array(R["grid"]); u, x = S["phi1"], S["dphi2"]; ev = lambda p, key="c": np.interp(u, G, R["tracks"][p][key])
    skc, skh = np.interp(u, G, R["sky_window"]["centre"]), np.interp(u, G, R["sky_window"]["half_width"])
    pmm = {p: np.abs(S[p]-ev(p)) < 2*np.sqrt(ev(p, "s")**2+S[p+"_error"]**2) for p in ("pmra", "pmdec")}
    Ey, lab, nb = ROWS[q]; E1 = np.linspace(-8, 32, nb); c1 = 0.5*(E1[1:]+E1[:-1]); cy = 0.5*(Ey[1:]+Ey[:-1])
    if q == "dphi2":
        sel = pmm["pmra"] & pmm["pmdec"]; val = x; H = np.histogram2d(u[sel], val[sel], (E1, Ey))[0]; B = np.zeros_like(H)
        for j, cc in enumerate(c1):
            ctr = np.interp(cc, G, R["sky_window"]["centre"]); off = (np.abs(cy-ctr) > 4) & (np.abs(cy-ctr) < 12)
            if off.sum() > 10:
                B[j] = np.clip(np.polyval(np.polyfit(cy[off], H[j, off], 2), cy), 0, None)
    else:
        other = "pmdec" if q == "pmra" else "pmra"; sel = pmm[other]; val = S[q]; yv = x-skc
        on = sel & (np.abs(yv) < skh); H = np.histogram2d(u[on], val[on], (E1, Ey))[0]
        Eb = np.arange(-14, 14.01, 1.); cb = 0.5*(Eb[1:]+Eb[:-1]); offb = np.abs(cb) >= 5
        H3 = np.histogramdd(np.column_stack((u[sel], yv[sel], val[sel])), (E1, Eb, Ey))[0]; B = np.zeros_like(H)
        for j in range(len(c1)):
            ob = offb & (H3[j].sum(1) > 0)
            if ob.sum() < 6:
                continue
            dg = 3 if ((ob & (cb < 0)).sum() >= 3 and (ob & (cb > 0)).sum() >= 3) else 1
            coef = np.linalg.pinv(np.vander(cb[ob], dg))@H3[j, ob, :]
            kc = (u >= E1[j]) & (u < E1[j+1]); h = float(np.median(skh[kc])) if kc.any() else 3.
            yy = np.linspace(-h, h, 41); B[j] = np.clip(np.vander(yy, dg)@coef, 0, None).mean(0)*(2*h)
    # model: expected stream counts per pixel from the fit's stars
    d = load(q); fit = F[q]["fit"]; knots = {k: v["knots"] for k, v in fit.items()}; M = Model(d, knots)
    th = np.concatenate([np.array(fit[k]["values"]) for k in M.sizes()])
    P = M.unpack(torch.tensor(th)); m = (M.B["m"]@P["m"]).numpy(); s = np.sqrt(np.exp(2*(M.B["s"]@P["s"]).numpy())+M.smin**2); f = 1/(1+np.exp(-(M.B["f"]@P["f"]).numpy()))
    sig = np.sqrt(s**2+d["e"]**2); W = d["W"]
    if q == "dphi2":
        ctr = np.interp(d["u"], G, R["sky_window"]["centre"])
        Z = ndtr((W-m)/sig)-ndtr((-W-m)/sig)-(ndtr((d["hhi"]-m)/sig)-ndtr((d["hlo"]-m)/sig))
        # stars removed by the upper-branch hole are not modelled: no model mass inside the hole
    else:
        ctr = np.interp(d["u"], G, R["tracks"][q]["c"]); Z = ndtr((W-m)/sig)-ndtr((-W-m)/sig)
    Smod = np.zeros_like(H); ju = np.clip(np.digitize(d["u"], E1)-1, 0, len(c1)-1)
    lo_e, hi_e = Ey[:-1][None, :]-ctr[:, None], Ey[1:][None, :]-ctr[:, None]
    a_, b_ = np.clip(lo_e, -W, W), np.clip(hi_e, -W, W)
    mass = (ndtr((b_-m[:, None])/sig[:, None])-ndtr((a_-m[:, None])/sig[:, None]))
    if q == "dphi2":
        ha, hb = np.clip(lo_e, d["hlo"][:, None], d["hhi"][:, None]), np.clip(hi_e, d["hlo"][:, None], d["hhi"][:, None])
        mass -= ndtr((hb-m[:, None])/sig[:, None])-ndtr((ha-m[:, None])/sig[:, None])
    mass = np.clip(mass, 0, None)*(f/np.maximum(Z, 1e-9))[:, None]
    np.add.at(Smod, ju, mass)
    inside = np.zeros_like(H, bool)
    for j, cc in enumerate(c1):
        if d["lo"] <= cc < d["hi"]:
            cj = np.interp(cc, G, R["sky_window"]["centre"] if q == "dphi2" else R["tracks"][q]["c"]); inside[j] = np.abs(cy-cj) < W
    X = H-B; Rz = np.where(inside, X-Smod, np.nan); Sm = np.where(inside, Smod, np.nan)
    if q == "dphi2":
        Xs = gaussian_filter(X, 1.); norm = np.clip(Xs, 0, None).sum(1, keepdims=True); norm[norm == 0] = 1
        imgs = (Xs/norm, gaussian_filter(np.nan_to_num(Sm), 1.)/norm, gaussian_filter(np.nan_to_num(Rz), 1.)/norm); vm = 0.08
        imgs = (imgs[0], np.where(inside, imgs[1], np.nan), np.where(inside, imgs[2], np.nan))
    else:
        sb = np.sqrt(np.maximum(B, 1)); imgs = (gaussian_filter(X/sb, 1.), np.where(inside, gaussian_filter(np.nan_to_num(Sm)/sb, 1.), np.nan),
                                                  np.where(inside, gaussian_filter(np.nan_to_num(Rz)/sb, 1.), np.nan)); vm = 4
    ext = (E1[0], E1[-1], Ey[0], Ey[-1]); Gf = np.array(F[q]["grid"]); trk = (np.interp(Gf, G, R["sky_window"]["centre"] if q == "dphi2" else R["tracks"][q]["c"])+np.array(F[q]["m"]))
    for a, im, tl in zip(axs, imgs, ("data excess", "model (stream component)", "residual (data - background - model)")):
        cm = plt.get_cmap("RdBu_r").copy(); cm.set_bad("0.92")
        a.imshow(np.asarray(im).T, origin="lower", aspect="auto", cmap=cm, vmin=-vm, vmax=vm, extent=ext)
        a.set(ylabel=lab, title=f"{q}: {tl}"); a.plot(Gf, trk, "k-", lw=0.6, alpha=0.5)
        for k_ in fit["m"]["knots"]:
            a.axvline(k_, color="k", lw=0.4, ls=":")
    inn = inside & np.isfinite(Rz); chi = (Rz/np.sqrt(np.maximum(B+np.clip(Smod, 0, None), 1)))[inn]
    print(f"{q}: residual (N - B - S)/sqrt(B + S) inside the fit region: mean {chi.mean():+.3f}, sd {chi.std():.3f}, n pixels {chi.size}; "
          f"total data excess {X[inside].sum():.0f}, model {Smod[inside].sum():.0f}", flush=True)


def main():
    qs = [sys.argv[sys.argv.index("--q")+1]] if "--q" in sys.argv else ["dphi2", "pmra", "pmdec"]
    S = catalogue(); R = json.loads((ROOT/"results/plot_data/stream_masks.json").read_text()); F = json.loads((ROOT/"results/plot_data/stream_spline_fit.json").read_text())
    fig, ax = plt.subplots(len(qs), 3, figsize=(20, 4.6*len(qs)), squeeze=False, sharex=True, constrained_layout=True)
    for i, q in enumerate(qs):
        panel(q, S, R, F, ax[i])
    for a in ax[-1]:
        a.set_xlabel(r"$\phi_1$ [deg]")
    fig.suptitle("Spline mixture model (track, width, stream fraction as splines in phi1; knots by AIC, dotted: track knots); black line: fitted track")
    out = ROOT/f"plots/stream_spline_{qs[0] if len(qs) == 1 else 'all'}.png"; fig.savefig(out, dpi=95); print(out.relative_to(ROOT))


if __name__ == "__main__":
    main()
