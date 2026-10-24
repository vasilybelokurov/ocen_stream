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


def panel(q, S, R, F, axs, key=None):
    key = key or q
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
    d = load(q); fit = F[key]["fit"]; knots = {k: v["knots"] for k, v in fit.items()}; M = Model(d, knots); M.smax = F[key].get("smax")
    th = np.concatenate([np.array(fit[k]["values"]) for k in M.sizes()])
    P = M.unpack(torch.tensor(th)); m = (M.B["m"]@P["m"]).numpy(); s = M.width((M.B["s"]@P["s"]).numpy()); f = 1/(1+np.exp(-(M.B["f"]@P["f"]).numpy()))
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
    if q == "dphi2":                                                  # observed-area weighted density on the fit grid, binned to pixels
        vg = d["vg"]; gz = np.exp(-0.5*((vg[None, :]-m[:, None])/sig[:, None])**2)*d["Aw"]; gz /= np.maximum(np.trapezoid(gz, vg, axis=1), 1e-12)[:, None]
        cum = np.concatenate((np.zeros((len(m), 1)), np.cumsum(0.5*(gz[:, 1:]+gz[:, :-1])*np.diff(vg)[None, :], 1)), 1)
        mass = np.clip(np.array([np.interp(hi_e[i], vg, cum[i])-np.interp(lo_e[i], vg, cum[i]) for i in range(len(m))]), 0, None)*f[:, None]
    if getattr(M, "ext", False) and q != "dphi2":                     # PM extended model: rate x truncated Gaussian on the phi1 grid
        Smod = np.zeros_like(H); ug, eg = d["ug"], d["eg"]; Pg = M.unpack(torch.tensor(th))
        mg = (M.Bg["m"]@Pg["m"]).numpy(); sgt = np.sqrt(M.width((M.Bg["s"]@Pg["s"]).numpy())**2+eg**2); rg = np.exp((M.Bg["f"]@Pg["f"]).numpy())
        trg = np.interp(ug, G, R["tracks"][q]["c"]); jg = np.clip(np.digitize(ug, E1)-1, 0, len(c1)-1)
        for i in range(len(ug)):
            a_, b_ = np.clip(Ey[:-1]-trg[i], -W, W), np.clip(Ey[1:]-trg[i], -W, W)
            Smod[jg[i]] += rg[i]*0.1*np.clip(ndtr((b_-mg[i])/sgt[i])-ndtr((a_-mg[i])/sgt[i]), 0, None)
    elif getattr(M, "ext", False):                                      # extended model: rate(phi1) x N x A integrated on the phi1 grid
        Smod = np.zeros_like(H); ug, Ag, vg = d["ug"], d["Ag"], d["vg"]; Pg = M.unpack(torch.tensor(th))
        mg = (M.Bg["m"]@Pg["m"]).numpy(); sgg = M.width((M.Bg["s"]@Pg["s"]).numpy()); rg = np.exp((M.Bg["f"]@Pg["f"]).numpy())
        ctg = np.interp(ug, G, R["sky_window"]["centre"]); jg = np.clip(np.digitize(ug, E1)-1, 0, len(c1)-1)
        gzs = [np.exp(-0.5*((vg[None, :]-mg[:, None])/sgg[:, None])**2)/(sgg[:, None]*np.sqrt(2*np.pi))*Ag*rg[:, None]*0.1]
        if "m2" in M.knots:                                               # + upper branch (kept separately for --components)
            m2g, s2g, r2g = (M.Bg["m2"]@Pg["m2"]).numpy(), M.width((M.Bg["s2"]@Pg["s2"]).numpy()), np.exp((M.Bg["f2"]@Pg["f2"]).numpy())*M.tapg.numpy()
            gzs.append(np.exp(-0.5*((vg[None, :]-m2g[:, None])/s2g[:, None])**2)/(s2g[:, None]*np.sqrt(2*np.pi))*Ag*r2g[:, None]*0.1)
            tr2 = (ug, ctg+m2g, r2g > 1.)
        comps = []
        for gz in gzs:
            Sc = np.zeros_like(H); cum = np.concatenate((np.zeros((len(ug), 1)), np.cumsum(0.5*(gz[:, 1:]+gz[:, :-1])*np.diff(vg)[None, :], 1)), 1)
            for i in range(len(ug)):
                Sc[jg[i]] += np.interp(Ey[1:]-ctg[i], vg, cum[i])-np.interp(Ey[:-1]-ctg[i], vg, cum[i])
            comps.append(Sc)
        Smod = sum(comps)
    else:
        np.add.at(Smod, ju, mass)
    inside = np.zeros_like(H, bool)
    for j, cc in enumerate(c1):
        if d["lo"] <= cc < d["hi"]:
            cj = np.interp(cc, G, R["sky_window"]["centre"] if q == "dphi2" else R["tracks"][q]["c"]); inside[j] = np.abs(cy-cj) < W
    X = H-B; Rz = np.where(inside, X-Smod, np.nan); Sm = np.where(inside, Smod, np.nan); titles = ("data excess", "model (stream component)", "residual (data - background - model)")
    if q == "dphi2":
        Xs = gaussian_filter(X, 1.); norm = np.clip(Xs, 0, None).sum(1, keepdims=True); norm[norm == 0] = 1
        nz = lambda A_: np.where(inside, gaussian_filter(np.nan_to_num(np.where(inside, A_, np.nan)), 1.)/norm, np.nan)
        imgs = (Xs/norm, nz(Smod), nz(X-Smod)); vm = 0.08
        if len(axs) >= 4 and len(comps) == 2:
            imgs = (Xs/norm, nz(comps[0]), nz(comps[1]), nz(X-Smod))
            titles = ("data excess", "model: main stream", "model: upper branch", "residual (data - background - both)")
            if len(axs) == 5:
                imgs = imgs[:3]+(nz(Smod),)+imgs[3:]; titles = titles[:3]+("model: main + branch",)+titles[3:]
    else:
        sb = np.sqrt(np.maximum(B, 1)); imgs = (gaussian_filter(X/sb, 1.), np.where(inside, gaussian_filter(np.nan_to_num(Sm)/sb, 1.), np.nan),
                                                  np.where(inside, gaussian_filter(np.nan_to_num(Rz)/sb, 1.), np.nan)); vm = 4
    ext = (E1[0], E1[-1], Ey[0], Ey[-1]); Gf = np.array(F[key]["grid"]); trk = (np.interp(Gf, G, R["sky_window"]["centre"] if q == "dphi2" else R["tracks"][q]["c"])+np.array(F[key]["m"]))
    if len(axs) >= 4 and len(imgs) == 3:                              # PM row in a --components figure: data | model | (empty) | residual
        axs[2].axis("off"); axs = [axs[0], axs[1], axs[3]]
    for a, im, tl in zip(axs, imgs, titles):
        cm = plt.get_cmap("RdBu_r").copy(); cm.set_bad("0.92")
        a.imshow(np.asarray(im).T, origin="lower", aspect="auto", cmap=cm, vmin=-vm, vmax=vm, extent=ext)
        a.set(ylabel=lab, title=f"{key}: {tl}"); a.plot(Gf, trk, "k-", lw=0.6, alpha=0.5)
        for k_ in fit["m"]["knots"]:
            a.axvline(k_, color="k", lw=0.4, ls=":")
        if "m2" in fit and q == "dphi2" and getattr(M, "ext", False):
            a.plot(tr2[0][tr2[2]], tr2[1][tr2[2]], "k--", lw=0.6, alpha=0.6)
    inn = inside & np.isfinite(Rz); chi = (Rz/np.sqrt(np.maximum(B+np.clip(Smod, 0, None), 1)))[inn]
    print(f"{q}: residual (N - B - S)/sqrt(B + S) inside the fit region: mean {chi.mean():+.3f}, sd {chi.std():.3f}, n pixels {chi.size}; "
          f"total data excess {X[inside].sum():.0f}, model {Smod[inside].sum():.0f}", flush=True)


def pm_two(q, S, R, F, axs):
    """PM row with separate main and branch components (fits <q>_main, <q>_branch: +-1 deg sky windows on the dphi2_sky_final
    tracks). Data = stars in the union of both windows (other-PM mask k = 2); background per column from 1-deg dphi2 bands at
    5-14 deg, quadratic in y, integrated over both windows; models = expected stream counts from each fit's stars."""
    from stream_spline_fit import sky_tracks, SKYWIN_HW, BRANCH_RANGE
    tm, tb = sky_tracks(); G = np.array(R["grid"]); u, x = S["phi1"], S["dphi2"]; ev = lambda p, key="c": np.interp(u, G, R["tracks"][p][key])
    skc = np.interp(u, G, R["sky_window"]["centre"]); yv = x-skc; other = "pmdec" if q == "pmra" else "pmra"
    sel = np.abs(S[other]-ev(other)) < 2*np.sqrt(ev(other, "s")**2+S[other+"_error"]**2)
    inb = (u >= BRANCH_RANGE[0]) & (u < BRANCH_RANGE[1]); wm, wb = np.abs(yv-tm(u)) < SKYWIN_HW, inb & (np.abs(yv-tb(u)) < SKYWIN_HW)
    Ey, lab, nb = ROWS[q]; E1 = np.linspace(-8, 32, nb); c1 = 0.5*(E1[1:]+E1[:-1]); cy = 0.5*(Ey[1:]+Ey[:-1])
    on = sel & (wm | wb); H = np.histogram2d(u[on], S[q][on], (E1, Ey))[0]
    Eb = np.arange(-14, 14.01, 1.); cb = 0.5*(Eb[1:]+Eb[:-1]); offb = np.abs(cb) >= 5
    H3 = np.histogramdd(np.column_stack((u[sel], yv[sel], S[q][sel])), (E1, Eb, Ey))[0]; B = np.zeros_like(H)
    for j in range(len(c1)):
        ob = offb & (H3[j].sum(1) > 0)
        if ob.sum() < 6:
            continue
        dg = 3 if ((ob & (cb < 0)).sum() >= 3 and (ob & (cb > 0)).sum() >= 3) else 1; coef = np.linalg.pinv(np.vander(cb[ob], dg))@H3[j, ob, :]
        cc = np.array([c1[j]]); yy = np.linspace(-SKYWIN_HW, SKYWIN_HW, 21)+tm(cc)[0]
        if BRANCH_RANGE[0] <= c1[j] < BRANCH_RANGE[1]:
            yy = np.concatenate((yy, np.linspace(-SKYWIN_HW, SKYWIN_HW, 21)+tb(cc)[0]))
        B[j] = np.clip(np.vander(yy, dg)@coef, 0, None).mean(0)*(2*SKYWIN_HW)*(2 if len(yy) > 21 else 1)
    comps, ins = [], np.zeros_like(H, bool)
    for w in ("main", "branch"):
        key = f"{q}_{w}"; d = load(q, skywin=w); fit = F[key]["fit"]; M = Model(d, {k: v["knots"] for k, v in fit.items()})
        th = np.concatenate([np.array(fit[k]["values"]) for k in M.sizes()]); P = M.unpack(torch.tensor(th))
        m = (M.B["m"]@P["m"]).numpy(); s_ = M.width((M.B["s"]@P["s"]).numpy()); f = 1/(1+np.exp(-(M.B["f"]@P["f"]).numpy()))
        sig = np.sqrt(s_**2+d["e"]**2); W = d["W"]; ctr = np.interp(d["u"], G, R["tracks"][q]["c"]); Z = ndtr((W-m)/sig)-ndtr((-W-m)/sig)
        a_, b_ = np.clip(Ey[:-1][None, :]-ctr[:, None], -W, W), np.clip(Ey[1:][None, :]-ctr[:, None], -W, W)
        mass = np.clip(ndtr((b_-m[:, None])/sig[:, None])-ndtr((a_-m[:, None])/sig[:, None]), 0, None)*(f/np.maximum(Z, 1e-9))[:, None]
        Sc = np.zeros_like(H); np.add.at(Sc, np.clip(np.digitize(d["u"], E1)-1, 0, len(c1)-1), mass); comps.append(Sc)
        Gf = np.array(F[key]["grid"]); comps[-1] = (Sc, Gf, np.interp(Gf, G, R["tracks"][q]["c"])+np.array(F[key]["m"]), fit["m"]["knots"])
        for j, cc in enumerate(c1):
            if d["lo"] <= cc < d["hi"]:
                ins[j] |= np.abs(cy-np.interp(cc, G, R["tracks"][q]["c"])) < W
    X = H-B; Sm = comps[0][0]+comps[1][0]; sb = np.sqrt(np.maximum(B, 1)); nz = lambda A_: np.where(ins, gaussian_filter(np.nan_to_num(np.where(ins, A_, 0.))/sb, 1.), np.nan)
    imgs = (gaussian_filter(X/sb, 1.), nz(comps[0][0]), nz(comps[1][0]), nz(Sm), nz(X-Sm))
    titles = ("data excess (main + branch windows)", "model: main stream", "model: upper branch", "model: main + branch", "residual (data - background - both)")
    if len(axs) == 4:
        imgs, titles = imgs[:3]+imgs[4:], titles[:3]+titles[4:]
    ext = (E1[0], E1[-1], Ey[0], Ey[-1])
    for a, im, tl in zip(axs, imgs, titles):
        cm = plt.get_cmap("RdBu_r").copy(); cm.set_bad("0.92"); a.imshow(np.asarray(im).T, origin="lower", aspect="auto", cmap=cm, vmin=-4, vmax=4, extent=ext)
        a.set(ylabel=lab, title=f"{q}: {tl}"); a.plot(comps[0][1], comps[0][2], "k-", lw=0.6, alpha=0.5); a.plot(comps[1][1], comps[1][2], "k--", lw=0.6, alpha=0.7)
    chi = ((X-Sm)/np.sqrt(np.maximum(B+np.clip(Sm, 0, None), 1)))[ins]
    print(f"{q} (main + branch): residual sd {chi.std():.3f}, mean {chi.mean():+.3f}; excess {X[ins].sum():.0f}, model main {comps[0][0][ins].sum():.0f} branch {comps[1][0][ins].sum():.0f}", flush=True)


def main():
    keys = sys.argv[sys.argv.index("--keys")+1].split(",") if "--keys" in sys.argv else None
    qs = keys or ([sys.argv[sys.argv.index("--q")+1]] if "--q" in sys.argv else ["dphi2", "pmra", "pmdec"])
    S = catalogue(); R = json.loads((ROOT/"results/plot_data/stream_masks.json").read_text()); F = json.loads((ROOT/"results/plot_data/stream_spline_fit.json").read_text())
    nc = (5 if "--combined" in sys.argv else 4) if "--components" in sys.argv else 3
    fig, ax = plt.subplots(len(qs), nc, figsize=(6.7*nc, 4.6*len(qs)), squeeze=False, sharex=True, constrained_layout=True)
    for i, q in enumerate(qs):
        if "--pmtwo" in sys.argv:
            pm_two(q, S, R, F, ax[i]); continue
        panel(q.split("_")[0], S, R, F, ax[i], key=q)
    for a in ax[-1]:
        a.set_xlabel(r"$\phi_1$ [deg]")
    fig.suptitle("Spline mixture model (track, width, stream fraction as splines in phi1; knots by AIC, dotted: track knots); black line: fitted track")
    out = ROOT/f"plots/stream_spline_{(qs[0] if len(qs) == 1 else 'all') if not keys else '_'.join(keys)}{'_components' if nc >= 4 else ''}{'_combined' if nc == 5 else ''}{'_pmtwo' if '--pmtwo' in sys.argv else ''}.png"; fig.savefig(out, dpi=95); print(out.relative_to(ROOT))


if __name__ == "__main__":
    main()
