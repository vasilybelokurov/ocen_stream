#!/usr/bin/env python3
"""Cross-selection view with masks drawn from the sequences the user marked by eye on the background-subtracted panels of
plots/gaia_xsel_60px.png (2026-10-23). Same method as bin/streams/gaia_cross_selection.py (60 phi1 pixels).

Marked sequences (read off the annotated image by axis calibration; ~+-0.3 in each coordinate):
  pmdec: segment 1 phi1 0.7-6.4 (-6.7 -> -7.4); segment 2 phi1 9.1-26.4 (-6.9 -> -9.6 at 19.8 -> -11.3); tightest signal.
  pmra*: phi1 16.7-30.5 (-7.6 -> -14.4 at 23.3 -> -21.6).
  sky:   lower branch phi1 15-28 (dphi2 0.35 -> 0.8 at 19.8 -> 3.0 at 28.1); upper branch phi1 14-23 (2.3 -> 6.0).
Tracks: smoothing splines through the marked points; outside the marked phi1 range the previous hand-placed PM knots
(gaia_cross_selection.KNOTS) and, on the sky, dphi2 = 0 before / the last marked value after the lower branch are used. Masks: |dPM| < 1.5 mas/yr (pmra*), < 1.0 (pmdec; the
marked sequence is ~1 mas/yr wide), sky |dphi2 - branch| < 1.25 deg.
Figures: gaia_xsel_marked.png -- rows dphi2 / pmra* / pmdec, columns: counts / column median, background-subtracted, masks + marks
  + STREAMFINDER members (sky mask = lower branch);
  gaia_xsel_branches.png -- PM panels (background-subtracted) for stars selected by the LOWER vs the UPPER sky branch (each with the
  other PM mask), and the CMD of each branch (all masks minus the same PM selection off the branches).
Usage: PYTHONPATH=src python bin/streams/gaia_xsel_marked.py [--cmd]  (--cmd: common cut = inside the isochrone CMD mask,
       bin/streams/gaia_cmd_mask.cmd_pass; outputs *_cmd.png)
"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import UnivariateSpline
from gaia_cross_selection import load, KNOTS

MARK = dict(
    pmdec=[(0.7, -6.7), (3.5, -7.0), (6.4, -7.4), (9.1, -6.9), (13.0, -7.9), (16.5, -8.8), (19.8, -9.6), (23.0, -10.3), (26.4, -11.3)],
    pmra=[(16.7, -7.6), (20.0, -10.8), (23.3, -14.4), (26.5, -17.2), (30.5, -21.6)],
    lower=[(15.0, 0.35), (17.5, 0.5), (19.8, 0.8), (24.0, 1.6), (28.1, 3.0)],
    upper=[(14.0, 2.3), (16.5, 3.5), (19.5, 4.8), (22.9, 6.0)])
W = dict(pmra=1.5, pmdec=1.0, sky=1.25)


def track(marks, prior, lo=-8, hi=32):
    """Smoothing spline through the marked points, padded outside their phi1 range with the prior track (callable)."""
    m = np.array(marks); a, b = m[:, 0].min(), m[:, 0].max()
    pad = [(p, float(prior(p))) for p in np.arange(lo, a-1.9, 2.)]+[(p, float(prior(p))) for p in np.arange(b+2, hi+0.1, 2.)]
    P = np.array(sorted(list(map(tuple, m))+pad)); w = np.where((P[:, 0] >= a) & (P[:, 0] <= b), 3., 1.)
    return UnivariateSpline(P[:, 0], P[:, 1], w=w, k=3, s=len(P)*0.05)


def main():
    from grid4_common import du, dx, d
    K = np.array(KNOTS); old_a = lambda p: np.interp(p, K[:, 0], K[:, 1]); old_d = lambda p: np.interp(p, K[:, 0], K[:, 2])
    ta, td = track(MARK["pmra"], old_a), track(MARK["pmdec"], old_d)
    ml = np.array(MARK["lower"]); tl = track(MARK["lower"], lambda p: np.where(p < ml[0, 0], 0., ml[-1, 1])+0.*p); tu = lambda p: np.interp(p, *np.array(MARK["upper"]).T, left=np.nan, right=np.nan)
    D = load(); u, x, pa, pd = D["phi1"], D["dphi2"], D["pmra"], D["pmdec"]
    CMD = "--cmd" in sys.argv
    if CMD:                                                                               # common cut: inside the CMD mask (k = 2)
        from gaia_cmd_mask import cmd_pass
        from gaia_footprint_plots import gc_masked
        raw = np.load(ROOT/"results/streams/gaia_footprint_dr3.npz"); lg = np.where(raw["l"] > 180, raw["l"]-360, raw["l"])
        cp = cmd_pass(2.0)[~gc_masked(lg.astype(float), raw["b"].astype(float))]
        print(f"CMD mask keeps {cp.sum()} of {cp.size} stars", flush=True)
        D = {k_: v[cp] for k_, v in D.items()}; u, x, pa, pd = D["phi1"], D["dphi2"], D["pmra"], D["pmdec"]
    tagc = "_cmd" if CMD else ""
    selA, selD = np.abs(pa-ta(u)) < W["pmra"], np.abs(pd-td(u)) < W["pmdec"]
    skyL = np.abs(x-tl(u)) < W["sky"]; skyU = np.abs(x-tu(u)) < W["sky"]                      # NaN outside 14-23 -> False
    offsky = (np.abs(x-tl(u)) > 4) & (np.abs(x-tl(u)) < 12) & ~(np.abs(x-tu(u)) < 2)
    E1 = np.linspace(-8, 32, 61); c1 = 0.5*(E1[1:]+E1[:-1]); ph = np.linspace(-6, 31, 300)
    n_on = np.histogram(u[skyL], E1)[0]; n_off = np.histogram(u[offsky], E1)[0]
    ratio = np.divide(n_on, n_off, out=np.zeros(len(c1)), where=n_off > 0)

    def panel(q, v, sel, Ey, off_sel=None, rat=None):
        H = np.histogram2d(u[sel], v[sel], (E1, Ey))[0]
        if q == "dphi2":
            cy = 0.5*(Ey[1:]+Ey[:-1]); B = np.zeros_like(H)
            for j, c in enumerate(c1):
                off = (np.abs(cy-tl(c)) > 4) & (np.abs(cy-tl(c)) < 12)
                if off.sum() > 10:
                    B[j] = np.clip(np.polyval(np.polyfit(cy[off], H[j, off], 2), cy), 0, None)
        else:
            B = np.histogram2d(u[off_sel], v[off_sel], (E1, Ey))[0]*rat[:, None]
        return H, B

    rows = (("dphi2", x, selA & selD, np.arange(-15, 15.01, 0.25), r"$\Delta\phi_2$ [deg]", "both PM masks", None),
            ("pmra", pa, skyL & selD, np.arange(-28, 4.01, 0.25), r"$\mu_{\alpha*}$ [mas/yr]", r"sky (lower branch) + $\mu_\delta$ masks", offsky & selD),
            ("pmdec", pd, skyL & selA, np.arange(-18, 2.01, 0.25), r"$\mu_\delta$ [mas/yr]", r"sky (lower branch) + $\mu_{\alpha*}$ masks", offsky & selA))
    trk = dict(dphi2=tl, pmra=ta, pmdec=td); wid = dict(dphi2=W["sky"], pmra=W["pmra"], pmdec=W["pmdec"])
    mem = dict(dphi2=dx, pmra=d["pmra"], pmdec=d["pmdec"])
    fig, ax = plt.subplots(3, 3, figsize=(21, 12.5), constrained_layout=True, sharex=True)
    for i, (q, v, sel, Ey, yl, lab, offs) in enumerate(rows):
        H, B = panel(q, v, sel, Ey, offs, ratio); e = (E1[0], E1[-1], Ey[0], Ey[-1]); Hs = gaussian_filter(H, 0.6)
        colmed = np.nanmedian(np.where(Hs > 0, Hs, np.nan), axis=1)[:, None]
        ax[i, 0].imshow((Hs/np.where(colmed > 0, colmed, np.nan)).T, origin="lower", extent=e, aspect="auto", cmap="Greys", vmin=0.5, vmax=3)
        ax[i, 1].imshow(gaussian_filter((H-B)/np.sqrt(np.maximum(B, 1)), 1.).T, origin="lower", extent=e, aspect="auto", cmap="RdBu_r", vmin=-4, vmax=4)
        for a_ in ax[i]:
            a_.fill_between(ph, trk[q](ph)-wid[q], trk[q](ph)+wid[q], color="C0", alpha=0.12 if a_ is not ax[i, 2] else 0.25, lw=0)
            if q == "dphi2":
                pu = np.linspace(14, 22.9, 50); a_.fill_between(pu, tu(pu)-W["sky"], tu(pu)+W["sky"], color="C1", alpha=0.12 if a_ is not ax[i, 2] else 0.3, lw=0)
            a_.set_ylabel(yl); a_.set_ylim(Ey[0], Ey[-1])
        ax[i, 2].plot(du, mem[q], ".", color="C2", ms=1.5, alpha=0.6)
        for key in (("lower", "upper") if q == "dphi2" else (q,)):
            m = np.array(MARK[key]); ax[i, 2].plot(m[:, 0], m[:, 1], "x", color="C3" if key != "upper" else "C1", ms=7, mew=2)
        ax[i, 2].set_xlim(E1[0], E1[-1])
        ax[i, 0].set_title(f"{yl.split(' [')[0]}: stars passing the {lab} (counts / column median)", fontsize=10)
        ax[i, 1].set_title("same, minus background from the unselected dimension: (N - B)/sqrt(B)", fontsize=10)
        ax[i, 2].set_title("masks (blue; orange: upper sky branch), marked points (x), STREAMFINDER members (green)", fontsize=9)
    for a_ in ax[-1]:
        a_.set_xlabel(r"$\phi_1$ [deg]")
    fig.suptitle("Gaia DR3 cross-selection with masks drawn from the marked sequences (60 phi1 pixels)"+(" + CMD mask (k = 2, dm(phi1)) applied to all panels" if CMD else ""), fontsize=11)
    fig.savefig(ROOT/f"plots/gaia_xsel_marked{tagc}.png", dpi=105); plt.close(fig)
    # --- lower vs upper sky branch: PMs and CMD
    fig, ax = plt.subplots(2, 3, figsize=(19, 9), constrained_layout=True)
    kU = (u > 13) & (u < 24); offU = offsky & kU
    for r, (name, sky) in enumerate((("lower branch", skyL & kU), ("upper branch", skyU & kU))):
        nn = np.histogram(u[sky], E1)[0]; nf = np.histogram(u[offU], E1)[0]; rt = np.divide(nn, nf, out=np.zeros(len(c1)), where=nf > 0)
        for j, (q, v, Ey, other, yl) in enumerate((("pmra", pa, np.arange(-28, 4.01, 0.25), selD, r"$\mu_{\alpha*}$"),
                                                   ("pmdec", pd, np.arange(-18, 2.01, 0.25), selA, r"$\mu_\delta$"))):
            H, B = panel(q, v, sky & other, Ey, offU & other, rt)
            ax[r, j].imshow(gaussian_filter((H-B)/np.sqrt(np.maximum(B, 1)), 1.).T, origin="lower", extent=(E1[0], E1[-1], Ey[0], Ey[-1]),
                            aspect="auto", cmap="RdBu_r", vmin=-4, vmax=4)
            ax[r, j].fill_between(ph, trk[q](ph)-wid[q], trk[q](ph)+wid[q], color="C0", alpha=0.15, lw=0)
            ax[r, j].set(xlim=(10, 28), ylabel=f"{yl} [mas/yr]", title=f"{name}: {yl} (sky {name} + other PM mask), background-subtracted")
        Ec, Eg = np.arange(0, 2.01, 0.05), np.arange(12, 20.01, 0.1); allsel = sky & selA & selD; offs = offU & selA & selD
        s = allsel.sum()/max((offs.sum()), 1) if False else np.sum(nn)/max(np.sum(nf), 1)
        Hn = np.histogram2d(D["bp_rp"][allsel], D["g"][allsel], (Ec, Eg))[0]; Hf = np.histogram2d(D["bp_rp"][offs], D["g"][offs], (Ec, Eg))[0]*s
        ax[r, 2].imshow(gaussian_filter(Hn-Hf, 1).T, origin="lower", extent=(Ec[0], Ec[-1], Eg[0], Eg[-1]), aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
        ax[r, 2].invert_yaxis(); ax[r, 2].set(xlabel="BP - RP", ylabel="G", title=f"{name}, phi1 13-24: CMD on {int(Hn.sum())} minus off {Hf.sum():.0f}")
    for a_ in ax[-1, :2]:
        a_.set_xlabel(r"$\phi_1$ [deg]")
    fig.suptitle("Lower vs upper sky branch (phi1 13-24): do they share the stream's proper motions and CMD?", fontsize=11)
    fig.savefig(ROOT/f"plots/gaia_xsel_branches{tagc}.png", dpi=105); plt.close(fig)
    print("plots/gaia_xsel_marked.png, plots/gaia_xsel_branches.png")


if __name__ == "__main__":
    main()
