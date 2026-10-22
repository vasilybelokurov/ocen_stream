#!/usr/bin/env python3
"""Gallery: real data vs the best model of each comparison fit (2026-10-22).

Columns (best chain by 256k mean; its best-scoring 256k seed is plotted): inflated bar 12.5 deg (step-2 profile), published bar at
12.5 / 17.5 / 27 deg (null_fits.py, fixedbar_fit.py), static bar, no bar (Hunter axisymmetric, disc mass free), no bar (McMillan17).
Rows vs phi1: Delta phi2, mu_alpha*, mu_delta, v_los: members (orange points) and model debris (chi > 0, age < 700 Myr) as
contours of the conditional density p(y | phi1) (as in the score): 2-D histogram (0.5 deg in phi1, ~1/60 of the y range), Gaussian
smoothing of 1.5 bins, each phi1 column normalised to its peak, contour levels LEVELS of the column peak;
and binned residuals data - model (2-deg bins; plot_bestfit_real.resid). Column titles: Delta ln L (+ prior, 256k) vs the
inflated 12.5-deg best.
Output: plots/nullfits_gallery.png; --paper: six columns (published bar at 17.5 deg left out), page width, short titles, no
residual legend text beyond the symbols -> plots/note_gallery.{pdf,png} (paper/stream_fits_note)
Usage: PYTHONPATH=src python bin/streams/nullfits_gallery.py [--paper]
"""
import glob, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from grid4_common import d, du, dx
from plot_bestfit_real import load, resid

NF, FB, PR = (ROOT/f"results/streams/{s}" for s in ("nullfits_20261022", "fixedbar_20261021", "profile_step2_20261020"))


LEVELS = (0.05, 0.2, 0.5, 0.8)


def contours(ax, u, y, ext, lim, color="navy", lw_scale=1.):
    """Contours of p(y | phi1): smoothed 2-D histogram, each phi1 column scaled to its peak."""
    H, xe, ye = np.histogram2d(u, y, bins=(int(round((ext[1]-ext[0])/0.5)), 60), range=(ext, lim))
    H = gaussian_filter(H, 1.5); pk = H.max(1, keepdims=True); H = np.divide(H, pk, out=np.zeros_like(H), where=pk > H.max()*1e-3)
    ax.contour(0.5*(xe[1:]+xe[:-1]), 0.5*(ye[1:]+ye[:-1]), H.T, levels=LEVELS, colors=color, linewidths=tuple(lw_scale*w for w in (0.5, 0.7, 1.1, 1.6)), linestyles=("dotted", "solid", "solid", "solid"), zorder=2)


def best_seed(pattern):
    recs = [json.loads(Path(f).read_text()) for f in glob.glob(str(pattern))]
    r = max(recs, key=lambda r: r["total_jk"]); return Path(pattern).parent/f"{r['run_id']}.npz", r["params"]


def columns():
    nf = json.loads((NF/"result.json").read_text()); fb = json.loads((FB/"result.json").read_text())
    prof = json.loads((PR/"profile.json").read_text()); ref = prof["12.5"]["mean"]+prof["12.5"]["ln_prior"]

    def best_chain(model):
        return max((k for k, v in nf.items() if v["model"] == model), key=lambda k: nf[k]["mean"])
    C = [("inflated bar, 12.5 deg", ref, PR/"runs/final_a12.5_[0-9].json")]
    for lab, model in (("published bar, 12.5 deg", "pub12.5"), ("published bar, 17.5 deg", "pub17.5")):
        k = best_chain(model); C.append((lab, nf[k]["mean"], NF/f"runs/final_{k}_[0-9].json"))
    C.append(("published bar, 27 deg", fb["from25_om42"]["mean"], FB/"runs/final_from25_om42_[0-9].json"))
    for lab, model in (("static bar (Omega_b = 0)", "static"), ("no bar (Hunter axi, disc free)", "nobar"), ("no bar (McMillan17)", "mcmillan")):
        k = best_chain(model); C.append((lab, nf[k]["mean"], NF/f"runs/final_{k}_[0-9].json"))
    return [(lab, tot-ref, pat) for lab, tot, pat in C]


PAPER_LABELS = {"inflated bar, 12.5 deg": "strong bar, 12.5$^\\circ$", "published bar, 12.5 deg": "published bar, 12.5$^\\circ$",
                "published bar, 27 deg": "published bar, 27$^\\circ$", "static bar (Omega_b = 0)": "non-rotating bar",
                "no bar (Hunter axi, disc free)": "no bar", "no bar (McMillan17)": "no bar, McMillan17"}


def main():
    paper = "--paper" in sys.argv
    C = columns(); ext = (-6, 30); e = np.arange(-2., 28.1, 2.)
    if paper:
        C = sorted([(PAPER_LABELS[lab], dl, pat) for lab, dl, pat in C if lab in PAPER_LABELS], key=lambda c: -c[1])
        plt.rcParams.update({"font.size": 6.5, "axes.labelsize": 6.5, "xtick.labelsize": 5.5, "ytick.labelsize": 5.5,
                             "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5})
    n = len(C)
    rows = (("x", dx, r"$\Delta\phi_2$ [deg]", (-12, 25)), ("pmra", d["pmra"], r"$\mu_{\alpha*}$ [mas/yr]", (-28, 2)),
            ("pmdec", d["pmdec"], r"$\mu_\delta$ [mas/yr]", (-17, -2)), ("v", d["v"], r"$v_{\rm los}$ [km/s]", (-50, 450)))
    fig, ax = plt.subplots(5, n, figsize=(7.1, 6.6) if paper else (3.3*n, 14), sharex=True, sharey="row", constrained_layout=True)
    if paper:
        fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.01)
    for j, (lab, dl, pat) in enumerate(C):
        npz, p = best_seed(pat); m = load(npz)
        for i, (q, dq, yl, lim) in enumerate(rows):
            ok = np.isfinite(dq)
            if q == "v":
                ax[i, j].plot(du[ok], dq[ok], "o", color="C1", mec="k", mew=0.3 if paper else 0.5, ms=2.2 if paper else 5, zorder=3)
            else:
                ax[i, j].plot(du[ok], dq[ok], ".", color="C1", ms=0.8 if paper else 2, alpha=0.4, zorder=1, rasterized=paper)
            contours(ax[i, j], m["u"], m[q], ext, lim, lw_scale=0.7 if paper else 1.)
            ax[i, j].set_ylim(lim)
            if j == 0:
                ax[i, j].set_ylabel(yl)
        r = resid(m, e)
        for q, mk, col in (("dphi2", "o", "C0"), ("pmra", "s", "C2"), ("pmdec", "^", "C3")):
            ax[4, j].plot(r["c"], r[q], mk+"-", color=col, ms=1.5 if paper else 3, lw=0.7 if paper else 1.5, label={"dphi2": r"$\Delta\phi_2$", "pmra": r"$\mu_{\alpha*}$", "pmdec": r"$\mu_\delta$"}[q])
        ax[4, j].axhline(0, color="k", lw=0.6); ax[4, j].set(xlabel=r"$\phi_1$ [deg]", xlim=ext, ylim=(-5, 4))
        if paper:
            ax[0, j].set_title(f"{lab}\n$\\Delta\\ln L$ = {dl:+.0f}", fontsize=6.5); continue
        bar = "" if p.get("amp", 0) == 0 else f"$\\Omega_b$ {p['omega']:.1f}, A {p['amp']:.2f}, S {p['size']:.2f}, ang {p['angle']:.0f}\n"
        ax[0, j].set_title(f"{lab}\n$\\Delta\\ln L$ = {dl:+.0f}\n{bar}d {p['dist']:.2f}, $v_c$ {p['vc']:.0f}, q {p['halo_q']:.2f}"
                           + (f", disc x{p['disc']:.2f}" if p.get("disc", 1) != 1 else ""), fontsize=9)
    ax[4, 0].set_ylabel("data $-$ model\n(deg; mas/yr)"); ax[4, 0].legend(*ax[4, 0].get_legend_handles_labels(), fontsize=5 if paper else 8, loc="upper left", ncol=3, frameon=False,
                    columnspacing=0.6, handlelength=1.2)
    if paper:
        for ext_ in ("pdf", "png"):
            fig.savefig(ROOT/f"plots/note_gallery.{ext_}", dpi=300)
        print(ROOT/"plots/note_gallery.pdf"); return
    fig.suptitle("Stream 54: members (orange) vs the best model of each kind (contours: debris < 700 Myr, p(y | $\\phi_1$) at 0.05 (dotted), 0.2, 0.5, 0.8 of the column peak); "
                 r"$\Delta\ln L$ (256k, + prior) vs the inflated-bar best; statistical sd ~80", fontsize=12)
    out = ROOT/"plots/nullfits_gallery.png"; fig.savefig(out, dpi=100); print(out)


if __name__ == "__main__":
    main()
