#!/usr/bin/env python3
"""The KDE model of a spray as used in score_chi_kde: p(x) = (1/K) sum_k KDE_k(x) over chi bins (5 kpc Myr; trailing arm, age < 700
Myr, >= 20 particles; Scott bandwidths with floors 0.5 deg, 0.2 mas/yr; v_los >= 5 km/s; distance kernel Scott with floor
0.1 kpc for display). Projected into 2D panels as log-density images (each particle: Gaussian kernel with its bin's bandwidths,
weight 1/(K n_k)); members overplotted. Row 1: all chi bins (as in the likelihood). Row 2: chi <= 105 only.
Usage: python bin/streams/plot_kde_model.py om34.5_an24_am1.2 [--contours] -> plots/kde_model_<tag>[_contours].png
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"bin/streams"))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from spray_bar_grid2 import load_data

d = load_data()
cmd = json.loads((ROOT/"results/plot_data/stream54_cmd_distance.json").read_text())["bins"]
PANELS = (("l", "b", (-75, -10), (5, 45)), ("b", "l", (10, 45), (-70, -30)), ("b", "pmra", (10, 45), (-22, 0)),
          ("b", "pmdec", (10, 45), (-15, -3)), ("b", "vlos", (10, 45), (120, 300)), ("b", "d", (10, 45), (2.5, 6.5)))
FLOOR = dict(l=0.5, b=0.5, pmra=0.2, pmdec=0.2, vlos=5., d=0.1)
# --robust: bandwidth = Scott factor x 1.4826 MAD, floors 0.2 deg / 0.1 mas/yr / 3 km/s / 0.05 kpc, caps 1 deg / 0.5 mas/yr / 10 km/s / 0.3 kpc
ROB_FLOOR = dict(l=0.2, b=0.2, pmra=0.1, pmdec=0.1, vlos=3., d=0.05)
ROB_CAP = dict(l=1.0, b=1.0, pmra=0.5, pmdec=0.5, vlos=10., d=0.3)
ROBUST = "--robust" in sys.argv


def bw(x, q, f):
    if ROBUST:
        return float(np.clip(f*1.4826*np.median(np.abs(x-np.median(x))), ROB_FLOOR[q], ROB_CAP[q]))
    return max(f*x.std(), FLOOR[q])


def kde_images(m, chi_max):
    k = (m["chi"] > 0) & (m["age"] < 700) & (m["chi"] <= chi_max)
    chi = m["chi"][k]; Q = {q: m[q][k] for q in FLOOR}
    edges = np.arange(0., chi.max()+5., 5.)
    bins = [(chi >= e0) & (chi < e0+5.) for e0 in edges]; bins = [s for s in bins if s.sum() >= 20]; K = len(bins)
    imgs = []
    for xq, yq, xr, yr in PANELS:
        xg = np.linspace(*xr, 300); yg = np.linspace(*yr, 300); img = np.zeros((300, 300))
        for s in bins:
            n = s.sum(); f = n**(-1/8.)
            hx = bw(Q[xq][s], xq, f); hy = bw(Q[yq][s], yq, f)
            gx = np.exp(-0.5*((xg[:, None]-Q[xq][s][None, :])/hx)**2)/(np.sqrt(2*np.pi)*hx)
            gy = np.exp(-0.5*((yg[:, None]-Q[yq][s][None, :])/hy)**2)/(np.sqrt(2*np.pi)*hy)
            img += (gy @ gx.T)/(K*n)
        imgs.append((xg, yg, img))
    return imgs, K


for tag in [x for x in sys.argv[1:] if not x.startswith("--")]:
    m = dict(np.load(ROOT/f"results/streams/spray_grid2/{tag}.npz"))
    fig, ax = plt.subplots(2, 6, figsize=(30, 9.5))
    for r, cm in enumerate((1e9, 105.)):
        imgs, K = kde_images(m, cm)
        for c, ((xq, yq, xr, yr), (xg, yg, img)) in enumerate(zip(PANELS, imgs)):
            a = ax[r, c]; lv = np.log10(img+1e-30); vmax = lv.max()
            CONT = "--contours" in sys.argv
            if not CONT:
                a.imshow(lv, origin="lower", aspect="auto", extent=(*xr, *yr), cmap="Blues", vmin=vmax-3.5, vmax=vmax)
            if yq in ("vlos",):
                hv = np.isfinite(d["v"]); a.errorbar(d["b"][hv], d["v"][hv], yerr=d["e_v"][hv], fmt="o", color="C3", mfc="w", ms=5, lw=0.8)
            elif yq == "d":
                a.errorbar([17.5]+[0.5*(c_["b"][0]+c_["b"][1]) for c_ in cmd], [5.43]+[5.43*c_["d_ratio"] for c_ in cmd],
                           yerr=[0]+[5.43*c_["d_ratio"]*np.log(10)/5*c_["dm_err"] for c_ in cmd], fmt="s", color="C3", mfc="w", ms=6)
            else:
                a.plot(d[xq], d[yq], ".", color="0.35" if CONT else "C3", ms=1.5 if CONT else 1.2, alpha=0.6 if CONT else 0.5, zorder=1)
            if CONT:
                xg = np.linspace(*xr, 200) if False else xg
                a.contour(xg, yg, lv, levels=vmax-np.array([2.5, 2.0, 1.5, 1.0, 0.5]), cmap="viridis", linewidths=1.3, zorder=3)
            a.set_xlim(*(xr[::-1] if xq == "l" else xr)); a.set_ylim(*yr); a.set_xlabel(xq); a.set_ylabel(yq)
            if c == 0:
                a.set_title(("all chi bins (as in the likelihood)" if r == 0 else "chi <= 105 only") + f", K = {K}", fontsize=9)
    fig.suptitle(f"KDE model of spray {tag} (blue, log density, 3.5 dex) vs stream-54 members (red; v_los stars and CMD distances open)", fontsize=12)
    sfx = ("_contours" if "--contours" in sys.argv else "") + ("_robust" if ROBUST else "")
    if sfx:
        fig.suptitle(f"KDE model of spray {tag}: contours at 10^-0.5 ... 10^-2.5 of the peak density (yellow = highest); stream-54 members as points", fontsize=12)
    fig.tight_layout(); fig.savefig(ROOT/f"plots/kde_model_{tag}{sfx}.png", dpi=62); print(f"plots/kde_model_{tag}{sfx}.png")
