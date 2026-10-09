#!/usr/bin/env python3
"""The KDE model of a spray as used in score_chi_kde: p(x) = (1/K) sum_k KDE_k(x) over chi bins (5 kpc Myr; trailing arm, age < 700
Myr, >= 20 particles; Scott bandwidths with floors 0.5 deg, 0.2 mas/yr; v_los >= 5 km/s; distance kernel Scott with floor
0.1 kpc for display). Projected into 2D panels as log-density images (each particle: Gaussian kernel with its bin's bandwidths,
weight 1/(K n_k)); members overplotted. Row 1: all chi bins (as in the likelihood). Row 2: chi <= 105 only.
Usage: python bin/streams/plot_kde_model.py om34.5_an24_am1.2 -> plots/kde_model_<tag>.png
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


def kde_images(m, chi_max):
    k = (m["chi"] > 0) & (m["age"] < 700) & (m["chi"] <= chi_max)
    chi = m["chi"][k]; Q = {q: m[q][k] for q in FLOOR}
    edges = np.arange(0., chi.max()+5., 5.)
    bins = [(chi >= e0) & (chi < e0+5.) for e0 in edges]; bins = [s for s in bins if s.sum() >= 20]; K = len(bins)
    imgs = []
    for xq, yq, xr, yr in PANELS:
        xg = np.linspace(*xr, 200); yg = np.linspace(*yr, 200); img = np.zeros((200, 200))
        for s in bins:
            n = s.sum(); f = n**(-1/8.)
            hx = max(f*Q[xq][s].std(), FLOOR[xq]); hy = max(f*Q[yq][s].std(), FLOOR[yq])
            gx = np.exp(-0.5*((xg[:, None]-Q[xq][s][None, :])/hx)**2)/(np.sqrt(2*np.pi)*hx)
            gy = np.exp(-0.5*((yg[:, None]-Q[yq][s][None, :])/hy)**2)/(np.sqrt(2*np.pi)*hy)
            img += (gy @ gx.T)/(K*n)
        imgs.append((xg, yg, img))
    return imgs, K


for tag in sys.argv[1:]:
    m = dict(np.load(ROOT/f"results/streams/spray_grid2/{tag}.npz"))
    fig, ax = plt.subplots(2, 6, figsize=(30, 9.5))
    for r, cm in enumerate((1e9, 105.)):
        imgs, K = kde_images(m, cm)
        for c, ((xq, yq, xr, yr), (xg, yg, img)) in enumerate(zip(PANELS, imgs)):
            a = ax[r, c]; lv = np.log10(img+1e-30); vmax = lv.max()
            a.imshow(lv, origin="lower", aspect="auto", extent=(*xr, *yr), cmap="Blues", vmin=vmax-3.5, vmax=vmax)
            if yq in ("vlos",):
                hv = np.isfinite(d["v"]); a.errorbar(d["b"][hv], d["v"][hv], yerr=d["e_v"][hv], fmt="o", color="C3", mfc="w", ms=5, lw=0.8)
            elif yq == "d":
                a.errorbar([17.5]+[0.5*(c_["b"][0]+c_["b"][1]) for c_ in cmd], [5.43]+[5.43*c_["d_ratio"] for c_ in cmd],
                           yerr=[0]+[5.43*c_["d_ratio"]*np.log(10)/5*c_["dm_err"] for c_ in cmd], fmt="s", color="C3", mfc="w", ms=6)
            else:
                a.plot(d[xq], d[yq], ".", color="C3", ms=1.2, alpha=0.5)
            a.set_xlim(*(xr[::-1] if xq == "l" else xr)); a.set_ylim(*yr); a.set_xlabel(xq); a.set_ylabel(yq)
            if c == 0:
                a.set_title(("all chi bins (as in the likelihood)" if r == 0 else "chi <= 105 only") + f", K = {K}", fontsize=9)
    fig.suptitle(f"KDE model of spray {tag} (blue, log density, 3.5 dex) vs stream-54 members (red; v_los stars and CMD distances open)", fontsize=12)
    fig.tight_layout(); fig.savefig(ROOT/f"plots/kde_model_{tag}.png", dpi=62); print(f"plots/kde_model_{tag}.png")
