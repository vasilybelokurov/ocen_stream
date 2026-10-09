#!/usr/bin/env python3
"""Figures for the chi-binned KDE likelihood (score_chi_kde):
(1) plots/chikde_<tag>.png: per observable, the model density along chi (sum of the per-bin KDE marginals, normalised per bin,
    log colour), with members placed at their most likely chi bin (argmax_k KDE_k(x_i)); members with v_los in the v_los panel;
(2) same file, last row: sky with model particles coloured by chi and members coloured by their most likely chi;
(3) plots/chikde_grid_maps.png: chi-KDE Delta lnL and sky-score Delta lnL over (Omega_b, angle) per amplitude.
Usage: python bin/streams/plot_chi_kde.py om34.5_an24_am1.2
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"bin/streams")); sys.path.insert(0, str(ROOT/"src"))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from spray_bar_grid2 import load_data
from ocen_dm.streams.score import score_chi_kde

d = load_data()
for tag in sys.argv[1:]:
    m = dict(np.load(ROOT/f"results/streams/spray_grid2/{tag}.npz"))
    s = score_chi_kde(d, m); bc = np.array(s["best_chi"])
    k = (m["chi"] > 0) & (m["age"] < 700); chi = m["chi"][k]
    fig, ax = plt.subplots(2, 3, figsize=(19, 10)); ax = ax.ravel()
    for a, (q, lab, rng, h) in zip(ax[:5], (("l", "l [deg]", (-75, -10), 0.5), ("b", "b [deg]", (0, 45), 0.5), ("pmra", "pmra [mas/yr]", (-22, 0), 0.2),
                                            ("pmdec", "pmdec [mas/yr]", (-16, -3), 0.2), ("vlos", "v_los [km/s]", (50, 300), 5.))):
        ce = np.arange(0, chi.max()+5, 5.); yg = np.linspace(*rng, 300); img = np.zeros((len(yg), len(ce)-1))
        for j, (e0, e1) in enumerate(zip(ce[:-1], ce[1:])):
            sel = (chi >= e0) & (chi < e1)
            if sel.sum() < 20:
                continue
            y = m[q][k][sel]; f = sel.sum()**(-1/8.); hh = max(f*y.std(), h)
            img[:, j] = np.exp(-0.5*((yg[:, None]-y[None, :])/hh)**2).sum(1)/(np.sqrt(2*np.pi)*hh*sel.sum())
        a.imshow(np.log10(img+1e-6*img.max()), origin="lower", aspect="auto", extent=(0, ce[-1], *rng), cmap="Greys", vmin=np.log10(img.max())-3)
        dq = {"l": d["l"], "b": d["b"], "pmra": d["pmra"], "pmdec": d["pmdec"], "vlos": d["v"]}[q]
        jit = np.random.default_rng(1).uniform(-2.5, 2.5, len(bc))
        a.scatter(bc+jit, dq, s=3 if q != "vlos" else 30, c="C3", alpha=0.5 if q != "vlos" else 1, edgecolors="k" if q == "vlos" else "none", lw=0.4,
                  label="members at their most likely chi bin")
        a.set_xlim(0, 200); a.set_ylim(*rng); a.set_xlabel("Gibbons phase chi [kpc Myr]"); a.set_ylabel(lab)
    ax[0].legend(fontsize=7, loc="lower right")
    a = ax[5]; sc = a.scatter(m["l"][k], m["b"][k], c=chi, s=1.5, cmap="viridis", vmin=0, vmax=150, alpha=0.6)
    a.scatter(d["l"], d["b"], c=bc, s=4, cmap="viridis", vmin=0, vmax=150, edgecolors="k", lw=0.15)
    a.set_xlim(-10, -80); a.set_ylim(5, 45); a.set_xlabel("l [deg]"); a.set_ylabel("b [deg]"); fig.colorbar(sc, ax=a, label="chi (model) / most likely chi (members, black edge)")
    fig.suptitle(f"chi-binned KDE likelihood, spray {tag}: grey = model density per chi bin (log); red = members at their most likely chi; lnL {s['total']:.0f}", fontsize=11)
    fig.tight_layout(); fig.savefig(ROOT/f"plots/chikde_{tag}.png", dpi=75); print(f"plots/chikde_{tag}.png")
out = json.loads((ROOT/"results/plot_data/spray_bar_grid2_chikde.json").read_text())
om = sorted({o["omega"] for o in out}); an = sorted({o["angle"] for o in out}); am = sorted({o["amp"] for o in out})
fig, ax = plt.subplots(2, len(am), figsize=(5*len(am), 8.5), squeeze=False)
for i, (key, lab) in enumerate((("chikde", "chi-KDE"), ("sky_score", "sky-conditional"))):
    best = max(o[key] for o in out)
    for j, A in enumerate(am):
        Z = np.full((len(an), len(om)), np.nan)
        for o in out:
            if o["amp"] == A:
                Z[an.index(o["angle"]), om.index(o["omega"])] = o[key]-best
        im = ax[i, j].imshow(Z, origin="lower", aspect="auto", cmap="viridis", extent=(om[0]-0.75, om[-1]+0.75, an[0]-2, an[-1]+2))
        for o in out:
            if o["amp"] == A:
                ax[i, j].text(o["omega"], o["angle"], f"{o[key]-best:.0f}", ha="center", va="center", fontsize=7, color="w")
        ax[i, j].set_title(f"{lab}: amp {A:g}, Delta lnL", fontsize=9); ax[i, j].set_xlabel("Omega_b [km/s/kpc]"); ax[i, j].set_ylabel("bar angle [deg]")
        fig.colorbar(im, ax=ax[i, j])
fig.tight_layout(); fig.savefig(ROOT/"plots/chikde_grid_maps.png", dpi=70); print("plots/chikde_grid_maps.png")
