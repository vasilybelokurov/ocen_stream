#!/usr/bin/env python3
"""Figures and diagnostics for the pericentric-release test (bin/streams/peri_release_test.py; 2026-10-21).

For 12.5 and 25 deg, release uniform (step-2 final, seed of the plotted best) vs peri sigma = 10 and 30 Myr (seed 2200001):
  plots/peri_release_debris.png : 4 rows (dphi2, pmra, pmdec, v_los) x 6 columns, debris (chi > 0, age < 700) coloured by
                                  release age, members in black;
  plots/peri_release_density.png: debris counts along phi1 (on-tube, |dphi2 - track| < 3 and |dPM| < 2) vs member counts, both
                                  normalised over phi1 14-22;
  stdout + results/plot_data/peri_release_diag.json: on-tube and branch fractions (phi1 6-22) and the Gaia-search
                                  branch/track particle ratio R_pred (windows of bin/streams/branch_search_gaia.py).
Usage: PYTHONPATH=src python bin/streams/peri_release_plots.py
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
from grid4_common import d, du, dx, project_gc, GC
import branch_search_gaia as BS

UNI = {"12.5": "results/streams/profile_step2_20261020/runs/final_a12.5_3.npz", "25": "results/streams/profile_step2_20261020/runs/final_a25_1.npz"}
PERI = "results/streams/peri_release_20261021/runs/a{lab}_s{sg}_seed2200001.npz"


def load(f):
    m = dict(np.load(ROOT/f)); u, x = project_gc(m["l"], m["b"], GC); k = (m["chi"] > 0) & (m["age"] < 700.)
    return dict(u=u[k], x=x[k], pmra=m["pmra"][k], pmdec=m["pmdec"][k], v=m["vlos"][k], age=m["age"][k])


def main():
    runs = {}
    for lab in ("12.5", "25"):
        runs[f"{lab} uniform"] = load(UNI[lab])
        for sg in (10, 30):
            runs[f"{lab} peri {sg}"] = load(PERI.format(lab=lab, sg=sg))
    mem = BS.members(); win, _ = BS.model_windows(mem); diag = {}
    print("run                 on-tube  branch  R_pred(Gaia windows)  [phi1 6-22, |dphi2|<15]")
    for key, p in runs.items():
        on = br = tot = 0; A = T = 0
        for (a0, a1), W in win.items():
            km = (du >= a0) & (du < a1); mx, mp, md = np.median(dx[km]), np.median(d["pmra"][km]), np.median(d["pmdec"][km])
            k = (p["u"] >= a0) & (p["u"] < a1) & (np.abs(p["x"]) < 15); tot += k.sum()
            on += np.sum(k & (np.abs(p["x"]-mx) < 3) & (np.hypot(p["pmra"]-mp, p["pmdec"]-md) < 2)); br += np.sum(k & (p["pmra"] < mp-3))
            kk = (p["u"] >= a0) & (p["u"] < a1)
            A += np.sum(kk & BS.sky_A(p["x"], W["track"]) & BS.in_pm(p["pmra"], p["pmdec"], W["bpm"]))
            T += np.sum(kk & BS.sky_T(p["x"], W["track"]) & BS.in_pm(p["pmra"], p["pmdec"], W["tpm"]))
        diag[key] = dict(on_tube=on/tot, branch=br/tot, R_pred=A/T)
        print(f"{key:18s}  {on/tot:6.2f}  {br/tot:6.2f}  {A/T:6.2f}")
    (ROOT/"results/plot_data/peri_release_diag.json").write_text(json.dumps(diag, indent=1))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    keys = ["12.5 uniform", "12.5 peri 30", "12.5 peri 10", "25 uniform", "25 peri 30", "25 peri 10"]
    fig, ax = plt.subplots(4, 6, figsize=(27, 15), sharex=True, constrained_layout=True)
    for j, key in enumerate(keys):
        p = runs[key]; k = (p["u"] > -6) & (p["u"] < 32); o = np.argsort(-p["age"][k])
        for i, (q, dq, yl, lim) in enumerate((("x", dx, r"$\Delta\phi_2$", (-12, 15)), ("pmra", d["pmra"], r"$\mu_{\alpha*}$", (-28, 0)),
                                              ("pmdec", d["pmdec"], r"$\mu_\delta$", (-17, -2)), ("v", d["v"], r"$v_{\rm los}$", (-50, 300)))):
            sc = ax[i, j].scatter(p["u"][k][o], p[q][k][o], c=p["age"][k][o], s=0.2, cmap="turbo", vmin=0, vmax=700, rasterized=True)
            okd = np.isfinite(dq); ax[i, j].plot(du[okd], dq[okd], "k.", ms=1.2 if q != "v" else 5, alpha=0.45)
            ax[i, j].set(ylim=lim, xlim=(-6, 32), ylabel=yl if j == 0 else None)
        ax[0, j].set_title(f"{key} (on-tube {diag[key]['on_tube']:.2f}, branch {diag[key]['branch']:.2f}, R_Gaia {diag[key]['R_pred']:.2f})", fontsize=10)
        ax[3, j].set_xlabel(r"$\phi_1$ [deg]")
    fig.colorbar(sc, ax=ax, shrink=0.4, label="release age [Myr ago]")
    fig.suptitle("Step-2 best models: uniform vs pericentric (bursty) release, same parameters; debris coloured by release age, members black", fontsize=12)
    fig.savefig(ROOT/"plots/peri_release_debris.png", dpi=80); print(ROOT/"plots/peri_release_debris.png")
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6), constrained_layout=True)
    e = np.arange(-4, 30.01, 1.); c = 0.5*(e[1:]+e[:-1]); nm = np.histogram(du, e)[0]; ref = (c >= 14) & (c < 22)
    trk = np.array([np.median(dx[(du >= a) & (du < a+1)]) if np.sum((du >= a) & (du < a+1)) > 5 else np.nan for a in e[:-1]])
    v = np.isfinite(trk); trk[~v] = np.interp(c[~v], c[v], trk[v])
    for j, lab in enumerate(("12.5", "25")):
        ax[j].step(c, nm/nm[ref].sum(), where="mid", color="k", lw=2, label="members")
        for key, col in ((f"{lab} uniform", "C7"), (f"{lab} peri 30", "C1"), (f"{lab} peri 10", "C3")):
            p = runs[key]; on = np.abs(p["x"]-np.interp(p["u"], c, trk)) < 3; h = np.histogram(p["u"][on], e)[0]
            ax[j].step(c, h/h[ref].sum(), where="mid", color=col, lw=1.3, label=key)
        ax[j].set(yscale="log", xlabel=r"$\phi_1$ [deg]", ylabel="counts (normalised over phi1 14-22)", title=f"{lab} deg: density along the stream (|dphi2 - track| < 3)")
        ax[j].legend(fontsize=8)
    fig.savefig(ROOT/"plots/peri_release_density.png", dpi=100); print(ROOT/"plots/peri_release_density.png")


if __name__ == "__main__":
    main()
