#!/usr/bin/env python3
"""Rescoring results (stream_model_rescore.csv): total chi2 vs bar/halo parameters, coloured by set (2026-10-25).

Rows without a score (the *_perstar.npz auxiliary files of search_real_wave3) are dropped. Replicate noise: groups of identical
parameters (>= 3 seeds) give the chi2 scatter at fixed model.
Usage: PYTHONPATH=src python bin/streams/stream_model_rescore_plot.py -> plots/stream_model_rescore.png
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]


def main():
    T = pd.read_csv(ROOT/"results/plot_data/stream_model_rescore.csv"); T = T[T.chi2_total.notna()]
    k = ["omega", "angle", "amp", "size", "halo_q", "vc", "dist", "pmra", "pmdec"]; g = T.groupby([T[x].round(4) for x in k]).chi2_total.agg(["count", "std", "mean"])
    g = g[g["count"] >= 3]; best = T.loc[T.chi2_total.idxmin()]
    P = (("omega", r"$\Omega_b$ [km/s/kpc]"), ("angle", "bar angle [deg]"), ("amp", "bar amplitude"), ("size", "bar size"), ("halo_q", "halo q"), ("vc", r"$v_c$ [km/s]"))
    fig, ax = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True); sets = sorted(T.set.unique())
    for a, (p, lab) in zip(ax.ravel(), P):
        for i, s in enumerate(sets):
            t = T[T.set == s]; a.scatter(t[p], t.chi2_total, s=6, color=f"C{i}", alpha=0.6, label=s.replace("_2026", " ").replace("1019", "").replace("1020", "").replace("1021", "").replace("1022", ""))
        a.scatter(best[p], best.chi2_total, marker="*", s=200, color="k", zorder=5)
        a.set(yscale="log", ylim=(100, 1e4), xlabel=lab, ylabel=r"$\chi^2$ (25 data points)")
    ax[0, 0].legend(fontsize=7, markerscale=2)
    fig.suptitle(f"Rescoring {len(T)} stored models with the single-component knot score; best (star) {Path(best.file).stem} in {best.set}: chi2 {best.chi2_total:.0f}; "
                 f"replicate scatter at fixed parameters: median sd {g['std'].median():.0f} ({len(g)} groups)")
    fig.savefig(ROOT/"plots/stream_model_rescore.png", dpi=85); print("plots/stream_model_rescore.png")


if __name__ == "__main__":
    main()
