#!/usr/bin/env python3
"""Progress of the fit_knot_nm.py chains: knot chi2 vs evaluation and the parameters visited (2026-10-25).

Reads OUT/runs/c*_e*.json (search evaluations) and OUT/result.json (256k finals, plotted at the right edge with their sd).
Usage: python bin/streams/fit_knot_plot.py results/streams/fitknot_20261025 -> plots/fit_knot_<dirname>.png
"""
import glob, json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
NAMES = ("angle", "omega", "amp", "size", "halo_q", "vc", "dist", "pmra", "pmdec")


def main():
    od = Path(sys.argv[1]); od = od if od.is_absolute() else ROOT/od
    R = json.loads((od/"result.json").read_text()); cids = sorted(R)
    fig, ax = plt.subplots(2, 5, figsize=(22, 8), constrained_layout=True); ax = ax.ravel()
    for j, cid in enumerate(cids):
        recs = sorted((json.loads(Path(f).read_text()) for f in glob.glob(str(od/f"runs/{cid}_e*.json"))), key=lambda r: r["run_id"])
        k = np.arange(len(recs)); y = np.array([r["chi2"]["total"] for r in recs]); col = f"C{j}"
        ax[0].plot(k, y, "o-", color=col, ms=3, lw=0.7, label=f"{cid}: final {R[cid]['mean']:.0f} $\\pm$ {R[cid]['sd']:.1f}")
        ax[0].errorbar(len(recs)+1+0.3*j, R[cid]["mean"], R[cid]["sd"], fmt="s", color=col, ms=6)
        for i, n in enumerate(NAMES):
            x = np.array([r["params"][n] for r in recs])
            ax[i+1].scatter(x, y, s=10, color=col, alpha=0.6); ax[i+1].plot(R[cid]["params"][n], R[cid]["mean"], "*", color=col, ms=14, mec="k")
    ax[0].set(yscale="log", xlabel="evaluation (128k releases); squares: 4 x 256k finals", ylabel=r"knot $\chi^2$ (25 data points)")
    ax[0].legend(fontsize=8)
    for i, n in enumerate(NAMES):
        ax[i+1].set(xlabel=n, ylabel=r"$\chi^2$", yscale="log", ylim=(140, 1500))
    fig.suptitle(f"{od.name}: Nelder-Mead chains, knot chi2 (stars: best point, 256k finals mean)")
    out = ROOT/f"plots/fit_knot_{od.name}.png"; fig.savefig(out, dpi=90); print(out.relative_to(ROOT))


if __name__ == "__main__":
    main()
