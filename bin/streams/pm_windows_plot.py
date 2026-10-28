#!/usr/bin/env python3
"""Proposed 3-deg PM window measurements vs the current spline data track, with each reviewer's drop rules (2026-10-28).

Input: results/plot_data/pm_windows_20261028.json (bin/streams/reviews/pm_windows_check_20261028.py: per 3-deg window a local
truncated Gaussian + linear trend + empirical background + per-star errors; the global spline track m(phi1) with its 1-sigma band
from the knot covariance). Values are offsets from the frozen PM track [mas/yr], as in the data vector.
Rows: Fable's and Codex's rules (docs/reviews/pm_windows_fable_codex_20261028.md), applied only where computable now:
  Fable -- stream stars >= 50, fraction f < 0.95, width < W/2;   Codex -- stream stars >= 30 (screen).
The mock-recovery and null tests of both reviewers have NOT been run: kept/dropped here is provisional.
Usage: python bin/streams/pm_windows_plot.py [--width 2] -> plots/pm_windows_20261028[_w2].png
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
WIDTH = float(sys.argv[sys.argv.index("--width")+1]) if "--width" in sys.argv else 3.
SUF = "" if WIDTH == 3 else f"_w{WIDTH:g}"
SPLINE, KEEP, DROP = "#2a78d6", "#1baf7a", "#eb6834"
RULES = dict(Fable=lambda w, W: [r for r, bad in (("< 50 stars", w["n_stream"] < 50), ("f at bound", w["f"] >= 0.95), ("width > W/2", w["width"] >= W/2)) if bad],
             Codex=lambda w, W: ["< 30 stars"] if w["n_stream"] < 30 else [])


def main():
    Z = json.loads((ROOT/f"results/plot_data/pm_windows_20261028{SUF}.json").read_text())
    fig, ax = plt.subplots(2, 2, figsize=(15, 9), sharex=True, gridspec_kw=dict(hspace=0.12, wspace=0.18))
    for i, (who, rule) in enumerate(RULES.items()):
        for j, q in enumerate(("pmra", "pmdec")):
            a = ax[i, j]; z = Z[q]; g, m, b = map(np.array, (z["grid"], z["m"], z["band"]))
            a.fill_between(g, m-b, m+b, color=SPLINE, alpha=0.18, lw=0); a.plot(g, m, color=SPLINE, lw=2, label="current data track (spline) ±1σ")
            a.errorbar(z["knots"], z["knot_values"], z["knot_err"], fmt="s", color=SPLINE, mfc="white", mew=1.5, ms=7, capsize=3, label="current spline knots (data vector)")
            for w in z["windows"]:
                x = 0.5*(w["a0"]+w["a1"]); why = rule(w, z["W"]); col = DROP if why else KEEP
                a.errorbar(x, w["c"], w["c_err"], fmt="o", color=col, mfc=col if not why else "white", mew=2, ms=8, capsize=3, lw=2, zorder=5)
                a.annotate(f"{w['n_stream']:.0f}", (x, w["c"]-w["c_err"]), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=8, color="0.3")
                if why:
                    a.annotate(", ".join(why), (x, w["c"]+w["c_err"]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, color=DROP)
            for e in np.arange(-2., 28.01, WIDTH):
                a.axvline(e, color="0.85", lw=0.8, zorder=0)
            a.axhline(0, color="0.6", lw=0.8); a.set_ylim(-1.2, 1.6) if q == "pmra" else a.set_ylim(-0.45, 0.6)
            a.set_title(f"{who}'s rules — {q}", fontsize=11); a.set_ylabel(f"{q} − frozen track [mas/yr]")
            if i == 1:
                a.set_xlabel("φ1 [deg]")
    h = [plt.Line2D([], [], color=SPLINE, lw=2), plt.Line2D([], [], color=SPLINE, marker="s", mfc="white", ls=""),
         plt.Line2D([], [], color=KEEP, marker="o", ls=""), plt.Line2D([], [], color=DROP, marker="o", mfc="white", mew=2, ls="")]
    fig.legend(h, ["current data track (spline) ±1σ", "current spline knots (data vector)", f"{WIDTH:g}° window: kept", f"{WIDTH:g}° window: dropped (reason above)"],
               loc="lower center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Proposed {WIDTH:g}° PM windows (local fit with background; number below = stream stars) vs the current spline track.\n"
                 "Fable: ≥ 50 stars, fraction < 0.95, width < W/2;  Codex: ≥ 30 stars.  Mock-recovery and null tests not yet run (provisional).", fontsize=11)
    fig.subplots_adjust(bottom=0.1)
    out = ROOT/f"plots/pm_windows_20261028{SUF}.png"; fig.savefig(out, dpi=110, bbox_inches="tight"); print(out)


if __name__ == "__main__":
    main()
