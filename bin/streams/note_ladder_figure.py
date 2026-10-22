#!/usr/bin/env python3
"""Two-panel figure for the stream-fitting note (paper/stream_fits_note; 2026-10-22).

(a) Likelihood ladder: Delta ln L (+ prior; 256k, mean of 4 seeds) of the best fit of each model relative to the best
    inflated-bar fit (12.5 deg), with the statistical sd (~80, star bootstrap) and the spread between optimiser chains (~300)
    shaded. Sources: results/streams/profile_step2_20261020/profile.json, fixedbar_20261021/result.json,
    nullfits_20261022/result.json.
(b) Non-axisymmetric force max_phi |F_m>0| / |F_axi| (z = 0) against R for the published bar and the fitted bars, with omega Cen's
    pericentre and apocentre ranges (results/plot_data/bar_options.json; bin/streams/diagnostics/bar_options.py).
Output: plots/note_ladder.png, plots/note_ladder.pdf
Usage: PYTHONPATH=src python bin/streams/note_ladder_figure.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
RS = ROOT/"results/streams"


def ladder():
    prof = json.loads((RS/"profile_step2_20261020/profile.json").read_text())
    fb = json.loads((RS/"fixedbar_20261021/result.json").read_text()); nf = json.loads((RS/"nullfits_20261022/result.json").read_text())
    ref = prof["12.5"]["mean"]+prof["12.5"]["ln_prior"]
    best = lambda m: max(v["mean"] for v in nf.values() if v["model"] == m)
    rows = [(f"strong bar, {a}$^\\circ$", prof[a]["mean"]+prof[a]["ln_prior"], "C3") for a in ("12.5", "17.5", "22.5", "25", "27.5", "32.5")]
    rows += [("published bar, 12.5$^\\circ$", best("pub12.5"), "k"), ("published bar, 17.5$^\\circ$", best("pub17.5"), "k"),
             ("published bar, 27$^\\circ$", fb["from25_om42"]["mean"], "k"),
             ("no bar (disc, halo free)", best("nobar"), "C0"), ("non-rotating bar", best("static"), "C2"), ("no bar (McMillan17)", best("mcmillan"), "C0")]
    return [(lab, v-ref, c) for lab, v, c in rows]


def main():
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 3.2), constrained_layout=True, gridspec_kw=dict(width_ratios=(1.25, 1)))
    rows = ladder(); y = np.arange(len(rows))[::-1]
    ax[0].barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.7)
    for yi, (_, v, _) in zip(y, rows):
        ax[0].text(v-120, yi, f"{v:.0f}", ha="right", va="center", fontsize=6.5)
    ax[0].axvspan(-300, 0, color="0.85", zorder=0)
    ax[0].set_yticks(y); ax[0].set_yticklabels([r[0] for r in rows], fontsize=7)
    ax[0].set(xlim=(-9800, 300)); ax[0].set_xlabel(r"$\Delta\ln L$ relative to the best fit", fontsize=8); ax[0].tick_params(labelsize=7)
    ax[0].set_title("(a) best fit of each model", fontsize=8)
    B = json.loads((ROOT/"results/plot_data/bar_options.json").read_text()); R = B["R"]; pr = B["profiles"]
    for key, lab, st in (("hunter fitted 12.5 deg (A 1.37 S 1.33)", r"strong bar (12.5$^\circ$ fit)", "C3-"),
                         ("hunter fitted 25 deg (A 1.38 S 1.24)", r"strong bar (25$^\circ$ fit)", "C3--"),
                         ("hunter published (A=S=1)", "published bar", "k-"),
                         ("sormani long bar length x1.3, non-axi x2", r"long bar $\times2$ mass, $\times1.3$ length", "C2:")):
        ax[1].plot(R, pr[key]["z0"], st, lw=1.5, label=lab)
    ax[1].axvspan(1.5, 2.2, color="0.92"); ax[1].axvspan(6.4, 7.2, color="0.85")
    ax[1].text(1.85, 0.015, "peri", ha="center", fontsize=6.5); ax[1].text(6.8, 0.015, "apo", ha="center", fontsize=6.5)
    ax[1].set(ylim=(0, 0.47), xlim=(1, 8.2)); ax[1].tick_params(labelsize=7)
    ax[1].set_xlabel(r"$R$ [kpc]", fontsize=8); ax[1].set_ylabel(r"max$_\phi |F_{m>0}|/|F_{\rm axi}|$", fontsize=8)
    ax[1].set_title("(b) non-axisymmetric force in the plane", fontsize=8); ax[1].legend(fontsize=6.5, loc="upper right")
    for ext in ("png", "pdf"):
        fig.savefig(ROOT/f"plots/note_ladder.{ext}", dpi=200)
    print(ROOT/"plots/note_ladder.png")


if __name__ == "__main__":
    main()
