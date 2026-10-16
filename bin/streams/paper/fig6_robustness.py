#!/usr/bin/env python3
"""Paper Fig. 6: other ingredients tested at fixed bar and cluster parameters (no refitting). Each entry is the change in
ln L relative to the same set-up without the change, at equal release-epoch number; where several seeds exist the mean
is shown with the seed scatter as an error bar. Grey band: +-150, typical seed scatter. Set-ups: '28' = 28-deg bar,
Omega_b 35.5, A 1.2, d 5.6 kpc, catalogue PMs (32000 epochs, seed 1; results/plot_data/phase2_variants.json); '28r' and
'16r' = refined candidates of 2026-10-13 (8000 epochs, seed 4; spiral_pilot_20261013, escape_pilot_20261013);
'16g' = grid-4 best (16 deg, 34.5, A 1.4, d 5.6; spin_exaggerate.json, 32000 epochs, seed 1); '16s' = sweep best
(16 deg, 34, S 1.0, A 1.2, d 5.6; lmc_spray_test.jsonl, 8000 epochs, seed 1); 'W' = 28 deg, 36, S 1.15, A 1.2, d 5.6
(mass_sweep.json and cluster_mass_test.json, 32000 epochs, seeds 1-3).
Output: plots/paper_fig6_robustness.pdf/.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"bin/streams/paper"))
import numpy as np

from paper_style import COLUMN, setup

plt = setup()
P = ROOT/"results/plot_data"


def entries():
    out = []                                   # (group, label, mean, sd or None)
    ph = json.loads((P/"phase2_variants.json").read_text()); b = ph["baseline"]["lnL"]
    for k, lab in (("M2 eta 0.002", r"slowing bar, $\eta=0.002$ (28)"), ("M2 eta 0.008", r"slowing bar, $\eta=0.008$ (28)"),
                   ("M3a Portail amp 1.2", "Portail et al. bar (28)"), ("M3b size 1.15", r"bar 15% longer (28)"),
                   ("M3b size 1.3", r"bar 30% longer (28)")):
        out.append(("Galactic bar", lab, ph[k]["lnL"]-b, None))
    sp = {r["name"]: r["lnL"] for r in json.loads((P/"spiral_pilot_20261013/pilot_an28_s4.json").read_text())["results"]}
    for k, lab in (("fiducial", "spiral arms, 22.5 km/s/kpc (28r)"), ("slow", "spiral arms, slow (28r)"), ("fast", "spiral arms, fast (28r)")):
        out.append(("Galactic bar", lab, sp[k]-sp["none"], None))
    se = json.loads((P/"spin_exaggerate.json").read_text())
    s16 = {r["factor"]: r["lnL"] for r in se if r["setup"].startswith("om34.5_an16")}
    for f, lab in ((1.0, "cluster rotation (16g)"), (10.0, r"cluster rotation $\times10$ (16g)")):
        out.append((r"$\omega$ Cen", lab, s16[f]-s16[0.0], None))
    cm = json.loads((P/"cluster_mass_test.json").read_text())
    ref = {r["seed"]: r["lnL"] for r in cm if r["mass_factor"] == 1.0}
    for f in (0.7, 1.5):
        dl = [r["lnL"]-ref[r["seed"]] for r in cm if r["mass_factor"] == f]
        out.append((r"$\omega$ Cen", rf"cluster mass $\times{f:g}$ (W)", np.mean(dl), np.std(dl, ddof=1)))
    for k, lab in (("M4b peri release s20", "release at pericentres (28)"), ("M4 window 1500, age<1000", "release over 1.5 Gyr (28)")):
        out.append((r"$\omega$ Cen", lab, ph[k]["lnL"]-b, None))
    for ang in (16, 28):
        e = {c["name"]: c["lnL"] for c in json.loads((P/f"escape_pilot_20261013/pilot_an{ang}_n8000_s4.json").read_text())["cases"]}
        out.append((r"$\omega$ Cen", f"Chen et al. escape law ({ang}r)", e["chen24_outer"]-e["fardal_outer"], None))
    lm = [json.loads(l) for l in (P/"lmc_spray_test.jsonl").open() if l.strip()]
    base = {r["seed"]: r["lnL"] for r in lm if r["an"] == 16 and r["mass"] == 0}
    for M in (1.0e11, 1.5e11, 2.0e11):
        dl = [r["lnL"]-base[r["seed"]] for r in lm if r["an"] == 16 and r["mass"] == M and r["seed"] in base]
        out.append(("Milky Way", rf"LMC, ${M/1e11:.1f}\times10^{{11}}\,{{\rm M}}_\odot$ (16s)", np.mean(dl), np.std(dl, ddof=1) if len(dl) > 1 else None))
    ms = json.loads((P/"mass_sweep.json").read_text())
    ref = {r["seed"]: r["lnL"] for r in ms if r["kind"] == "H" and abs(r["vc"]-228.8) < 0.1}
    for vc in (220.0, 240.0):
        dl = [r["lnL"]-ref[r["seed"]] for r in ms if r["kind"] == "H" and r["vc"] == vc]
        out.append(("Milky Way", rf"halo mass, $v_{{\rm c}}(R_0)={vc:.0f}$ km/s (W)", np.mean(dl), np.std(dl, ddof=1)))
    return out


def main():
    rows = entries()
    fig, ax = plt.subplots(figsize=(COLUMN, 4.3))
    y, labels, groups = [], [], []
    yy = 0
    for g in dict.fromkeys(r[0] for r in rows):
        groups.append((g, yy))
        for r in rows:
            if r[0] == g:
                y.append(yy); labels.append(r[1]); yy -= 1
                ax.errorbar(r[2], yy+1, xerr=r[3], fmt="o", ms=3.2, color="k", capsize=0, lw=0.8)
        yy -= 0.7
    ax.axvspan(-150, 150, color="0.88", lw=0); ax.axvline(0, color="k", lw=0.5)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6.3)
    for g, y0 in groups:
        ax.text(-3750, y0+0.55, g, fontsize=7, fontweight="bold", ha="left")
    ax.set_xlim(-3800, 1000); ax.set_ylim(yy+0.3, 1.2)
    ax.set_xlabel(r"change in $\ln L$ at fixed bar and cluster parameters")
    ax.tick_params(axis="y", length=0)
    for ext in ("pdf", "png"):
        fig.savefig(ROOT/f"plots/paper_fig6_robustness.{ext}")
    for r in rows:
        print(f"{r[0]:14s} {r[1]:55s} {r[2]:8.0f} {'' if r[3] is None else f'+- {r[3]:.0f}'}")


if __name__ == "__main__":
    main()
