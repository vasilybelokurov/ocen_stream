#!/usr/bin/env python3
"""Time evolution of the stellar density profile in the isolation tests: restricted runner vs live N-body.

Rows: model A (no DM), model B (photometric-mass DM). Columns:
 (1) restricted runner, rho(r, t) / rho(r, 0) for t = 10..100 Myr (one-hue ramp, light = early);
 (2) live pyfalcon run of the same ICs, t = 2..20 Myr (its isolation test length);
 (3) restricted runner, shell ratio vs time for selected shells, with the +-1 sigma Poisson band of
     each shell (dashed) and the +-5% acceptance band (grey).
Spherical shells about the median of stars + remnants, 16 log bins 0.3-100 pc; stars only.
Data: results/plot_data/streams_isolation_evolution.json. Usage:
  python bin/streams/plot_isolation_evolution.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "ocen-df-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from ocen_stream.analysis import load, run_particles, snapshot_times

EDGES = np.logspace(np.log10(0.3), 2, 17)
MID = np.sqrt(EDGES[1:]*EDGES[:-1])
RAMP = LinearSegmentedColormap.from_list("blue", ["#b7d3f6", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"])
SHELL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]      # categorical slots 1-5
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"
MODELS = (("A_nodm", "A: no DM"), ("B_dm_phot", "B: DM, photometric stellar mass"))


def profile(xv, species):
    lum = species <= 1
    c = np.median(xv[lum, :3], axis=0)
    r = np.linalg.norm(xv[species == 0, :3]-c, axis=1)*1e3
    return np.histogram(r, EDGES)[0].astype(float)


def series(run):
    mass, species = run_particles(run)
    _, times = snapshot_times(run)
    rows = []
    for t in np.unique(times):
        _, xv = load(run, t)
        rows.append(profile(xv, species))
    return np.unique(times), np.array(rows)


def style(ax):
    ax.set_xscale("log")
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)


def main():
    out = {"edges_pc": EDGES.tolist()}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.2))
    for row, (model, title) in enumerate(MODELS):
        runs = {"restricted": ROOT/f"results/streams/checks/{model}_iso", "live": ROOT/f"results/nbody/{model}/isolated"}
        res = {}
        for col, (tag, run) in enumerate(runs.items()):
            t, H = series(run)
            res[tag] = dict(t_myr=t.tolist(), counts=H.tolist())
            ax = axes[row, col]
            ok = H[0] >= 50
            ax.axhspan(0.95, 1.05, color="#efeee8", zorder=0)
            ax.axhline(1, color=MUTED, lw=0.8, zorder=1)
            sig = 1/np.sqrt(H[0, ok])
            ax.plot(MID[ok], 1+sig, ":", color=MUTED, lw=1); ax.plot(MID[ok], 1-sig, ":", color=MUTED, lw=1)
            later = np.where(t > 0)[0]
            for k in later:
                ax.plot(MID[ok], H[k, ok]/H[0, ok], "-", color=RAMP((t[k]-t[later[0]])/max(t[later[-1]]-t[later[0]], 1e-9)),
                        lw=1.6, marker="o", ms=3)
            ax.set_ylim(0.8, 1.2)
            style(ax)
            ax.set_xlabel("r [pc]", color=INK)
            ax.set_ylabel(r"$\rho(r,t)/\rho(r,0)$, stars", color=INK)
            ax.set_title(f"{title}\n{'restricted runner' if tag == 'restricted' else 'live N-body (pyfalcon)'}: "
                         f"t = {t[later[0]]:.0f}-{t[later[-1]]:.0f} Myr (light -> dark)", fontsize=10, color=INK, loc="left")
            ax.text(0.98, 0.04, "grey band: +-5%   dotted: +-1 sigma Poisson", transform=ax.transAxes, ha="right",
                    fontsize=8, color=MUTED)
        # shell ratio vs time, restricted
        t, H = np.array(res["restricted"]["t_myr"]), np.array(res["restricted"]["counts"])
        ax = axes[row, 2]
        ax.axhspan(0.95, 1.05, color="#efeee8", zorder=0); ax.axhline(1, color=MUTED, lw=0.8)
        for c, i in zip(SHELL_COLORS, (1, 2, 3, 6, 10)):
            ratio = H[:, i]/H[0, i]
            s = 1/np.sqrt(H[0, i])
            ax.plot(t, ratio, "-o", color=c, lw=1.8, ms=4, label=f"{EDGES[i]:.2f}-{EDGES[i+1]:.2f} pc (N0 = {H[0, i]:.0f})")
            ax.plot(t, 1+s+0*t, "--", color=c, lw=0.8, alpha=0.6); ax.plot(t, 1-s+0*t, "--", color=c, lw=0.8, alpha=0.6)
        ax.set_xscale("linear"); ax.set_ylim(0.8, 1.2)
        ax.grid(True, color=GRID, lw=0.6)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_xlabel("t [Myr]", color=INK); ax.set_ylabel("shell density / initial", color=INK)
        ax.set_title(f"{title}\nrestricted runner: selected shells vs time (dashed: +-1 sigma)", fontsize=10, color=INK, loc="left")
        ax.legend(fontsize=8, frameon=False, labelcolor=INK)
        out[model] = res
    fig.suptitle("Isolation tests: stellar density evolution, restricted runner (100 Myr) vs live N-body (20 Myr), same ICs",
                 fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(ROOT/"plots/streams_isolation_evolution.png", dpi=120)
    (ROOT/"results/plot_data/streams_isolation_evolution.json").write_text(json.dumps(out))
    print("wrote plots/streams_isolation_evolution.png")


if __name__ == "__main__":
    main()
