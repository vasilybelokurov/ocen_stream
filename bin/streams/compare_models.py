#!/usr/bin/env python3
"""Tidal tails of the three DM cases (prescribed progenitor potential, 200k star tracers each), side by side.

Runs (bin/streams/run_prescribed.py): A = no DM, B = moderate DM (photometric stellar mass), C = 5x B's DM inside
200 pc (bin/streams/make_shell_model.py). Same orbit (McMillan17, 1955.58 Myr, point mass), same star-tracer
counting (fractions per tracer, frozen r_max < 30 pc stars counted as bound).
Panels: (a) unbound stellar fraction vs time; (b) robust spreads of debris energy and Lz relative to the cluster;
(c) width and normal velocity dispersion of debris 0.3-2 kpc from the cluster; (d-f) present-day sky (l, b) of
the debris of each model, with omega Cen marked.
Usage: python bin/streams/compare_models.py
Output: plots/streams_dm_models.png, results/plot_data/streams_dm_models.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"bin"/"streams"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ocen-df-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ocen_dm.streams.analysis import frozen_star_mass, load, run_core, run_particles, snapshot_times, state, tail_metrics
from ocen_dm.streams.restricted import host_potential
from compare_live import sky

MODELS = [("A_nodm", "A: no DM", "#2a78d6"), ("B_dm_phot", "B: moderate DM", "#eb6834"),
          ("C_dm5x_shell", "C: 5x DM inside 200 pc", "#1baf7a")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"


def main():
    host = host_potential()
    out = {}
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3)
    axa, axb, axc = (fig.add_subplot(gs[0, i]) for i in range(3))
    for key, lab, col in MODELS:
        R = ROOT/"results/streams/prescribed"/key
        m, s = run_particles(R); core = run_core(R); fsm = frozen_star_mass(R)
        files, times = snapshot_times(R)
        rows = []
        for t in times:
            tt, xv = load(R, t)
            c, b, _ = state(xv, m, s, core=core)
            tm = tail_metrics(host, xv, c, b, s, mass=m, frozen_star_mass=fsm)
            rows.append(dict(t_myr=tt, **tm))
        out[key] = rows
        T = np.array([r["t_myr"] for r in rows])
        g = lambda k: np.array([r.get(k, np.nan) for r in rows], float)
        axa.plot(T, g("unbound_star_fraction"), "-", color=col, lw=2, label=lab)
        axb.plot(T, g("dLz_mad"), "-", color=col, lw=2, label=f"{lab}: dLz")
        axb2 = axb
        axc.plot(T, g("width_normal_pc"), "-", color=col, lw=2, label=f"{lab}")
        # sky today
        tt, xv = load(R, name="snap_today.npz")
        c, b, _ = state(xv, m, s, core=core)
        l, bb = sky(xv[~b]); lc, bc = sky(c[None, :])
        a = fig.add_subplot(gs[1, MODELS.index((key, lab, col))])
        a.plot(l, bb, ".", color=col, ms=1.2, alpha=0.4, rasterized=True)
        a.plot(lc, bc, "*", color=INK, ms=12)
        a.set_xlim(90, -90); a.set_ylim(-60, 60)
        a.set_xlabel("l [deg]", color=INK); a.set_ylabel("b [deg]", color=INK)
        a.set_title(f"{lab}: unbound tracers today, N = {(~b).sum():,} of {len(m)+int(round(fsm/1e-6)):,}",
                    fontsize=10, color=INK, loc="left")
        out[key+"_today_sky"] = dict(n_unbound=int((~b).sum()))
    axa.set_title("unbound stellar fraction (of all stars)", fontsize=10, color=INK, loc="left")
    axb.set_title("debris Lz spread about the cluster (1.48 MAD) [kpc km/s]", fontsize=10, color=INK, loc="left")
    axc.set_title("debris width normal to the orbit, 0.3-2 kpc from cluster [pc]", fontsize=10, color=INK, loc="left")
    for a in (axa, axb, axc):
        a.set_xlabel("t [Myr] (today = 1955.6)", color=INK); a.grid(True, color=GRID, lw=0.6); a.legend(fontsize=8, frameon=False)
    for a in fig.axes:
        a.tick_params(colors=MUTED, labelsize=9)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle("Tidal tails of omega Cen for three DM cases: prescribed progenitor potential, 200k massless star tracers, "
                 "McMillan17, 1.96 Gyr", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(ROOT/"plots/streams_dm_models.png", dpi=120)
    (ROOT/"results/plot_data/streams_dm_models.json").write_text(json.dumps(out, indent=1))
    for key, lab, col in MODELS:
        r = out[key][-1]
        print(f"{lab:26s} today: unbound {r['unbound_star_fraction']:.4f}, dE {r['dE_mad']:.0f}, dLz {r['dLz_mad']:.1f}, "
              f"width {r.get('width_normal_pc', np.nan):.0f} pc, sigma_N {r.get('sigma_v_normal', np.nan):.1f}")


if __name__ == "__main__":
    main()
