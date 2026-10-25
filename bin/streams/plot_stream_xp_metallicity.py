#!/usr/bin/env python3
"""Gaia XP (Andrae+2023 XGBoost) [M/H] of Ibata+2024 stream 54/55 members by segment vs omega Cen members at matched G.
Inputs (WSDB pulls, see JOURNAL 2026-10-08): results/plot_data/stream_xp_metallicity_members.npz (members q3c_join 1'' to
gaia_dr3, left join koposov.andrae2023_v21_table1), results/plot_data/ocen_members_andrae.npz (omega Cen PM members 0.2-0.6 deg).
Panels: [M/H] histograms for G < 15.5 and G 16.5-17.6 (omega Cen shaded), and stream-54 sky map coloured by [M/H] for G < 15.5
(filled) over G >= 15.5 (small).
Usage: python bin/streams/plot_stream_xp_metallicity.py -> plots/stream_xp_metallicity.png
"""
import os
from pathlib import Path
import numpy as np, astropy.coordinates as coord, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[2]
import sys; sys.path.insert(0, str(ROOT/"src")); import ocen_stream  # noqa: E401,E402,F401  (loads the local .env)
t = Table.read(os.path.expandvars("$OCEN_CATALOGUES/streamfinder_ibata2024_dr3.fits")); t = t[np.isin(np.asarray(t["Stream"]), [54, 55])]
d = np.load(ROOT/"results/plot_data/stream_xp_metallicity_members.npz"); oc = np.load(ROOT/"results/plot_data/ocen_members_andrae.npz")
ok = (d["source_id"] == np.asarray(t["Gaia"], np.int64)[d["xid"]]) & np.isfinite(d["mh"])
x = d["xid"][ok]; mh = d["mh"][ok]; G = d["g"][ok]
g = coord.SkyCoord(np.asarray(t["RAdeg"], float)[x], np.asarray(t["DEdeg"], float)[x], unit="deg").galactic
l = np.where(g.l.deg > 180, g.l.deg-360, g.l.deg); b = g.b.deg; st = np.asarray(t["Stream"])[x]
grp = {"54 b < 25 (next to cluster)": ((st == 54) & (b < 25), "C0"), "54 b 25-32": ((st == 54) & (b >= 25) & (b < 32), "C2"),
       "54 knee b > 32": ((st == 54) & (b >= 32), "C3"), "55 (Fimbulthul-S)": (st == 55, "C1")}
fig, ax = plt.subplots(1, 3, figsize=(19, 5.2)); bins = np.arange(-2.6, 0.31, 0.15)
for a, (lo, hi) in zip(ax[:2], ((0, 15.5), (16.5, 17.6))):
    k0 = (oc["g"] >= lo) & (oc["g"] < hi)
    a.hist(oc["mh"][k0], bins, density=True, color="0.85", label=f"omega Cen members (N={k0.sum()}, median {np.median(oc['mh'][k0]):+.2f})")
    for n, (m, c) in grp.items():
        k = m & (G >= lo) & (G < hi)
        if k.sum() < 5:
            continue
        p = ks_2samp(mh[k], oc["mh"][k0]).pvalue
        a.hist(mh[k], bins, density=True, histtype="step", lw=2, color=c, label=f"{n}: N={k.sum()}, median {np.median(mh[k]):+.2f}, KS p={p:.2g}")
    a.set_title(f"G {'< 15.5' if lo == 0 else f'{lo}-{hi}'}" + (" (giants; XP reliable)" if lo == 0 else " (dwarf-dominated, XP biased)"))
    a.set_xlabel("[M/H] (Andrae+2023 XGBoost)"); a.set_ylabel("density"); a.legend(fontsize=7, loc="upper left"); a.grid(alpha=0.3)
k = st == 54
ax[2].scatter(l[k & (G >= 15.5)], b[k & (G >= 15.5)], s=4, color="0.75", label="G >= 15.5")
sc = ax[2].scatter(l[k & (G < 15.5)], b[k & (G < 15.5)], c=mh[k & (G < 15.5)], cmap="coolwarm", vmin=-2.2, vmax=-0.6, s=45, ec="k", lw=0.4,
                   label="G < 15.5")
ax[2].set_xlim(-25, -65); ax[2].set_xlabel("l [deg]"); ax[2].set_ylabel("b [deg]"); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
fig.colorbar(sc, ax=ax[2], label="[M/H]"); ax[2].set_title("stream 54 members with XP [M/H]")
fig.suptitle("Ibata+2024 streams 54/55: Gaia XP metallicities vs omega Cen at matched G", fontsize=11)
fig.tight_layout(); fig.savefig(ROOT/"plots/stream_xp_metallicity.png", dpi=85)
