#!/usr/bin/env python3
"""Benchmark of the MW-LMC orbit (src/ocen_dm/streams/lmc.py, rigid MW = Hunter+2024 axi) against Vasiliev+2021 (Tango) N-body
runs (Zenodo 4300977, $OCEN_CATALOGUES/vasiliev21_tango_zenodo4300977, potentials_{axisym,triax}/trajlmc.txt, accel.txt;
time in AGAMA units of 0.9778 Gyr, today t = 0; x mirrored to our frame). Compared vs lookback time: LMC distance, and the MW
reflex velocity dv_MW(t) = v_MW(t) - v_MW(today) = integral_t^0 acc dt' (acc = minus the MW-centre acceleration, both codes).
M_LMC = 1.0, 1.5 (Tango's value), 2.0 e11. Output plots/lmc_vs_tango.png and a table on stdout.
Usage: PYTHONPATH=src python bin/streams/lmc_vs_tango.py   (1 core, ~10 s)"""
import os
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from ocen_dm.streams.restricted import AGAMA_T_MYR
from ocen_dm.streams.lmc import mw_lmc_orbit

ROOT = Path(__file__).resolve().parents[2]; TD = Path(os.path.expanduser("$OCEN_CATALOGUES/vasiliev21_tango_zenodo4300977"))
T = 1955.58/AGAMA_T_MYR; MIR = np.array([-1, 1, 1])


def reflex(t, acc):   # dv_MW(t) = int_t^0 acc dt' for lookback t < 0 (t ascending)
    m = t <= 0; tt, aa = t[m], acc[m]
    cum = cumulative_trapezoid(aa, tt, axis=0, initial=0.); return tt, cum[-1]-cum


rows = {}
for s in ("axisym", "triax"):
    tr = np.loadtxt(TD/f"potentials_{s}/trajlmc.txt"); ac = np.loadtxt(TD/f"potentials_{s}/accel.txt")
    rows[f"Tango {s} (1.5e11)"] = dict(t=tr[:, 0], d=np.linalg.norm(tr[:, 1:4], axis=1), x0=tr[np.argmin(abs(tr[:, 0])), 1:7]*np.r_[MIR, MIR],
                                       rv=reflex(ac[:, 0], ac[:, 1:4]*MIR))
for M in (1.0e11, 1.5e11, 2.0e11):
    o = mw_lmc_orbit(M, T); t = o["t"]-T
    rows[f"this work {M:.1e}"] = dict(t=t, d=np.linalg.norm(o["lmc"][:, :3], axis=1), x0=o["lmc"][np.argmin(abs(t))], rv=reflex(t, o["acc"]))

print(f"{'model':26s} {'x_LMC today [kpc]':>24s} {'v_LMC today [km/s]':>26s} peri[kpc] age[Myr] D(-0.5,-1,-2 Gyr)  |dv_MW|(-0.25,-0.5,-1,-2 Gyr) [km/s]")
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
for k, r in rows.items():
    t, d = r["t"], r["d"]; m = (t > -1) & (t <= 0); i = np.argmin(np.where(m, d, np.inf))
    tv, dv = r["rv"]; dvm = np.linalg.norm(dv, axis=1)
    at = lambda q, tt, x: float(np.interp(x/(AGAMA_T_MYR/1e3), tt, q))
    print(f"{k:26s} {np.array2string(r['x0'][:3], precision=1):>24s} {np.array2string(r['x0'][3:], precision=1):>26s} {d[i]:8.1f} {-t[i]*AGAMA_T_MYR:7.0f}  "
          + " ".join(f"{at(d, t, x):5.0f}" for x in (-0.5, -1, -2)) + "   " + " ".join(f"{at(dvm, tv, x):5.1f}" for x in (-0.25, -0.5, -1, -2)))
    ls = "--" if k.startswith("Tango") else "-"
    ax[0].plot(t*AGAMA_T_MYR/1e3, d, ls, label=k); ax[1].plot(tv*AGAMA_T_MYR/1e3, dvm, ls, label=k)
ax[0].set(xlim=(-3, 0.05), ylim=(0, 400), xlabel="t - t_today [Gyr]", ylabel="LMC Galactocentric distance [kpc]")
ax[1].set(xlim=(-3, 0.05), xlabel="t - t_today [Gyr]", ylabel="|v_MW(t) - v_MW(today)| [km/s]")
ax[0].legend(fontsize=8); fig.suptitle("MW-LMC orbit: rigid Hunter24-axi MW + truncated-NFW LMC (solid) vs Vasiliev+2021 N-body (dashed)")
fig.savefig(ROOT/"plots/lmc_vs_tango.png", dpi=130); print("plots/lmc_vs_tango.png")
