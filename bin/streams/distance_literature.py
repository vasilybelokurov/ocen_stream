#!/usr/bin/env python3
"""Published heliocentric distances to omega Cen (NGC 5139) vs the main-run posterior and the prior box (2026-11-01).

Values are taken from the papers' abstracts (checked on ADS); distance moduli converted with d = 10^(mu/5 + 1) pc and
sigma_d = 0.2 ln(10) d sigma_mu (statistical and systematic terms added in quadrature where both are quoted).
Usage: OCEN_RUNS_DIR=... PYTHONPATH=src python bin/streams/distance_literature.py  -> plots/distance_literature.png
"""
import os
from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ocen_stream import pilots as PL, jaxspray as J

ROOT = Path(__file__).resolve().parents[2]
mu = lambda m, e: (10**(m/5+1)/1e3, 0.2*np.log(10)*10**(m/5+1)/1e3*e)
# (label, method, d [kpc], sigma [kpc], ADS bibcode)
LIT = [("Thompson+2001", "eclipsing binary", 5.36, 0.30, "2001AJ....121.3089T"),
       ("van de Ven+2006", "dynamical", 4.8, 0.3, "2006A&A...445..513V"),
       ("Del Principe+2006", "RR Lyrae NIR PL", *mu(13.70, np.hypot(0.06, 0.06)), "2006ApJ...652..362D"),
       ("Navarrete+2017", "RR Lyrae + T2C NIR PL", 5.52, 0.27, "2017A&A...604A.120N"),
       ("Braga+2018", "RR Lyrae NIR PLZ", *mu(13.674, np.hypot(0.008, 0.038)), "2018AJ....155..137B"),
       ("Bono+2019", "RR Lyrae opt+NIR", *mu(13.720, np.hypot(0.002, 0.030)), "2019ApJ...870..115B"),
       ("Braga+2020", "RR Lyrae + T2C NIR PL", *mu(13.65, np.hypot(0.07, 0.01)), "2020A&A...644A..95B"),
       ("Soltis+2021", "Gaia EDR3 parallax", 5.24, 0.11, "2021ApJ...908L...5S"),
       ("Baumgardt & Vasiliev 2021", "combination of methods", 5.43, 0.05, "2021MNRAS.505.5957B"),
       ("Haeberle+2025", "kinematic (oMEGACat)", 5.494, 0.061, "2025ApJ...983...95H")]


def main():
    ht = J.load_host(ROOT/"results/gpu/host_tables.npz"); z = np.load(Path(os.environ["OCEN_RUNS_DIR"])/"smc_main_n1200/smc_final.npz")
    d = np.array([PL.to_params(u, ht)["dist"] for u in z["x"]]); q = np.percentile(d, [2.5, 16, 50, 84, 97.5])
    fig, ax = plt.subplots(figsize=(11, 5.5)); y = np.arange(len(LIT))[::-1]
    ax.axvspan(*PL.LIM["dist"], color="0.93", zorder=0, label=f"prior box {PL.LIM['dist'][0]}-{PL.LIM['dist'][1]} kpc")
    ax.axvspan(q[0], q[4], color="tab:orange", alpha=0.25, zorder=1, label="stream fit, 95%"); ax.axvspan(q[1], q[3], color="tab:orange", alpha=0.6, zorder=1, label="stream fit, 68%")
    for yi, (lab, meth, di, ei, bib) in zip(y, LIT):
        c = "tab:red" if "combination" in meth else "k"; ax.errorbar(di, yi, xerr=ei, fmt="o", color=c, capsize=3, ms=5, zorder=3)
        ax.text(6.02, yi, f"{lab}: {meth}, {di:.2f} ± {ei:.2f}", va="center", fontsize=8)
    ax.set(yticks=[], xlim=(4.4, 6.0), xlabel="heliocentric distance [kpc]", title="omega Cen distance: literature vs the stream fit (smc_main_n1200)")
    ax.legend(loc="lower left", fontsize=8); fig.subplots_adjust(right=0.58)
    out = ROOT/"plots/distance_literature.png"; fig.savefig(out, dpi=110); print("plot:", out)
    for lab, meth, di, ei, bib in LIT:
        print(f"{lab:26s} {meth:26s} {di:6.3f} ± {ei:5.3f}  (fit median - this) / sigma = {(q[2]-di)/ei:5.1f}   https://ui.adsabs.harvard.edu/abs/{bib}")
    print("stream fit 2.5/16/50/84/97.5%:", np.round(q, 3))


if __name__ == "__main__":
    main()
