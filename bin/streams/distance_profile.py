#!/usr/bin/env python3
"""What does the distance trade against? Profile of the main run's evaluations in distance (2026-11-01).

Reads every evaluated model of a run (OUT/log/*.npz: u, chi2; bank 7 unless rescored) and bins them in distance (0.025 kpc).
Per bin: the lowest chi2 and the median chi2 of the 10 best models (less sensitive to single lucky draws; bank sd ~4), and the
median parameters of those 10 best. Caveat: the SMC spends its cold-stage effort near the posterior, so bins far from it were only
explored at high temperature -- their minima are upper bounds on the true profile.
Plot: plots/distance_profile_<run>.png (chi2 profile with the literature distances; best-model parameters vs distance).
Usage: OCEN_RUNS_DIR=... PYTHONPATH=src python bin/streams/distance_profile.py --run smc_main_n1200
"""
import argparse, json, os
from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ocen_stream import pilots as PL, jaxspray as J

ROOT = Path(__file__).resolve().parents[2]; RUNS = Path(os.environ["OCEN_RUNS_DIR"])
LAB = dict(angle="bar angle [deg]", omega="Omega_b [km/s/kpc]", amp="A", size="S", pmra="pmra [mas/yr]", pmdec="pmdec [mas/yr]",
           vc="v_c(R0) [km/s]", halo_q="halo q")
NBEST = 10


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); a = ap.parse_args()
    ht = J.load_host(ROOT/"results/gpu/host_tables.npz"); fs = sorted((RUNS/a.run/"log").glob("*.npz"))
    U = np.concatenate([np.load(f)["u"] for f in fs]); C = np.concatenate([np.load(f)["chi2"] for f in fs]); B = np.concatenate([np.load(f)["bank"] for f in fs])
    ok = np.isfinite(C) & (B == 7); U, C = U[ok], C[ok]; print(f"{len(C)} models (bank 7, finite chi2) from {len(fs)} log files")
    lo, hi = PL.LIM["dist"]; d = lo+(hi-lo)*U[:, 4]; E = np.arange(lo, hi+1e-9, 0.025); c = 0.5*(E[1:]+E[:-1]); rows = []
    for i in range(len(c)):
        k = np.where((d >= E[i]) & (d < E[i+1]))[0]
        if len(k) < NBEST:
            continue
        b = k[np.argsort(C[k])[:NBEST]]; P = [PL.to_params(u, ht) for u in U[b]]
        rows.append(dict(d=float(c[i]), n=int(len(k)), cmin=float(C[b[0]]), c10=float(np.median(C[b])),
                         **{n: float(np.median([p[n] for p in P])) for n in LAB}))
    R = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    print(f"{'d':>6s} {'n':>6s} {'min':>7s} {'med10':>7s} " + " ".join(f"{n:>7s}" for n in LAB))
    for r in rows:
        print(f"{r['d']:6.3f} {r['n']:6d} {r['cmin']:7.1f} {r['c10']:7.1f} " + " ".join(f"{r[n]:7.4g}" for n in LAB))
    json.dump(rows, open(ROOT/f"results/plot_data/distance_profile_{a.run}.json", "w"), indent=1)
    fig, ax = plt.subplots(3, 3, figsize=(14, 10.5), constrained_layout=True); ax = ax.ravel()
    A = ax[0]; A.plot(R["d"], R["cmin"], "o-", c="0.5", ms=3, label="lowest chi2 in bin"); A.plot(R["d"], R["c10"], "o-", c="k", ms=4, label=f"median of {NBEST} best")
    A.axvspan(5.43-0.05, 5.43+0.05, color="tab:red", alpha=0.2, label="Baumgardt & Vasiliev 2021: 5.43 +- 0.05")
    A.axvspan(5.494-0.061, 5.494+0.061, color="tab:blue", alpha=0.15, label="Haeberle+2025: 5.49 +- 0.06")
    A.set(xlabel="distance [kpc]", ylabel="chi2 (bank 7)", yscale="log", title="chi2 profile in distance"); A.legend(fontsize=7)
    for A, n in zip(ax[1:], LAB):
        A.plot(R["d"], R[n], "o-", c="tab:purple", ms=4); A.axvspan(5.38, 5.48, color="tab:red", alpha=0.2); A.set(xlabel="distance [kpc]", ylabel=LAB[n])
    fig.suptitle(f"{a.run}: all {len(C)} evaluated models binned in distance (0.025 kpc); parameters = median of the {NBEST} best per bin\n"
                 "far from the posterior the bins were explored only at high temperature: their minima are upper bounds", fontsize=10)
    out = ROOT/f"plots/distance_profile_{a.run}.png"; fig.savefig(out, dpi=95); print("plot:", out)


if __name__ == "__main__":
    main()
