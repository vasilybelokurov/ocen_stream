#!/usr/bin/env python3
"""Posterior summary of a production SMC run on the real data, plus the emcee drift check (2026-11-01).

Reads from the private runs repository ($OCEN_RUNS_DIR/<run>): smc_final.npz (beta = 1, equally weighted after resampling), the
beta = 0.1 stage, and polish.h5 (emcee started from the final population).
Reports, per parameter: best particle, 16/50/84% of the population; chi2 of the population (best, median).
Drift check: walker mean of each parameter every 50 emcee steps, in units of the SMC sd, (mean_emcee - mean_SMC)/sd_SMC, and the
ratio of walker sd to SMC sd. A converged SMC population is a sample of the target, so the walkers should not drift (|shift| of order
1/sqrt(40 walkers / tau-correlated) ~ 0.2-0.3 sd is noise).
Plot: plots/main_posterior_<run>.png (pairs: beta = 0.1 grey, beta = 1 blue, last emcee walkers orange; chi2 vs angle; drift traces).
Usage: OCEN_RUNS_DIR=... PYTHONPATH=src python bin/streams/main_posterior.py --run smc_main_n1200
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
LAB = dict(angle="bar angle [deg]", omega="Omega_b [km/s/kpc]", amp="A", size="S", dist="distance [kpc]", pmra="pmra [mas/yr]",
           pmdec="pmdec [mas/yr]", vc="v_c(R0) [km/s]", halo_q="halo q")
PAIRS = (("angle", "omega"), ("angle", "halo_q"), ("angle", "vc"), ("dist", "vc"), ("amp", "size"))


def phys(U, ht):
    return {k: np.array([PL.to_params(u, ht)[k] for u in U]) for k in PL.NAMES}


def main():
    import h5py
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); a = ap.parse_args()
    run = RUNS/a.run; ht = J.load_host(ROOT/"results/gpu/host_tables.npz")
    z = np.load(run/"smc_final.npz"); x, c = z["x"], z["c"]; P = phys(x, ht); b = int(np.argmin(c))
    lo = next((np.load(f) for f in sorted(run.glob("smc_stage*.npz")) if abs(float(np.load(f)["beta"])-0.1) < 1e-9), None)
    with h5py.File(run/"polish.h5") as h:
        ch = h["mcmc/chain"][:]; lp = h["mcmc/log_prob"][:]                                    # (steps, walkers, D), (steps, walkers)
    rows = {}
    for k in PL.NAMES:
        q = np.percentile(P[k], [16, 50, 84]); rows[k] = dict(best=float(P[k][b]), p16=float(q[0]), p50=float(q[1]), p84=float(q[2]))
    # drift: emcee walker mean / sd vs the SMC population, in physical units, every 50 steps
    mu, sd = {k: P[k].mean() for k in PL.NAMES}, {k: P[k].std() for k in PL.NAMES}
    steps = np.arange(0, len(ch)+1, 50); steps[0] = 1
    W = [phys(ch[s-1], ht) for s in steps]
    drift = {k: dict(shift_sd=[float((w[k].mean()-mu[k])/sd[k]) for w in W], width_ratio=[float(w[k].std()/sd[k]) for w in W]) for k in PL.NAMES}
    chi2_w = -2*lp
    res = dict(n=len(x), unique=int(len(np.unique(x, axis=0))), best_chi2=float(c[b]), median_chi2=float(np.median(c)),
               emcee=dict(steps=len(ch), walkers=ch.shape[1], best_chi2=float(chi2_w.min()), median_chi2_last=float(np.median(chi2_w[-1])),
                          steps_report=steps.tolist()), params=rows, drift=drift)
    print(json.dumps(res, indent=1))
    fig, ax = plt.subplots(2, len(PAIRS)+1, figsize=(4.2*(len(PAIRS)+1), 8.4))
    Wl = phys(ch[-1], ht)
    for A, (p, q) in zip(ax[0], PAIRS):
        if lo is not None:
            Q = phys(lo["x"], ht); A.scatter(Q[p], Q[q], s=3, c="0.75", label="SMC beta = 0.1")
        A.scatter(P[p], P[q], s=3, c="tab:blue", label="SMC beta = 1")
        A.scatter(Wl[p], Wl[q], s=10, c="tab:orange", edgecolor="k", lw=0.3, label="emcee step 400")
        A.scatter([P[p][b]], [P[q][b]], s=120, c="tab:red", marker="*", edgecolor="k", label="best particle")
        A.set_xlabel(LAB[p]); A.set_ylabel(LAB[q])
    ax[0, 0].legend(fontsize=7)
    A = ax[0, -1]
    if lo is not None:
        A.scatter(phys(lo["x"], ht)["angle"], lo["c"], s=3, c="0.75")
    A.scatter(P["angle"], c, s=3, c="tab:blue"); A.set_yscale("log"); A.set_xlabel(LAB["angle"]); A.set_ylabel("chi2")
    for i, k in enumerate(PL.NAMES):
        A = ax[1, min(i, len(PAIRS))] if i < len(PAIRS) else ax[1, len(PAIRS)]
    A = ax[1, 0]
    for k in PL.NAMES:
        A.plot(steps, drift[k]["shift_sd"], "-o", ms=3, label=k)
    A.axhline(0, c="k", lw=0.8); A.set_xlabel("emcee step"); A.set_ylabel("(walker mean - SMC mean) / SMC sd"); A.legend(fontsize=7, ncol=2)
    A = ax[1, 1]
    for k in PL.NAMES:
        A.plot(steps, drift[k]["width_ratio"], "-o", ms=3, label=k)
    A.axhline(1, c="k", lw=0.8); A.set_xlabel("emcee step"); A.set_ylabel("walker sd / SMC sd")
    A = ax[1, 2]
    A.plot(np.median(chi2_w, 1), c="tab:orange", label="emcee walkers, median"); A.plot(chi2_w.min(1), c="tab:orange", ls=":", label="emcee, min")
    A.axhline(np.median(c), c="tab:blue", label="SMC median"); A.axhline(c.min(), c="tab:blue", ls=":", label="SMC min")
    A.set_xlabel("emcee step"); A.set_ylabel("chi2"); A.legend(fontsize=7)
    for k, A in zip(("angle", "omega", "vc"), ax[1, 3:]):
        A.hist(P[k], 40, density=True, color="tab:blue", alpha=0.6, label="SMC beta = 1")
        A.hist(Wl[k], 15, density=True, histtype="step", color="tab:orange", lw=1.5, label="emcee step 400"); A.set_xlabel(LAB[k])
    ax[1, 3].legend(fontsize=7)
    fig.suptitle(f"{a.run}: posterior (SMC, N = {len(x)}, beta = 1) and emcee drift check ({len(ch)} steps x {ch.shape[1]} walkers)")
    fig.tight_layout(); out = ROOT/f"plots/main_posterior_{a.run}.png"; fig.savefig(out, dpi=100); print("plot:", out)


if __name__ == "__main__":
    main()
