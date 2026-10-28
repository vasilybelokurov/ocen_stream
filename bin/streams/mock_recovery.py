#!/usr/bin/env python3
"""Asimov mock recovery check for the production SMC (2026-10-28; docs/reviews/final_run_fable_codex_20261027.md).

Reads a run of ocen_stream.smc on a mock made by gpu_mock_export.py, from the private runs repository ($OCEN_RUNS_DIR):
the final population (smc_final.npz, beta = 1; or the latest stage if the run is unfinished) and the beta = 0.1 stage, plus the
truth (inputs/<mock>/truth.json). Particles are equally weighted (resampled at every stage).
Reports, per parameter: truth, best particle, 16/50/84% of the population, (truth - median)/sd.
Joint check: squared Mahalanobis distance of the truth from the population (mean, covariance, 9-D) and the fraction of particles
nearer than the truth (truth inside the joint 68% region if < 0.68). Best chi2 vs the chi2 at the truth (0.27, bank 7).
Plot: plots/mock_recovery_<run>.png (pairs of the main parameters, beta = 0.1 grey and beta = 1 blue, truth red; chi2 vs angle).
Usage: OCEN_RUNS_DIR=... PYTHONPATH=src python bin/streams/mock_recovery.py --run smc_mock_scanbest_n300_v2_20261027 \
         --mock mock_scanbest_20261027
"""
import argparse, glob, json, os
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


def population(run):
    f = run/"smc_final.npz"
    if not f.exists():
        f = sorted(run.glob("smc_stage*.npz"))[-1]
    z = np.load(f); return z["x"], z["c"], float(z["beta"]), f.name


def stage_at(run, beta):
    for f in sorted(run.glob("smc_stage*.npz")):
        z = np.load(f)
        if abs(float(z["beta"])-beta) < 1e-9:
            return z["x"], z["c"]
    return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--mock", required=True); a = ap.parse_args()
    run = RUNS/a.run; ht = J.load_host(ROOT/"results/gpu/host_tables.npz")
    truth = json.loads((RUNS/"inputs"/a.mock/"truth.json").read_text())["truth"]
    x, c, beta, src = population(run); P = {k: np.array([PL.to_params(u, ht)[k] for u in x]) for k in PL.NAMES}; b = int(np.argmin(c))
    rows = {}
    for k in PL.NAMES:
        q = np.percentile(P[k], [16, 50, 84]); rows[k] = dict(truth=truth[k], best=float(P[k][b]), p16=float(q[0]), p50=float(q[1]), p84=float(q[2]),
                                                               z=float((truth[k]-q[1])/np.std(P[k])))
    ut = PL.to_unit(truth, ht); mu = x.mean(0); Ci = np.linalg.inv(np.cov(x.T))
    d2 = lambda u: float((u-mu)@Ci@(u-mu)); dt = d2(ut); frac = float(np.mean([d2(u) < dt for u in x]))
    res = dict(source=src, beta=beta, n=len(x), unique=int(len(np.unique(x, axis=0))), best_chi2=float(c[b]), median_chi2=float(np.median(c)),
               truth_mahalanobis2=dt, frac_particles_nearer_than_truth=frac, truth_inside_joint_68=frac < 0.68, params=rows)
    print(json.dumps(res, indent=1))
    # plot
    lo = stage_at(run, 0.1); fig, ax = plt.subplots(1, len(PAIRS)+1, figsize=(4.2*(len(PAIRS)+1), 4))
    for A, (p, q) in zip(ax, PAIRS):
        if lo is not None:
            Q = {k: np.array([PL.to_params(u, ht)[k] for u in lo[0]]) for k in (p, q)}; A.scatter(Q[p], Q[q], s=4, c="0.7", label="beta = 0.1")
        A.scatter(P[p], P[q], s=4, c="tab:blue", label=f"beta = {beta:.2g}"); A.scatter([truth[p]], [truth[q]], s=120, c="tab:red", marker="*", edgecolor="k", label="truth")
        A.set_xlabel(LAB[p]); A.set_ylabel(LAB[q])
    ax[0].legend(fontsize=8)
    if lo is not None:
        ax[-1].scatter([PL.to_params(u, ht)["angle"] for u in lo[0]], lo[1], s=4, c="0.7")
    ax[-1].scatter(P["angle"], c, s=4, c="tab:blue"); ax[-1].axvline(truth["angle"], c="tab:red"); ax[-1].axhline(0.27, c="tab:red", ls=":")
    ax[-1].set_yscale("log"); ax[-1].set_xlabel(LAB["angle"]); ax[-1].set_ylabel("chi2 (bank 7); dotted: chi2 at truth")
    fig.suptitle(f"Asimov mock recovery ({a.run}): population at beta = 0.1 (grey) and beta = {beta:.2g} (blue); truth = red star")
    out = ROOT/f"plots/mock_recovery_{a.run}.png"; fig.savefig(out, dpi=110, bbox_inches="tight"); print("plot:", out)


if __name__ == "__main__":
    main()
