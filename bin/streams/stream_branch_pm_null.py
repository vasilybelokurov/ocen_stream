#!/usr/bin/env python3
"""Calibrated null for the branch PM fits of stream_branch_compare.py (2026-10-24).

Same selection and 2-D PM mixture (SB.fit_pm2) in empty sky windows |y - (m(phi1) + k)| < HW, k = -2.0 ... -5.5 deg (0.5 steps,
below the main stream; the branch is above), in the bins phi1 12-16, 16-20 (20-22 dropped: branch fit spike-like).
1. False detections: distribution of the fit significance in the null windows vs the branch's (8.9, 6.5 sigma).
2. Error calibration: into each null window inject N_branch stars drawn from the MAIN fit (mu, s) convolved with errors of
   random window stars; refit; pulls of mu_ra, mu_de and of the along-PM offset; fraction of fits that land > 1 mas/yr off.
Usage: PYTHONPATH=src python bin/streams/stream_branch_pm_null.py -> results/plot_data/stream_branch_pm_null.json
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
import stream_branch_compare as SB

KS = np.arange(-2.0, -5.51, -0.5); NINJ = 4


def main():
    R, sc, m, m2, pm = SB.tracks(); B = json.loads((ROOT/"results/plot_data/stream_branch_compare.json").read_text())
    from gaia_cmd_mask import cmd_pass
    from gaia_footprint_plots import gc_masked
    raw = dict(np.load(ROOT/"results/streams/gaia_footprint_dr3.npz")); lg = np.where(raw["l"] > 180, raw["l"]-360, raw["l"]).astype(float)
    keep = cmd_pass(2.0) & ~gc_masked(lg, raw["b"].astype(float)); C = {k: v[keep].astype(float) for k, v in raw.items() if v.dtype.kind == "f"}
    u = C["phi1"]; y = C["dphi2"]-sc(u); vra, vde = C["pmra"]-pm["pmra"](u), C["pmdec"]-pm["pmdec"](u)
    box = (np.abs(vra) < SB.WRA) & (np.abs(vde) < SB.WDE); rng = np.random.default_rng(11); out = []
    for row in B["bins"][:2]:
        a0, a1 = row["phi1"]; c = 0.5*(a0+a1); kb = box & (u >= a0) & (u < a1); bg = kb & (np.abs(y) >= SB.OFF[0]) & (np.abs(y) < SB.OFF[1])
        vb = np.column_stack((vra[bg], vde[bg])); mn, br = row["main"], row["branch"]; nin = int(round(br["N"]))
        a_, d_ = pm["pmra"](np.array([c]))[0]+mn["mu_ra"], pm["pmdec"](np.array([c]))[0]+mn["mu_de"]; ux, uy = np.array([a_, d_])/np.hypot(a_, d_)
        sig0, pulls = [], []
        for k in KS:
            s = kb & (np.abs(y-(m(u)+k)) < SB.HW); v = np.column_stack((vra[s], vde[s])); e = np.column_stack((C["pmra_error"][s], C["pmdec_error"][s]))
            r = SB.fit_pm2(v, e, vb); sig0.append(r["sig"])
            for _ in range(NINJ):
                j = rng.integers(0, len(v), nin); ei = e[j]
                vi = np.column_stack((mn["mu_ra"]+np.sqrt(mn["s_ra"]**2+ei[:, 0]**2)*rng.standard_normal(nin), mn["mu_de"]+np.sqrt(mn["s_de"]**2+ei[:, 1]**2)*rng.standard_normal(nin)))
                ok = (np.abs(vi[:, 0]) < SB.WRA) & (np.abs(vi[:, 1]) < SB.WDE)
                ri = SB.fit_pm2(np.vstack((v, vi[ok])), np.vstack((e, ei[ok])), vb)
                dra, dde = ri["mu_ra"]-mn["mu_ra"], ri["mu_de"]-mn["mu_de"]; along = dra*ux+dde*uy
                ea = np.hypot(ux*ri["mu_ra_err"], uy*ri["mu_de_err"])
                pulls.append((dra/ri["mu_ra_err"], dde/ri["mu_de_err"], along, along/ea, ri["sig"]))
        sig0, P = np.array(sig0), np.array(pulls)
        res = dict(phi1=(a0, a1), null_sig=sig0.tolist(), branch_sig=br["sig"], n_inj=nin, pull_ra_sd=float(np.std(P[:, 0])), pull_de_sd=float(np.std(P[:, 1])),
                   along_mean=float(P[:, 2].mean()), along_sd=float(np.std(P[:, 2])), pull_along_sd=float(np.std(P[:, 3])), frac_off_1=float(np.mean(np.hypot(P[:, 0]*0, 0)+(np.abs(P[:, 2]) > 1))),
                   inj_sig_median=float(np.median(P[:, 4])))
        out.append(res)
        print(f"phi1 {a0:.0f}-{a1:.0f}: null sig {np.round(np.sort(sig0), 1).tolist()} (branch {br['sig']:.1f}) | injected N {nin} at the main PM: recovered sig median {res['inj_sig_median']:.1f}; "
              f"pull sd mu_ra {res['pull_ra_sd']:.2f}, mu_de {res['pull_de_sd']:.2f}; along-PM offset {res['along_mean']:+.2f} sd {res['along_sd']:.2f} mas/yr "
              f"(pull sd {res['pull_along_sd']:.2f}); |along| > 1: {res['frac_off_1']:.2f}", flush=True)
    json.dump(out, open(ROOT/"results/plot_data/stream_branch_pm_null.json", "w"), indent=1)


if __name__ == "__main__":
    main()
