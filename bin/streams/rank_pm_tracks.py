#!/usr/bin/env python3
"""Find saved candidates that fit the PM ridge rather than only the KDE score.

This is a descriptive diagnostic, not a new likelihood or a calibrated
model ranking. Compare both PMs over phi1=10-28 deg, with equal bin weight.
Sky predictions are marginalized over PM. Require at least 20 model
particles in each displayed sky tube. The marginal sky median is recorded,
but does not identify the observed branch when several branches coexist.
Multiple saved seeds are averaged at the level of the predicted ridge.
All input filenames and parameters are recorded for subsequent verification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bin/streams"))

import numpy as np
from grid4_common import PMRA, PMDEC, d
from diagnose_pm_profiles import pm_basis, summarize
from refine_bar_profiles import atomic_json


def candidates(refinement_label):
    groups = []
    expression = re.compile(r"om([\d.]+)_an([\d.]+)_am([\d.]+)_d([\d.]+)\.npz")
    for f in sorted((ROOT / "results/streams/spray_grid4").glob("*.npz")):
        match = expression.fullmatch(f.name)
        if match is None:
            continue
        omega, angle, amp, dist = map(float, match.groups())
        if angle not in (16, 28):
            continue
        groups.append(dict(name="grid4/" + f.stem, angle=angle, files=[str(f.relative_to(ROOT))],
                           parameters=dict(omega=omega, amp=amp, size=1., dist=dist, pmra=PMRA, pmdec=PMDEC), seeds=[1]))
    broad = json.loads((ROOT / "results/plot_data/profile_stage1.json").read_text())
    for p in broad:
        files = [f"results/streams/profile_s1/{p['tag']}_s{s}.npz" for s in (1, 2, 3)]
        if all((ROOT / f).exists() for f in files):
            groups.append(dict(name="profile_s1/" + p["tag"], angle=p["an"], files=files, seeds=[1, 2, 3],
                               parameters={k: p[src] for k, src in zip(("omega", "amp", "size", "dist", "pmra", "pmdec"),
                                                                    ("om", "am", "size", "dist", "pmra", "pmdec"))}))
    for angle in (16, 28):
        payload = json.loads((ROOT / "results/plot_data" / refinement_label / f"fit_an{angle}.json").read_text())
        for p in payload["log"]:
            groups.append(dict(name=refinement_label + "/" + Path(p["spray"]).stem, angle=angle,
                               files=[p["spray"]], seeds=[payload["seed"]], original_lnL=p["lnL"],
                               parameters={k: p[k] for k in ("omega", "amp", "size", "dist", "pmra", "pmdec")}))
    return groups


def assess(group, basis):
    rows_by_seed = [summarize(dict(np.load(ROOT / f)), np.arange(10., 28.1, 2.), basis) for f in group["files"]]
    if any(len(rows) != 9 for rows in rows_by_seed):
        return dict(group, valid=False, reason="missing a comparison bin")
    observed = np.array([[r["observed"][key][1] for key in ("pmra", "pmdec")] for r in rows_by_seed[0]])
    predicted = np.array([[[r["model"][key][1] for key in ("pmra", "pmdec")] for r in rows] for rows in rows_by_seed])
    sky_obs = np.array([r["observed"]["across_sky"][1] for r in rows_by_seed[0]])
    sky_pred = np.array([[r["model"]["across_sky"][1] for r in rows] for rows in rows_by_seed]).mean(axis=0)
    residual = predicted.mean(axis=0) - observed
    tubes = np.array([[r["n_model_in_sky_tube"] for r in rows] for rows in rows_by_seed])
    sky_rms = float(np.sqrt(np.mean((sky_pred - sky_obs)**2)))
    return dict(group, valid=True, sky_rms_deg=sky_rms, min_model_tube_count=int(tubes.min()),
                follows_sky=bool(sky_rms < 1. and tubes.min() >= 20),
                has_sky_support=bool(tubes.min() >= 20),
                pm_rms=float(np.sqrt(np.mean(residual**2))),
                mid_pm_rms=float(np.sqrt(np.mean(residual[:4]**2))),
                knee_pm_rms=float(np.sqrt(np.mean(residual[4:]**2))),
                max_abs_pm_residual=float(np.max(np.abs(residual))),
                per_seed_pm_rms=np.sqrt(np.mean((predicted - observed)**2, axis=(1, 2))).tolist(),
                bin_centers=list(np.arange(11., 28., 2.)), predicted_pm=predicted.mean(axis=0).tolist(),
                observed_pm=observed.tolist(), residual_pm=residual.tolist(), sky_residual=(sky_pred - sky_obs).tolist())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="bar_profile_refinement_20261013")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    out = ROOT / "results/plot_data" / a.label / "ridge_bank.json"
    payload = json.loads(out.read_text()) if out.exists() else dict(
        metric="Equal-bin RMS over phi1 10-28 in both PMs; descriptive only; sky-only weights; |dphi2|<2 display tube", rows=[])
    payload["sky_gate"] = "At least 20 particles per sky tube; marginal sky median is not a valid branch-matching gate"
    for r in payload["rows"]:
        if r.get("valid"):
            r["has_sky_support"] = r["min_model_tube_count"] >= 20
    done = {r["name"] for r in payload["rows"]}
    basis = pm_basis(d["l"], d["b"])
    groups = candidates(a.label)
    for i, group in enumerate(groups[:a.limit]):
        if group["name"] in done:
            continue
        t0 = time.time()
        row = assess(group, basis)
        row["seconds"] = time.time() - t0
        payload["rows"].append(row)
        atomic_json(out, payload)
        if (i + 1) % 20 == 0:
            print(f"{i+1}/{len(groups)}: {group['name']} pm RMS {row.get('pm_rms', np.nan):.3f}", flush=True)
    for angle in (16, 28):
        rows = sorted((r for r in payload["rows"] if r["angle"] == angle and r.get("has_sky_support")), key=lambda r: r["pm_rms"])
        print("BEST", angle, json.dumps([{k: r[k] for k in ("name", "parameters", "pm_rms", "mid_pm_rms", "knee_pm_rms", "sky_rms_deg", "max_abs_pm_residual")} for r in rows[:5]]), flush=True)
    atomic_json(out, payload)


if __name__ == "__main__":
    main()
