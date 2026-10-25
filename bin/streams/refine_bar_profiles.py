#!/usr/bin/env python3
"""Joint nuisance refinement at fixed bar angles, with held-out spray checks.

Uses grid4_common.score unchanged. All six nuisance coordinates move together.
The corotation bound uses Hunter's measured A2 half-maximum length, 4.25 kpc;
the original 5 kpc length can be selected. Every evaluation and its spray are
preserved. This is an exploratory profile, not a calibrated angle measurement.

Example: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python
  bin/streams/refine_bar_profiles.py fit --angle 28 --maxfev 160
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bin/streams"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from scipy.optimize import minimize

from grid4_common import OCEN_OBS, make_spray, score
from ocen_stream.cluster import OCEN_DISTANCE_KPC
from ocen_stream.restricted import HUNTER24_DIR, agama_kpc

NAMES = ("omega", "amp", "size", "dist", "pmra", "pmdec")
OFFSET = np.array([34., 1.2, 1.1, 5.6, OCEN_OBS["pmra"], OCEN_OBS["pmdec"]])
SCALE = np.array([1., .1, .1, .05, .027, .027])
LOW = np.array([30., .8, .85, 5.25, OCEN_OBS["pmra"] - .081, OCEN_OBS["pmdec"] - .081])
HIGH = np.array([40., 1.7, 1.5, 5.95, OCEN_OBS["pmra"] + .081, OCEN_OBS["pmdec"] + .081])


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=1))
    os.replace(temp, path)


def coordinates(row):
    return np.array([row[k] for k in ("om", "am", "size", "dist", "pmra", "pmdec")])


def measurement_term(x, distance_sigma):
    return float(-.5 * (((x[3] - OCEN_DISTANCE_KPC) / distance_sigma)**2
                        + ((x[4] - OCEN_OBS["pmra"]) / .027)**2
                        + ((x[5] - OCEN_OBS["pmdec"]) / .027)**2))


def corotation_function():
    ag = agama_kpc()
    axi = ag.Potential(file=str(Path(HUNTER24_DIR) / "MWPotentialHunter24_axi.ini"))
    rr = np.linspace(2., 12., 4001)
    om = np.sqrt(-rr * axi.force(np.column_stack((rr, rr * 0, rr * 0)))[:, 0]) / rr
    return lambda omega: float(np.interp(-omega, -om, rr))


def fit(args):
    directory = ROOT / "results/plot_data" / args.label
    out = directory / f"fit_an{args.angle:g}.json"
    if out.exists():
        raise FileExistsError(f"Choose a fresh label; refusing to overwrite {out}")
    rcr = corotation_function()
    archive = ROOT / "results/streams" / args.label
    archive.mkdir(parents=True, exist_ok=True)
    source = json.loads((ROOT / "results/plot_data/profile_stage1.json").read_text())
    broad = sorted((r for r in source if r["an"] == args.angle), key=lambda r: r["lnL"], reverse=True)
    starts = [coordinates(r) for r in broad[:2]]
    if args.angle == 28:
        starts += [np.array([36., 1.4, 1.3, 5.6, OCEN_OBS["pmra"], OCEN_OBS["pmdec"]]),
                   np.array([36., 1.2, 1.15, 5.6, OCEN_OBS["pmra"], OCEN_OBS["pmdec"]])]
    elif args.angle == 16:
        starts += [np.array([34., 1.2, 1., 5.6, OCEN_OBS["pmra"], OCEN_OBS["pmdec"]])]
    log, cached = [], {}
    payload = dict(args=vars(args), names=NAMES, bounds=[LOW.tolist(), HIGH.tolist()],
                   parameter_offset=OFFSET.tolist(), parameter_scale=SCALE.tolist(),
                   score="grid4_common.score / unchanged score_conditional",
                   distance_measurement=[OCEN_DISTANCE_KPC, args.distance_sigma], pm_sigma=.027,
                   nrel=args.nrel, seed=args.seed, log=log)

    def objective(y, stage="local"):
        x = OFFSET + SCALE * y
        ratio = rcr(x[0]) / (args.rbar0 * x[2])
        if ratio < 1:
            return 1000. + 100. * (1. - ratio)
        key = hashlib.sha256(np.asarray(x, dtype="<f8").tobytes()).hexdigest()[:20]
        if key in cached:
            return cached[key]
        t0 = time.time()
        m = make_spray(x[0], args.angle, x[1], x[3], nrel=args.nrel, seed=args.seed,
                       pm=x[4:6], host_kw=dict(bar_size=x[2]), max_age_myr=700.)
        sc = score(m)
        np.savez_compressed(archive / f"an{args.angle:g}_{key}.npz", **m)
        lp = measurement_term(x, args.distance_sigma)
        # Rescaling affects optimizer tolerances, not the optimum.
        f = -(sc["total"] + lp) / 100.
        row = dict(stage=stage, **dict(zip(NAMES, map(float, x))),
                   lnL=float(sc["total"]), lnmeasurement=lp, lnpenalized=float(sc["total"] + lp),
                   K=int(sc["K"]), frac_bg=float(sc["frac_bg"]), rcr_over_rbar=ratio,
                   seconds=time.time() - t0, spray=str((archive / f"an{args.angle:g}_{key}.npz").relative_to(ROOT)))
        log.append(row)
        cached[key] = f
        payload["best"] = max(log, key=lambda r: r["lnpenalized"])
        atomic_json(out, payload)
        print(f"an{args.angle:g} {len(log):3d} {stage}: lnL {row['lnL']:.1f} "
              f"pen {row['lnpenalized']:.1f} " + " ".join(f"{k}={v:.4f}" for k, v in zip(NAMES, x))
              + f" [{row['seconds']:.1f}s]", flush=True)
        return f

    for x in starts:
        objective((x - OFFSET) / SCALE, "initial")
    x0 = np.array([payload["best"][k] for k in NAMES])
    result = minimize(objective, (x0 - OFFSET) / SCALE, method="COBYQA",
                      bounds=list(zip((LOW - OFFSET) / SCALE, (HIGH - OFFSET) / SCALE)),
                      options=dict(maxfev=args.maxfev, initial_tr_radius=.8, final_tr_radius=.025))
    payload["optimizer"] = dict(success=bool(result.success), message=str(result.message),
                                 nfev=int(result.nfev), fun=float(result.fun), x=result.x.tolist())
    atomic_json(out, payload)
    print("RESULT", json.dumps(dict(angle=args.angle, best=payload["best"], optimizer=payload["optimizer"])), flush=True)


def validate(args):
    directory = ROOT / "results/plot_data" / args.label
    suffix = f"_an{args.angles[0]:g}" if len(args.angles) == 1 else ""
    outfile = directory / f"heldout{suffix}.json"
    rows = json.loads(outfile.read_text()) if outfile.exists() else []
    done = {(r["angle"], r["candidate"], r["seed"], r["nrel"]) for r in rows}
    archive = ROOT / "results/streams" / args.label
    for an in args.angles:
        payload = json.loads((directory / f"fit_an{an:g}.json").read_text())
        candidates = {"refined": payload["best"],
                      "initial": max((r for r in payload["log"] if r["stage"] == "initial"), key=lambda r: r["lnpenalized"])}
        for name, p in candidates.items():
            for seed in args.seeds:
                if (an, name, seed, args.nrel) in done:
                    continue
                t0 = time.time()
                f = archive / f"heldout_an{an:g}_{name}_n{args.nrel}_s{seed}.npz"
                if f.exists():
                    m = dict(np.load(f))
                else:
                    m = make_spray(p["omega"], an, p["amp"], p["dist"], nrel=args.nrel, seed=seed,
                                   pm=(p["pmra"], p["pmdec"]), host_kw=dict(bar_size=p["size"]), max_age_myr=700.)
                    np.savez_compressed(f, **m)
                sc = score(m)
                row = dict(angle=an, candidate=name, seed=seed, nrel=args.nrel, lnL=float(sc["total"]),
                           frac_bg=float(sc["frac_bg"]), K=int(sc["K"]), parameters={k: p[k] for k in NAMES},
                           seconds=time.time()-t0, spray=str(f.relative_to(ROOT)))
                rows.append(row)
                atomic_json(outfile, rows)
                print(json.dumps(row), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--angle", type=float, required=True, choices=(16., 28.))
    f.add_argument("--maxfev", type=int, default=160)
    f.add_argument("--nrel", type=int, default=8000)
    f.add_argument("--seed", type=int, default=1)
    f.add_argument("--distance-sigma", type=float, default=.05)
    f.add_argument("--rbar0", type=float, default=4.25)
    v = sub.add_parser("validate")
    v.add_argument("--angles", type=float, nargs="+", default=(16., 28.))
    v.add_argument("--seeds", type=int, nargs="+", default=(4, 5, 6))
    v.add_argument("--nrel", type=int, default=32000)
    for p in (f, v):
        p.add_argument("--label", default="bar_profile_refinement_20261013")
    args = ap.parse_args()
    (fit if args.mode == "fit" else validate)(args)


if __name__ == "__main__":
    main()
