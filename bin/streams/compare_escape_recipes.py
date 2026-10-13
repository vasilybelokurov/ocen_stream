#!/usr/bin/env python3
"""Paired escape-law and launch-arm pilot, at frozen bar/cluster parameters.

No host/state/likelihood tuning. Compare Fardal and Chen+2024 at the same
release epochs, seed and progenitor potential; repeat with both launch arms.
The scorer selects positive chi, so a launch-arm cut is a separate assumption.
Checks preserve the Fardal default and verify Chen's energy and covariance.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bin/streams"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from grid4_common import make_spray, score, r_kpc, M, du, d
from ocen_dm.streams.escape import MEAN, SIGMA, chen_parameters
from ocen_dm.streams.spray import release_ic
from ocen_dm.streams.restricted import FrozenCore, satellite_potential
from diagnose_pm_profiles import pm_basis, summarize
from refine_bar_profiles import atomic_json, NAMES


def checks(args):
    # The committed pre-change Fardal function is the local numerical reference.
    source = subprocess.run(["git", "show", "6444082:src/ocen_dm/streams/spray.py"], capture_output=True, text=True, check=True).stdout
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "release_ic")
    scope = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "committed_fardal_reference", "exec"), scope)
    n = 1000
    orbit = np.tile([5., 1., .5, 30., 100., 20.], (n, 1))
    radius = np.full(n, .1)
    velocity = np.full(n, 4.)
    basis = np.repeat(np.eye(3)[None], n, axis=0)
    old = scope["release_ic"](orbit, radius, velocity, basis, np.random.default_rng(11))
    new = release_ic(orbit, radius, velocity, basis, np.random.default_rng(11))
    identical = all(np.array_equal(x, y) for x, y in zip(old, new))
    satellite = satellite_potential(np.empty((0, 3)), np.empty(0), core=FrozenCore(r_kpc, M))
    escape = lambda r: np.sqrt(-2 * satellite.potential(np.column_stack((r, r*0, r*0))))
    ic, arm = release_ic(orbit, radius, velocity, basis, np.random.default_rng(11), escape_recipe="chen24", escape_speed=escape)
    relative = ic-np.repeat(orbit, 2, axis=0)
    r = np.linalg.norm(relative[:, :3], axis=1)
    e = .5*np.sum(relative[:, 3:]**2, axis=1)+satellite.potential(relative[:, :3])
    error = float(np.max(abs(e)) / np.max(escape(r)**2))
    sample = chen_parameters(np.random.default_rng(20261013), 100000)
    mean_z = (sample.mean(0)-MEAN)/(SIGMA/np.sqrt(len(sample)))
    result = dict(fardal_reference_commit="6444082", fardal_default_bitwise_identical=identical,
                  chen_isolated_energy_relative_error=error, sample_mean=sample.mean(0).tolist(),
                  sample_sigma=sample.std(0).tolist(), sample_mean_standard_errors=mean_z.tolist(),
                  sample_r_alpha_correlation=float(np.corrcoef(sample[:, 0], sample[:, 3])[0, 1]),
                  source="https://arxiv.org/html/2408.01496v2#S3", radial_std=.35, angle_units="degrees",
                  formula_uses_actual_satellite_escape_speed=True)
    result["passed"] = bool(identical and error < 1e-12)
    atomic_json(ROOT / "results/plot_data" / args.label / "sampler_checks.json", result)
    print(json.dumps(result, indent=1))
    if not result["passed"]:
        raise RuntimeError("Release-law check failed")


def pilot(args):
    directory = ROOT / "results/plot_data" / args.label
    if not json.loads((directory / "sampler_checks.json").read_text())["passed"]:
        raise RuntimeError("Run sampler checks first")
    output = directory / f"pilot_an{args.angle:g}_n{args.nrel}_s{args.seed}.json"
    archive = ROOT / "results/streams" / args.label
    archive.mkdir(parents=True, exist_ok=True)
    if output.exists():
        payload = json.loads(output.read_text())
        p = payload["parameters"]
    else:
        fit = json.loads((ROOT / "results/plot_data" / args.refinement_label / f"fit_an{args.angle:g}.json").read_text())
        p = {k: fit["best"][k] for k in NAMES}
        payload = dict(args=vars(args), parameters=p, cases=[], source_files_sha256={
            str(f): hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
            ("src/ocen_dm/streams/escape.py", "src/ocen_dm/streams/spray.py", "src/ocen_dm/streams/score.py", "bin/streams/grid4_common.py")})
        atomic_json(output, payload)
    done = {r["name"] for r in payload["cases"]}
    basis = pm_basis(d["l"], d["b"])
    for recipe, arms in (("fardal", 1), ("chen24", 1), ("fardal", None), ("chen24", None)):
        name = recipe + ("_outer" if arms == 1 else "_both")
        if name in done:
            continue
        f = archive / f"an{args.angle:g}_{name}_n{args.nrel}_s{args.seed}.npz"
        t0 = time.time()
        if f.exists():
            m = dict(np.load(f))
        else:
            m = make_spray(p["omega"], args.angle, p["amp"], p["dist"], nrel=args.nrel, seed=args.seed,
                           pm=(p["pmra"], p["pmdec"]), host_kw=dict(bar_size=p["size"]), max_age_myr=700.,
                           escape_recipe=recipe, arm_only=arms)
            np.savez_compressed(f, **m)
        sc = score(m)
        bins = summarize(m, np.arange(10., 28.1, 2.), basis)
        residual = np.array([[r["model"][k][1]-r["observed"][k][1] for k in ("pmra", "pmdec")] for r in bins])
        selected = (m["chi"] > 0) & (m["age"] < 700)
        inner = (selected & (m["arm_origin"] < 0)).sum() if arms is None else 0
        row = dict(name=name, recipe=recipe, arm_only=arms, lnL=float(sc["total"]),
                   segments=[float(sc["per_star"][(du>=lo)&(du<hi)].sum()) for lo,hi in ((-99,4),(4,17),(17,99))],
                   pm_rms=float(np.sqrt(np.mean(residual**2))), bins=bins, K=int(sc["K"]),
                   frac_bg=float(sc["frac_bg"]), positive_chi_particles=int(selected.sum()),
                   inner_launch_positive_chi_particles=int(inner), seconds=time.time()-t0,
                   spray=str(f.relative_to(ROOT)))
        if recipe == "fardal" and arms is None:
            old = dict(np.load(archive / f"an{args.angle:g}_fardal_outer_n{args.nrel}_s{args.seed}.npz"))
            outer = m["arm_origin"] == 1
            row["outer_subset_max_abs_differences"] = {k: float(np.max(abs(m[k][outer]-old[k])))
                                                       for k in ("l", "b", "pmra", "pmdec", "vlos", "d", "chi", "age")}
        payload["cases"].append(row)
        atomic_json(output, payload)
        print(json.dumps({k: v for k,v in row.items() if k != "bins"}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("check")
    p = sub.add_parser("pilot")
    p.add_argument("--angle", type=int, required=True, choices=(16,28))
    p.add_argument("--seed", type=int, default=4)
    p.add_argument("--nrel", type=int, default=8000)
    p.add_argument("--refinement-label", default="bar_profile_refinement_20261013")
    for x in (c,p):
        x.add_argument("--label", default="escape_pilot_20261013")
    args = ap.parse_args()
    (checks if args.mode == "check" else pilot)(args)


if __name__ == "__main__":
    main()
