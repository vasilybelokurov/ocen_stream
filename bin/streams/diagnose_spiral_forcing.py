#!/usr/bin/env python3
"""Validate the reflected Hunter spiral and run a small fixed-phase pilot.

The reference spiral phase is 25 deg, amplitude 1, pattern speed 22.5
km/s/kpc. Pilot variations use Hunter's published 17.5-25 pattern-speed
range, keeping phase and all cluster/bar parameters fixed. Data and scoring
are unchanged. This is not an optimized spiral fit or a bar-angle inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bin/streams"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from grid4_common import T, make_spray, score, du
from ocen_dm.streams.restricted import HUNTER24_DIR, agama_kpc, host_potential
from ocen_dm.streams.spiral import reflected_file, with_spiral
from refine_bar_profiles import atomic_json, NAMES


def check(args):
    ag = agama_kpc()
    source = Path(HUNTER24_DIR) / "MWPotentialHunter24_spiral.ini"
    original = ag.Potential(file=str(source))
    reflected = ag.Potential(file=reflected_file(str(source)))
    rng = np.random.default_rng(20261013)
    points = rng.uniform([-8., -8., -2.], [8., 8., 2.], (80, 3))
    mirror = np.array([-1., 1., 1.])
    f1, f2 = reflected.force(points), original.force(points * mirror) * mirror
    p1, p2 = reflected.potential(points), original.potential(points * mirror)
    force_error = float(np.max(np.abs(f1 - f2)) / np.max(np.abs(f2)))
    potential_error = float(np.max(np.abs(p1 - p2)) / np.max(np.abs(p2)))
    phi = np.linspace(0., 2 * np.pi, 1024, endpoint=False)
    ring = np.column_stack((6. * np.cos(phi), 6. * np.sin(phi), np.full_like(phi, .3)))
    potentials = reflected.potential(ring)
    monopole_ratio = float(abs(potentials.mean()) / np.max(np.abs(potentials)))
    source_monopole_ratio = float(abs(original.potential(ring).mean()) / np.max(np.abs(potentials)))
    monopole_preservation_error = float(abs(potentials.mean() - original.potential(ring).mean()) / np.max(np.abs(potentials)))
    host = host_potential("hunter24_axi")
    zero_identity = with_spiral(host, T, amplitude=0.) is host
    added = with_spiral(host, T, phase_deg=0., omega=0.)
    added_error = float(np.max(abs(added.force(points) - host.force(points) - reflected.force(points))) / np.max(abs(f2)))
    # A rotating potential at t_today must have its specified present phase.
    dynamic = with_spiral(host, T, phase_deg=25., omega=22.5)
    static = with_spiral(host, T, phase_deg=25., omega=0.)
    today_error = float(np.max(abs(dynamic.force(points, t=T) - static.force(points, t=T))) / np.max(abs(f2)))
    phi_past = ag.Potential(potential=reflected, rotation=np.radians(25.) - 22.5 * .1)
    past_error = float(np.max(abs(dynamic.force(points, t=T - .1) - host.force(points) - phi_past.force(points))) / np.max(abs(f2)))
    result = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  reflected_file=reflected_file(str(source)), reflection_force_relative_error=force_error,
                  reflection_potential_relative_error=potential_error, azimuthal_monopole_ratio=monopole_ratio,
                  source_azimuthal_monopole_ratio=source_monopole_ratio,
                  monopole_preservation_error=monopole_preservation_error,
                  zero_amplitude_identity=zero_identity, addition_relative_error=added_error,
                  today_phase_error=today_error, past_phase_error=past_error)
    result["passed"] = bool(max(force_error, potential_error, added_error, today_error, past_error) < 1e-9
                            and monopole_preservation_error < 1e-12 and zero_identity)
    atomic_json(ROOT / "results/plot_data" / args.label / "force_checks.json", result)
    print(json.dumps(result, indent=1))
    if not result["passed"]:
        raise RuntimeError("Spiral force checks failed")


def pilot(args):
    checks = ROOT / "results/plot_data" / args.label / "force_checks.json"
    if not json.loads(checks.read_text())["passed"]:
        raise RuntimeError("Run force checks first")
    out = ROOT / "results/plot_data" / args.label / f"pilot_an{args.angle:g}_s{args.seed}.json"
    archive = ROOT / "results/streams" / args.label
    archive.mkdir(parents=True, exist_ok=True)
    if out.exists():
        payload = json.loads(out.read_text())
        parameters = payload["parameters"]
        if payload["nrel"] != args.nrel:
            raise ValueError("Resume requires the same particle count")
    else:
        fit = json.loads((ROOT / "results/plot_data" / args.refinement_label / f"fit_an{args.angle:g}.json").read_text())
        parameters = {k: fit["best"][k] for k in NAMES}
        payload = dict(parameters=parameters, reference_fit=args.refinement_label,
                       reference_optimizer=fit.get("optimizer", "still running when snapshot was taken"),
                       angle=args.angle, seed=args.seed, nrel=args.nrel, phase_deg=25., results=[])
        atomic_json(out, payload)
    done = {r["name"] for r in payload["results"]}
    cases = [("none", None), ("fiducial", dict(amplitude=1., omega=22.5, phase_deg=25.)),
             ("slow", dict(amplitude=1., omega=17.5, phase_deg=25.)),
             ("fast", dict(amplitude=1., omega=25., phase_deg=25.))]
    for name, spiral in cases:
        if name in done:
            continue
        t0 = time.time()
        f = archive / f"an{args.angle:g}_{name}_n{args.nrel}_s{args.seed}.npz"
        if f.exists():
            m = dict(np.load(f))
        else:
            hk = dict(bar_size=parameters["size"])
            if spiral is not None:
                hk["spiral"] = spiral
            m = make_spray(parameters["omega"], args.angle, parameters["amp"], parameters["dist"],
                           pm=(parameters["pmra"], parameters["pmdec"]), host_kw=hk,
                           nrel=args.nrel, seed=args.seed, max_age_myr=700.)
            np.savez_compressed(f, **m)
        sc = score(m)
        segments = [float(sc["per_star"][(du >= lo) & (du < hi)].sum()) for lo, hi in ((-99, 4), (4, 17), (17, 99))]
        row = dict(name=name, spiral=spiral, lnL=float(sc["total"]), segments=segments,
                   K=int(sc["K"]), frac_bg=float(sc["frac_bg"]), seconds=time.time()-t0,
                   spray=str(f.relative_to(ROOT)))
        payload["results"].append(row)
        atomic_json(out, payload)
        print(json.dumps(row), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)
    check_parser = sub.add_parser("check")
    pilot_parser = sub.add_parser("pilot")
    pilot_parser.add_argument("--angle", type=float, required=True)
    pilot_parser.add_argument("--seed", type=int, default=4)
    pilot_parser.add_argument("--nrel", type=int, default=8000)
    pilot_parser.add_argument("--refinement-label", default="bar_profile_refinement_20261013")
    for p in (check_parser, pilot_parser):
        p.add_argument("--label", default="spiral_pilot_20261013")
    args = ap.parse_args()
    (check if args.mode == "check" else pilot)(args)


if __name__ == "__main__":
    main()
