#!/usr/bin/env python3
"""Tidal tails with a PRESCRIBED (frozen) progenitor potential: massless star tracers in host + moving satellite.

The satellite potential is the fitted model's own spherical mass profile (results/nbody/<model>/model_profiles.json:
total enclosed mass of stars + remnants + halo on 300 radii, 0.05-3000 pc), fixed in time and moving along the
point-mass orbit of the centre (integrated back from today's phase-space point by --tback, no friction). Tracers:
--nstars star particles drawn at random from the model's equilibrium ICs (DF-sampled stars, bin/nbody/build_ics.py).
Because the potential is fixed, a star whose radial apocentre r_max(E) in the satellite potential is below --rfreeze
can never escape; such stars are counted but not integrated (exact in this mode). All others are integrated in ONE
agama.orbit call over the whole time span, with output every ~--snap Myr.

Outputs (same format as bin/streams/run_restricted.py, read by src/ocen_dm/streams/analysis.py): snap_XXXX.npz
(t_myr, pos [pc], vel [km/s], Galactocentric, integrated tracers), snap_today.npz, particles.npz (mass = 1e-6 Msun
per tracer: massless, equal weights; species = 0; index into the ICs), frozen_core.npz (the satellite model as a
FrozenCore; its 'stars' entry = the weight of the non-integrated tracers, so fractions are per tracer), run.json.

Usage: python bin/streams/run_prescribed.py --model A_nodm --out results/streams/prescribed/A_nodm --nstars 200000
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))

import numpy as np

from ocen_dm.streams.restricted import (AGAMA_T_MYR, OCEN_TODAY, FrozenCore, agama_kpc, host_potential,
                                        rmax_of_energy, satellite_potential)

M_TRACER = 1e-6


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, help="results/nbody/<model> with ics.npz and model_profiles.json")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--nstars", type=int, default=200000)
    p.add_argument("--ics", type=Path, default=None, help="star-tracer ICs (default results/nbody/<model>/ics.npz)")
    p.add_argument("--rfreeze", type=float, default=30., help="[pc] stars with r_max(E) below this are not integrated")
    p.add_argument("--mw", default="McMillan17")
    p.add_argument("--tback", type=float, default=2.0*AGAMA_T_MYR, help="Myr before today (default 1955.58)")
    p.add_argument("--snap", type=float, default=50.)
    p.add_argument("--accuracy", type=float, default=1e-8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--bar-omega", type=float, default=None, help="rotating-bar host: Hunter+2024 barred MW "
                   "(../oCen_bar/agama_potentials/MWPotentialHunter24_full.ini) with constant pattern speed [km/s/kpc, >0 prograde]; "
                   "use --mw hunter24_axi for its axisymmetric control")
    p.add_argument("--bar-angle", type=float, default=28.0, help="present-day bar angle [deg] from the Sun-GC line (near end at l > 0)")
    p.add_argument("--bar-amp", type=float, default=None, help="bar amplitude A: axi + A (barred - axisymmetrised baryons), as spray grids")
    p.add_argument("--dist", type=float, default=None, help="today's omega Cen distance [kpc] (with --pm; catalogue RA/Dec/v_los, baumgardt frame)")
    p.add_argument("--pm", type=float, nargs=2, default=None, help="today's omega Cen pmra pmdec [mas/yr] (with --dist)")
    p.add_argument("--frame", default=None, choices=("baumgardt", "ibata19"), help="solar frame + omega Cen distance "
                   "(src/ocen_dm/streams/frames.py) for today's centre; default: legacy OCEN_TODAY (= baumgardt within 0.6 km/s)")
    p.add_argument("--spin", type=Path, default=None, help="results/streams/spin/<model>.json from bin/streams/fit_spin.py: "
                   "Lynden-Bell rotation (src/ocen_dm/streams/rotation.py), axis fixed in the Galactic frame at today's orientation")
    args = p.parse_args()
    t_wall = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    agama = agama_kpc()
    mdir = ROOT/"results/nbody"/args.model
    prof = json.loads((mdir/"model_profiles.json").read_text())
    r_kpc = np.array(prof["r_pc"])/1e3
    M_tot = np.array(prof["enclosed"]["total"])
    sat_model = FrozenCore(r_kpc, M_tot)
    sat = satellite_potential(np.zeros((0, 3)), np.zeros(0), core=sat_model)

    ics_path = args.ics or (mdir/"ics.npz")
    ics = np.load(ics_path)
    stars = np.where(ics["species"] == 0)[0]
    rng = np.random.default_rng(args.seed)
    pick = np.sort(rng.choice(stars, min(args.nstars, len(stars)), replace=False))
    xv_rel = np.hstack((ics["pos"][pick]/1e3, ics["vel"][pick])).astype(float)
    E = sat.potential(xv_rel[:, :3]) + 0.5*np.sum(xv_rel[:, 3:]**2, axis=1)
    rmax_pc = rmax_of_energy(sat, E)*1e3
    from ocen_dm.streams.frames import ocen_today
    today = OCEN_TODAY.copy() if args.frame is None else ocen_today(args.frame)
    if args.dist is not None:
        from ocen_dm.streams.frames import to_model, OCEN_OBS
        today = to_model(OCEN_OBS["ra"], OCEN_OBS["dec"], args.dist, args.pm[0], args.pm[1], OCEN_OBS["vlos"])[0]
    spin = None
    if args.spin is not None:
        from ocen_dm.streams.rotation import spin_up
        spin = json.loads(args.spin.read_text())
        xv_rel, flipped = spin_up(xv_rel, rmax_pc, today, spin["q0"], spin["rq_pc"], pa_deg=spin["pa_deg"],
                                  incl_towards_deg=spin["incl_towards_deg"], r1_pc=spin["r1_pc"], frame=args.frame or "baumgardt")
        print(f"spin: {args.spin}: flipped {flipped.mean():.3f} of the tracers", flush=True)
    active = rmax_pc >= args.rfreeze
    n_frozen = int((~active).sum())
    print(f"{args.model}: {len(pick)} star tracers; {n_frozen} with r_max < {args.rfreeze:g} pc counted only; "
          f"integrating {active.sum()}", flush=True)

    T = args.tback/AGAMA_T_MYR
    host = host_potential(args.mw, bar_omega=args.bar_omega, bar_angle_deg=args.bar_angle, t_today=T, bar_amp=args.bar_amp)
    # backward from t = T (today) to 0: for a time-dependent (rotating-bar) host the clock must start at T
    _, traj = agama.orbit(potential=host, ic=today, timestart=T, time=-T, trajsize=2, accuracy=1e-12)   # default 1e-8 loses 5-12 pc per round trip
    start = traj[-1]
    tc, orb = agama.orbit(potential=host, ic=start, timestart=0., time=T, trajsize=int(np.ceil(args.tback/0.05))+1, accuracy=1e-12)
    total = agama.Potential(host, agama.Potential(potential=sat, center=np.column_stack((tc, orb))))
    nsnap = int(round(args.tback/args.snap))
    ic = xv_rel[active] + start
    t0 = time.time()
    res = agama.orbit(ic=ic, potential=total, timestart=0., time=T, trajsize=nsnap+1, accuracy=args.accuracy, verbose=False)
    t_int = time.time()-t0
    times = np.asarray(res[0, 0])*AGAMA_T_MYR
    traj_all = np.stack([np.asarray(r) for r in res[:, 1]], axis=1)          # (nsnap+1, N, 6)
    for k, tk in enumerate(times):
        np.savez(args.out/f"snap_{k:04d}.npz", t_myr=tk, pos=(traj_all[k, :, :3]*1e3).astype(np.float32),
                 vel=traj_all[k, :, 3:].astype(np.float32))
    np.savez(args.out/"snap_today.npz", t_myr=times[-1], pos=(traj_all[-1, :, :3]*1e3).astype(np.float32),
             vel=traj_all[-1, :, 3:].astype(np.float32))
    np.savez(args.out/"particles.npz", mass=np.full(active.sum(), M_TRACER, np.float32),
             species=np.zeros(active.sum(), np.int8), index=pick[active])
    FrozenCore(r_kpc, M_tot, {0: n_frozen*M_TRACER}).save(args.out/"frozen_core.npz")
    meta = dict(created_utc=datetime.now(timezone.utc).isoformat(), method="prescribed (frozen) satellite potential, "
                "massless star tracers", model=args.model, bar_omega=args.bar_omega, bar_angle_deg=args.bar_angle if args.bar_omega else None, bar_amp=args.bar_amp, model_profiles=str(mdir/"model_profiles.json"),
                ics=str(ics_path), mw=args.mw, tback_myr=args.tback, t_today_myr=args.tback, nstars=int(len(pick)),
                n_integrated=int(active.sum()), n_counted_only=n_frozen, rfreeze_pc=args.rfreeze, seed=args.seed,
                accuracy=args.accuracy, spin=spin, spin_file=str(args.spin) if args.spin else None, snap_times_myr=times.tolist(), start=start.tolist(), ocen_today=today.tolist(), frame=args.frame or "baumgardt",
                satellite_mass=float(M_tot[-1]), integration_s=t_int, wall_s=time.time()-t_wall,
                centre_offset_today_pc=float(np.linalg.norm(orb[-1, :3]-today[:3])*1e3))
    (args.out/"run.json").write_text(json.dumps(meta, indent=1))
    print(f"integration {t_int:.1f} s, total {time.time()-t_wall:.1f} s; centre today {meta['centre_offset_today_pc']:.2f} pc "
          f"from omega Cen", flush=True)


if __name__ == "__main__":
    main()
