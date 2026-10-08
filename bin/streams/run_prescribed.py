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
    active = rmax_pc >= args.rfreeze
    n_frozen = int((~active).sum())
    print(f"{args.model}: {len(pick)} star tracers; {n_frozen} with r_max < {args.rfreeze:g} pc counted only; "
          f"integrating {active.sum()}", flush=True)

    host = host_potential(args.mw)
    _, traj = agama.orbit(potential=host, ic=OCEN_TODAY, time=-args.tback/AGAMA_T_MYR, trajsize=2)
    start = traj[-1]
    T = args.tback/AGAMA_T_MYR
    tc, orb = agama.orbit(potential=host, ic=start, timestart=0., time=T, trajsize=int(np.ceil(args.tback/0.05))+1)
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
                "massless star tracers", model=args.model, model_profiles=str(mdir/"model_profiles.json"),
                ics=str(ics_path), mw=args.mw, tback_myr=args.tback, t_today_myr=args.tback, nstars=int(len(pick)),
                n_integrated=int(active.sum()), n_counted_only=n_frozen, rfreeze_pc=args.rfreeze, seed=args.seed,
                accuracy=args.accuracy, snap_times_myr=times.tolist(), start=start.tolist(), ocen_today=OCEN_TODAY.tolist(),
                satellite_mass=float(M_tot[-1]), integration_s=t_int, wall_s=time.time()-t_wall,
                centre_offset_today_pc=float(np.linalg.norm(orb[-1, :3]-OCEN_TODAY[:3])*1e3))
    (args.out/"run.json").write_text(json.dumps(meta, indent=1))
    print(f"integration {t_int:.1f} s, total {time.time()-t_wall:.1f} s; centre today {meta['centre_offset_today_pc']:.2f} pc "
          f"from omega Cen", flush=True)


if __name__ == "__main__":
    main()
