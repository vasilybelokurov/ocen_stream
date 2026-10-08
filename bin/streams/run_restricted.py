#!/usr/bin/env python3
"""Restricted N-body run of an omega Cen model on its orbit (src/ocen_dm/streams/restricted.py).

Particles come from an equilibrium IC file of bin/nbody/build_ics.py (pc, km/s, Msun, species).
The satellite centre is integrated back from today's phase-space point (OCEN_TODAY) by --tback
in the host potential (point mass; no dynamical friction) and the particles start there.
Every --tupd Myr the satellite potential is refitted (spherical multipole) from the bound particles;
with --frozen it is fitted once at t = 0 and only moves along the orbit (control run).

--no-host gives an isolation test: satellite at rest at the origin, no host, same machinery.

Outputs in --out (same snapshot format as bin/nbody/run_nbody.py, so the N-body analysis can read
them): snap_XXXX.npz every --snap Myr (t_myr; pos [pc] and vel [km/s], Galactocentric; float32),
snap_today.npz at t = tback, diagnostics.jsonl (per update: t, centre, r_gal, bound mass per
species, offset of the bound stars' median from the prescribed centre, stellar r_half),
run.json, restart.npz.

Usage:
  python bin/streams/run_restricted.py --ics results/nbody/A_nodm/ics.npz --out results/streams/A_validation
  python bin/streams/run_restricted.py --ics results/nbody/A_nodm/ics.npz --out results/streams/A_iso --no-host --tstop 200
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

from ocen_dm.streams.restricted import (AGAMA_T_MYR, OCEN_TODAY, RestrictedState, advance, agama_kpc,
                                        bound_set, host_potential)

SPECIES = ("stars", "remnants", "halo")


def null_host():
    """A negligible potential so the same orbit machinery runs without a host (isolation test)."""
    agama = agama_kpc()
    return agama.Potential(type="Plummer", mass=1e-20, scaleRadius=1.0)


def diagnostics(state, mass, species, sat=None):
    rel = state.xv - state.centre
    row = dict(t_myr=state.t_myr, centre=state.centre.tolist(), r_gal_kpc=float(np.linalg.norm(state.centre[:3])),
               bound_mass={SPECIES[s]: float(mass[state.bound & (species == s)].sum()) for s in np.unique(species)})
    if sat is not None:      # central potential and density of the refitted satellite (guards against spurious cusps)
        rr = np.array([1e-7, 1e-6, 1e-5, 1e-4, 1e-3])
        xyz = np.column_stack((rr, 0*rr, 0*rr))
        row["sat_phi"] = sat.potential(xyz).tolist()
        row["sat_rho"] = sat.density(xyz).tolist()
    bs = state.bound & (species == 0)
    if bs.sum() > 10:
        row["bound_star_median_offset_pc"] = (np.median(rel[bs, :3], axis=0)*1e3).tolist()
        r = np.linalg.norm(rel[bs, :3], axis=1)*1e3
        row["r_half_stars_pc"] = float(np.median(r))
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ics", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--mw", default="McMillan17")
    p.add_argument("--tback", type=float, default=2.0*AGAMA_T_MYR, help="Myr before today to start (default 1955.58, as the live runs)")
    p.add_argument("--tstop", type=float, default=None, help="stop time [Myr from start]; default tback")
    p.add_argument("--tupd", type=float, default=2.0, help="satellite-potential update interval [Myr]")
    p.add_argument("--snap", type=float, default=10.0, help="snapshot interval [Myr]")
    p.add_argument("--accuracy", type=float, default=1e-8)
    p.add_argument("--no-host", action="store_true")
    p.add_argument("--frozen", action="store_true", help="never refit the satellite potential (fitted once at t = 0)")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    tstop = args.tback if args.tstop is None else args.tstop
    args.out.mkdir(parents=True, exist_ok=True)
    agama = agama_kpc()

    d = np.load(args.ics)
    mass = d["mass"].astype(float); species = d["species"].astype(np.int8)
    xv0 = np.hstack((d["pos"]/1e3, d["vel"])).astype(float)          # pc -> kpc
    if args.no_host:
        host = null_host(); start = np.zeros(6)
    else:
        host = host_potential(args.mw)
        _, traj = agama.orbit(potential=host, ic=OCEN_TODAY, time=-args.tback/AGAMA_T_MYR, trajsize=2)
        start = traj[-1]
    restart = args.out/"restart.npz"
    if args.resume and restart.exists():
        r = np.load(restart)
        state = RestrictedState(float(r["t_myr"]), r["xv"], r["centre"], r["bound"])
        if args.frozen:
            raise SystemExit("--resume with --frozen needs the t = 0 potential; restart the run instead")
        print(f"resumed at t = {state.t_myr:.2f} Myr", flush=True)
        diag_mode = "a"
    else:
        state = RestrictedState(0.0, xv0 + start, start.copy(), np.ones(len(mass), bool))
        diag_mode = "w"
    bound, sat = bound_set(state.xv, state.centre, mass, start=state.bound)
    state.bound = bound
    meta = dict(created_utc=datetime.now(timezone.utc).isoformat(), ics=str(args.ics), mw=None if args.no_host else args.mw,
                tback_myr=args.tback, tstop_myr=tstop, tupd_myr=args.tupd, frozen=args.frozen, snap_myr=args.snap, accuracy=args.accuracy,
                n=len(mass), counts={SPECIES[s]: int(np.sum(species == s)) for s in np.unique(species)},
                start=start.tolist(), ocen_today=OCEN_TODAY.tolist(), t_today_myr=args.tback, method="restricted N-body",
                agama_t_myr=AGAMA_T_MYR)
    (args.out/"run.json").write_text(json.dumps(meta, indent=1))
    np.savez(args.out/"particles.npz", mass=mass.astype(np.float32), species=species)
    diag = open(args.out/"diagnostics.jsonl", diag_mode)

    def snapshot(name, st):
        np.savez(args.out/name, t_myr=st.t_myr, pos=(st.xv[:, :3]*1e3).astype(np.float32), vel=st.xv[:, 3:].astype(np.float32))

    if state.t_myr == 0:
        diag.write(json.dumps(diagnostics(state, mass, species, sat))+"\n"); diag.flush()
        snapshot("snap_0000.npz", state)
    eps = 1e-9
    next_snap = (np.floor(state.t_myr/args.snap+eps)+1)*args.snap
    t_wall = time.time()
    while state.t_myr < tstop - eps:
        dt = min(args.tupd, tstop-state.t_myr, next_snap-state.t_myr)
        t0 = time.time()
        state, sat, _ = advance(state, host, mass, sat, dt, accuracy=args.accuracy, frozen=args.frozen)
        row = diagnostics(state, mass, species, sat)
        diag.write(json.dumps(row)+"\n"); diag.flush()
        if state.t_myr >= next_snap - eps:
            k = int(round(state.t_myr/args.snap))
            snapshot(f"snap_{k:04d}.npz", state)
            np.savez(restart, t_myr=state.t_myr, xv=state.xv, centre=state.centre, bound=state.bound)
            next_snap += args.snap
            eta = (time.time()-t_wall)/max(state.t_myr, 1e-9)*(tstop-state.t_myr)/3600
            print(f"t = {state.t_myr:8.2f} Myr  r_gal {row['r_gal_kpc']:.2f} kpc  bound: "
                  + ", ".join(f"{k} {v:.4e}" for k, v in row["bound_mass"].items())
                  + f"  offset {np.round(row.get('bound_star_median_offset_pc', [np.nan]*3), 2).tolist()} pc"
                  + f"  [{time.time()-t0:.1f} s/update, ETA {eta:.2f} h]", flush=True)
    diag.close()
    if not args.no_host and abs(tstop-args.tback) < 1e-6:
        snapshot("snap_today.npz", state)
        off = np.linalg.norm(state.centre[:3]-OCEN_TODAY[:3])*1e3
        print(f"present day: centre {off:.2f} pc from OCEN_TODAY", flush=True)
        meta["centre_offset_today_pc"] = off
    (args.out/"run.json").write_text(json.dumps(dict(meta, finished_utc=datetime.now(timezone.utc).isoformat(),
                                                    wall_hours=(time.time()-t_wall)/3600), indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
