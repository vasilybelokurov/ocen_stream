#!/usr/bin/env python3
"""Validation of the restricted N-body runs against the live (pyfalcon) runs of the same ICs.

For each (live, restricted) pair the same centre finder and bound criterion
(src/ocen_dm/streams/analysis.py) are applied to snapshots at a common set of times. Measured:
bound mass per species versus time; unbound stellar fraction; rms of host energy and Lz of the
stellar debris relative to the cluster; width and velocity dispersion of unbound stars 0.3-2 kpc
from the cluster in the local (along / in-plane / normal) frame; sky distribution today.

Acceptance criteria (stated before the comparison, 2026-10-08): unbound stellar fraction within
20%, dE and dLz rms within 10%, shell width and dispersions within 15%, at the times where the live
cluster is within 50 pc of the point-mass orbit (all of A; B early only, since live B drifts by
stripped-DM self-gravity, which the restricted scheme does not contain).

Usage:
  python bin/streams/compare_live.py --name streams_validation \
      --pair results/nbody/A_nodm/orbit results/streams/A_validation \
      --pair results/nbody/B_dm_phot/orbit results/streams/B_validation [--times 100 300 ...]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ocen-df-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ocen_dm.streams.analysis import frozen_star_count, load, run_core, run_particles, snapshot_times, state, tail_metrics
from ocen_dm.streams.restricted import AGAMA_T_MYR, OCEN_TODAY, agama_kpc, centre_orbit, host_potential

R_SUN_KPC, V_SUN = 8.178, (11.1, 12.24+240.0, 7.25)     # as bin/nbody/analyse_tails.py


def sky(xv):
    """Simulation frame (X toward the GC from the Sun) -> l, b [deg]."""
    import astropy.coordinates as coord
    import astropy.units as u
    gc = coord.Galactocentric(x=-xv[:, 0]*u.kpc, y=xv[:, 1]*u.kpc, z=xv[:, 2]*u.kpc,
                              v_x=-xv[:, 3]*u.km/u.s, v_y=xv[:, 4]*u.km/u.s, v_z=xv[:, 5]*u.km/u.s,
                              galcen_distance=R_SUN_KPC*u.kpc, z_sun=0*u.pc,
                              galcen_v_sun=coord.CartesianDifferential(V_SUN*u.km/u.s))
    g = gc.transform_to(coord.Galactic())
    return np.asarray(g.l.wrap_at(180*u.deg).deg), np.asarray(g.b.deg)


def point_mass_track(host, tback, times):
    _, traj = agama_kpc().orbit(potential=host, ic=OCEN_TODAY, time=-tback/AGAMA_T_MYR, trajsize=2)
    start = traj[-1]
    t, orb = centre_orbit(host, start, 0.0, 1.0, int(np.ceil(max(times)))+1)
    return np.interp(times, t*AGAMA_T_MYR, np.arange(len(t))), orb


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--name", default="streams_validation")
    p.add_argument("--pair", nargs=2, action="append", required=True, metavar=("LIVE", "RESTRICTED"))
    p.add_argument("--times", nargs="+", type=float, default=[50, 100, 200, 300, 500, 750, 1000, 1500, 1955.58])
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--tags", nargs=2, default=["live", "restricted"], help="names of the first and second run of each pair (plot legend)")
    p.add_argument("--title", default="Restricted vs live N-body: same ICs, same orbit start, same bound criterion")
    args = p.parse_args()
    host = host_potential("McMillan17")
    labels = args.labels or [Path(r).parent.name if Path(r).name == "orbit" else Path(r).name for r, _ in args.pair]
    tback = 2.0*AGAMA_T_MYR
    out = dict(created_by="bin/streams/compare_live.py", pairs=[], times=args.times)
    fig, ax = plt.subplots(3, 4, figsize=(17, 11.5))
    for ip, ((live, restr), lab) in enumerate(zip(args.pair, labels)):
        rec = dict(label=lab, live=live, restricted=restr, rows=[])
        mass_l, sp_l = run_particles(live); mass_r, sp_r = run_particles(restr)
        _, t_r = snapshot_times(restr)
        idx, orb = point_mass_track(host, tback, args.times)
        for t, k in zip(args.times, idx):
            if t > t_r.max() + 1:
                continue
            row = dict(t_myr=t)
            for tag, run, m, s in (("live", live, mass_l, sp_l), ("restricted", restr, mass_r, sp_r)):
                tt, xv = load(run, t)
                c, b, bm = state(xv, m, s, guess=orb[int(round(k))], core=run_core(run))
                tm = tail_metrics(host, xv, c, b, s, n_frozen_stars=frozen_star_count(run))
                drift = float(np.linalg.norm(c[:3]-orb[int(round(k)), :3])*1e3)
                row[tag] = dict(t_snap=tt, bound_mass=bm, orbit_offset_pc=drift, **tm)
            rec["rows"].append(row)
            print(lab, f"t={t:7.1f}", " | ".join(
                f"{tag}: unb {row[tag]['unbound_star_fraction']:.4f} dE {row[tag]['dE_rms']:.0f} dLz {row[tag]['dLz_rms']:.1f} "
                f"wN {row[tag].get('width_normal_pc', np.nan):.0f} sN {row[tag].get('sigma_v_normal', np.nan):.1f} off {row[tag]['orbit_offset_pc']:.0f}"
                for tag in ("live", "restricted")), flush=True)
        out["pairs"].append(rec)
        rows = rec["rows"]; T = np.array([r["t_myr"] for r in rows])
        col = f"C{ip}"
        def series(tag, key):
            return np.array([r[tag].get(key, np.nan) for r in rows], float)
        panels = [("unbound_star_fraction", "unbound stellar fraction"), ("dE_rms", "debris dE rms [km$^2$ s$^{-2}$]"),
                  ("dLz_rms", "debris dLz rms [kpc km/s]"), ("width_normal_pc", "debris width normal, 0.3-2 kpc [pc]"),
                  ("width_inplane_pc", "debris width in-plane [pc]"), ("sigma_v_normal", r"$\sigma_v$ normal [km/s]"),
                  ("sigma_v_inplane", r"$\sigma_v$ in-plane [km/s]"), ("sigma_v_along", r"$\sigma_v$ along [km/s]"),
                  ("orbit_offset_pc", "cluster offset from point-mass orbit [pc]")]
        for a, (key, ttl) in zip(ax.flat, panels):
            a.plot(T, series("live", key), "o-", color=col, label=f"{lab} {args.tags[0]}")
            a.plot(T, series("restricted", key), "s--", color=col, mfc="none", label=f"{lab} {args.tags[1]}")
            a.set_title(ttl, fontsize=10); a.set_xlabel("t [Myr]")
        for sp_name in ("stars", "halo"):
            a = ax.flat[9]
            bl = np.array([r["live"]["bound_mass"].get(sp_name, np.nan) for r in rows])
            br = np.array([r["restricted"]["bound_mass"].get(sp_name, np.nan) for r in rows])
            if np.all(np.isnan(bl)):
                continue
            ls = "-" if sp_name == "stars" else ":"
            a.plot(T, bl/bl[0], ls, color=col, marker="o", ms=3, label=f"{lab} {sp_name} {args.tags[0]}")
            a.plot(T, br/br[0], ls, color=col, marker="s", ms=3, mfc="none", label=f"{lab} {sp_name} {args.tags[1]}")
            a.set_title("bound mass / initial", fontsize=10); a.set_xlabel("t [Myr]")
        # sky today
        if T.max() > tback - 1:
            for j, (tag, run, m, s) in enumerate((("live", live, mass_l, sp_l), ("restricted", restr, mass_r, sp_r))):
                tt, xv = load(run, tback)
                c, b, _ = state(xv, m, s, guess=orb[-1], core=run_core(run))
                sel = (s == 0) & ~b
                l, bb = sky(xv[sel])
                a = ax.flat[10+j]
                a.plot(l, bb, ",", color=col, alpha=0.3)
                lc, bc = sky(c[None, :])
                a.plot(lc, bc, "k*", ms=10)
                a.set_title(f"unbound stars today, {args.tags[j]}", fontsize=10); a.set_xlabel("l [deg]"); a.set_ylabel("b [deg]")
                a.set_xlim(90, -90); a.set_ylim(-60, 60)
    ax.flat[0].legend(fontsize=7); ax.flat[9].legend(fontsize=6)
    fig.suptitle(args.title, fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT/f"plots/{args.name}.png", dpi=110)
    (ROOT/f"results/plot_data/{args.name}.json").write_text(json.dumps(out, indent=1))
    print("wrote", f"plots/{args.name}.png")


if __name__ == "__main__":
    main()
