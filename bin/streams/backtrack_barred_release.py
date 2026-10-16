#!/usr/bin/env python3
"""Can measured RV stars return to the cluster in the fitted barred host?

Integrate backward in host + the same moving, prescribed progenitor potential
as the spray. Distances are scanned, not taken from parallax or CMD shifts.
The dimensionless encounter diagnostic uses actual release scales: position
2*r_J and velocity 2*Omega*r_J. It is not a likelihood or ancestry probability.
Mock spray particles selected by sky position provide a numerical/reference
check. A few observational-error draws are descriptive, not a posterior or a
complete exclusion test over allowed bar histories and cluster states.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bin/streams"))
sys.path.insert(0, str(ROOT / "src"))

import astropy.coordinates as coord
import astropy.units as u
import numpy as np

from grid4_common import T, d, du, dx, GC, project_gc, r_kpc, M
from ocen_dm.cluster import OCEN_DISTANCE_KPC
from ocen_dm.streams.frames import OCEN_OBS, to_model
from ocen_dm.streams.restricted import AGAMA_T_MYR, FrozenCore, agama_kpc, host_potential, satellite_potential
from ocen_dm.streams.spray import rj_vj_R
from refine_bar_profiles import atomic_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--angle", type=int, required=True, choices=(16, 28))
    ap.add_argument("--label", default="bar_profile_refinement_20261013")
    ap.add_argument("--back-myr", type=float, default=700.)
    ap.add_argument("--eta", type=float, default=0., help="dimensionless bar deceleration, using the existing host prescription")
    ap.add_argument("--catalogue-state", action="store_true", help="use D=5.43 and catalogue PMs instead of the fitted cluster state")
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--distance-step", type=float, default=.05)
    a = ap.parse_args()
    directory = ROOT / "results/plot_data" / a.label
    suffix = (f"_b{a.back_myr:g}" if a.back_myr != 700 else "") + (f"_eta{a.eta:g}" if a.eta else "") + ("_catalogue" if a.catalogue_state else "")
    output = directory / f"barred_backtrack_an{a.angle}{suffix}.json"
    if output.exists():
        raise FileExistsError(output)
    fit = json.loads((directory / f"fit_an{a.angle}.json").read_text())
    p = dict(fit["best"])
    if a.catalogue_state:
        p.update(dist=OCEN_DISTANCE_KPC, pmra=OCEN_OBS["pmra"], pmdec=OCEN_OBS["pmdec"])
    hk = dict(bar_size=p["size"])
    if a.eta:
        hk["bar_eta"] = a.eta
    host = host_potential("x", bar_omega=p["omega"], bar_angle_deg=a.angle, bar_amp=p["amp"], t_today=T, **hk)
    ag = agama_kpc()
    today = to_model(OCEN_OBS["ra"], OCEN_OBS["dec"], p["dist"], p["pmra"], p["pmdec"], OCEN_OBS["vlos"])[0]
    n_fine = int(round(a.back_myr / .05)) + 1
    tc, center = ag.orbit(potential=host, ic=today, timestart=T, time=-a.back_myr / AGAMA_T_MYR,
                          trajsize=n_fine, accuracy=1e-12, verbose=False)
    step = 10
    times, orbit = tc[::step], center[::step]
    rj, vj, rotation = rj_vj_R(host, orbit, times, lambda rr: np.interp(np.log(rr), np.log(r_kpc), M))
    satellite = satellite_potential(np.empty((0, 3)), np.empty(0), core=FrozenCore(r_kpc, M))
    total = ag.Potential(host, ag.Potential(potential=satellite, center=np.column_stack((tc[::-1], center[::-1]))))
    distances = np.arange(2.5, 7.001, a.distance_step)
    data_sky = coord.SkyCoord(l=d["l"] * u.deg, b=d["b"] * u.deg, frame="galactic").icrs
    rv_indices = np.where(np.isfinite(d["v"]) & np.isfinite(d["e_v"]))[0]
    # Reference particles are chosen only by proximity on the sky, never by PM/RV.
    if not a.eta and not a.catalogue_state:
        control_file = ROOT / f"results/streams/{a.label}/heldout_an{a.angle}_refined_n32000_s4.npz"
        m = dict(np.load(control_file))
    else:
        from grid4_common import make_spray
        control_file = ROOT / f"results/streams/{a.label}/backtrack_control_an{a.angle}{suffix}.npz"
        if control_file.exists():
            m = dict(np.load(control_file))
        else:
            m = make_spray(p["omega"], a.angle, p["amp"], p["dist"], nrel=8000, seed=4,
                           pm=(p["pmra"], p["pmdec"]), host_kw=hk, max_age_myr=700.)
            np.savez_compressed(control_file, **m)
    mu, mx = project_gc(m["l"], m["b"], GC)
    keep = (m["chi"] > 0) & (m["age"] < a.back_myr)
    candidates = np.where(keep)[0]
    mock_indices = sorted(set(int(candidates[np.argmin((mu[candidates]-du[i])**2+(mx[candidates]-dx[i])**2)]) for i in rv_indices))
    mock_sky = coord.SkyCoord(l=m["l"] * u.deg, b=m["b"] * u.deg, frame="galactic").icrs
    rows = []
    payload = dict(args=vars(a), cluster_parameters={k: p[k] for k in ("omega", "amp", "size", "dist", "pmra", "pmdec")},
                   diagnostic="min sqrt[(r/(2 rJ))^2+(dv/(2 Omega rJ))^2]; not a likelihood",
                   distances=distances.tolist(), back_myr=a.back_myr, sample_step_myr=.5,
                   error_draw_seed=20261013, control_file=str(control_file.relative_to(ROOT)), rows=rows)
    rng = np.random.default_rng(20261013)

    def backtrack(ra, dec, pmra, pmdec, velocity):
        ics = to_model(np.full(len(distances), ra), np.full(len(distances), dec), distances,
                       np.full(len(distances), pmra), np.full(len(distances), pmdec), np.full(len(distances), velocity))
        trajectories = ag.orbit(potential=total, ic=ics, timestart=T, time=-a.back_myr/AGAMA_T_MYR,
                                trajsize=len(times), accuracy=1e-8, verbose=False)
        best = None
        for j, (tt, xv) in enumerate(trajectories):
            if np.max(abs(tt-times)) > 1e-10:
                raise ValueError("Star and cluster times differ")
            relative = xv-orbit
            r = np.linalg.norm(relative[:, :3], axis=1)
            v = np.linalg.norm(relative[:, 3:], axis=1)
            metric = np.sqrt((r/(2*rj))**2+(v/(2*vj))**2)
            q = int(metric.argmin())
            row = dict(metric=float(metric[q]), distance=float(distances[j]), radius_pc=float(r[q]*1000),
                       speed_kms=float(v[q]), rj_pc=float(rj[q]*1000), vj_kms=float(vj[q]),
                       radius_over_rj=float(r[q]/rj[q]), speed_over_vj=float(v[q]/vj[q]),
                       age_myr=float((T-times[q])*AGAMA_T_MYR), distance_boundary=j in (0, len(distances)-1),
                       age_boundary=q in (0, len(times)-1))
            if best is None or row["metric"] < best["metric"]:
                best = row
        return best

    for kind, indices in (("data", rv_indices), ("mock", mock_indices)):
        for index in indices:
            t0 = time.time()
            if kind == "data":
                sky, pmra, pmdec, velocity = data_sky[index], d["pmra"][index], d["pmdec"][index], d["v"][index]
                covariance = np.array([[d["e_pmra"][index]**2, d["rho"][index]*d["e_pmra"][index]*d["e_pmdec"][index]],
                                       [d["rho"][index]*d["e_pmra"][index]*d["e_pmdec"][index], d["e_pmdec"][index]**2]])
                values = [[float(pmra), float(pmdec), float(velocity)]]
                for _ in range(a.draws):
                    delta = rng.multivariate_normal([0., 0.], covariance)
                    values.append([float(pmra+delta[0]), float(pmdec+delta[1]), float(velocity+rng.normal()*d["e_v"][index])])
                extra = dict(phi1=float(du[index]), e_v=float(d["e_v"][index]), pm_covariance=covariance.tolist())
            else:
                sky = mock_sky[index]
                values = [[float(m["pmra"][index]), float(m["pmdec"][index]), float(m["vlos"][index])]]
                extra = dict(phi1=float(mu[index]), true_distance=float(m["d"][index]), true_release_age=float(m["age"][index]))
            encounters = [backtrack(float(sky.ra.deg), float(sky.dec.deg), *values_) for values_ in values]
            row = dict(kind=kind, index=int(index), ra=float(sky.ra.deg), dec=float(sky.dec.deg), observables=values,
                       encounters=encounters, nominal=encounters[0], best_error_draw=min(encounters, key=lambda e:e["metric"]),
                       seconds=time.time()-t0, **extra)
            rows.append(row)
            atomic_json(output, payload)
            print(json.dumps(dict(angle=a.angle, kind=kind, index=int(index), phi1=extra["phi1"],
                                  nominal=row["nominal"], best=row["best_error_draw"], seconds=row["seconds"])), flush=True)
    print("DONE", output, flush=True)


if __name__ == "__main__":
    main()
