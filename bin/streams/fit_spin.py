#!/usr/bin/env python3
"""Fit the Lynden-Bell flip probability q(r_max) = q0 x^2/(1+x^2)/(1+(r_max/r_q)^2), x = r_max/r_1, of src/ocen_dm/streams/rotation.py to the Gaia DR3
plane-of-sky rotation profile (results/plot_data/gaia_dr3_ocen_rotation.json), per model, for the same 200k star tracers
as bin/streams/run_prescribed.py (same ICs, seed, satellite potential). The model is projected at omega Cen's present
position; pole sky PA 192 deg (Ibata+2019 axis, sense from our data); the tilt towards us (Ibata+2019: 45 deg) is fitted
jointly with q0, r_q to Gaia PM rotation and the oMEGACat VI LOS rotation amplitudes (r > 30''). Checks: E and |v| unchanged by flipping; the
LOS rotation of the projected model is maximal (receding) near PA 103 deg as in oMEGACat VI.
Usage: python bin/streams/fit_spin.py --model A_nodm [--ics ...]  -> results/streams/spin/<model>.json
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
import numpy as np

from ocen_dm.streams.restricted import OCEN_TODAY, FrozenCore, agama_kpc, rmax_of_energy, satellite_potential
from ocen_dm.streams.rotation import flip_probability, projected_kinematics, rotation_profile, spin_up


def tracers(model, ics_path, nstars=200000, seed=1):
    agama_kpc()
    prof = json.loads((ROOT/"results/nbody"/model/"model_profiles.json").read_text())
    sat = satellite_potential(np.zeros((0, 3)), np.zeros(0),
                              core=FrozenCore(np.array(prof["r_pc"])/1e3, np.array(prof["enclosed"]["total"])))
    ics = np.load(ics_path)
    stars = np.where(ics["species"] == 0)[0]
    pick = np.sort(np.random.default_rng(seed).choice(stars, min(nstars, len(stars)), replace=False))
    xv = np.hstack((ics["pos"][pick]/1e3, ics["vel"][pick])).astype(float)
    E = sat.potential(xv[:, :3]) + 0.5*np.sum(xv[:, 3:]**2, axis=1)
    return xv, rmax_of_energy(sat, E)*1e3, sat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--ics", type=Path, default=None)
    ap.add_argument("--tilts", type=float, nargs="+", default=[40, 45, 50, 55, 60, 65])
    ap.add_argument("--r1s", type=float, nargs="+", default=[0, 2, 4, 6, 10])
    a = ap.parse_args()
    xv, rmax, sat = tracers(a.model, a.ics or ROOT/"results/nbody"/a.model/"ics.npz")
    obs = json.loads((ROOT/"results/plot_data/gaia_dr3_ocen_rotation.json").read_text())["bins"]
    edges = [obs[0]["lo"]] + [b["hi"] for b in obs]
    y, e = np.array([b["vtan"] for b in obs]), np.array([b["vtan_err"] for b in obs])
    from astropy.table import Table
    los = Table.read(ROOT/"data/processed/kinematics/omegacat_vi_los_rotation.ecsv")
    los = los[los["r_lower"] >= 30]                                   # arcsec; the paper's axis fit uses r > 30''
    le = list(np.asarray(los["r_lower"])/60) + [float(los["r_upper"][-1])/60]
    ly = np.asarray(los["v_rot"]); lerr = 0.5*(np.asarray(los["v_rot_err_lo"])+np.asarray(los["v_rot_err_hi"]))
    best = None
    for tilt in a.tilts:
      for r1 in a.r1s:
        for q0 in np.linspace(0.4, 1.0, 7):
            for rq in np.geomspace(20, 200, 11):
                xs, _ = spin_up(xv, rmax, OCEN_TODAY, q0, rq, incl_towards_deg=tilt, r1_pc=r1)
                kin = projected_kinematics(xs, OCEN_TODAY)
                prof = rotation_profile(*kin, edges)
                lp = rotation_profile(*kin, le)
                m = np.array([p["vtan"] for p in prof]); em = np.array([p["vtan_err"] for p in prof])
                lm = np.array([p["vlos_amp"] for p in lp])
                c_pm = float(np.sum((m-y)**2/(e**2+em**2))); c_los = float(np.sum((lm-ly)**2/lerr**2))
                if best is None or c_pm+c_los < best[0]+best[1]:
                    best = (c_pm, c_los, tilt, q0, rq, prof, lp, r1)
    c_pm, c_los, tilt, q0, rq, prof, lp, r1 = best
    chi2 = c_pm+c_los
    xs, flip = spin_up(xv, rmax, OCEN_TODAY, q0, rq, incl_towards_deg=tilt, r1_pc=r1)
    dE = np.abs(np.linalg.norm(xs[:, 3:], axis=1)-np.linalg.norm(xv[:, 3:], axis=1)).max()
    esc = rmax >= 48.5
    wide = rotation_profile(*projected_kinematics(xs, OCEN_TODAY), [0.5, 1.5, 3, 10, 36, 60, 120])
    print(f"{a.model}: tilt {tilt:g} deg, q0 = {q0:.2f}, r_1 = {r1:g} pc, r_q = {rq:.1f} pc, chi2 PM {c_pm:.1f} (5 bins) + LOS {c_los:.1f} ({len(ly)} bins); flipped {flip.mean():.3f}; max | |v| change | = {dE:.1e} km/s")
    for p, o in zip(prof, obs):
        print(f"  {p['lo']:2g}-{p['hi']:2g}': model {p['vtan']:+.3f} +- {p['vtan_err']:.3f}  Gaia {o['vtan']:+.3f} +- {o['vtan_err']:.3f}")
    for p, v, ve in zip(lp, ly, lerr):
        print(f"  LOS {p['lo']*60:5.1f}-{p['hi']*60:5.1f}\": model amp {p['vlos_amp']:.2f} km/s  oMEGACat {v:.2f} +- {ve:.2f}")
    for p in wide:
        print(f"  {p['lo']:4g}-{p['hi']:4g}': N {p['n']:6d}  v_tan {p['vtan']:+.3f} mas/yr  v_los amp {p['vlos_amp']:.2f} km/s, max receding PA {p['pa_max_receding']:.0f}")
    print(f"  flip probability at r_max = 48.5 / 92 pc: {flip_probability(48.5, q0, rq, r1):.3f} / {flip_probability(92., q0, rq, r1):.3f}; escaper-region "
          f"(r_max >= 48.5 pc) stars: {esc.sum()}, flipped {flip[esc].mean():.3f}")
    out = ROOT/"results/streams/spin"; out.mkdir(parents=True, exist_ok=True)
    (out/f"{a.model}.json").write_text(json.dumps(dict(model=a.model, q0=q0, rq_pc=rq, r1_pc=float(r1), chi2=chi2, chi2_pm=c_pm, chi2_los=c_los, pa_deg=192., incl_towards_deg=float(tilt), los_bins=lp,
        fit_bins=prof, gaia_bins=obs, wide_bins=wide, flipped_fraction=float(flip.mean())), indent=1))


if __name__ == "__main__":
    main()
