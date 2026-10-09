#!/usr/bin/env python3
"""Mean rotation velocity v_rot(r) about the spin axis of the spun-up omega Cen model (A_nodm ICs + Lynden-Bell flips with the
fitted spin, results/streams/spin/A_nodm.json), in bins of 3D radius r. Used by the rotating spray (prescription A): released
particles get + v_rot(|x_rel|) e_phi. Also prints the dispersion and the flip fraction per bin.
Output: results/streams/spin/A_nodm_vrot.json (r_pc, vrot_kms, sigma_kms, n)
Usage: python bin/streams/spin_vrot_profile.py
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"src"))
import numpy as np
from ocen_dm.streams.restricted import FrozenCore, satellite_potential, rmax_of_energy
from ocen_dm.streams.rotation import spin_up, spin_axis_icrs, icrs_to_model_matrix
from ocen_dm.streams.frames import ocen_today

spin = json.loads((ROOT/"results/streams/spin/A_nodm.json").read_text())
prof = json.loads((ROOT/"results/nbody/A_nodm/model_profiles.json").read_text())
sat = satellite_potential(np.zeros((0, 3)), np.zeros(0), core=FrozenCore(np.array(prof["r_pc"])/1e3, np.array(prof["enclosed"]["total"])))
ics = np.load(ROOT/"results/nbody/A_nodm/ics.npz"); st = np.where(ics["species"] == 0)[0]
xv = np.hstack((ics["pos"][st]/1e3, ics["vel"][st])).astype(float)
rmax = rmax_of_energy(sat, sat.potential(xv[:, :3])+0.5*np.sum(xv[:, 3:]**2, 1))*1e3
today = ocen_today("baumgardt")
xs, flip = spin_up(xv, rmax, today, spin["q0"], spin["rq_pc"], pa_deg=spin["pa_deg"], incl_towards_deg=spin["incl_towards_deg"], r1_pc=spin["r1_pc"])
s = icrs_to_model_matrix("baumgardt") @ spin_axis_icrs(today, spin["pa_deg"], spin["incl_towards_deg"])   # model-frame spin vector
ephi = np.cross(s[None, :], xs[:, :3]); n = np.linalg.norm(ephi, axis=1); ephi /= np.where(n > 0, n, 1)[:, None]
vphi = np.einsum("ij,ij->i", xs[:, 3:], ephi); R = np.linalg.norm(xs[:, :3], axis=1)*1e3
edges = np.array([0, 2, 5, 10, 20, 30, 45, 60, 80, 100, 130, 160, 200, 250, 300, 400])
out = dict(r_pc=[], vrot_kms=[], sigma_kms=[], n=[], flip_frac=[], spin_model=str(ROOT/"results/streams/spin/A_nodm.json"), spin_vector_model=s.tolist())
for lo, hi in zip(edges[:-1], edges[1:]):
    k = (R >= lo) & (R < hi)
    if k.sum() < 50: continue
    out["r_pc"].append(float(np.median(R[k]))); out["vrot_kms"].append(float(vphi[k].mean())); out["n"].append(int(k.sum()))
    out["sigma_kms"].append(float(np.sqrt(np.mean(np.sum(xs[k, 3:]**2, 1))/3))); out["flip_frac"].append(float(flip[k].mean()))
    print(f"r {lo:4.0f}-{hi:4.0f} pc  n {k.sum():7d}  v_rot {vphi[k].mean():6.2f} +- {vphi[k].std()/np.sqrt(k.sum()):.2f} km/s  sigma_1D {out['sigma_kms'][-1]:5.2f}  flipped {flip[k].mean():.3f}")
(ROOT/"results/streams/spin/A_nodm_vrot.json").write_text(json.dumps(out, indent=1))
