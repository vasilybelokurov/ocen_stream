#!/usr/bin/env python3
"""Scoping diagnostic for the restricted-N-body stream experiments: enclosed mass and Jacobi radius
of each fitted omega Cen model (no DM, fixed-rho20 scan, photometric-mass DM) at the pericentre
and apocentre of omega Cen's orbit in McMillan17.

r_J solves r^3 = G M(<r) / (Omega^2 - d2Phi/dr2) at the orbital radius (spherically averaged host,
circular-orbit tidal term; for an eccentric orbit this is the instantaneous estimate).
Host terms are computed in a subprocess (AGAMA kpc units) so they never mix with the pc-unit DF
models of this process.

Usage: python bin/streams/scope_jacobi.py   (writes results/plot_data/streams_scope_jacobi.json)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(key, "4")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))

import numpy as np
from scipy.optimize import brentq

MODELS = {
    "A_nodm": "results/df/dftwo_counts_20260923::observed_no_halo_start0",
    **{f"rho20_{v}": f"results/df/rho20scan_{v}_20260923::observed_free_halo_start0" for v in ("0.25", "0.5", "1", "2", "4")},
    "B_mps9": "results/df/mpsscan_9_20260923::observed_free_halo_start0",
}
R_ORB_KPC = (1.42, 7.11)          # pericentre, apocentre in McMillan17 (JOURNAL 2026-09-24)

HOST_CODE = r"""
import agama, glob, os, json, numpy as np
agama.setUnits(mass=1, length=1, velocity=1)
ini = glob.glob(os.path.dirname(agama.__file__)+'/**/McMillan17.ini', recursive=True)[0]
pot = agama.Potential(ini)
out = {}
for R in %s:
    # spherically averaged: average over directions of the radial force and its radial derivative
    n = 2000; rng = np.random.default_rng(1); u = rng.normal(size=(n, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
    h = 1e-3*R
    fr = lambda r: np.mean(np.sum(pot.force(u*r)*u, axis=1))     # (km/s)^2/kpc, negative inward
    omega2 = -fr(R)/R
    d2phi = (fr(R-h)-fr(R+h))/(2*h)                              # d2Phi/dr2 = -d(F_r)/dr
    out[str(R)] = dict(omega2=omega2, d2phi=d2phi, tidal=omega2-d2phi)     # (km/s/kpc)^2
print(json.dumps(out))
"""


def host_terms():
    res = subprocess.run([sys.executable, "-c", HOST_CODE % (list(R_ORB_KPC),)], capture_output=True, text=True, check=True)
    return json.loads(res.stdout.strip().splitlines()[-1])


def main():
    host = host_terms()
    from ocen_dm.kinematics.compact_recovery import mass_profiles
    from ocen_dm.kinematics.df_fit import build_df_model, model_config_from_dict
    from ocen_dm.kinematics.positive_df import agama_pc
    G = agama_pc().G                                        # pc (km/s)^2 / Msun
    radii = np.logspace(-1, 3.3, 200)                       # 0.1 - 2000 pc
    out = dict(host=host, models={})
    for name, spec in MODELS.items():
        batch, job = spec.split("::")
        summary = json.loads((ROOT/batch/"fits"/job/"summary.json").read_text())
        model = build_df_model(model_config_from_dict(summary["best"]["config"]))
        prof = mass_profiles(model, radii)
        tot = np.array(prof["total"])
        Mr = lambda r: np.interp(np.log(r), np.log(radii), tot)
        row = dict(rho20=prof["rho20"], M_star=prof["M_star"],
                   M_halo_total=float(np.array(prof["halo"])[-1]),
                   M_tot_le={str(R): float(Mr(R)) for R in (20, 50, 100, 200, 500, 1000)})
        for key, h in host.items():
            k = h["tidal"]*1e-6                              # (km/s/pc)^2
            rJ = brentq(lambda r: r**3 - G*Mr(r)/k, 1., 1990.)
            row[f"rJ_pc_at_{key}kpc"] = float(rJ); row[f"M_lt_rJ_at_{key}kpc"] = float(Mr(rJ))
        out["models"][name] = row
        print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != "M_tot_le"}),
              {k: f"{v:.3g}" for k, v in row["M_tot_le"].items()}, flush=True)
    path = ROOT/"results/plot_data/streams_scope_jacobi.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
