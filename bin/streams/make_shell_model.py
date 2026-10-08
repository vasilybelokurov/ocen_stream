#!/usr/bin/env python3
"""Mass profile of the "5x DM inside the apocentre Jacobi radius" model (user choice, 2026-10-08).

Start from model B (photometric-mass DM fit; results/nbody/B_dm_phot/model_profiles.json). Keep it unchanged
inside the kinematic data (r < R_IN = 63 pc, the outer edge of the Gaia profiles: 2400'' at 5.43 kpc). In the
shell R_IN < r < R_OUT = 200 pc (apocentre Jacobi radius) set the DM density to a constant rho_sh chosen so that
M_DM(< R_OUT) = FACTOR x M_DM,B(< R_OUT); beyond R_OUT keep B's halo. Edges smoothed with tanh of width EDGE
(3 pc). Mass added outside 63 pc exerts no force inside 63 pc, so the fitted kinematics are unchanged; the
star tracers (B's DF sample) are in equilibrium there, while the few stars beyond 63 pc become more bound.
Writes results/nbody/C_dm5x_shell/model_profiles.json (same format: r_pc, density, enclosed per component).

Usage: python bin/streams/make_shell_model.py [--factor 5]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq

R_IN, R_OUT, EDGE = 63., 200., 3.


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--factor", type=float, default=5.)
    p.add_argument("--out", default="results/nbody/C_dm5x_shell")
    args = p.parse_args()
    B = json.loads((ROOT/"results/nbody/B_dm_phot/model_profiles.json").read_text())
    r = np.array(B["r_pc"])
    rf = np.unique(np.concatenate([r, np.linspace(40, 260, 2201)]))           # fine grid across the shell edges
    rho_B = np.exp(np.interp(np.log(rf), np.log(r), np.log(np.maximum(np.array(B["density"]["halo"]), 1e-300))))
    w = 0.5*(np.tanh((rf-R_IN)/EDGE) - np.tanh((rf-R_OUT)/EDGE))               # 1 inside the shell, 0 outside

    def rho_C(rho_sh):
        return rho_B*(1-w) + rho_sh*w

    def M_of(rho):
        return np.concatenate([[0.], cumulative_trapezoid(4*np.pi*rf**2*rho, rf)]) + np.interp(rf[0], r, B["enclosed"]["halo"])

    MB_out = np.interp(R_OUT, rf, M_of(rho_B))
    target = args.factor*MB_out
    rho_sh = brentq(lambda x: np.interp(R_OUT, rf, M_of(rho_C(x)))-target, 1e-3, 10.)
    rho = rho_C(rho_sh)
    Mh = M_of(rho)
    out = dict(r_pc=rf.tolist(), density={}, enclosed={},
               notes=dict(base="B_dm_phot", r_in_pc=R_IN, r_out_pc=R_OUT, edge_pc=EDGE, factor=args.factor, rho_shell=rho_sh,
                          M_DM_B_lt_200=MB_out, M_DM_C_lt_200=float(np.interp(R_OUT, rf, Mh)),
                          M_DM_B_total=float(B["enclosed"]["halo"][-1]), M_DM_C_total=float(Mh[-1]),
                          rho_B_at_63=float(np.interp(R_IN, rf, rho_B))))
    for comp in ("stars", "remnants"):
        out["density"][comp] = np.interp(rf, r, B["density"][comp]).tolist()
        out["enclosed"][comp] = np.interp(rf, r, B["enclosed"][comp]).tolist()
    out["density"]["halo"] = rho.tolist(); out["enclosed"]["halo"] = Mh.tolist()
    out["density"]["total"] = (np.array(out["density"]["stars"])+np.array(out["density"]["remnants"])+rho).tolist()
    out["enclosed"]["total"] = (np.array(out["enclosed"]["stars"])+np.array(out["enclosed"]["remnants"])+Mh).tolist()
    d = ROOT/args.out
    d.mkdir(parents=True, exist_ok=True)
    (d/"model_profiles.json").write_text(json.dumps(out))
    n = out["notes"]
    print(f"rho_shell = {rho_sh:.4f} Msun/pc^3 (B at 63 pc: {n['rho_B_at_63']:.4f}); M_DM(<200) B {MB_out:.3e} -> C {n['M_DM_C_lt_200']:.3e}; "
          f"total DM B {n['M_DM_B_total']:.3e} -> C {n['M_DM_C_total']:.3e}; M_total(<200) C {np.interp(200, rf, out['enclosed']['total']):.3e}")
    for R in (20, 63, 80, 200, 1000):
        print(f"  r<{R:4d}: M_DM B {np.interp(R, r, B['enclosed']['halo']):.3e}  C {np.interp(R, rf, Mh):.3e}")


if __name__ == "__main__":
    main()
