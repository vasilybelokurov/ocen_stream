#!/usr/bin/env python3
"""Export the data side of the knot score (stream_model_compare.py) for the GPU score (pipeline stage 3, 2026-10-25).

Everything the model measurement needs that does not depend on the model: the stream frame (rotation, phi2 polynomial), the
frozen PM / sky tracks and windows on their phi1 grid, the Gaia PM-error table of the data stars in the sky window (sorted by
phi1), the 25-point data vector and its covariances, and per local sky window (stream_sky_local_nostrip, all 8 windows) the
0.1-deg x vg grid, area map Ag, fixed background lam_b, histogram edges, slope g, data rate r and track c.
Usage: PYTHONPATH=src python bin/streams/gpu_score_export.py -> results/gpu/score_inputs.npz
"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
sys.argv = [sys.argv[0]]
import stream_model_compare as MC
import stream_sky_local as SL
from ocen_stream.path import gc_rotation
OUT = ROOT/"results/gpu/score_inputs.npz"


def main():
    D = MC.data_vector(); O = MC.Obs(); o = np.argsort(O.err_u)
    W = []
    for wd in O.L["windows"]:
        a0, a1 = wd["phi1"]; w = SL.window(O.d, a0, a1, O.m2, O.m1); dv = w["vg"][1]-w["vg"][0]
        jg = np.clip(np.digitize(w["ug"], w["Eu"])-1, 0, len(w["Eu"])-2)
        lb = np.clip(w["cf"][jg]@np.vander(w["vg"], 3).T, 0, None)/w["cell"]*w["Ag"]
        W.append(dict(ug=w["ug"], vg=w["vg"], Ag=w["Ag"], lb=lb, eu0=w["ug"][0]-0.05, ev0=w["vg"][0]-0.5*dv, dv=dv, du=w["du"], pk=w["pk"],
                      g=w["g"], r=wd["r"], c=wd["c"], a0=a0, a1=a1, drop=tuple(wd["phi1"]) in MC.DROP))
    st = lambda k: np.array([w[k] for w in W])
    R = O.R; tr = R["tracks"]
    np.savez(OUT, Rm=gc_rotation(O.GC["pole"], O.GC["origin"]), coeffs=np.asarray(O.GC["coeffs"]), G=O.G,
             pmra_c=tr["pmra"]["c"], pmra_s=tr["pmra"]["s"], pmdec_c=tr["pmdec"]["c"], pmdec_s=tr["pmdec"]["s"],
             win_c=R["sky_window"]["centre"], win_hw=R["sky_window"]["half_width"], err_u=O.err_u[o], err=O.err[o],
             **{f"D_{k}": np.asarray(v) for k, v in D.items() if k != "win"},
             **{f"sky_{k}": st(k) for k in ("ug", "vg", "Ag", "lb", "eu0", "ev0", "dv", "du", "pk", "g", "r", "c", "a0", "a1", "drop")},
             smin=SL.SMIN, smax=SL.SMAX, nboot=MC.NBOOT)
    print(f"{OUT.relative_to(ROOT)}: {len(W)} sky windows, {len(O.err_u)} error stars, {OUT.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
