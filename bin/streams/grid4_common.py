"""Shared set-up for the grid4 scripts: data (members, PM covariances, chord-frame coordinates), omega Cen profile, spray and
scoring helpers with the grid4 conventions (PMs fixed at the free-angle best, model A spray, release over the last 1000 Myr)."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"bin/streams")); sys.path.insert(0, str(ROOT/"src"))
import numpy as np
from spray_bar_grid2 import load_data, w
from ocen_dm.streams.restricted import AGAMA_T_MYR, host_potential
from ocen_dm.streams.frames import observables, to_model, OCEN_OBS
from ocen_dm.streams.spray import spray_unwrapped
from ocen_dm.streams.score import score_conditional
from ocen_dm.streams.path import project_gc

PMRA, PMDEC = -3.2223, -6.7517
d = load_data(); nd = len(d["l"])
cov = np.zeros((nd, 2, 2)); cov[:, 0, 0] = d["e_pmra"]**2; cov[:, 1, 1] = d["e_pmdec"]**2; cov[:, 0, 1] = cov[:, 1, 0] = d["rho"]*d["e_pmra"]*d["e_pmdec"]
GC = json.loads((ROOT/"results/plot_data/stream54_gc_frame_chord.json").read_text())
du, dx = project_gc(d["l"], d["b"], GC); dW = np.column_stack((dx, d["pmra"], d["pmdec"])); win = (du.min(), du.max())
prof = json.loads((ROOT/"results/nbody/A_nodm/model_profiles.json").read_text()); r_kpc, M = np.array(prof["r_pc"])/1e3, np.array(prof["enclosed"]["total"])
T = 1955.58/AGAMA_T_MYR
_vr = json.loads((ROOT/"results/streams/spin/A_nodm_vrot.json").read_text())
SPIN = dict(r_kpc=np.array(_vr["r_pc"])/1e3, vrot_kms=np.array(_vr["vrot_kms"]), s=np.array(_vr["spin_vector_model"]))


def make_spray(om, an, am, dist, nrel=8000, seed=1, spin=None):
    trel = np.linspace(T-1000/AGAMA_T_MYR, T, nrel)
    host = host_potential("x", bar_omega=om, bar_angle_deg=an, t_today=T, bar_amp=am)
    today = to_model(OCEN_OBS["ra"], OCEN_OBS["dec"], dist, PMRA, PMDEC, OCEN_OBS["vlos"])[0]
    sp = spray_unwrapped(host, today, T, r_kpc, M, trel, seed=seed, spin=spin); tr = sp["arm"] == 1; o = observables(sp["xv"][tr])
    return dict(l=w(o["l"]), b=o["b"], pmra=o["pmra"], pmdec=o["pmdec"], vlos=o["vlos"], d=o["dist"], chi=sp["chi"][tr], age=sp["t_release_myr_ago"][tr])


def score(m, **kw):
    mu, mx = project_gc(m["l"], m["b"], GC)
    return score_conditional(du, dW, d["v"], d["e_v"], cov, mu, np.column_stack((mx, m["pmra"], m["pmdec"])), m["vlos"], m["chi"], m["age"], u_window=win, **kw)
