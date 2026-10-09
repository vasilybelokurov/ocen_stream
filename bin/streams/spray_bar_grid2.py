#!/usr/bin/env python3
"""Stage 2a re-run with the per-star sky-conditional score (src/ocen_dm/streams/score.py) and Gibbons-chi unwrapping
(spray_unwrapped). Same 45 bar models (Omega_b 30-36 x angle 24/28/32 x amplitude 0.8/1.0/1.2), model A, 1.96 Gyr, omega Cen at
OCEN_TODAY; release over the last 1000 Myr at fixed common epochs (common random numbers: same seed for all models).
Data: Ibata+2024 stream 54 members with b > 15, -75 < l < -20 (Gaia DR3 PM errors and correlations; 29 v_los).
Saves per model: trailing particles (l, b, pmra, pmdec, vlos, d, chi, age) to results/streams/spray_grid2/<tag>.npz;
summary results/plot_data/spray_bar_grid2.json.
Usage: python bin/streams/spray_bar_grid2.py --nrel 8000
"""
import argparse, itertools, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"src"))
import numpy as np, astropy.coordinates as coord
from astropy.table import Table
from ocen_dm.streams.restricted import AGAMA_T_MYR, OCEN_TODAY, host_potential
from ocen_dm.streams.frames import observables
from ocen_dm.streams.spray import spray_unwrapped
from ocen_dm.streams.score import score, overshoot_fraction, chi_track

w = lambda x: np.where(x > 180, x-360, x)
OMEGA = [30., 31.5, 33., 34.5, 36.]; ANGLE = [24., 28., 32.]; AMP = [0.8, 1.0, 1.2]


def load_data():
    t = Table.read(os.path.expanduser("$OCEN_CATALOGUES/streamfinder_ibata2024_dr3.fits")); t = t[np.isin(np.asarray(t["Stream"]), [54, 55])]
    er = np.load(ROOT/"results/plot_data/stream5455_gaia_errors.npz"); assert np.all(er["source_id"] == np.asarray(t["Gaia"], np.int64))
    g = coord.SkyCoord(np.asarray(t["RAdeg"], float), np.asarray(t["DEdeg"], float), unit="deg").galactic; l, b = w(g.l.deg), g.b.deg
    k = (np.asarray(t["Stream"]) == 54) & (b > 15) & (l < -20) & (l > -75)
    v = np.asarray(t["VHel"], float); ev = np.asarray(t["e_VHel"], float); v[~(np.isfinite(ev) & (ev < 300))] = np.nan
    return dict(l=l[k], b=b[k], pmra=np.asarray(t["pmRA"], float)[k], pmdec=np.asarray(t["pmDE"], float)[k], e_pmra=er["pmra_error"][k],
                e_pmdec=er["pmdec_error"][k], rho=er["pmra_pmdec_corr"][k], v=v[k], e_v=ev[k])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nrel", type=int, default=8000); ap.add_argument("--window", type=float, default=1000.)
    a = ap.parse_args()
    data = load_data()
    prof = json.loads((ROOT/"results/nbody/A_nodm/model_profiles.json").read_text()); r_kpc, M = np.array(prof["r_pc"])/1e3, np.array(prof["enclosed"]["total"])
    T = 1955.58/AGAMA_T_MYR; trel = np.linspace(T-a.window/AGAMA_T_MYR, T, a.nrel)
    outd = ROOT/"results/streams/spray_grid2"; outd.mkdir(parents=True, exist_ok=True); res = []
    for om, an, am in itertools.product(OMEGA, ANGLE, AMP):
        t0 = time.time()
        host = host_potential("x", bar_omega=om, bar_angle_deg=an, t_today=T, bar_amp=am)
        sp = spray_unwrapped(host, OCEN_TODAY.copy(), T, r_kpc, M, trel, seed=1)
        tr = sp["arm"] == 1; o = observables(sp["xv"][tr]); l = w(o["l"])
        mod = dict(l=l, b=o["b"], pmra=o["pmra"], pmdec=o["pmdec"], vlos=o["vlos"])
        sc = score(data, mod)
        ov = overshoot_fraction(l, o["b"], sp["t_release_myr_ago"][tr], data["l"], data["b"])
        k = sp["chi"][tr] > 0
        ct = chi_track(sp["chi"][tr][k], dict(l=l[k], b=o["b"][k], pmra=o["pmra"][k], pmdec=o["pmdec"][k], vlos=o["vlos"][k], d=o["dist"][k],
                                               age=sp["t_release_myr_ago"][tr][k]))
        tag = f"om{om:g}_an{an:g}_am{am:g}"
        np.savez_compressed(outd/f"{tag}.npz", l=l, b=o["b"], pmra=o["pmra"], pmdec=o["pmdec"], vlos=o["vlos"], d=o["dist"], chi=sp["chi"][tr],
                            age=sp["t_release_myr_ago"][tr])
        res.append(dict(omega=om, angle=an, amp=am, score=sc, overshoot=ov, chi_track=ct, seconds=time.time()-t0))
        print(f"Omega {om:5.1f} angle {an:4.0f} amp {am:3.1f}: lnL {sc['total']:10.1f} (sky {sc['sky']:9.1f}, pm {sc['pm']:9.1f}, vlos {sc['vlos']:7.1f}; "
              f"bg-only sky {sc['n_bg_sky']}, pm {sc['n_bg_pm']})  overshoot {ov:.2f}  [{time.time()-t0:.0f} s]", flush=True)
        (ROOT/"results/plot_data/spray_bar_grid2.json").write_text(json.dumps(dict(n_data=len(data["l"]), grid=res)))


if __name__ == "__main__":
    main()
