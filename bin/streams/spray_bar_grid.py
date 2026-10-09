#!/usr/bin/env python3
"""Stage 2a: spray grid over bar pattern speed, angle and amplitude (Hunter+2024; model A mass profile; 1.96 Gyr; omega Cen at
OCEN_TODAY). Score per observable: chi2 = sum (model - data)^2 / (sig_data^2 + sig_model^2) over b bins where both are finite;
model bins flagged (masked) if the mixture fit is unreliable (sig_mu > 1 mas/yr or 1 deg, or stream fraction < 0.5).
No extra error floor (project rule). Output: results/plot_data/spray_bar_grid.json (all tracks + chi2).
Usage: python bin/streams/spray_bar_grid.py --nrel 10000
"""
import argparse, itertools, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"src"))
import numpy as np
from ocen_dm.streams.restricted import AGAMA_T_MYR, OCEN_TODAY, host_potential
from ocen_dm.streams.frames import observables
from ocen_dm.streams.spray import spray
from ocen_dm.streams.track import measure

w = lambda x: np.where(x > 180, x-360, x)
OMEGA = [30., 31.5, 33., 34.5, 36.]; ANGLE = [24., 28., 32.]; AMP = [0.8, 1.0, 1.2]


def score(tr, data):
    out = {}
    for q in ("l", "pmra", "pmdec"):
        m, sm, fr = (np.asarray(tr[q][k], float) for k in ("mu", "sig_mu", "frac"))
        d, sd = np.asarray(data[q]["mu"], float), np.asarray(data[q]["sig_mu"], float)
        bad = ~np.isfinite(m) | ~np.isfinite(sm) | (sm > 1.0) | (fr < 0.5)
        ok = ~bad & np.isfinite(d) & np.isfinite(sd)
        out[q] = dict(chi2=float(np.sum((m[ok]-d[ok])**2/(sd[ok]**2+sm[ok]**2))), nbins=int(ok.sum()), masked=int((bad & np.isfinite(d)).sum()))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nrel", type=int, default=10000); ap.add_argument("--tgyr", type=float, default=1.95558)
    a = ap.parse_args()
    data = json.loads((ROOT/"results/plot_data/stream54_track.json").read_text())["all"]
    prof = json.loads((ROOT/"results/nbody/A_nodm/model_profiles.json").read_text())
    r_kpc, M = np.array(prof["r_pc"])/1e3, np.array(prof["enclosed"]["total"])
    T = a.tgyr*1e3/AGAMA_T_MYR; res = []
    outp = ROOT/"results/plot_data/spray_bar_grid.json"
    for om, an, am in itertools.product(OMEGA, ANGLE, AMP):
        t0 = time.time()
        host = host_potential("hunter24_bar", bar_omega=om, bar_angle_deg=an, t_today=T, bar_amp=am)
        sp = spray(host, OCEN_TODAY.copy(), T, r_kpc, M, n_release=a.nrel, seed=1)
        xv = sp["xv"][sp["arm"] == 1]; o = observables(xv); l = w(o["l"]); k = (o["b"] > 15) & (l < -20) & (l > -75)
        tr = measure(l[k], o["b"][k], o["pmra"][k], o["pmdec"][k], nboot=30)
        sc = score(tr, data); tot = sum(v["chi2"] for v in sc.values())
        res.append(dict(omega=om, angle=an, amp=am, chi2=sc, chi2_total=tot, n_arm=int(k.sum()), seconds=time.time()-t0,
                        track={q: {kk: np.asarray(v).tolist() for kk, v in tr[q].items()} for q in ("l", "pmra", "pmdec")}, n=tr["n"].tolist()))
        print(f"Omega {om:5.1f} angle {an:4.0f} amp {am:3.1f}: chi2 l {sc['l']['chi2']:9.1f} ({sc['l']['nbins']},{sc['l']['masked']})  "
              f"pmra {sc['pmra']['chi2']:8.1f} ({sc['pmra']['nbins']},{sc['pmra']['masked']})  pmdec {sc['pmdec']['chi2']:8.1f} "
              f"({sc['pmdec']['nbins']},{sc['pmdec']['masked']})  N {k.sum()}  [{time.time()-t0:.0f} s]", flush=True)
        outp.write_text(json.dumps(dict(data_b=data["b"], grid=res)))


if __name__ == "__main__":
    main()
