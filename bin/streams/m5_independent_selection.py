#!/usr/bin/env python3
"""M5 (docs/STREAM_DIAGNOSTIC_PLAN.md): is the knee in the members' PMs shaped by the STREAMFINDER (orbit-template) selection?
Independent Gaia DR3 selection without orbit templates in the knee region (chord-frame phi1 > 8 deg):
  sky: RA/Dec box around the stream-54 members with phi1 > 8 deg, +3 deg margin (q3c_poly_query on gaia_dr3.gaia_source);
  quality: G < 20, ruwe < 1.4; distance: parallax - 3 parallax_error < 0.4 mas (rejects nearby stars, d > ~2.5 kpc allowed);
  PMs: broad window pmra -25..2, pmdec -17..-1 mas/yr (covers the whole model and member range; no orbit information);
  CMD: within max(0.05, 1.5 sigma) mag in BP-RP of the members' running-median locus in G (sigma = members' colour scatter, MAD),
  G 16-20 only; parallax < 0.25 + 2 parallax_error mas (in addition to the SQL cut).
Signal per 2-deg phi1 bin: PM density of on-stream stars (|dphi2 - members' local median| < 1.5 deg) minus off-stream control strips
(3 < |offset| < 7 deg, scaled by area f = 3/8), counts in a 0.75 mas/yr radius aperture around every 0.25 mas/yr pixel
(top-hat sums); significance map (N_on - f N_off)/sqrt(N_on + f^2 N_off); ridge = peak of the significance map at least 1 mas/yr inside the PM window edges,
compared with the members' median PMs. Query cached in results/streams/m5_gaia_knee.npz.
Outputs plots/m5_independent_selection.png, results/plot_data/m5_independent_selection.json.
Usage: python bin/streams/m5_independent_selection.py
"""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"bin/streams"))
import numpy as np, astropy.coordinates as coord, astropy.units as u
from astropy.table import Table
from scipy.ndimage import gaussian_filter, convolve
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from ocen_stream.path import project_gc

w = lambda x: np.where(x > 180, x-360, x)
GC = json.loads((ROOT/"results/plot_data/stream54_gc_frame_chord.json").read_text())
t = Table.read(os.path.expandvars("$OCEN_CATALOGUES/streamfinder_ibata2024_dr3.fits")); t = t[np.asarray(t["Stream"]) == 54]
ra, dec = np.asarray(t["RAdeg"], float), np.asarray(t["DEdeg"], float); g = coord.SkyCoord(ra, dec, unit="deg").galactic
l, b = w(g.l.deg), g.b.deg; keep = (b > 15) & (l < -20) & (l > -75)
mem = dict(sid=np.asarray(t["Gaia"], np.int64)[keep], ra=ra[keep], dec=dec[keep], l=l[keep], b=b[keep], pmra=np.asarray(t["pmRA"], float)[keep],
           pmdec=np.asarray(t["pmDE"], float)[keep], G=np.asarray(t["Gmag"], float)[keep], br=np.asarray(t["(B-R)"], float)[keep])
mem["u"], mem["x"] = project_gc(mem["l"], mem["b"], GC)
cache = ROOT/"results/streams/m5_gaia_knee.npz"
if cache.exists():
    q = dict(np.load(cache))
else:
    import sqlutilpy as sqlutil
    k = mem["u"] > 8; r0, r1, d0, d1 = mem["ra"][k].min(), mem["ra"][k].max(), mem["dec"][k].min(), mem["dec"][k].max()
    dm = 3.; rm = 3./np.cos(np.radians(max(abs(d0), abs(d1))+3))
    poly = [r0-rm, d0-dm, r1+rm, d0-dm, r1+rm, d1+dm, r0-rm, d1+dm]
    print("query box RA %.1f..%.1f Dec %.1f..%.1f" % (r0-rm, r1+rm, d0-dm, d1+dm), flush=True)
    sql = f"""SELECT source_id, ra, dec, pmra, pmdec, pmra_error, pmdec_error, parallax, parallax_error, phot_g_mean_mag AS g, bp_rp, ruwe
              FROM gaia_dr3.gaia_source
              WHERE q3c_poly_query(ra, dec, ARRAY[{','.join(f'{v:.4f}' for v in poly)}])
                AND phot_g_mean_mag < 20 AND ruwe < 1.4 AND parallax - 3*parallax_error < 0.4
                AND pmra BETWEEN -25 AND 2 AND pmdec BETWEEN -17 AND -1 AND bp_rp IS NOT NULL AND bp_rp <> 'NaN'::float8"""
    q = sqlutil.get(sql, asDict=True); q = {k_: np.asarray(v) for k_, v in q.items()}
    np.savez_compressed(cache, **q)
print("Gaia stars after quality/PM/parallax cuts:", len(q["ra"]))
gq = coord.SkyCoord(q["ra"], q["dec"], unit="deg").galactic; q["u"], q["x"] = project_gc(w(gq.l.deg), gq.b.deg, GC)
# CMD locus of the members (running median of BP-RP in G)
Gb = np.arange(14., 20.25, 0.5); mc = [np.nanmedian(mem["br"][(mem["G"] >= a) & (mem["G"] < a+0.5)]) for a in Gb[:-1]]
ms = [1.4826*np.nanmedian(np.abs(mem["br"][(mem["G"] >= a) & (mem["G"] < a+0.5)]-c)) for a, c in zip(Gb[:-1], mc)]
loc = np.interp(q["g"], Gb[:-1]+0.25, mc); wid = np.maximum(0.05, 1.5*np.interp(q["g"], Gb[:-1]+0.25, ms))
cmd = (np.abs(q["bp_rp"]-loc) < wid) & (q["g"] > 16) & (q["parallax"] < 0.25+2*q["parallax_error"])
insel = np.isin(q["source_id"], mem["sid"]); print(f"CMD-selected: {cmd.sum()}; members recovered in the query: {insel.sum()} "
                                                   f"(of {np.sum(mem['u'] > 8)} with phi1 > 8), of which pass CMD: {(insel & cmd).sum()}")
edges = np.arange(8., 30.01, 2.); cen = 0.5*(edges[1:]+edges[:-1]); pe = np.arange(-25, 2.01, 0.25), np.arange(-17, -0.99, 0.25)
res = dict(phi1=cen.tolist(), ridge_pmra=[], ridge_pmdec=[], mem_pmra=[], mem_pmdec=[], n_on=[], n_off_scaled=[])
fig, ax = plt.subplots(2, len(cen), figsize=(3.2*len(cen), 6.5))
for j, (a0, a1) in enumerate(zip(edges[:-1], edges[1:])):
    mk = (mem["u"] >= a0) & (mem["u"] < a1); xc = np.median(mem["x"][mk]) if mk.sum() > 5 else 0.
    s = cmd & (q["u"] >= a0) & (q["u"] < a1); off = q["x"]-xc
    on, offm = s & (np.abs(off) < 1.5), s & (np.abs(off) > 3.) & (np.abs(off) < 7.); fa = 3./8.
    yy, xx = np.mgrid[-3:4, -3:4]; disc = ((xx**2+yy**2) <= 9).astype(float)          # 0.75 mas/yr radius in 0.25 pixels
    Ho = convolve(np.histogram2d(q["pmra"][on], q["pmdec"][on], bins=pe)[0], disc, mode="constant")
    Hf = convolve(np.histogram2d(q["pmra"][offm], q["pmdec"][offm], bins=pe)[0], disc, mode="constant")
    D = (Ho-fa*Hf)/np.sqrt(np.maximum(Ho+fa**2*Hf, 1.))
    ce = 0.5*(pe[0][1:]+pe[0][:-1]), 0.5*(pe[1][1:]+pe[1][:-1]); inner = (ce[0][:, None] > -24) & (ce[0][:, None] < 1) & (ce[1][None, :] > -16) & (ce[1][None, :] < -2)
    i, k_ = np.unravel_index(np.argmax(np.where(inner, D, -np.inf)), D.shape); rp, rd = 0.5*(pe[0][i]+pe[0][i+1]), 0.5*(pe[1][k_]+pe[1][k_+1])
    res["ridge_pmra"].append(float(rp)); res["ridge_pmdec"].append(float(rd)); res["n_on"].append(int(on.sum())); res["n_off_scaled"].append(float(offm.sum()*fa)); res.setdefault("peak_sig", []).append(float(D[i, k_]))
    res["mem_pmra"].append(float(np.median(mem["pmra"][mk])) if mk.sum() else np.nan); res["mem_pmdec"].append(float(np.median(mem["pmdec"][mk])) if mk.sum() else np.nan)
    ax[0, j].imshow(D.T, origin="lower", extent=(-25, 2, -17, -1), aspect="auto", cmap="RdBu_r", vmin=-6, vmax=6)
    ax[0, j].plot(mem["pmra"][mk], mem["pmdec"][mk], "k.", ms=1.5, alpha=0.5); ax[0, j].plot(rp, rd, "y*", ms=12, mec="k")
    ax[0, j].set_title(f"phi1 {a0:g}-{a1:g}: on {on.sum()}, off {offm.sum()*fa:.0f}, peak {D[i, k_]:.1f} sig", fontsize=8); ax[0, j].set_xlabel("pmra"); ax[0, j].set_ylabel("pmdec")
    ax[1, j].plot(q["x"][s]-xc, q["pmdec"][s], ",", color="0.5"); ax[1, j].plot(mem["x"][mk]-xc, mem["pmdec"][mk], "r.", ms=2)
    ax[1, j].set_xlim(-8, 8); ax[1, j].set_ylim(-17, -1); ax[1, j].set_xlabel("dphi2 - members' median"); ax[1, j].set_ylabel("pmdec")
    print(f"phi1 {a0:4.0f}-{a1:4.0f}: on {on.sum():5d} off(scaled) {offm.sum()*fa:7.1f} peak {D[i, k_]:4.1f} sig  ridge pmra {rp:6.2f} pmdec {rd:6.2f} | members "
          f"median pmra {res['mem_pmra'][-1]:6.2f} pmdec {res['mem_pmdec'][-1]:6.2f} (n {mk.sum()})")
fig.suptitle("M5: independent Gaia DR3 selection (no orbit templates; CMD + broad PM + parallax): on-stream minus scaled off-stream PM density "
             "significance (top; star = peak; black = STREAMFINDER members) and dphi2 vs pmdec (bottom; red = members)", fontsize=10)
fig.tight_layout(); fig.savefig(ROOT/"plots/m5_independent_selection.png", dpi=70)
(ROOT/"results/plot_data/m5_independent_selection.json").write_text(json.dumps(res, indent=1)); print("plots/m5_independent_selection.png")
