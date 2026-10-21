#!/usr/bin/env python3
"""Independent Gaia DR3 search for the off-track 'second branch' predicted by the 25-deg bar model (2026-10-21).

Pre-registered in JOURNAL.md (2026-10-21) before the query; design: docs/reviews/fable_selection_tests_20261021.md.
The best step-2 models (results/streams/profile_step2_20261020/runs/final_a25_1, final_a12.5_3) put debris released
~250-500 Myr ago into a branch south of the stream track (dphi2 -4..-9 deg), 5-6 mas/yr more negative in pmra, at 4.0-4.6 kpc;
the 25-deg model has ~2-3x more of it. STREAMFINDER's orbit-template selection could miss it. M5 (2026-10-10) did not test
this: it looked only within 1.5 deg of the track and used 3-7 deg strips as background.
Windows (per 4-deg phi1 bin 6-10, 10-14, 14-18, 18-22; chord great-circle frame):
  branch PM window = 16-84% of the 25-deg model's branch particles (pmra < members' median - 3, |dphi2| < 15);
  sky A = dphi2 in [-13, -1] minus |dphi2 - track| < 2.5 (track = members' median dphi2);
  track: sky |dphi2 - track| < 2.5, PM window = members' 16-84%.
Gaia cuts (SQL): 15.5 < G < 20, ruwe < 1.4, parallax - 2 err < 0.26, broad PM window; parallax (+0.017 mas ZP) consistent with [0.15, 0.27] at 2 sigma, CMD within
max(0.05, 1.5 sigma) of the members' running-median BP-RP locus (raw Gaia photometry of the members found in the query),
G 16-19.8; branch CMD shifted brighter by delta (primary 0.4; also 0, 0.2, 0.6). Members removed from A.
Background ABCD: B = same sky, PM window shifted by -5 (branch) / +5 (track) mas/yr in pmra; C = same PM window in sky sidebands
dphi2 -25..-13 and +4..+9; D = both; background = B C / D (separable sky x PM field). Leakage check: C from +4..+9 only.
Predictions: model particles in the same sky + PM windows (no CMD/parallax), R_pred = branch / track for each model.
Query cached to results/streams/branch_search_gaia.npz (reloaded if present).
Output: results/plot_data/branch_search_gaia.json, plots/branch_search_gaia.png
Usage: PYTHONPATH=src python bin/streams/branch_search_gaia.py
"""
import json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
import astropy.coordinates as coord, astropy.units as u
from astropy.table import Table
from ocen_dm.streams.path import project_gc

w = lambda x: np.where(x > 180, x-360, x)
GC = json.loads((ROOT/"results/plot_data/stream54_gc_frame_chord.json").read_text())
BINS = ((6, 10), (10, 14), (14, 18), (18, 22))
MODELS = {"25": "results/streams/profile_step2_20261020/runs/final_a25_1.npz", "12.5": "results/streams/profile_step2_20261020/runs/final_a12.5_3.npz"}
CACHE = ROOT/"results/streams/branch_search_gaia.npz"
PLX_ZP = 0.017


def members():
    t = Table.read(os.path.expandvars("$OCEN_CATALOGUES/streamfinder_ibata2024_dr3.fits")); t = t[np.asarray(t["Stream"]) == 54]
    ra, dec = np.asarray(t["RAdeg"], float), np.asarray(t["DEdeg"], float); g = coord.SkyCoord(ra, dec, unit="deg").galactic
    l, b = w(g.l.deg), g.b.deg; keep = (b > 15) & (l < -20) & (l > -75)
    m = dict(sid=np.asarray(t["Gaia"], np.int64)[keep], ra=ra[keep], dec=dec[keep], pmra=np.asarray(t["pmRA"], float)[keep],
             pmdec=np.asarray(t["pmDE"], float)[keep])
    m["u"], m["x"] = project_gc(l[keep], b[keep], GC)
    return m


def hull_polygon(ra, dec, pad=0.5):
    """Convex hull of the region in a local tangent plane, padded by pad deg, as a flat RA/Dec vertex list."""
    from scipy.spatial import ConvexHull
    r0, d0 = np.mean(ra), np.mean(dec); X = (ra-r0)*np.cos(np.radians(d0)); Y = dec-d0
    h = ConvexHull(np.column_stack((X, Y))); P = np.column_stack((X, Y))[h.vertices]; c = P.mean(0)
    P = c+(P-c)*(1+pad/np.linalg.norm(P-c, axis=1))[:, None]
    return [v for x, y in P for v in (r0+x/np.cos(np.radians(d0)), d0+y)]


def query():
    if CACHE.exists():
        return dict(np.load(CACHE))
    import sqlutilpy as sqlutil
    # polygon in RA/Dec around chord-frame phi1 5..23, dphi2 -26..+10 (convex hull of a dense grid, 0.5-deg pad)
    L, B = np.meshgrid(np.arange(-110., 10.01, 0.25), np.arange(-20., 75.01, 0.25)); L, B = L.ravel(), B.ravel()
    uu, xx = project_gc(L, B, GC); k = (uu >= 5) & (uu <= 23) & (xx >= -26) & (xx <= 10)
    c = coord.SkyCoord(L[k]*u.deg, B[k]*u.deg, frame="galactic").icrs; ra, dec = c.ra.deg, c.dec.deg
    poly = hull_polygon(ra, dec)
    print(f"query polygon: {len(poly)//2} vertices, RA {min(poly[0::2]):.1f}..{max(poly[0::2]):.1f}", flush=True)
    sql = f"""SELECT source_id, ra, dec, pmra, pmdec, pmra_error, pmdec_error, parallax, parallax_error, phot_g_mean_mag AS g, bp_rp, ruwe
              FROM gaia_dr3.gaia_source
              WHERE q3c_poly_query(ra, dec, ARRAY[{','.join(f'{v:.4f}' for v in poly)}])
                AND phot_g_mean_mag BETWEEN 15.5 AND 20 AND ruwe < 1.4 AND parallax - 2*parallax_error < 0.26
                AND pmra BETWEEN -25 AND 2 AND pmdec BETWEEN -17 AND -1 AND bp_rp IS NOT NULL AND bp_rp <> 'NaN'::float8"""
    q = sqlutil.get(sql, asDict=True); q = {k: np.asarray(v) for k, v in q.items()}
    np.savez_compressed(CACHE, **q)
    return q


def model_windows(mem):
    """Per bin: track dphi2, members' PM window, 25-deg branch PM window; and model particle counts for the predictions."""
    win, pred = {}, {k: {} for k in MODELS}
    M = {}
    for lab, f in MODELS.items():
        m = dict(np.load(ROOT/f)); uu, xx = project_gc(m["l"], m["b"], GC); k = (m["chi"] > 0) & (m["age"] < 700.) & (np.abs(xx) < 15)
        M[lab] = dict(u=uu[k], x=xx[k], pmra=m["pmra"][k], pmdec=m["pmdec"][k])
    for a0, a1 in BINS:
        km = (mem["u"] >= a0) & (mem["u"] < a1); tr = float(np.median(mem["x"][km])); mp = float(np.median(mem["pmra"][km]))
        p = M["25"]; kb = (p["u"] >= a0) & (p["u"] < a1) & (p["pmra"] < mp-3)
        win[(a0, a1)] = dict(track=tr, mem_pmra_med=mp,
                             tpm=(np.percentile(mem["pmra"][km], [16, 84]).tolist(), np.percentile(mem["pmdec"][km], [16, 84]).tolist()),
                             bpm=(np.percentile(p["pmra"][kb], [16, 84]).tolist(), np.percentile(p["pmdec"][kb], [16, 84]).tolist()))
        W = win[(a0, a1)]
        for lab, p in M.items():
            kk = (p["u"] >= a0) & (p["u"] < a1)
            A = kk & sky_A(p["x"], tr) & in_pm(p["pmra"], p["pmdec"], W["bpm"])
            T = kk & sky_T(p["x"], tr) & in_pm(p["pmra"], p["pmdec"], W["tpm"])
            pred[lab][(a0, a1)] = (int(A.sum()), int(T.sum()))
    return win, pred


sky_A = lambda x, tr: (x >= -13) & (x <= -1) & (np.abs(x-tr) >= 2.5)
sky_T = lambda x, tr: np.abs(x-tr) < 2.5
sky_C = lambda x, north_only=False: ((x >= 4) & (x <= 9)) | (((x >= -25) & (x <= -13)) if not north_only else False)
in_pm = lambda pa, pd, W, dpa=0.: (pa >= W[0][0]+dpa) & (pa <= W[0][1]+dpa) & (pd >= W[1][0]) & (pd <= W[1][1])


def abcd(nA, nB, nC, nD):
    bg = nB*nC/max(nD, 1); var = nA+bg**2*(1/max(nB, 1)+1/max(nC, 1)+1/max(nD, 1))
    return nA-bg, float(np.sqrt(var)), bg


def main():
    mem = members(); q = query(); print(f"Gaia stars after the SQL cuts: {len(q['ra'])}", flush=True)
    gq = coord.SkyCoord(q["ra"], q["dec"], unit="deg").galactic; q["u"], q["x"] = project_gc(w(gq.l.deg), gq.b.deg, GC)
    ismem = np.isin(q["source_id"], mem["sid"]); print(f"members found in the query: {ismem.sum()} of {len(mem['sid'])}", flush=True)
    # members' CMD locus in raw Gaia photometry
    gm, bm = q["g"][ismem], q["bp_rp"][ismem]; Gb = np.arange(14., 20.25, 0.5)
    mc = np.array([np.nanmedian(bm[(gm >= a) & (gm < a+0.5)]) for a in Gb[:-1]])
    ms = np.array([1.4826*np.nanmedian(np.abs(bm[(gm >= a) & (gm < a+0.5)]-c)) for a, c in zip(Gb[:-1], mc)])
    ok = np.isfinite(mc) & np.isfinite(ms); gx, mc, ms = Gb[:-1][ok]+0.25, mc[ok], ms[ok]
    plx = q["parallax"]+PLX_ZP; pplx = (plx-2*q["parallax_error"] < 0.27) & (plx+2*q["parallax_error"] > 0.15)

    def cmd(delta):
        g0 = q["g"]+delta                                    # a star at 0.4 mag brighter maps onto the members' locus
        loc = np.interp(g0, gx, mc); wid = np.maximum(0.05, 1.5*np.interp(g0, gx, ms))
        return (np.abs(q["bp_rp"]-loc) < wid) & (q["g"] > 16) & (q["g"] < 19.8) & pplx
    win, pred = model_windows(mem); out = dict(bins={}, predictions={}, summary={})
    for lab in MODELS:
        A = sum(v[0] for v in pred[lab].values()); T = sum(v[1] for v in pred[lab].values())
        out["predictions"][lab] = dict(per_bin={f"{a}-{b}": v for (a, b), v in pred[lab].items()}, R_pred=A/T)
        print(f"model {lab} deg: particles branch-window {A}, track-window {T}, R_pred = {A/T:.2f}  per bin " +
              ", ".join(f"{a}-{b}: {v[0]}/{v[1]}" for (a, b), v in pred[lab].items()))
    for delta in (0.4, 0.0, 0.2, 0.6):
        cb, ct = cmd(delta), cmd(0.0); tot = dict(eb=0., vb=0., et=0., vt=0., eb_n=0., vb_n=0.)
        for (a0, a1), W in win.items():
            s = (q["u"] >= a0) & (q["u"] < a1); tr = W["track"]
            A_ = s & cb & ~ismem & sky_A(q["x"], tr) & in_pm(q["pmra"], q["pmdec"], W["bpm"])
            B_ = s & cb & ~ismem & sky_A(q["x"], tr) & in_pm(q["pmra"], q["pmdec"], W["bpm"], -5.)
            C_ = s & cb & sky_C(q["x"]) & in_pm(q["pmra"], q["pmdec"], W["bpm"]); D_ = s & cb & sky_C(q["x"]) & in_pm(q["pmra"], q["pmdec"], W["bpm"], -5.)
            Cn = s & cb & sky_C(q["x"], True) & in_pm(q["pmra"], q["pmdec"], W["bpm"]); Dn = s & cb & sky_C(q["x"], True) & in_pm(q["pmra"], q["pmdec"], W["bpm"], -5.)
            eb, sb, bgb = abcd(A_.sum(), B_.sum(), C_.sum(), D_.sum()); ebn, sbn, _ = abcd(A_.sum(), B_.sum(), Cn.sum(), Dn.sum())
            TA = s & ct & sky_T(q["x"], tr) & in_pm(q["pmra"], q["pmdec"], W["tpm"]); TB = s & ct & sky_T(q["x"], tr) & in_pm(q["pmra"], q["pmdec"], W["tpm"], 5.)
            TC = s & ct & sky_C(q["x"]) & in_pm(q["pmra"], q["pmdec"], W["tpm"]); TD = s & ct & sky_C(q["x"]) & in_pm(q["pmra"], q["pmdec"], W["tpm"], 5.)
            et, st, bgt = abcd(TA.sum(), TB.sum(), TC.sum(), TD.sum())
            plx_ex = float(np.mean(plx[A_])-np.mean(plx[B_])*1.0) if A_.sum() and B_.sum() else np.nan
            pm_ex = float((np.sum(q["pmra"][A_])-bgb*np.mean(q["pmra"][B_]+5.))/max(eb, 1)) if B_.sum() else np.nan
            out["bins"].setdefault(str(delta), {})[f"{a0}-{a1}"] = dict(N_A=int(A_.sum()), bg_A=float(bgb), excess_branch=float(eb), sig_branch=float(sb),
                                                                         excess_branch_northC=float(ebn), sig_branch_northC=float(sbn),
                                                                         N_T=int(TA.sum()), bg_T=float(bgt), excess_track=float(et), sig_track=float(st),
                                                                         branch_pm_window=W["bpm"], track_pm_window=W["tpm"], track_dphi2=tr,
                                                                         excess_pmra_centroid=pm_ex, raw_mean_plx_A_minus_B=plx_ex)
            tot["eb"] += eb; tot["vb"] += sb**2; tot["et"] += et; tot["vt"] += st**2; tot["eb_n"] += ebn; tot["vb_n"] += sbn**2
        R = tot["eb"]/tot["et"]; sR = abs(R)*np.sqrt(tot["vb"]/max(tot["eb"]**2, 1e-9)+tot["vt"]/tot["et"]**2)
        out["summary"][str(delta)] = dict(excess_branch=tot["eb"], sig_branch=np.sqrt(tot["vb"]), z_branch=tot["eb"]/np.sqrt(tot["vb"]),
                                          excess_branch_northC=tot["eb_n"], z_branch_northC=tot["eb_n"]/np.sqrt(tot["vb_n"]),
                                          excess_track=tot["et"], sig_track=np.sqrt(tot["vt"]), R=R, sig_R=sR)
        S = out["summary"][str(delta)]
        print(f"delta {delta:.1f}: branch excess {S['excess_branch']:+8.1f} +- {S['sig_branch']:.1f} (z {S['z_branch']:+.1f}; north-C only z "
              f"{S['z_branch_northC']:+.1f}) | track excess {S['excess_track']:.1f} +- {S['sig_track']:.1f} | R = {R:+.3f} +- {sR:.3f}", flush=True)
        for k_, v in out["bins"][str(delta)].items():
            print(f"     phi1 {k_:5s}: A {v['N_A']:5d} bg {v['bg_A']:7.1f} excess {v['excess_branch']:+7.1f} +- {v['sig_branch']:5.1f} | "
                  f"track A {v['N_T']:5d} bg {v['bg_T']:7.1f} excess {v['excess_track']:+7.1f} +- {v['sig_track']:5.1f} | "
                  f"branch PM window pmra {v['branch_pm_window'][0][0]:.1f}..{v['branch_pm_window'][0][1]:.1f}", flush=True)
    (ROOT/"results/plot_data/branch_search_gaia.json").write_text(json.dumps(out, indent=1, default=float))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, len(BINS), figsize=(4.2*len(BINS), 8), constrained_layout=True)
    cb = cmd(0.4); ct = cmd(0.0)
    for j, ((a0, a1), W) in enumerate(win.items()):
        s = (q["u"] >= a0) & (q["u"] < a1)
        for i, (sel, title) in enumerate(((cb & ~ismem & sky_A(q["x"], W["track"]), "sky A (branch), CMD +0.4"),
                                          (cb & sky_C(q["x"]), "sky C sidebands, CMD +0.4"))):
            ax[i, j].hist2d(q["pmra"][s & sel], q["pmdec"][s & sel], bins=(np.arange(-25, 2.1, 0.5), np.arange(-17, -0.9, 0.5)), cmap="Greys")
            for Wn, col in ((W["bpm"], "r"), (W["tpm"], "b")):
                ax[i, j].add_patch(plt.Rectangle((Wn[0][0], Wn[1][0]), Wn[0][1]-Wn[0][0], Wn[1][1]-Wn[1][0], fill=False, ec=col, lw=1.5))
            ax[i, j].set(title=f"phi1 {a0}-{a1}: {title}", xlabel="pmra", ylabel="pmdec")
    fig.suptitle("Gaia DR3 independent search: red = predicted branch PM window, blue = members' track window", fontsize=10)
    fig.savefig(ROOT/"plots/branch_search_gaia.png", dpi=100); print(ROOT/"plots/branch_search_gaia.png")


if __name__ == "__main__":
    main()
