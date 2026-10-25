#!/usr/bin/env python3
"""Sweep the stored stream simulations for the upper sky branch (2026-10-24).

Per particle file (l, b, pmra, pmdec, d, chi, age): debris chi > 0 (trailing, as score.py / branch_search_gaia), age < age_cut_myr
(stored) or 700 Myr, chord frame (project_gc), b > 8, |dphi2| < 15. y = dphi2 - data sky-window centre (stream_masks.json).
Metrics in phi1 12-20 (data values in brackets; stream_branch_*.py):
  R_win    particles in |y - m2_data| < 1 / in |y - m_data| < 1 (data tracks of dphi2_sky_final) [~0.63 from the PM-fit counts,
           CMD-selected: branch ~0.25 mag fainter, so a lower limit for an all-debris count]
  peak     secondary peak of the (y - model main track) histogram (0.25-deg bins, smoothed 0.5 deg) at +1.5..+5 deg: offset and
           height / main-peak height (model main track = median y within 2 deg of the data main track per 1-deg phi1 slice)
  dmu      median |mu| in the branch window - in the main window, along the main PM direction, 16-20 [-0.78 +- 0.27; 12-20 -0.60 +- 0.20]
  ddm      5 log10(median d branch / median d main), 12-20 [+0.26 +- 0.16]
  leading  fraction of chi < 0 particles in the branch window (diagnostic: leading-arm debris projected there)
Parameters: tokens in the file name (letters + number) and, for runs/ directories, the record in manifest/state/design json if
it names the file. Output: results/plot_data/branch_sweep.csv (one row per file).
Usage: PYTHONPATH=src python bin/streams/branch_sweep.py [--dirs spray_grid4,...] [--all]
"""
import csv, json, re, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
from scipy.ndimage import gaussian_filter1d
from ocen_stream.path import project_gc
from stream_spline_fit import sky_tracks

GC = json.loads((ROOT/"results/plot_data/stream54_gc_frame_chord.json").read_text())
SKIP = re.compile(r"T2c0|T1c|T3c|recovery|mocks|perstar|_jk|snap_|hunter_|db98|prescribed|validation|checks")
ALL_DIRS = ["spray_grid4", "spray_grid4_spin", "spray_grid2", "training_9d_20261017", "search_real_wave1_20261019", "search_real_wave2_20261019",
            "search_real_wave3_20261019", "profile_real_20261019", "profile_step2_20261020", "obsbar_20261022", "fixedbar_20261021", "nullfits_20261022",
            "sun_dist_bar", "paper", "excess_pattern_20261021", "peri_release_20261021", "mass_sweep", "profile_s1", "bar_profile_refinement_20261013",
            "phase2b", "emulator_round1_20261018", "emulator_round2_20261018", "training_9d_reps_20261018", "bar_prescreen_20261021", "d3_free_pm"]
HW = 1.0; LO, HI = 12., 20.


def params_from_name(stem):
    return {k: float(v) for k, v in re.findall(r"(?:^|_)([a-z]+)([-+]?\d+(?:\.\d+)?)(?=_|$)", stem)}


def json_params(d):
    """file stem -> parameter dict from the directory's json records (best effort: any dict entry carrying a 'file'/'name'/'run'
    key or keyed by the stem, with numeric leaves)."""
    out = {}
    for fn in ("manifest.json", "state.json", "design.json", "result.json", "grid.json"):
        f = d/fn
        if not f.exists():
            continue
        try:
            J = json.loads(f.read_text())
        except Exception:
            continue
        def walk(o, key=None):
            if isinstance(o, dict):
                nums = {k: v for k, v in o.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
                for kk in ("best_params", "params", "theta"):
                    if isinstance(o.get(kk), dict):
                        nums |= {k: v for k, v in o[kk].items() if isinstance(v, (int, float))}
                tag = next((str(o[k]) for k in ("file", "name", "run", "tag", "id") if k in o and isinstance(o[k], (str, int))), key)
                if tag is not None and nums:
                    out.setdefault(Path(str(tag)).stem, {}).update(nums)
                for k, v in o.items():
                    walk(v, k)
            elif isinstance(o, list):
                for v in o:
                    walk(v, key)
        walk(J)
    return out

from scipy.optimize import minimize
from scipy.special import ndtr, logsumexp
YL, YH = -8., 8.; SLICES = ((12., 14.), (14., 16.), (16., 18.), (18., 20.))


def _lin_bg(y, b):
    """linear background density on [YL, YH], slope b clipped so it stays positive."""
    b = np.clip(b, -0.99/8, 0.99/8); return (1+b*y)/(YH-YL)


def _gauss(y, m, s):
    return np.exp(-0.5*((y-m)/s)**2)/(s*np.sqrt(2*np.pi))/max(ndtr((YH-m)/s)-ndtr((YL-m)/s), 1e-9)


def twoband(y):
    """unbinned fits on [YL, YH]: one Gaussian + linear background vs two Gaussians (m2 >= m1 + 1) + linear background.
    Returns dBIC (positive favours two), the two bands (lower, upper: mean, sigma, count), and the dip = 2G model density at its
    minimum between the means / density at the weaker band's mean (1 = no dip)."""
    y = y[(y > YL) & (y < YH)]; n = len(y)
    if n < 30:
        return None
    def nll1(p):
        m, ls, lf, b = p; f = 1/(1+np.exp(-lf)); s = np.exp(ls)
        return -np.sum(np.log(f*_gauss(y, m, s)+(1-f)*_lin_bg(y, b)+1e-300))
    def unpack2(p):
        m1, ls1, d, ls2, a1, a2, b = p; w = np.exp([a1, a2, 0.]-logsumexp([a1, a2, 0.])); return m1, np.exp(ls1), m1+1+np.exp(d), np.exp(ls2), w, b
    def nll2(p):
        m1, s1, m2, s2, w, b = unpack2(p)
        return -np.sum(np.log(w[0]*_gauss(y, m1, s1)+w[1]*_gauss(y, m2, s2)+w[2]*_lin_bg(y, b)+1e-300))
    med = np.median(y)
    r1 = min((minimize(nll1, (m0, np.log(1.), 0., 0.), method="Nelder-Mead", options=dict(maxiter=4000)) for m0 in (med-1, med, med+1)), key=lambda r: r.fun)
    st = [(m0, np.log(0.8), np.log(dd), np.log(0.8), 0., 0., 0.) for m0 in (med-3, med-2, med-1, med) for dd in (1., 2., 3.)]
    r2 = min((minimize(nll2, p0, method="Nelder-Mead", options=dict(maxiter=8000, xatol=1e-4, fatol=1e-4)) for p0 in st), key=lambda r: r.fun)
    m1, s1, m2, s2, w, b = unpack2(r2.x); dbic = 2*(r1.fun-r2.fun)-3*np.log(n)
    yy = np.linspace(m1, m2, 200); dens = lambda t: w[0]*_gauss(t, m1, s1)+w[1]*_gauss(t, m2, s2)+w[2]*_lin_bg(t, b)
    weak = m1 if w[0]*_gauss(m1, m1, s1) < w[1]*_gauss(m2, m2, s2) else m2; dip = float(dens(yy).min()/dens(np.array([weak]))[0])
    return dict(dbic=float(dbic), m_lo=m1, s_lo=s1, n_lo=w[0]*n, m_up=m2, s_up=s2, n_up=w[1]*n, dip=dip, sep=m2-m1, n=n)


def classify(T):
    """per-slice two-band results -> model summary (medians over slices with a fit) and class."""
    T = [t for t in T if t is not None]
    if not T:
        return dict(cls="none")
    two = [t for t in T if t["dbic"] > 10 and 1.5 <= t["sep"] <= 5 and t["dip"] < 0.5 and min(t["n_lo"], t["n_up"]) > 0.2*max(t["n_lo"], t["n_up"])]
    out = dict(n_slices=len(T), n_two=len(two), sep=float(np.median([t["sep"] for t in T])), dip=float(np.median([t["dip"] for t in T])),
               ratio_up_lo=float(np.median([t["n_up"]/max(t["n_lo"], 1e-9) for t in T])), dbic=float(np.median([t["dbic"] for t in T])),
               m_lo=float(np.median([t["m_lo"] for t in T])), m_up=float(np.median([t["m_up"] for t in T])))
    out["cls"] = "two" if len(two) >= 3 else ("fan" if np.median([t["s_lo"] for t in T]) > 1.5 else "single")
    return out



def measure(f, mt, bt, sc_grid, G):
    z = np.load(f);
    if "l" not in z.files:
        return None
    age_cut = float(z["age_cut_myr"]) if "age_cut_myr" in z.files else 700.
    u, x = project_gc(z["l"], z["b"], GC); y = x-np.interp(u, G, sc_grid)
    base = (z["b"] > 8) & (np.abs(x) < 15) & (z["age"] < age_cut) & (u >= LO) & (u < HI); tr = base & (z["chi"] > 0)
    um = mt(u); ub = bt(u); wm, wb = tr & (np.abs(y-um) < HW), tr & (np.abs(y-ub) < HW)
    r = dict(n_trail=int(tr.sum()), n_main=int(wm.sum()), n_branch=int(wb.sum()))
    r["R_win"] = wb.sum()/max(wm.sum(), 1)
    T = [twoband(y[tr & (u >= a0) & (u < a1)]-mt(np.array([0.5*(a0+a1)]))[0]) for a0, a1 in SLICES]; C = classify(T)
    r |= {f"tb_{k}": v for k, v in C.items()}; r["tb_slices"] = json.dumps([None if t is None else {k: round(float(v), 3) for k, v in t.items()} for t in T])
    if C.get("cls") == "two":                                    # PM / distance differences of the model's own two bands
        dl, du_ = [], []
        for (a0, a1), t in zip(SLICES, T):
            k = tr & (u >= a0) & (u < a1); yy = y[k]-mt(np.array([0.5*(a0+a1)]))[0]
            dl.append(k.nonzero()[0][np.abs(yy-t["m_lo"]) < min(t["s_lo"], 1.)]); du_.append(k.nonzero()[0][np.abs(yy-t["m_up"]) < min(t["s_up"], 1.)])
        il, iu = np.concatenate(dl), np.concatenate(du_)
        if len(il) >= 10 and len(iu) >= 10:
            A, D = np.median(z["pmra"][il]), np.median(z["pmdec"][il]); e = np.array([A, D])/np.hypot(A, D)
            r["tb_dmu"] = float((np.median(z["pmra"][iu])-A)*e[0]+(np.median(z["pmdec"][iu])-D)*e[1]); r["tb_ddm"] = float(5*np.log10(np.median(z["d"][iu])/np.median(z["d"][il])))
    wbl = base & (z["chi"] <= 0) & (np.abs(y-ub) < HW); r["leading"] = wbl.sum()/max(wbl.sum()+wb.sum(), 1)
    # model main track and secondary peak
    near = tr & (np.abs(y-um) < 2.); sl = np.floor(u).astype(int); mm = np.full(u.size, np.nan)
    for s in range(int(LO), int(HI)):
        k = near & (sl == s)
        if k.sum() >= 5:
            mm[sl == s] = np.median(y[k])
    ok = tr & np.isfinite(mm); E = np.arange(-6, 8.01, 0.25); h = gaussian_filter1d(np.histogram(y[ok]-mm[ok], E)[0].astype(float), 2.); c = 0.5*(E[1:]+E[:-1])
    hm = h[np.abs(c) < 0.75].max() if ok.any() else 0.; sec = (c >= 1.5) & (c <= 5.)
    if ok.sum() > 20 and hm > 0:
        j = np.argmax(np.where(sec, h, -1)); trough = h[(c > 0.5) & (c < c[j])].min() if np.any((c > 0.5) & (c < c[j])) else h[j]
        r.update(peak_off=float(c[j]), peak_ratio=float(h[j]/hm), peak_contrast=float(h[j]/max(trough, 1e-9)))
    else:
        r.update(peak_off=np.nan, peak_ratio=np.nan, peak_contrast=np.nan)
    # PM amplitude and distance differences (medians)
    for tag, a0, a1 in (("", LO, HI), ("_1620", 16., 20.)):
        km, kb = wm & (u >= a0) & (u < a1), wb & (u >= a0) & (u < a1)
        if km.sum() >= 10 and kb.sum() >= 10:
            A, D = np.median(z["pmra"][km]), np.median(z["pmdec"][km]); e = np.array([A, D])/np.hypot(A, D)
            r["dmu"+tag] = float((np.median(z["pmra"][kb])-A)*e[0]+(np.median(z["pmdec"][kb])-D)*e[1])
            r["ddm"+tag] = float(5*np.log10(np.median(z["d"][kb])/np.median(z["d"][km])))
        else:
            r["dmu"+tag] = r["ddm"+tag] = np.nan
    return r


def main():
    dirs = sys.argv[sys.argv.index("--dirs")+1].split(",") if "--dirs" in sys.argv else (ALL_DIRS if "--all" in sys.argv else ["spray_grid4"])
    R = json.loads((ROOT/"results/plot_data/stream_masks.json").read_text()); G = np.array(R["grid"]); sc = np.array(R["sky_window"]["centre"]); mt, bt = sky_tracks()
    rows = []; keys = set()
    for dn in dirs:
        d = ROOT/"results/streams"/dn; files = sorted(p for p in d.rglob("*.npz") if not SKIP.search(str(p.relative_to(ROOT/"results/streams"))))
        jp = json_params(d); n0 = len(rows)
        for f in files:
            try:
                r = measure(f, mt, bt, sc, G)
            except Exception as ex:
                print(f"  skip {f.name}: {ex}", flush=True); continue
            if r is None:
                continue
            p = params_from_name(f.stem) | jp.get(f.stem, {}); row = dict(dir=dn, file=f.stem, **{f"p_{k}": v for k, v in p.items()}, **r); rows.append(row); keys |= set(row)
        sub = rows[n0:]; Rw = np.array([q["R_win"] for q in sub]); dmu = np.array([q["dmu_1620"] for q in sub], float); ddm = np.array([q["ddm"] for q in sub], float)
        if sub:
            print(f"{dn}: {len(sub)} models | R_win median {np.median(Rw):.2f} [{np.percentile(Rw, 5):.2f}, {np.percentile(Rw, 95):.2f}] | "
                  f"dmu(16-20) median {np.nanmedian(dmu):+.2f} | ddm median {np.nanmedian(ddm):+.2f}", flush=True)
    order = ["dir", "file"]+sorted(k for k in keys if k.startswith("p_"))+[k for k in rows[0] if k not in ("dir", "file") and not k.startswith("p_")]
    out = ROOT/f"results/plot_data/branch_sweep{'_' + '_'.join(dirs) if '--dirs' in sys.argv or '--all' not in sys.argv else ''}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=order); w.writeheader(); [w.writerow({k: q.get(k, "") for k in order}) for q in rows]
    print(out.relative_to(ROOT), len(rows), "rows")


if __name__ == "__main__":
    main()
