#!/usr/bin/env python3
"""Mock stream-54 catalogues drawn exactly from the likelihood model at known parameters (2026-10-18; design reviewed by
Codex, docs/reviews/codex_cutoff_and_mocks_2026-10-18.md).

For a truth spray (make_spray, 256k release epochs, random epoch-grid start, trailing arm, age < 700 Myr) and the fixed
chi bins 0-200 of the score (bin weight 1/K for every bin, particle weight 1/(K n_k) in occupied bins), each real member i
keeps its along-stream coordinate u_i, Gaia PM covariance C_i and velocity availability; then
  with probability P_bg = eps g_u / ((1 - eps) f_u(u_i) + eps g_u): (x, pmra, pmdec) uniform in BG_BOUNDS, v uniform in
  V_BOUNDS (if member i has v);
  else particle j with probability proportional to w_j N(u_i - U_j; h_u), and x = X_j + N(0, h_x^2),
  PM = MU_j + N(0, H + C_i) with H = diag(h_pm^2), v = V_j + N(0, h_v^2 + e_v,i^2).
This is the conditional KDE mixture p(w | u) that the current score_conditional evaluates (finite-spray, raw score, no
selection): the component is chosen once and nothing is rejected (Codex 2026-10-18: rejecting foreground draws after
choosing the component samples a different mixture). Members drawn outside the background box are kept and are scored with
outside_bounds='zero' (no background density there); their number is recorded.
Truths: T1 fast low-angle bar at the measured distance, T2 conventional bar, T3 weak bar. Per truth: one generating spray
(seed 500000 + 10 t), catalogues c = 0, 1 (sampling seeds), and an independent truth-check spray (seed 500000 + 10 t + 5).
Output: $OCEN_SIMS-linked results/streams/mocks_20261018/T<t>/{truth_spray.npz, check_spray.npz, mock_c<c>.npz,
mock_c<c>.json}
Usage: OMP_NUM_THREADS=1 PYTHONPATH=src python bin/streams/mock_data.py --workers 6
"""
import argparse, hashlib, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
OUT = ROOT/"results/streams/mocks_20261018"
E = np.arange(0., 205., 5.)
EPS, H = 0.05, (0.5, 0.5, 0.2, 0.2)
HV = 5.
TRUTHS = {1: dict(angle=14.0, omega=34.8, amp=1.35, size=1.2, dist=5.48, pmra=-3.26, pmdec=-6.79, vc=235.0, halo_q=0.95),
          2: dict(angle=27.0, omega=35.0, amp=1.0, size=1.0, dist=5.43, pmra=-3.250, pmdec=-6.746, vc=229.0, halo_q=1.0),
          3: dict(angle=27.0, omega=35.0, amp=0.7, size=1.0, dist=5.43, pmra=-3.250, pmdec=-6.746, vc=229.0, halo_q=1.0)}
KEYS = ("l", "b", "pmra", "pmdec", "vlos", "d", "chi", "age")
# wide background box containing ~all model debris (0.01-99.99 % over 63 sprays, padded): mocks drawn from the full model
# PDF are scored with this box, so generation and scoring use the same likelihood (2026-10-18)
WIDE_W = ((-95., 55.), (-50., 10.), (-45., 15.)); WIDE_V = (-420., 450.)


def spray(p, seed):
    from grid4_common import make_spray
    off = float(np.random.default_rng([seed, 777]).uniform())
    m = make_spray(p["omega"], p["angle"], p["amp"], p["dist"], nrel=256000, seed=seed, pm=(p["pmra"], p["pmdec"]),
                   host_kw=dict(bar_size=p["size"], axi_variant=dict(halo_q=p["halo_q"], vc_target=p["vc"])), max_age_myr=700.,
                   epoch_offset=off)
    out = {k: np.asarray(m[k], np.float64) for k in KEYS}; out["age_cut_myr"] = np.float64(700.)
    return out, off


def draw(m, rng, box="wide"):
    """One mock catalogue from spray m (see module docstring). box: 'wide' (WIDE_W/WIDE_V, default) or 'data' (BG_BOUNDS)."""
    from grid4_common import du, dW, d, cov, win, BG_BOUNDS, V_BOUNDS, project_gc, GC
    if box == "wide":
        BG_BOUNDS, V_BOUNDS = WIDE_W, WIDE_V
    mu, mx = project_gc(m["l"], m["b"], GC)
    sel = (m["chi"] > 0) & (m["age"] < 700.)
    K = len(E)-1; kb = np.digitize(m["chi"], E)-1; inb = sel & (kb >= 0) & (kb < K)
    nk = np.bincount(kb[inb], minlength=K); w = np.zeros(len(mu)); w[inb] = 1./(K*nk[kb[inb]])
    U, X, PA, PD, V = mu[inb], mx[inb], m["pmra"][inb], m["pmdec"][inb], m["vlos"][inb]; wj = w[inb]
    g_u = 1./(win[1]-win[0]); hu, hx, hpa, hpd = H
    B = np.asarray(BG_BOUNDS, float); VB = np.asarray(V_BOUNDS, float)
    hasv = np.isfinite(d["v"]); n = len(du)
    out_w = np.zeros((n, 3)); out_v = np.full(n, np.nan); n_bg = n_re = 0
    for i in range(n):
        ker = wj*np.exp(-0.5*((du[i]-U)/hu)**2)/(np.sqrt(2*np.pi)*hu)
        fu = ker.sum()
        pbg = EPS*g_u/((1-EPS)*fu+EPS*g_u)
        if rng.uniform() < pbg:
            n_bg += 1
            out_w[i] = rng.uniform(B[:, 0], B[:, 1])
            if hasv[i]:
                out_v[i] = rng.uniform(*VB)
            continue
        Ci = np.array([[hpa**2+cov[i, 0, 0], cov[i, 0, 1]], [cov[i, 1, 0], hpd**2+cov[i, 1, 1]]])
        cdf = np.cumsum(ker/fu)
        j = min(int(np.searchsorted(cdf, rng.uniform())), len(cdf)-1)
        x = X[j]+rng.normal(0, hx); pm = rng.multivariate_normal([PA[j], PD[j]], Ci)
        v = V[j]+rng.normal(0, np.sqrt(HV**2+d["e_v"][i]**2)) if hasv[i] else np.nan
        inside = B[0, 0] <= x <= B[0, 1] and B[1, 0] <= pm[0] <= B[1, 1] and B[2, 0] <= pm[1] <= B[2, 1] and \
            (not hasv[i] or VB[0] <= v <= VB[1])
        n_re += not inside                      # counted, not redrawn
        out_w[i] = (x, pm[0], pm[1]); out_v[i] = v
    return dict(du=du.copy(), dw=out_w, dv=out_v, dev=d["e_v"].copy(), cov=cov.copy(), n_bg=n_bg, n_outside_box=n_re)


def spray_task(args):
    t, name, seed = args
    od = OUT/f"T{t}"; od.mkdir(parents=True, exist_ok=True); f = od/f"{name}.npz"; t0 = time.time()
    if f.exists():
        raise SystemExit(f"{f} exists; refusing to overwrite")
    m, off = spray(TRUTHS[t], seed); np.savez_compressed(od/f"{name}_tmp.npz", **m); os.replace(od/f"{name}_tmp.npz", f)
    (od/f"{name}.json").write_text(json.dumps(dict(truth=t, params=TRUTHS[t], seed=seed, epoch_offset=off, nrel=256000,
                                                   sha256=hashlib.sha256(f.read_bytes()).hexdigest(), seconds=time.time()-t0), indent=1))
    return t, name, time.time()-t0


def catalogues(t, box="wide"):
    od = OUT/f"T{t}"; m = dict(np.load(od/"truth_spray.npz")); sp = json.loads((od/"truth_spray.json").read_text())
    for c in (0, 1):
        f = od/f"mock_{box}_c{c}.npz"
        if f.exists():
            raise SystemExit(f"{f} exists; refusing to overwrite")
        cat = draw(m, np.random.default_rng([600000, t, c]), box=box)
        np.savez_compressed(f, **{k: v for k, v in cat.items() if isinstance(v, np.ndarray)})
        f.with_suffix(".json").write_text(json.dumps(dict(truth=t, params=TRUTHS[t], eps=EPS, h=H, hv=HV, chi_edges=E.tolist(),
                                                          truth_spray=sp, catalogue=c, sampling_seed=[600000, t, c],
                                                          n_bg=cat["n_bg"], n_outside_box=cat["n_outside_box"], box=box,
                                                          sha256=hashlib.sha256(f.read_bytes()).hexdigest()), indent=1))
        print(f"T{t} c{c}: background {cat['n_bg']}, outside the box {cat['n_outside_box']}", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=6); ap.add_argument("--truths", nargs="*", type=int, default=[1, 2, 3])
    ap.add_argument("--box", default="wide", choices=("wide", "data")); a = ap.parse_args()
    ap2 = a
    jobs = [(t, n, 500000+10*t+(5 if n == "check_spray" else 0)) for t in a.truths for n in ("truth_spray", "check_spray")
            if not (OUT/f"T{t}"/f"{n}.npz").exists()]
    if jobs:
        with Pool(min(a.workers, len(jobs)), maxtasksperchild=1) as pool:
            for t, n, s in pool.imap_unordered(spray_task, jobs):
                print(f"T{t} {n} done in {s:.0f} s", flush=True)
    for t in a.truths:
        catalogues(t, box=a.box)


if __name__ == "__main__":
    main()
