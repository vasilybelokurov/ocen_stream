#!/usr/bin/env python3
"""PM knot estimator test (2026-10-28): is the end-knot misfit an estimator mismatch?

Data knots are values of natural cubic splines fitted to all stars at once (stream_spline_fit.py: mean m and log-width log s
splines with the data's AIC knots, truncated to |v| < W, plus background). Model knots (jaxscore.pm == stream_model_compare.pm)
are local fits: truncated Gaussian with a linear trend in phi1 within +-1.5 deg of each knot.
Here one model's particles (masks and selection exactly as stream_model_compare.Obs) are measured both ways:
  local  -- stream_model_compare.Obs.pm (the production estimator);
  spline -- the data's estimator without background: truncated-Gaussian ML with m and log s natural splines on the DATA's knots
            (stream_spline_fit.json), particles with phi1 in the data range [-2, 28].
Reported per knot: data, local, spline, (spline - local) and (data - local), (data - spline) in units of the data sigma; particle counts
within +-1.5 deg. If spline - local ~ data - local at the end knots, the misfit there is the estimator, not the physics.
Particles: regenerated with the C spray (same code as the CUDA kernel; tolerance JM.SPRAY_ACC) at the parameters of a run JSON.
Usage: PYTHONPATH=src python bin/streams/knot_estimator_test.py --params results/streams/fitknot_20261025/runs/final_c0_2.json [--threads 8]
"""
import argparse, json, os, sys, tempfile
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(); ap.add_argument("--params", required=True); ap.add_argument("--threads", type=int, default=8); A = ap.parse_args()
sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]; sys.argv = [sys.argv[0]]
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import minimize
from scipy.special import ndtr, log_ndtr
import stream_model_compare as SMC
from stream_spline_fit import basis
from ocen_stream import jaxspray as J, jaxmodel as JM, pilots as PL, cspray

I = ROOT/"results/gpu"


def particles_file(params, path):
    ht, inp = J.load_host(I/"host_tables.npz"), JM.load_inputs(I/"spray_inputs.npz")
    off = json.loads((I/"points.json").read_text())["_epoch_offset"]; trel, keep = JM.schedule(inp, off)
    tb, ic, t0, _ = JM.release(params, ht, inp, trel, keep)
    y, chi, n = cspray.integrate(np.asarray(ic), np.asarray(t0), tb, JM.SPRAY_ACC, nthreads=A.threads)
    o = JM.observables(jnp.asarray(y), inp); age = (float(inp["T"])-np.asarray(t0))*JM.T_MYR
    np.savez(path, age_cut_myr=700., chi=np.asarray(chi)*JM.T_MYR, age=age, l=np.asarray(o["l"]), b=np.asarray(o["b"]), pmra=np.asarray(o["pmra"]),
             pmdec=np.asarray(o["pmdec"]), vlos=np.asarray(o["vlos"]), d=np.asarray(o["dist"]))
    return int(np.sum(n < 0))


def spline_fit(u, v, km, ks, W):
    """truncated-Gaussian ML: v ~ N(m(u), s(u)^2) on |v| < W, m and log s natural splines on knots km, ks."""
    Bm, Bs = basis(km, u), basis(ks, u)

    def nll(p):
        m = Bm@p[:len(km)]; ls = Bs@p[len(km):]; s = np.exp(ls); z = (v-m)/s
        lZ = np.log(np.maximum(ndtr((W-m)/s)-ndtr((-W-m)/s), 1e-300))
        return np.sum(0.5*z**2+ls+lZ)
    p0 = np.r_[np.full(len(km), np.median(v)), np.full(len(ks), np.log(np.std(v)))]
    r = minimize(nll, p0, method="L-BFGS-B", options=dict(maxiter=5000, maxfun=200000))
    r = minimize(nll, r.x, method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-6, fatol=1e-6)) if not r.success else r
    return r.x[:len(km)], r.x[len(km):], r


def main():
    z = json.loads(Path(A.params).read_text()); p = {k: float(z.get("params", z)[k]) for k in PL.NAMES}
    tmp = Path(tempfile.mkdtemp())/"model.npz"; nf = particles_file(p, tmp); print("params", {k: round(v, 3) for k, v in p.items()}, "failed particles", nf)
    D = SMC.data_vector(); O = SMC.Obs(); P = O.particles(str(tmp), np.random.default_rng(3)); n = P["u"].size
    W = {q: D[q+"_W"] for q in ("pmra", "pmdec")}; loc = O.pm(P, np.arange(n), D, W); F = SMC.J("stream_spline_fit"); out = {}
    for q, oth in (("pmra", "pmdec"), ("pmdec", "pmra")):
        vq = P[q]-O.ev(q, P["u"]); k = P["inwin"] & P["m_"+oth] & (np.abs(vq) < W[q]) & (P["u"] >= -2) & (P["u"] <= 28)
        km, ks = F[q]["fit"]["m"]["knots"], F[q]["fit"]["s"]["knots"]; m, ls, r = spline_fit(P["u"][k], vq[k], km, ks, W[q])
        d, sd = D[q], np.sqrt(np.diag(D[q+"_C"])); L = loc[q]; cnt = [int(np.sum(k & (np.abs(P["u"]-c) < 1.5))) for c in km]
        print(f"\n{q}: {k.sum()} particles, spline fit converged {r.success}; model width exp(log s) at knots {np.round(np.exp(np.interp(km, ks, ls)), 3).tolist()}")
        print(" knot |  data   local  spline | (spline-local)/sd (data-local)/sd (data-spline)/sd | particles +-1.5")
        for j, c in enumerate(km):
            print(f"{c:5.1f} | {d[j]:6.3f} {L[j]:7.3f} {m[j]:7.3f} | {(m[j]-L[j])/sd[j]:+16.2f} {(d[j]-L[j])/sd[j]:+16.2f} {(d[j]-m[j])/sd[j]:+16.2f} | {cnt[j]:6d}")
        rl = d-L; rs = d-m; C = D[q+"_C"]
        out[q] = dict(chi2_local=float(rl@np.linalg.solve(C, rl)), chi2_spline=float(rs@np.linalg.solve(C, rs)))
        print(f" chi2 vs data (data covariance only): local {out[q]['chi2_local']:.1f}, spline {out[q]['chi2_spline']:.1f}")
    os.remove(tmp)


if __name__ == "__main__":
    main()
