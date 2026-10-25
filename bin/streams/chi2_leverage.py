#!/usr/bin/env python3
"""Why does chi2 of poor-fit models move by 5-15 under small numerical changes? Per-summary decomposition (2026-10-25).

For a pair of sprays of the same parameters that differ only numerically (JAX DOP853 1e-8 vs AGAMA, same bank, each own chi),
each summary's residual r_i = (data - model) / sigma_tot and shift d_i = (m_2 - m_1) / sigma_tot, and its exact chi2 change
(diagonal sky terms; PM terms through the full covariance, attributed per knot by the linearised 2 (C^-1 r)_i dm_i, which sums to
the PM block's change up to second order). Tests the leverage hypothesis dchi2_i ~ -2 r_i d_i: small summary shifts times large
residuals at poor fits.
Usage: PYTHONPATH=src python bin/streams/chi2_leverage.py -> results/plot_data/chi2_leverage.json
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
sys.argv = [sys.argv[0]]
import stream_model_compare as MC
from ocen_stream.jaxscore import make_bank

D = MC.data_vector(); O = MC.Obs(); S = ROOT/"results/gpu/spray_check"; res = {}
lab = [f"c{x:g}" for x in D["pk"]]+[f"s{x:g}" for x in D["pk"]]+[f"ra{x:g}" for x in D["pmra_k"]]+[f"de{x:g}" for x in D["pmdec_k"]]
k = len(D["pk"]); na = len(D["pmra_k"])
for name in ("c0_final", "start_a12p5", "start_a25", "start_obsbar"):
    b = make_bank(179200, MC.NBOOT, 7)
    M1, M2 = (MC.measure(O, D, str(S/f), None, bank=b) for f in (f"{name}_agama.npz", f"{name}_jax_dop853.npz"))
    c1, c2 = MC.chi2(D, M1)["total"], MC.chi2(D, M2)["total"]
    contrib, resid, shift = np.zeros(25), np.zeros(25), np.zeros(25)
    for q, e, sl in (("sky_c", "sky_ce", slice(0, k)), ("sky_s", "sky_se", slice(k, 2*k))):
        v1, v2 = D[e]**2+M1[e]**2, D[e]**2+M2[e]**2
        contrib[sl] = (D[q]-M2[q])**2/v2-(D[q]-M1[q])**2/v1; resid[sl] = (D[q]-M1[q])/np.sqrt(v1); shift[sl] = (M2[q]-M1[q])/np.sqrt(v1)
    for q, sl in (("pmra", slice(2*k, 2*k+na)), ("pmdec", slice(2*k+na, 25))):
        C = D[q+"_C"]+np.diag(M1[q+"_e"]**2); r = D[q]-M1[q]; dm = M2[q]-M1[q]; s = np.sqrt(np.diag(C))
        contrib[sl] = -2*np.linalg.solve(C, r)*dm; resid[sl] = r/s; shift[sl] = dm/s
    o = np.argsort(-np.abs(contrib))[:5]
    res[name] = dict(chi2_agama=c1, dchi2=c2-c1, sum_contrib=float(np.nansum(contrib)), top=[dict(summary=lab[i], resid_sigma=float(resid[i]),
                     shift_sigma=float(shift[i]), dchi2=float(contrib[i])) for i in o], median_abs_shift=float(np.nanmedian(np.abs(shift))),
                     max_abs_shift=float(np.nanmax(np.abs(shift))), rms_resid=float(np.sqrt(np.nanmean(resid**2))))
    r_ = res[name]; print(f"{name}: chi2 {c1:.1f}, dchi2 {r_['dchi2']:+.2f} (sum of terms {r_['sum_contrib']:+.2f}); rms residual {r_['rms_resid']:.1f} sigma; "
                          f"summary shifts median {r_['median_abs_shift']:.3f}, max {r_['max_abs_shift']:.3f} sigma", flush=True)
    for t in r_["top"]:
        print(f"    {t['summary']:>6}: residual {t['resid_sigma']:+6.2f} sigma, shift {t['shift_sigma']:+.3f} sigma -> dchi2 {t['dchi2']:+.2f}", flush=True)
(ROOT/"results/plot_data/chi2_leverage.json").write_text(json.dumps(res, indent=1))
