#!/usr/bin/env python3
"""Bias and noise of the fixed-chi-bin score on the training-set calibration runs (2026-10-17).

Points 14 and 20 of results/streams/training_9d_20261017 have 8 independent seeds at 256k and 8 at 128k release epochs.
For every run: raw total (chi bins 0-200), jackknife-corrected total (score_bias.jackknife_total, G = 10, interleaved), and
delta-method corrected total (score_bias.chi_bin_bias). Test: the corrected means at 128k and 256k should agree (the raw
ones differ by the shot-noise bias); seed sd at each resolution; jackknife bias estimate vs the observed 128k->256k shift.
Output: results/plot_data/calibration_bias_20261017.json
Usage: OMP_NUM_THREADS=1 PYTHONPATH=src python bin/streams/calibration_bias.py --workers 4
"""
import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
RUNS = ROOT/"results/streams/training_9d_20261017/runs"
OUT = ROOT/"results/plot_data/calibration_bias_20261017.json"
E = np.arange(0., 205., 5.)


def task(f):
    from grid4_common import score, du, dW, d, cov, win, BG_BOUNDS, V_BOUNDS, project_gc, GC
    from ocen_stream.score_bias import jackknife_total, chi_bin_bias
    rec = json.loads(Path(f).read_text()); m = dict(np.load(str(f).replace(".json", ".npz")))
    j = jackknife_total(lambda x: score(x, chi_edges=E, chunk=50), m)
    mu, mx = project_gc(m["l"], m["b"], GC)
    dm = chi_bin_bias(du, dW, d["v"], d["e_v"], cov, mu, np.column_stack((mx, m["pmra"], m["pmdec"])), m["vlos"], m["chi"], m["age"], E,
                      u_window=win, w_bounds=BG_BOUNDS, v_bounds=V_BOUNDS)
    assert abs(j["total"]-rec["scores"]["chi200"]["total"]) < 1e-6
    return dict(run_id=rec["run_id"], point=rec["point"], nrel=rec["nrel"], seed=rec["seed"], raw=j["total"],
                jackknife=j["total_corrected"], delta=dm["total_corrected"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
    design = json.loads((RUNS.parent/"design.json").read_text())
    files = sorted(f for f in RUNS.glob("p0[0-9][0-9]_r*_n*k.json") if json.loads(f.read_text())["point"] in (14, 20))
    with Pool(a.workers, maxtasksperchild=1) as pool:
        rows = pool.map(task, files)
    out = dict(rows=rows, summary=[])
    for p in (14, 20):
        for meth in ("raw", "jackknife", "delta"):
            s = {}
            for n in (128000, 256000):
                x = np.array([r[meth] for r in rows if r["point"] == p and r["nrel"] == n])
                s[n] = (x.mean(), x.std(ddof=1), len(x))
            shift = s[256000][0]-s[128000][0]; se = np.hypot(s[256000][1]/np.sqrt(s[256000][2]), s[128000][1]/np.sqrt(s[128000][2]))
            out["summary"].append(dict(point=p, method=meth, mean_128k=s[128000][0], sd_128k=s[128000][1], n_128k=s[128000][2],
                                       mean_256k=s[256000][0], sd_256k=s[256000][1], n_256k=s[256000][2], shift=shift, shift_se=se))
            print(f"point {p} {meth:9s}: 128k {s[128000][0]:9.1f} (sd {s[128000][1]:5.1f}, n {s[128000][2]})  256k {s[256000][0]:9.1f} "
                  f"(sd {s[256000][1]:5.1f}, n {s[256000][2]})  shift 256k-128k {shift:+6.1f} +- {se:4.1f}", flush=True)
    out["params"] = {p: design["points"][p]["params"] for p in (14, 20)}
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
