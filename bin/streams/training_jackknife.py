#!/usr/bin/env python3
"""Jackknife bias correction of every training-set run (2026-10-18).

For each run in results/streams/training_9d_20261017/runs: the fixed-chi-bin score (chi 0-200 in 5, as training_set.py) of
the full spray and of the 10 leave-one-group-out sprays (group g = particles with index % 10 == g, interleaved in release
order; score_bias.jackknife_total convention). Stored per run (runs/<run_id>_jk.npz + <run_id>_jk.json):
  total (raw, must equal the stored chi200 total), total_jk = G l - (G - 1) mean_g l_(-g), leave-out totals,
  var_jk = (G - 1)/G sum_g (l_(-g) - mean)^2 (jackknife variance of the total: a single-run noise estimate, to be checked
  against the seed replicates), per-star raw and jackknife-corrected ln L (float64).
Provenance: sha256 of the spray file (must match the run record) and of the scoring sources. Resumable: a run is skipped
if its _jk.json exists with the same spray and source hashes.
Usage: OMP_NUM_THREADS=1 PYTHONPATH=src python bin/streams/training_jackknife.py --workers 8
"""
import argparse, hashlib, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
RUNS = ROOT/"results/streams/training_9d_20261017/runs"
E = np.arange(0., 205., 5.)
G = 10
SOURCES = ["src/ocen_dm/streams/score.py", "src/ocen_dm/streams/score_bias.py", "bin/streams/grid4_common.py",
           "bin/streams/training_jackknife.py"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def task(args):
    f, src = args
    from grid4_common import score
    rec = json.loads(Path(f).read_text()); rid = rec["run_id"]
    npz = RUNS/f"{rid}.npz"
    if sha(npz) != rec["spray_sha256"]:
        return rid, "spray hash mismatch"
    m = dict(np.load(npz)); n = len(m["chi"]); t0 = time.time()
    full = score(m, chi_edges=E, chunk=50)
    if abs(full["total"]-rec["scores"]["chi200"]["total"]) > 1e-6:
        return rid, f"raw total {full['total']} != stored {rec['scores']['chi200']['total']}"
    idx = np.arange(n); lo_tot, lo_per = [], []
    for g in range(G):
        keep = idx % G != g
        s = score({k: (v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n else v) for k, v in m.items()},
                  chi_edges=E, chunk=50)
        lo_tot.append(s["total"]); lo_per.append(s["per_star"])
    lo_tot = np.array(lo_tot); lo_per = np.array(lo_per)
    tot_jk = G*full["total"]-(G-1)*lo_tot.mean()
    var_jk = (G-1)/G*np.sum((lo_tot-lo_tot.mean())**2)
    per_jk = G*full["per_star"]-(G-1)*lo_per.mean(axis=0)
    tmp = RUNS/f"{rid}_jk_tmp.npz"; np.savez_compressed(tmp, per_star_raw=full["per_star"], per_star_jk=per_jk, leave_out_totals=lo_tot)
    os.replace(tmp, RUNS/f"{rid}_jk.npz")
    out = dict(run_id=rid, point=rec["point"], replicate=rec["replicate"], nrel=rec["nrel"], seed=rec["seed"], params=rec["params"],
               G=G, chi_edges=E.tolist(), total=float(full["total"]), total_jk=float(tot_jk), bias_jk=float(full["total"]-tot_jk),
               var_jk=float(var_jk), leave_out_totals=lo_tot.tolist(), spray_sha256=rec["spray_sha256"],
               jk_npz_sha256=sha(RUNS/f"{rid}_jk.npz"), sources=src, seconds=time.time()-t0)
    tmpj = RUNS/f"{rid}_jk.json.tmp"; tmpj.write_text(json.dumps(out, indent=1)); os.replace(tmpj, RUNS/f"{rid}_jk.json")
    return rid, None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
    src = {s: sha(ROOT/s) for s in SOURCES}
    todo = []
    for f in sorted(RUNS.glob("p*_r*_n*k.json")):
        jk = f.with_name(f.stem+"_jk.json")
        if jk.exists():
            j = json.loads(jk.read_text())
            if j["sources"] == src and j["spray_sha256"] == json.loads(f.read_text())["spray_sha256"]:
                continue
        todo.append(str(f))
    print(f"{len(todo)} runs to jackknife, {a.workers} workers", flush=True)
    with Pool(a.workers, maxtasksperchild=4) as pool:
        for k, (rid, err) in enumerate(pool.imap_unordered(task, [(f, src) for f in todo]), 1):
            print(f"{time.strftime('%H:%M:%S')} {k}/{len(todo)} {rid} {'ERROR '+err if err else 'ok'}", flush=True)


if __name__ == "__main__":
    main()
