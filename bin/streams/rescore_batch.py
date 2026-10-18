#!/usr/bin/env python3
"""Re-score every stored emulator spray against a catalogue (real members or a mock), fast (2026-10-18).

Sprays: the emulator's runs -- training_9d_20261017 replicate 0 (256k), training_9d_reps_20261018, emulator_round1/2,
emulator_check (all 256k). Score: score_conditional with fixed chi bins 0-200, cutoff 7, one-pass jackknife G = 10
(matches the exact brute-force scores to < 1e-3; tests/test_score_cutoff.py), background box and u window of the real data.
Catalogue: 'real' (grid4_common member arrays) or a mock npz from mock_data.py (du, dw, dv, dev, cov). --box wide scores with
mock_data.WIDE_W/WIDE_V (the box the mock's background was drawn from); --box data uses the real data's box (mocks then
with outside_bounds='zero').
Output: OUTDIR/rescore.json = dict(manifest, rows): per run key, unit, purpose, seed, epoch_offset, nrel, params, spray path
and sha256, total, total_jk, var_jk. Manifest: catalogue sha256, settings, source hashes.
Usage: OMP_NUM_THREADS=1 PYTHONPATH=src python bin/streams/rescore_batch.py --catalogue real|PATH --outdir DIR --workers 8
"""
import argparse, hashlib, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
E = np.arange(0., 205., 5.)
CUTOFF, G = 7., 10
TRAIN = ROOT/"results/streams/training_9d_20261017"
BATCHES = ["results/streams/training_9d_reps_20261018", "results/streams/emulator_round1_20261018",
           "results/streams/emulator_round2_20261018", "results/streams/emulator_check_20261018"]
SOURCES = ["src/ocen_dm/streams/score.py", "bin/streams/grid4_common.py", "bin/streams/rescore_batch.py"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def runs():
    D = json.loads((TRAIN/"design.json").read_text())["points"]
    out = []
    for f in sorted(TRAIN.glob("runs/*_r0_n256k.json")):
        j = json.loads(f.read_text())
        out.append(dict(key=f"t{j['point']}", unit=D[j["point"]]["unit"], purpose="training", seed=j["seed"], epoch_offset=None,
                        nrel=j["nrel"], params=j["params"], spray=str(f.with_suffix(".npz").relative_to(ROOT))))
    for b in BATCHES:
        for f in sorted((ROOT/b/"runs").glob("*.json")):
            if f.name.endswith(".error.json"):
                continue
            j = json.loads(f.read_text())
            key, unit = (j["key"], j["unit"]) if "unit" in j and j["unit"] is not None else (f"t{j['point']}", D[j["point"]]["unit"])
            out.append(dict(key=key, unit=unit, purpose=j.get("purpose", ""), seed=j["seed"], epoch_offset=j.get("epoch_offset"),
                            nrel=j["nrel"], params=j["params"], spray=str(f.with_suffix(".npz").relative_to(ROOT))))
    return out


def catalogue(spec):
    if spec == "real":
        from grid4_common import du, dW, d, cov
        return dict(du=du, dw=dW, dv=d["v"], dev=d["e_v"], cov=cov), "real", "raise"
    c = dict(np.load(spec)); return dict(du=c["du"], dw=c["dw"], dv=c["dv"], dev=c["dev"], cov=c["cov"]), sha(spec), "zero"


def bounds(box):
    from grid4_common import BG_BOUNDS, V_BOUNDS
    from mock_data import WIDE_W, WIDE_V
    return (WIDE_W, WIDE_V) if box == "wide" else (BG_BOUNDS, V_BOUNDS)


def task(args):
    r, spec, box = args
    from grid4_common import win, project_gc, GC
    BG_BOUNDS, V_BOUNDS = bounds(box)
    from ocen_dm.streams.score import score_conditional
    c, _, ob = catalogue(spec)
    ob = "raise" if box == "wide" else ob
    m = dict(np.load(ROOT/r["spray"])); mu, mx = project_gc(m["l"], m["b"], GC)
    s = score_conditional(c["du"], c["dw"], c["dv"], c["dev"], c["cov"], mu, np.column_stack((mx, m["pmra"], m["pmdec"])), m["vlos"],
                          m["chi"], m["age"], u_window=win, w_bounds=BG_BOUNDS, v_bounds=V_BOUNDS, chi_edges=E, cutoff=CUTOFF,
                          jackknife_groups=G, chunk=50, outside_bounds=ob)
    return dict(r, spray_sha256=sha(ROOT/r["spray"]), total=s["total"], total_jk=s["total_jk"], var_jk=s["var_jk"], frac_bg=s["frac_bg"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--catalogue", required=True); ap.add_argument("--outdir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--box", default="data", choices=("data", "wide"), help="background box: the real data's, or the wide mock box")
    a = ap.parse_args()
    od = Path(a.outdir) if Path(a.outdir).is_absolute() else ROOT/a.outdir; od.mkdir(parents=True, exist_ok=True)
    if (od/"rescore.json").exists():
        raise SystemExit(f"{od/'rescore.json'} exists; refusing to overwrite")
    _, csha, ob = catalogue(a.catalogue); R = runs(); t0 = time.time()
    with Pool(a.workers, maxtasksperchild=50) as pool:
        rows = pool.map(task, [(r, a.catalogue, a.box) for r in R], chunksize=4)
    man = dict(catalogue=a.catalogue if a.catalogue == "real" else str(Path(a.catalogue).resolve().relative_to(ROOT.resolve()))
               if str(Path(a.catalogue).resolve()).startswith(str(ROOT.resolve())) else Path(a.catalogue).name,
               catalogue_sha256=csha, outside_bounds="raise" if a.box == "wide" else ob, box=a.box,
               bounds=[list(map(list, bounds(a.box)[0])), list(bounds(a.box)[1])], chi_edges=E.tolist(), cutoff=CUTOFF, jackknife_groups=G,
               sources={s: sha(ROOT/s) for s in SOURCES}, n_runs=len(rows), seconds=time.time()-t0)
    tmp = od/"rescore.json.tmp"; tmp.write_text(json.dumps(dict(manifest=man, rows=rows))); os.replace(tmp, od/"rescore.json")
    print(f"{len(rows)} runs re-scored in {time.time()-t0:.0f} s -> {od/'rescore.json'}")


if __name__ == "__main__":
    main()
