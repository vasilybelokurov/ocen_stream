#!/usr/bin/env python3
"""Rescore stored spray models with the single-component knot score of stream_model_compare.py (2026-10-24; screening only).

Sets: nullfits, obsbar, fixedbar, profile_* (89.6k-179k particles; profile_s1 with 8k particles excluded), search_real_wave3.
Each model measured as in stream_model_compare.measure (same masks, Asimov sky fits, PM fits at the spline knots, NBOOT particle
bootstraps for the model errors); parameters from the run's JSON. Workers fork from one parent that holds the data (copy-on-write).
Usage: PYTHONPATH=src python bin/streams/stream_model_rescore.py [--workers 8] [--limit N] [--tag NAME]
       -> results/plot_data/stream_model_rescore{_tag}.csv (chi2 per observable + per-point model values/errors, v_* JSON columns)
"""
import csv, glob, json, os, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
NW = int(sys.argv[sys.argv.index("--workers")+1]) if "--workers" in sys.argv else 8
LIMIT = int(sys.argv[sys.argv.index("--limit")+1]) if "--limit" in sys.argv else None
TAG = ("_"+sys.argv[sys.argv.index("--tag")+1]) if "--tag" in sys.argv else ""
VEC = ("sky_c", "sky_s", "pmra", "pmdec", "sky_ce", "sky_se", "pmra_e", "pmdec_e")                    # per-point model values and errors
sys.argv = [sys.argv[0]]
import stream_model_compare as MC

SETS = ["nullfits_20261022", "obsbar_20261022", "fixedbar_20261021", "profile_real_20261019", "profile_T2c0_20261019", "profile_step2_20261020",
        "profile_omega37_20261021_a22.5", "profile_omega37_20261021_a25", "profile_omega37_20261021_a27.5", "search_real_wave3_20261019"]
PAR = ("angle", "omega", "amp", "size", "halo_q", "vc", "dist", "pmra", "pmdec")
G = {}


def work(f):
    t0 = time.time(); rng = np.random.default_rng(abs(hash(f)) % 2**32)
    try:
        M = MC.measure(G["O"], G["D"], f, rng); c = MC.chi2(G["D"], M)
    except Exception as e:                                                           # noqa: BLE001 -- record and continue
        return dict(file=f, error=repr(e)[:200])
    j = Path(f).with_suffix(".json"); p = json.loads(j.read_text()).get("params", {}) if j.exists() else {}
    return dict(file=str(Path(f).relative_to(ROOT)), set=Path(f).parts[-3], n_particles=M["n_particles"], **{k: p.get(k, np.nan) for k in PAR},
                **{"chi2_"+k: v for k, v in c.items()}, **{"v_"+k: json.dumps(np.round(M[k], 5).tolist()) for k in VEC}, seconds=round(time.time()-t0, 1))


def main():
    import multiprocessing as mp
    files = [f for s in SETS for f in sorted(glob.glob(str(ROOT/f"results/streams/{s}/runs/*.npz")))][:LIMIT]
    G["D"] = MC.data_vector(); G["O"] = MC.Obs()
    print(f"{len(files)} models, {NW} workers", flush=True)
    out = ROOT/f"results/plot_data/stream_model_rescore{TAG}.csv"; rows = []; t0 = time.time()
    with mp.get_context("fork").Pool(NW) as pool:
        for i, r in enumerate(pool.imap_unordered(work, files, chunksize=2)):
            rows.append(r)
            if (i+1) % 50 == 0 or i+1 == len(files):
                keys = sorted({k for x in rows for k in x}, key=lambda k: (k != "file", k))
                with open(out, "w", newline="") as fh:
                    w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rows)
                ok = [x for x in rows if "chi2_total" in x]; b = min(ok, key=lambda x: x["chi2_total"]) if ok else {}
                print(f"{i+1}/{len(files)} done, {time.time()-t0:.0f} s; errors {len(rows)-len(ok)}; best so far {b.get('file')} chi2 {b.get('chi2_total', np.nan):.1f}", flush=True)
    print(out.relative_to(ROOT))


if __name__ == "__main__":
    main()
