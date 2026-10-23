#!/usr/bin/env python3
"""Iterate the CMD mask (dm relation) and the sky + PM masks of the omega Cen stream (2026-10-23).

Protocol (docs/reviews/iteration_readiness_fable_codex_20261023.md):
  rel -> results/plot_data/gaia_cmd_mask.json dm_relation -> cmd_pass (content-hash cache) -> stream_masks.py (marked start)
      -> stream_dm_fit.py (frozen reference domain/grid, nulls +-6 mas/yr) -> anchored linear relation (errors x2) -> rel
Frozen: template, PM-control scaling, dm-fit domain / grid / MS-TO split (reference line), off-sky bands, bins, thresholds,
k = 2, g_bright = 17, error model, reddening, anchor dm(0) = 0 and the form (held constant beyond the last accepted bin).
Converged when, between successive iterations: |delta dm| < 0.5 x inflated sigma in every bin accepted in both, the accepted set
is unchanged, the relation slope moves < 0.5 x its error, PM tracks move < 0.05 mas/yr (phi1 -1..27), the sky track < 0.1 deg
(phi1 13..24), and the CMD-pass sample by < 2%. Reaching --nmax without that is reported as NOT converged.
Each iteration's mask and dm-fit outputs are copied to results/streams/iterate_<start>/iter<k>/; the original gaia_cmd_mask.json
is saved once as results/plot_data/gaia_cmd_mask_pre_iterate.json.
Usage: PYTHONPATH=src python bin/streams/stream_iterate.py --start current|anchored [--nmax 4]
"""
import json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
PD = ROOT/"results/plot_data"; CMJ = PD/"gaia_cmd_mask.json"
START = sys.argv[sys.argv.index("--start")+1] if "--start" in sys.argv else "current"
NMAX = int(sys.argv[sys.argv.index("--nmax")+1]) if "--nmax" in sys.argv else 4
ANCH0 = dict(form="anchored_linear", slope=-0.0266, phi1_min=0., phi1_max=24., note="Fable 2026-10-23 anchored template line (start)")


def run(script, *args):
    env = dict(os.environ, PYTHONPATH=str(ROOT/"src"), OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "2"))
    r = subprocess.run([sys.executable, str(ROOT/"bin/streams"/script), *args], cwd=ROOT, env=env, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"{script} failed:\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
    return r.stdout


def set_relation(rel):
    P = json.loads(CMJ.read_text()); P["dm_relation"] = rel; CMJ.write_text(json.dumps(P, indent=1))


def main():
    if not (PD/"gaia_cmd_mask_pre_iterate.json").exists():
        shutil.copy(CMJ, PD/"gaia_cmd_mask_pre_iterate.json")
    pre = json.loads((PD/"gaia_cmd_mask_pre_iterate.json").read_text())["dm_relation"]
    rel = dict(pre) if START == "current" else dict(ANCH0)
    out = ROOT/f"results/streams/iterate_{START}"; out.mkdir(parents=True, exist_ok=True); prev, log = None, []
    for it in range(NMAX):
        set_relation(rel); log_m = run("stream_masks.py"); ncmd = int(log_m.split(" stars in the CMD mask")[0].split()[-1])
        log_d = run("stream_dm_fit.py"); M = json.loads((PD/"stream_masks.json").read_text()); F = json.loads((PD/"stream_dm_fit.json").read_text())
        d = out/f"iter{it}"; d.mkdir(exist_ok=True)
        for f in ("stream_masks.json", "stream_dm_fit.json", "gaia_cmd_mask.json"):
            shutil.copy(PD/f, d/f)
        (d/"stream_masks.log").write_text(log_m); (d/"stream_dm_fit.log").write_text(log_d)
        G = np.array(M["grid"]); bins = {r["phi1"]: r for r in F["bins"]}; A = F["relation_anchored"]
        rec = dict(it=it, rel_in=rel, rel_out=A, ncmd=ncmd, bins=bins, tracks={q: np.array(M["tracks"][q]["c"]) for q in ("pmra", "pmdec", "dphi2")})
        line = (f"iter {it}: relation in {rel.get('slope', rel.get('coef'))}; CMD-pass {ncmd}; accepted {A['accepted']}; "
                f"dm " + " ".join(f"{c:g}:{b['all_dm']:+.2f}{'' if b['accepted'] else '(r)'}" for c, b in bins.items())
                + f"; relation out {A['slope']:+.4f} +- {A['slope_err']:.4f} to {A['phi1_max']:g} (chi2 {A['chi2']:.1f}/{A['dof']})")
        conv = None
        if prev is not None:
            both = [c for c in bins if bins[c]["accepted"] and prev["bins"][c]["accepted"]]
            sig = lambda b: 2.0*max(0.5*(b["all_hi"]-b["all_lo"]), 0.02)
            ddm = {c: abs(bins[c]["all_dm"]-prev["bins"][c]["all_dm"])/sig(bins[c]) for c in both}
            pmw, skw = (G >= -1) & (G <= 27), (G >= 13) & (G <= 24)
            dtr = {q: float(np.max(np.abs(rec["tracks"][q]-prev["tracks"][q])[pmw if q != "dphi2" else skw])) for q in rec["tracks"]}
            dn = abs(ncmd-prev["ncmd"])/prev["ncmd"]; dsl = abs(A["slope"]-prev["rel_out"]["slope"])/A["slope_err"]
            same = set(A["accepted"]) == set(prev["rel_out"]["accepted"])
            conv = (max(ddm.values(), default=0) < 0.5) and same and dsl < 0.5 and dtr["pmra"] < 0.05 and dtr["pmdec"] < 0.05 and dtr["dphi2"] < 0.1 and dn < 0.02
            line += (f"\n   vs iter {it-1}: max |ddm|/sigma {max(ddm.values(), default=0):.2f}; accepted set same {same}; slope change {dsl:.2f} sigma; "
                     f"tracks pmra {dtr['pmra']:.3f} pmdec {dtr['pmdec']:.3f} sky {dtr['dphi2']:.3f}; CMD-pass change {100*dn:.1f}% -> {'CONVERGED' if conv else 'not converged'}")
        print(line, flush=True); log.append(line)
        if conv:
            break
        prev = rec; rel = {k: A[k] for k in ("form", "slope", "phi1_min", "phi1_max")} | dict(note=f"iterate_{START} iter {it} output", slope_err=A["slope_err"])
    else:
        log.append(f"NOT converged within {NMAX} iterations")
        print(log[-1], flush=True)
    (out/"summary.txt").write_text("\n".join(log)+"\n")


if __name__ == "__main__":
    main()
