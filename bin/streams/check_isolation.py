#!/usr/bin/env python3
"""Isolation test of the restricted runner: does the model stay in equilibrium without a host?

Acceptance (same as the live N-body isolation tests, JOURNAL 2026-09-24): stellar spherically
averaged density within 5% of t = 0 at 0.3-30 pc (shells with >= 400 particles), bound stellar mass
change < 0.1%, stellar r_half drift < 2%.

Usage: python bin/streams/check_isolation.py results/streams/checks/A_nodm_iso [...]
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
import numpy as np

from ocen_dm.streams.analysis import load, run_particles, snapshot_times


def main():
    report = {}
    for run in sys.argv[1:]:
        mass, species = run_particles(run)
        files, times = snapshot_times(run)
        _, xv0 = load(run, 0.0)
        _, xv1 = load(run, times.max())
        edges = np.logspace(np.log10(0.3), np.log10(30.), 13)
        st = species == 0
        r0 = np.linalg.norm(xv0[st, :3], axis=1)*1e3
        r1 = np.linalg.norm(xv1[st, :3], axis=1)*1e3
        n0, _ = np.histogram(r0, edges); n1, _ = np.histogram(r1, edges)
        ok = n0 >= 400
        ratio = n1[ok]/n0[ok]
        rh0, rh1 = np.median(r0), np.median(r1)
        diag = [json.loads(l) for l in open(Path(run)/"diagnostics.jsonl")]
        bm0, bm1 = diag[0]["bound_mass"]["stars"], diag[-1]["bound_mass"]["stars"]
        res = dict(t_end=float(times.max()), density_ratio=ratio.round(4).tolist(),
                   max_density_dev=float(np.max(np.abs(ratio-1))), r_half_drift=float(rh1/rh0-1),
                   bound_star_change=float(bm1/bm0-1))
        res["pass"] = bool(res["max_density_dev"] < 0.05 and abs(res["r_half_drift"]) < 0.02 and abs(res["bound_star_change"]) < 1e-3)
        report[run] = res
        print(run, json.dumps(res))
    out = ROOT/"results/plot_data/streams_isolation_checks.json"
    out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
