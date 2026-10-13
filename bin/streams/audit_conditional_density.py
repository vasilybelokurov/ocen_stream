#!/usr/bin/env python3
"""Measure along-stream density dependence of the foreground/background mix.

Legacy L=[(1-eps) f(u,w)+eps g_u g_w]/[(1-eps) f_u+eps g_u].
Its conditional foreground weight is pi(u)=(1-eps)f_u/[(1-eps)f_u+eps g_u].
Compare to (1-eps)*[f(u,w)/f_u(u)] + eps*g_w, with the same kernels and
background constants. This isolates the mixing rule, not background support
or physical stream changes. The fixed-mixture alternative is a diagnostic;
it is not installed as the production likelihood, and can extrapolate where
the foreground is weak. Its limitations must be reported with the ranking.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bin/streams"))

import numpy as np

from grid4_common import score, du
from refine_bar_profiles import atomic_json


def evaluate(row):
    m = dict(np.load(ROOT / row["spray"]))
    s = score(m, return_components=True)
    f, fu, gu, gw = (s[k] for k in ("foreground_joint", "foreground_u", "background_u", "background_w"))
    eps = .05
    foreground = np.divide(f, fu, out=np.zeros_like(f), where=fu > 1e-300)
    fixed = np.where(fu > 1e-300, (1 - eps) * foreground + eps * gw, gw)
    new = np.log(fixed)
    pi = (1 - eps) * fu / ((1 - eps) * fu + eps * gu)
    out = dict(row, legacy_lnL=float(s["total"]), fixed_mix_lnL=float(new.sum()),
               difference=float(new.sum() - s["total"]),
               legacy_foreground_weight=np.percentile(pi, [5, 50, 95]).tolist(),
               segments=[dict(lo=lo, hi=hi, n=int(((du >= lo) & (du < hi)).sum()),
                              legacy=float(s["per_star"][(du >= lo) & (du < hi)].sum()),
                              fixed=float(new[(du >= lo) & (du < hi)].sum()),
                              mean_pi=float(pi[(du >= lo) & (du < hi)].mean()))
                         for lo, hi in ((-99, 4), (4, 17), (17, 99))])
    if "saved_lnL" in row:
        out["replay_error"] = float(s["total"] - row["saved_lnL"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="bar_profile_refinement_20261013")
    a = ap.parse_args()
    toy_u = np.linspace(.01, .99, 100)
    toy_p, toy_g = 1 / np.sqrt(2 * np.pi), 1 / 20
    legacy = lambda fu: ((1 - .05) * fu * toy_p + .05 * toy_g) / ((1 - .05) * fu + .05)
    constant = np.full(len(toy_u), .95 * toy_p + .05 * toy_g)
    toy = dict(legacy_max_per_star_log_change=float(np.max(abs(np.log(legacy(np.ones(100))) - np.log(legacy(2 * toy_u))))),
               fixed_mix_max_per_star_log_change=float(np.max(abs(np.log(constant) - np.log(.95 * (2 * toy_u * toy_p) / (2 * toy_u) + .05 * toy_g)))))
    groups = []
    for om, an, am in itertools.product((33, 34, 34.5, 35, 35.5, 36, 37), (16, 20, 24, 28), (1.2, 1.4)):
        tag = f"om{om:g}_an{an:g}_am{am:g}_d5.6"
        groups.append(dict(sample="grid4", angle=an, omega=om, amp=am, dist=5.6, seed=1, nrel=8000,
                           spray=f"results/streams/spray_grid4/{tag}.npz"))
    for f in sorted((ROOT / "results/plot_data" / a.label).glob("heldout_an*.json")):
        for r in json.loads(f.read_text()):
            groups.append(dict(sample="heldout", angle=r["angle"], candidate=r["candidate"], seed=r["seed"],
                               nrel=r["nrel"], parameters=r["parameters"], spray=r["spray"], saved_lnL=r["lnL"]))
    pilot = ROOT / "results/plot_data/spiral_pilot_20261013/pilot_an28_s4.json"
    if pilot.exists():
        p = json.loads(pilot.read_text())
        groups += [dict(sample="spiral_pilot", angle=28, candidate=r["name"], seed=4, nrel=8000,
                        parameters=p["parameters"], spray=r["spray"], saved_lnL=r["lnL"]) for r in p["results"]]
    rows = []
    path = ROOT / "results/plot_data" / a.label / "density_mixture_audit.json"
    payload = dict(toy=toy, definition=__doc__, rows=rows)
    for i, row in enumerate(groups):
        if not (ROOT / row["spray"]).exists():
            continue
        r = evaluate(row)
        rows.append(r)
        atomic_json(path, payload)
        if i % 10 == 0:
            print(f"{i+1}/{len(groups)}: {r['sample']} angle {r['angle']} legacy {r['legacy_lnL']:.1f} fixed {r['fixed_mix_lnL']:.1f}", flush=True)
    print("TOY", json.dumps(toy))
    for key in ("legacy_lnL", "fixed_mix_lnL"):
        grid = [r for r in rows if r["sample"] == "grid4"]
        best = max(r[key] for r in grid)
        print("GRID", key, {an: round(max(r[key] for r in grid if r["angle"] == an) - best, 1) for an in (16, 20, 24, 28)})
        held = [r for r in rows if r["sample"] == "heldout" and r["candidate"] == "refined"]
        print("HELDOUT", key, {an: [round(r[key], 1) for r in held if r["angle"] == an] for an in (16, 28)})
    print("MAX_REPLAY_ERROR", max(abs(r.get("replay_error", 0)) for r in rows))


if __name__ == "__main__":
    main()
