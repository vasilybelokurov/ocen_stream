#!/usr/bin/env python3
"""D1 (docs/STREAM_DIAGNOSTIC_PLAN.md): what drives the bar-angle preference? Best grid4 model at each angle (16, 20, 24, 28 deg),
32000 release epochs, seed 1. Per-star scores:
  all: score_conditional (joint dphi2, PMs, v_los given phi1);
  x:   conditional of dphi2 alone; pm: conditional of the PM pair alone (kernel 0.2 mas/yr + star's Gaia covariance);
  v:   conditional of v_los alone (stars with v_los; kernel 5 km/s + error).
Each single-observable score uses the same particles, chi-bin weights, phi1 kernel (0.5 deg) and a 5% background mixture with
that observable's box (12 deg, 30x30 mas/yr, 400 km/s) as score_conditional. Sums per phi1 segment (< 4, 4-17, > 17 deg) are
reported relative to the 28-deg model. Figure: cumulative Delta lnL along phi1 per observable (plots/d1_segments.png).
Usage: OMP_NUM_THREADS=8 python bin/streams/d1_segments.py
"""
import json
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from grid4_common import ROOT, make_spray, score, project_gc, GC, du, dx, d, cov, win

MODELS = {16: ("seed_test/om34.5_an16_am1.4_d5.6_n32000_s1.npz", None), 20: ("seed_test/om34.5_an20_am1.2_d5.6_n32000_s1.npz", None),
          24: ("d1/om35_an24_am1.2_d5.6_n32000_s1.npz", (35., 24., 1.2, 5.6)), 28: ("spin_exag/om35.5_an28_am1.2_d5.6_f0.npz", None)}
SEG = [(-99, 4, "phi1 < 4"), (4, 17, "phi1 4-17"), (17, 99, "phi1 > 17")]
from obs_split import single_scores, EPS, HU, g_u, hasv

S = {}
for a, (path, setup) in MODELS.items():
    f = ROOT/"results/streams"/path
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(f, **make_spray(*setup, nrel=32000, seed=1))
    m = dict(np.load(f)); S[a] = dict(all=score(m)["per_star"], **single_scores(m)); print(f"angle {a}: done", flush=True)
out = {}
print("Delta lnL relative to the 28-deg model (positive = better than 28 deg)")
for q in ("all", "x", "pm", "v"):
    for a in (16, 20, 24):
        dl = S[a][q]-S[28][q]; row = {name: float(np.nansum(dl[(du >= lo) & (du < hi)])) for lo, hi, name in SEG}; row["total"] = float(np.nansum(dl))
        out[f"{q}_{a}"] = row
        print(f"{q:3s} {a} deg: " + "  ".join(f"{k} {v:+7.0f}" for k, v in row.items()))
print("members per segment:", {name: int(((du >= lo) & (du < hi)).sum()) for lo, hi, name in SEG}, " with v_los:",
      {name: int((hasv & (du >= lo) & (du < hi)).sum()) for lo, hi, name in SEG})
(ROOT/"results/plot_data/d1_segments.json").write_text(json.dumps(out, indent=1))
o = np.argsort(du); fig, ax = plt.subplots(1, 4, figsize=(24, 4.5))
for a_, q in zip(ax, ("all", "x", "pm", "v")):
    for a, c in zip((16, 20, 24), ("C3", "C1", "C2")):
        a_.plot(du[o], np.nancumsum((S[a][q]-S[28][q])[o]), color=c, label=f"{a} deg - 28 deg")
    a_.axhline(0, color="0.6", lw=0.8); a_.set_title({"all": "joint (likelihood)", "x": "dphi2 only", "pm": "PM pair only", "v": "v_los only"}[q])
    a_.set_xlabel(r"$\phi_1$ [deg]"); a_.set_ylabel(r"cumulative $\Delta\ln L$ vs 28 deg")
    for x0 in (4, 17): a_.axvline(x0, color="0.8", ls=":")
ax[0].legend(); fig.suptitle("D1: where along the stream and in which observable the low-angle preference arises (4x particles, seed 1)")
fig.tight_layout(); fig.savefig(ROOT/"plots/d1_segments.png", dpi=80); print("plots/d1_segments.png")
