#!/usr/bin/env python3
"""Which observable carries the frame gain? Per-observable conditional scores (obs_split.single_scores) for the W model at d 5.6: baumgardt vs ibata19 vs baumgardt with V_sun,y 232.24 vs the IC/projection
crosses. Sums per phi1 segment relative to baumgardt/baumgardt."""
import numpy as np
from grid4_common import ROOT, score, du
from obs_split import single_scores
P = ROOT/"results/streams"
F = {"B/B": P/"sun_dist_bar/T1_W_om36_an28_am1.2_sz1.15_baumgardt_d5.6.npz", "I/I": P/"sun_dist_bar/T1_W_om36_an28_am1.2_sz1.15_ibata19_d5.6.npz",
     "vy232": P/"frame_decomp_edges/C_om36_an28_am1.2_sz1.15_ic-b_vy232_proj-b_vy232.npz",
     "IC I/proj B": P/"frame_decomp_edges/F_om36_an28_am1.2_sz1.15_ic-ibata19_proj-baumgardt.npz",
     "IC B/proj I": P/"frame_decomp_edges/F_om36_an28_am1.2_sz1.15_ic-baumgardt_proj-ibata19.npz"}
SEG = [(-99, 4), (4, 17), (17, 99)]
S = {}
for k, f in F.items():
    m = dict(np.load(f)); S[k] = dict(all=score(m)["per_star"], **single_scores(m))
for k in F:
    if k == "B/B":
        continue
    print(k, " | ".join(f"{q}: " + " ".join(f"{np.nansum((S[k][q]-S['B/B'][q])[(du >= a) & (du < b)]):+6.0f}" for a, b in SEG) for q in ("all", "x", "pm", "v")))
