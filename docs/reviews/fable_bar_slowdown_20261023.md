# Fable: bar slowdown and the omega Cen stream -- importance, rewinding lessons, strategy (2026-10-23)

Read-only review. Claude checked the arithmetic (phase lag c tau^2 with c = eta Omega^2 / 2: 12 / 25 / 48 deg at 350 / 500 / 700 Myr
for eta 0.003, Omega 33.1; A S^3 of the slowing bar 700 Myr ago 0.80; alias time pi / 12 t.u. = 256 Myr). The orbit/phase
numbers come from Fable's scratchpad AGAMA check (not in the repo).

## Findings
- eta ~ 0.003 (Mannu+2026 4.5 km/s/kpc/Gyr; Dillamore+2026) is an "Omega_b +-1"-class effect on the 700-Myr window:
  bar-phase lag vs a constant bar 10 deg (330 Myr), 30 (500), 40-50 (700); constant Omega +1.5 gives 15-30 / 50-80 deg.
  Orbit offset 0.19 kpc / 6 km/s at 350 Myr, 0.74 kpc / 45 km/s at 700 Myr.
- To first order degenerate with Omega and angle: Omega_fit = Omega_today + ~1.2, alpha_fit = alpha_today + ~10 deg. So a slowing
  bar pushes the equivalent present-day angle LOWER (strong-bar 12.5 -> ~2-5 deg today); it sharpens, not relieves, the angle tension.
- It cannot mimic the strong bar: with fixed amplitude and S ~ 1/Omega the past bar is weaker (A S^3 0.90 at 350 Myr, 0.80 at
  700 Myr); the demand is an rms outer torque (R 4-7 kpc), not a phase. Expected effect after re-optimisation: hundreds, not 2500.
- The 10-10 slowing test (old frame, fixed params, Omega 31-34 not bracketing) is not a verdict.
- No resonance crossing in the window (Omega 36 -> 33 over 700 Myr; 1:1 at ~24.4). Inference to check: the published-bar peaks at
  32-33 and 44-46 are an m = 2 alias pinned by the bar phase at the ~235-256-Myr pericentre.

## Lessons from metal_poor_disk
Rewinds are phase-sensitive; model (bar-history) uncertainty dominates data errors (island width 10x errors); mock closure must
include the mismatched history; coherence over (Omega, eta) is patchy -> scan Omega at <= 0.5 with CRN, never NM from one start;
forward modelling beats rewinding. The stream's memory of the bar ~ 2 circulation periods of the near-1:1 angle (360-380 Myr): it
measures the window-mean Omega and pericentre phases at 230-350 Myr; eta only weakly (through the oldest, worst-fitted debris).
Treat eta as a nuisance with prior 0.003 +- 0.001-0.0015; re-parametrise by the window-mean Omega_w = Omega_today + eta Omega^2 <tau>.

## Strategy (decision tree)
A. Fixed-parameter slowing-bar sprays (CRN, 32k): 4 bases (strong 12.5, 25; published 27/44.8, 22/33) x eta {0.001,0.002,0.003,0.005}
   x Omega_today offsets {-2.4..+0.6} + constant-Omega controls: ~100 sprays, ~20-35 min on 8 cores.
   Within +-150 -> nuisance; published gain > 300 -> go to C first; loss > 500 everywhere -> only B matters.
B. Mock closure (decisive for "strong bar = artefact of constant Omega"): truth = published bar 27 deg, eta 0.003, Omega_today 33
   (and 39); fit with the constant-Omega design (A, S free) + control on a constant published-bar truth in the present set-up;
   3 runs ~15 h. A S^3 >= 2.5 and gap >~ 1000 -> artefact; A S^3 <= 1.6 and gap <= 300 -> strong bar stands. Fable predicts 1.4-1.7.
C. Joint refit, A = S = 1, (Omega_w, eta) free with eta prior, angle 27 and 12.5: 2 x ~5 h. Gap < 500 -> strong-bar claim withdrawn;
   > 1500 -> strong bar required regardless of history.
D. If C is ambiguous: re-score at age cuts 500 / 700 (no new sprays).
Pitfalls: amplitude/length convention; 1-Myr time table (1.9 deg of bar per step; check once with a finer table); Omega_today vs
Omega_fit when comparing with the literature; PM prior matters (LMC degeneracy); differences below ~300 need 4 x 256k at matched seeds.
Against: A-C may only confirm a few-hundred nuisance; misfits common to every model will not be touched by eta.
