# Fable: STREAMFINDER selection imprints, Gaia DR3 re-selection, selection-aware modelling (2026-10-23)

Read-only plan. Paper facts from arXiv 1804.11339 (Malhan & Ibata 2018), 2012.05245 (Ibata+2021), 2311.17202 (Ibata+2024).
Claude re-checked the members' per-bin statistics (3681 members): counts per 2 deg from phi1 -2: 656, 685, 341, 120, 43, 19, 28,
49, 142, 243, 344, 329, 257, 188, 57; sd(dphi2) 2.85, 2.82, 2.61, 2.04, 0.82, 0.60, 0.60, 0.72, 1.09, 1.38, 1.43, 1.87, 2.16, 2.84, 3.59;
template width 50 pc at 5.4 kpc = 0.53 deg.

## STREAMFINDER selection (verified from the papers)
Per-star likelihood: 20-deg DB98 orbit template (Gaussian width 50 pc) x 12.5-Gyr PARSEC SSP at one of 7 [Fe/H] (-2.2..-0.7, sets
the distance) x parallax x PM x v_los vs an empirical per-tile GMM field; threshold ln L-hat > 19.8 (6 sigma); G0 < 20, A_V < 2;
GCs masked at 2 r_t; Ibata+2021: "for |b| < 20 the sample is highly incomplete and the selection is not remotely objective".
Members are orbit-, isochrone-, distance- and contrast-dependent; the threshold is non-linear in local density.

## Imprints on stream 54 and tests
1. Threshold non-linearity: in the phi1 6-14 gap sd(dphi2) collapses to 0.6-0.8 deg (~template width) vs 2-2.9 near the cluster
   and 1.1-2.2 at the knee -> only template-ridge stars survive; G and extinction do not explain it. Gap likely a real dip
   amplified by the threshold.
2. Orbit-template bias of the sky ridge at phi1 6-13 (where the dphi2 misfit lives).
3. Distance/parallax/CMD coupling: STREAMFINDER per-star distances 5.3 -> 3.2 kpc along the stream; member parallaxes and CMDs are
   shaped by them (explains the +0.063 mas near-cluster excess; knee distance from members unreliable).
4. Single-[Fe/H] isochrone vs omega Cen's spread; RGB nearly absent (29 stars at G < 15).
5. No members at b < 13.2; 1.6-deg cluster mask.
6. Gaia completeness/quality fractions vary with scanning; measure from the field (gaiaunlimited M10 map installed).
Not an artefact: the knee PM (M5 check, 2026-10-10).

## Re-selection design
WSDB gaia_dr3 footprint phi1 -10..35, |dphi2| < 15, b > 8, G < 20, 5-param+, parallax - 3 sigma < 0.4, broad PM box; ~15-30M rows.
(a) CMD window = KDE of omega Cen's own Gaia CMD shifted by a fitted distance ratio f(phi1); (b) spline tracks in PM, ridge,
width, parallax, iterated from wide windows; (c) per-bin mixture (stream Gaussian x PM x parallax; field = sideband KDE with a
dphi2 gradient) -> p_member, N(phi1), explicit S(phi1, phi2, G, colour, PM); (d) cross-match with the 3681 members.
Extra: Gaia RVS, XP [M/H] (G < 16), GALAH DR4, APOGEE DR17; DESI does not reach the field.

## Modelling with S
Weight spray particles by an isochrone/MF-sampled S; extended (Poisson) mixture likelihood in (phi1, dphi2, PMs[, v_los]) with
a stars-per-particle nuisance. Gains: far-end density, branch, gap, knee counts. Risks: escaper MF, release history, field model
at ~1 star/deg2 in the gap. Keep the conditional score as a cross-check.

## Staged plan
S0 (2-4 h, 1 core, existing 4.8M-row cache results/streams/branch_search_gaia.npz, phi1 5-23): fixed broad windows, per 1-deg bin
  Gaussian + quadratic in dphi2 -> excess, ridge, width. Excess >= 3x members at 6-13 -> gap is selection; ridge vs members' median
  tests the dphi2 misfit.
S1 (1 h) footprint query; S2 (1-2 d) selection + S + cross-match (recovery > 90% at 14-26, > 70% elsewhere, purity > 70%);
S3 (1 d) density profile, ridge, width, ZP-corrected statistical parallax, stream-excess CMD; S4 (2-4 d) forward-model S on stored
sprays (3 LF/MF variants, 2 release histories) before any refit.
Pitfalls: crowding/scanning at b 8-15; cluster mask and extra-tidal halo; parallax ZP (L21 per star); DR3 PM systematics ~0.02
mas/yr; CMD width from the cluster, not an isochrone; SFD only (Bayestar stops at dec -30); no circularity (do not initialise
tracks or CMD from members only).
Against: the bar signal sits in the knee kinematics (already shown not to be an artefact); the density adds information but is
degenerate with mass-loss history, escaper MF and the field model. Build the density likelihood only if S3 measures the profile
at >= 3 sigma per 2-deg bin and S4 shows LF/release sensitivity below the model differences.
