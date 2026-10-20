# Fable review: step-2 design (2026-10-20)

Read-only review of commit 508b80e (gravity21 frame, v_c prior, step2 design, profile_nm changes). Claude verified items 1, 2, 4
against the code before acting; items 2 and 4 fixed in the following commit (profile_nm --transfer; strict manifest check).

## Verified issues
1. Box: the real-data profile (profile_real_20261019) and run_real_chain.sh use --box data; the 32k frame tests used the wide
   box (~3.7 ln L per background star more volume). Wide-box numbers are not comparable with -20498; use --box data.
2. No family transfer across angles: starts() only takes runs within +-2.5 deg, so the 12.5-deg family is never tried at
   17.5-27.5; chains at 25 and 27.5 share starts. Fix: second chain = global best moved to the angle.
3. All step2 starts sit at the band edge (d unit 1.0) and vc/q edges; with STEP0 0.1 and budget 30 a chain that must climb
   ~7000 measures how far it got, not the profile.
4. Resume check compared only keys of the new manifest (optional keys vc_prior could be dropped silently). Cached
   evaluations are keyed by run id: do not change start directories between resumes.
5. W_sun = 7.25 (Schoenrich) while Sgr A* mu_b = -0.219 gives 4.74047 x 0.219 x 8.275 = 8.59 km/s; one 32k spray settles it.
6. Frame plumbing otherwise consistent (placement, projection, r0 in axi_variant and r_cr, segment_profile vc_prior).

## Design points
7. The v_c prior N(233, 7) is cosmetic under the untempered score (1.7 at 220 vs score gradients 1e2-1e3 per km/s);
   effectively a hard band 220-250. Same for the d prior inside the band.
8. R0 has angle-sized leverage: ~1000-1800 ln L per 1 sigma (33 pc) at fixed parameters, comparable with the angle contrast
   (640-910). Free R0 with prior N(8.275, 0.034) and V_tot tied, or at least a +-33 pc sensitivity; Oyama+2024 mu_l is a
   smaller version (~-600).
9. d, vc, q edges all point one way (less bound omega Cen orbit); banding d pushes the compensation into vc, q, A, S.
   Inner-potential terms (disc_scale, bar mass/eta) are the physical absorbers; test them before the full profile.

## Recommended order
A. Recovery test (~2-3 h, 8 cores): 32k NM, 4 angles (12.5, 17.5, 27.5, 32.5) x 2 chains (one from the other family), budget
   60-80, data box, gravity21, step2 band. Recovery to within ~500-1000 of the d=5.70 best -> band harmless, run the profile;
   deficit > ~2000 with edges -> misspecification, go to C.
B. 128k profile only after A(i): data box, budget >= 40, starts from A, 4 x 256k finals, segment_profile readout.
C. Inner-potential variants in the same campaign: disc_scale as a 10th coordinate (optionally bar_eta) in A-style runs at
   12.5 and 27.5; does any remove the Delta phi2 offsets and the d/vc/q edge pull? LMC after C.
D. One-spray checks (32k): W_sun 8.59; R0 +-33 pc; Oyama V_tot. They set the systematic floor for any angle contrast.
E. Fix items 1, 4 (and 2) before launching.
