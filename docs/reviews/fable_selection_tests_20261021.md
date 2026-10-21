# Fable review: selection-aware score and a Gaia search for the 25-deg second branch (2026-10-21)

Claude verified: on-tube (|dphi2-med|<3 deg, |dPM|<2) / second-branch (pmra < member median - 3) debris fractions per
4-deg phi1 bin 6-22: 12.5 deg 0.56-0.70 / 0.12-0.18; 25 deg 0.39-0.47 / 0.28-0.38; bin/streams/m5_independent_selection.py
(ABCD background method) exists.
## Findings
- The 12.5-vs-25 gap (1071) is mostly debris off the members' tube, not track shape. First-order tube normalisation
  gives the 25-deg model back +805 (S = +-2 MAD), +684 (3), +595 (4), +398 (6 MAD); 60-70% from phi1 18-28.
- The branch occupies the same region in both models (amplitude differs): d 4.0-4.6 kpc (track 5.0-5.4), dphi2 -4..-9 deg
  south of the track (only 10-19% overlaps the members' sky), pmra 5-6 mas/yr more negative, pmdec within ~1, v_los 30-60
  km/s lower at phi1 18-27; 24% below the data box (dphi2 < -9.5).
- Member catalogue (Ibata+2024 MRT) has no membership probability or rejected candidates.
## Test 2 (ranked first): Gaia DR3 search
Region A per 4-deg phi1 bin from the model's branch PMs (e.g. phi1 6-10: pmra -14.3..-9.1, pmdec -10.2..-8.0; ... 18-22:
-17.1..-14.2, -11.6..-10.2), dphi2 -13..-1 excluding +-2.5 deg of the track, parallax soft cut (0.18-0.27 incl. ZP), members'
CMD locus shifted ~0.4 mag brighter (fit the shift), G 16-19.8, ruwe < 1.4, dereddened. One WSDB pull, selections in Python.
Background ABCD: PM sidebands -5 mas/yr, sky sidebands dphi2 -25..-13 and +4..+9 (avoid stream 45 at +10..+14), subtract
model leakage (report with/without). Statistic R = branch excess / track excess: predicted ~1.0 (25 deg) vs ~0.35 (12.5),
>= 6 sigma apart; pre-register 5-sigma excess, R, CMD shift + mean parallax (4.0-4.6 kpc), pmra gradient along phi1
(strongest discriminant vs field), Gaia RVS v_los at phi1 > 18. Clean bins phi1 6-14.
Outcomes: R ~ 1 -> branch real, current score unfair, angle open, score track+branch with a full Gaia selection;
R ~ 0.3 -> 12.5 stands with its own weak branch confirmed; no excess -> both models over-produce 250-500 Myr off-track
debris -> release history suspect.
## Test 1: selection-aware score (sensitivity bound, not a likelihood)
L_i = [(1-eps) f(w_i|u_i) + eps/|S(u_i)|] / [(1-eps) F_S(u_i) + eps], F_S = integral_S f(w|u); S empirical and fixed
before scoring: per 2-deg phi1 bin, +-k MAD around members' medians in (dphi2, pmra, pmdec), k = 2,3,4,6 as a family;
F_S as hard particle count first; floor F_S at 10 effective particles; no model-dependent S. Mock check on the 20 T2
catalogues truncated by the same recipe (maximum must stay at 27). Can soften the 12.5 preference, never establish 25.
## Against its own ranking
Test 2 rests on the models' absolute 250-500 Myr debris (release history never varied); a null means "release model
wrong", and the leakage correction re-introduces the model. Test 1 is cheaper (hours) and bounds the sensitivity now.
