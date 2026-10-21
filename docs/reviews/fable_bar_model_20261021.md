# Fable review: why the 25-deg bar scatters its debris, and a targeted alternative-bar plan (2026-10-21)

Read-only; diagnostics bin/streams/diagnostics/bar_diag.py, bar_diag2.py (Claude re-ran bar_diag2: force ratios, branch
fractions and Jacobi energies reproduce).
## Mechanism (checked)
- Not chaos (shadow orbits e-fold > 300 Myr) and not a low-order resonance: omega Cen is retrograde (Lz -510), peri 1.5-2.2,
  apo 6.4-7.2 kpc, T_r 92 Myr; (Omega_b - Omega_phi)/Omega_r = 1.17 (25) / 1.13 (12.5); CR 6.29 / 6.79 kpc, OLR 10-11 kpc.
- Branch = the less-bound (higher Jacobi energy) leading part of the debris released ~300-350 Myr ago: 25 deg E_J -105,160 vs
  track -107,130; it separates deterministically through later pericentre torques (Pearson+2017-type differential torquing).
- The angles differ in the branch FRACTION (age 300-350, phi1 6-22: 60% vs 24%) and the released E_J width: the 25-deg model's
  331-Myr release pericentre lies on the bar major axis (deepest of the last Gyr, r 1.54 kpc); the 12.5-deg one at -45 deg.
  Inferred (untested): bar-aligned release at the deepest pericentre widens E_J and feeds the branch -> set by angle,
  Omega_b and the INNER bar.
## Suspect bar parametrisation (checked)
- |F_nonaxi|max/|F_axi| at R 1.5-7 kpc (z 0): published Hunter24/Portail17 (A = S = 1) 0.23-0.29 / 0.19 / 0.13 / 0.08 / 0.04
  (R <= 3 / 4 / 5 / 6 / 7); best 25 deg (A 1.38, S 1.24) 0.29-0.40 / 0.34 / 0.25 / 0.19 / 0.135; best 12.5 (A 1.37, S 1.33)
  0.27-0.37 / 0.36 / 0.27 / 0.21 / 0.16 -> 1.2-1.4x the published m>0 force inside 3 kpc but 2.3-3.5x at 5-7 kpc (omega Cen's
  apocentre). Quadrupole moment ~ A S^3 = 2.6 / 3.2 x published. S 1.14-1.33 puts the bar end at 5.7-6.6 kpc vs Wegg+2015
  half-length 5.0 +- 0.2 kpc (arXiv:1504.01401).
- The fits buy extra non-axisymmetric force at R > 4 kpc with bar parameters: first suspect for the common misfits (knee pmdec,
  Delta phi2 offsets, d at the band top, vc 223-225). The inflated OUTER bar is not what makes the branch.
- Available: Hunter24 spiral (2 arms, Omega_sp 22.5, amplitude ~ (R/8.18)^2), implemented in src/ocen_dm/streams/spiral.py
  (pilot 2026-10-13 at frozen 28-deg params: +277/+317). Portail17 m>0 force within 2-8% of Hunter's (interchangeable).
## Plan (targeted)
M-A "published bar + spiral arms": A = S = 1 fixed; angle, Omega_b free; spiral amplitude a_sp in [0, 1.5] and phase free
(pattern speed 22.5). M-B null: A = S = 1, no spiral. Pre-screen (<= 1.5 h, 1-2 cores): (a) can the spiral supply ~0.1 of
F_axi at 5-7 kpc; (b) four 32k sprays at the 12.5/25 best params with A=S=1, a_sp=1, phase 0/90; (c) cluster orbit in the
spiral host. Then profile_nm --design step3 at 12.5, 20, 27.5 deg (2 chains x 40 at 128k, 4 x 256k): ~4.5 h; M-B at 2 angles ~2 h.
Criteria: angle gap 12.5 vs 27.5 (< 300 -> angle not determined; > 1000 -> robust); knee pmdec residual < 0.3 mas/yr; Delta phi2
offsets halve; far-end density within 2x; branch R_pred at 25-27.5 <= 0.2 (Gaia-compatible) or ~0.5 (intrinsic); d and vc leave
the edges; M-A better than M-B by > 500.
Against: the branch mechanism (inner bar, release pericentre phase) is untouched by M-A; spiral phase/pattern speed degenerate
with (Omega_b, angle); pilot gain small vs the 1885 gap. Cheaper complementary check: one 32k spray at the 25-deg best with
Omega_b +-1.5 (release pericentre off the major axis) -> does the branch fraction respond?
