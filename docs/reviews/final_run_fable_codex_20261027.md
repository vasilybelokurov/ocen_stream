# Final GPU run design: Fable + Codex (2026-10-27)

Brief: scratchpad final_run_brief.md (smc.py, pilots, tests, Asimov mock). Fixes applied: 4230fc2 (Codex), 2cd5dc9 (Fable).

## Codex
- Reflection at the box faces is not symmetric for correlated DE steps; simultaneous updates with moving donors are not invariant
  (counterexample: variance 1.000 -> 0.886, 3 particles). Fixed: half-ensembles with frozen donors, out-of-box proposals rejected.
- Make beta = 0.1 an explicit stage saved after its moves; guard the box/config/sources on resume; interrupted-resume test. Fixed.
- One Asimov mock is a smoke test, not coverage (needs noisy realisations); N 300 to beta 1 useful; beta 0.1 cannot validate T = 1.
- Go to beta 1, T ~ 10 = sensitivity only; chi2-only target ("log|C| irrelevant" too strong: spread 1.4 ~ factor 2 in weights).
- Distance: Gaussian prior if 5.43 +- 0.05 is trusted (untempered), plus a widened flat-prior sensitivity run.
- Skip the automatic polish; extend only on mixing evidence.

## Fable (measured on CPU fakes, scratchpad smc_probe*.py)
- Bugs: polish ignored the prior; no guard when > N/2 particles are inf. Fixed.
- Costs: gauss 52k evaluations / 12 stages; curved valley 98k / 11; a valley with a flat floor (the real case) 96-105k; beta-1
  acceptance 0.05-0.06; 55-60% of the cost is spent before beta 0.1.
- Flat-floor valley at N 600: one end under-populated x3-10 (3 seeds; not jitter; not cured by 60 extra sweeps or the polish);
  N 1200 fixes it (ends 88/104 of 120 expected) at x2 cost. Local-donor DE rejected (fragments, log Z off by 4).
- Polish 40 x 1000: 36k evaluations, tau 60-90 steps, ~500 effective samples, does not fill a depleted end -> 40 x 400 as a drift
  diagnostic only.
- Mock: peaked (machinery test). At N 300 to beta 1 ~6.5 A100-h (est.). Pass criteria: best chi2 <= 0.27 + 2; truth inside the 95%
  1-D intervals of all 9 and |mean - truth| < 1 sd for >= 7; no T = 1 particles at angle > 25; 10-14 stages; beta-1 acceptance >= 0.1;
  0 inf; one kill + resume on the GPU reproducing the state bit for bit.
- Temperature: beta 1, report T = 1 and T = 10 (sensitivity). Distance: Gaussian prior for the main run (flat box piles at an
  arbitrary edge; post-hoc reweighting collapses the ESS).
- Plan: mock N 300 (~6.5 h), main N 600 + Gaussian distance prior + polish 400 as a diagnostic (~30 h); N 1200 or a second seed if the
  claim rests on the valley ends.

## Agree
beta 1 with T = 10 kept as a sensitivity contour; the Asimov mock is a machinery test; Gaussian distance prior preferred (Codex: if
trusted, plus a flat sensitivity run); polish not as an automatic posterior.

## Differ (user to decide)
1. Polish: Codex none by default; Fable 40 x 400 as a drift diagnostic (~3-4 A100-h).
2. Validation depth: Codex wants noisy-realisation coverage tests (many runs); Fable one N 300 Asimov mock with explicit pass criteria.
3. Population: Fable's flat-valley finding -> N 1200 (x2) or two seeds if the valley ends matter; Codex did not test this.
