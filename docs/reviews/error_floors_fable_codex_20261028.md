# S5-style error floors in the stream likelihood? Fable and Codex (2026-10-28)

Question: add per-observable nuisance floors (Koposov+2023, [arXiv:2211.04495](https://arxiv.org/abs/2211.04495), Eq. 13) to the
25-knot chi2 (chi2/dof ~ 8)? Brief: per-observable chi2 at the pilots' best fits (sky track 13/13, width 38/53, pmra 22/46,
pmdec 53/21 at angle 30/7), proposal = 4 additive floors marginalised per model (sampler stays 9-D).

## Agreed
- chi2 tempering (T ~ 10) is a sensitivity analysis, not a normalised error model; a global scale leaves the chi2 ranking unchanged.
- If floors are used: marginalise per model (cheap; logged m, e can be rescored), keep log|C|, proper priors; needs a fresh versioned run.
- The noise-free mock cannot validate floors. Before the main run: a noisy mock from C_data (coverage), and a mock with injected
  model error (another host / release prescription), scored with the production likelihood.
- Bootstrap e: small at interior knots (e/sigma_data 0.03-0.10), 0.26-0.35 at the last pmdec knots and last sky windows (Fable,
  computed); 8 replicates are noisy; some knots get e = 0 from a single surviving replicate (Codex; checked: 5-10 models per run).

## Disagreement
- **Codex:** provisionally adopt four additive floors with Gaussian normalisation (half-normal priors), compare with multiplicative
  scales; postpone the main run until the marginalised floor model is validated. Strongest counterargument (its own): four floors on
  25 summaries may conceal a wrong physical model and make the inference prior-dominated. Also: export the sky track-width covariance
  (computed in stream_sky_local.py, only marginal errors exported; checked); replace the 1e3 missing-knot penalty.
- **Fable (computed on 21,747 logged real-data models; reproduced):** do not add floors. With floors the best model moves from angle 30
  (chi2 126) to 20 (plain chi2 153) and the fitted pmdec floor is 6.3 sigma_data (0.28 mas/yr): pmdec is switched off. Angle 5-32
  becomes flat to < 6 and 35-50 only 6-20 above (vs 54-135 with chi2); multiplicative scales give the same. The misfit is structured:
  at the best fits at 7, 20 and 30 deg the same knots fail - pmdec first and last knots (2.6-5.8 sigma) and sky width in the last two
  windows (-3.3 to -5.1 sigma, model wider). Dropping the two end PM knots moves the best to 14 deg. Floors fitted by the misfitting
  knots silence them; this is data removal, not honest uncertainty. Recommendation: chi2 at beta 1 (+ beta 0.1), per-knot residuals,
  noisy mock, and investigate the four end knots as a data/model question first.
