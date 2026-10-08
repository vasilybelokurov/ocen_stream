# How stream models are fitted to data: methods review (2026-10-09)

Full texts read (pdftotext of arXiv versions; StreaMAX code read at commit 9fd520e, not run). Extraction notes per paper in the
session scratchpad; the key points, with locations, are below.

| | Gibbons+2014 (arXiv:1406.2243) | Erkal+2019 (arXiv:1812.08192) | Koposov+2023 (arXiv:2211.04495) | Dillamore+2022 (arXiv:2205.13547) | Chemaly+2026 (arXiv:2601.15373; 2607.05510; StreaMAX) |
|---|---|---|---|---|---|
| Stream | Sgr | Orphan (RR Lyrae) | Orphan-Chenab (6D) | GD-1 | extragalactic (2D images) |
| Generator | mLCS: release at +-r_t (Eq. 2), equal time steps, isotropic sigma; host + moving Plummer progenitor (essential: without it stream half as long, Fig. 3) | mLCS (as G14); 1e7 Msun Plummer; strip Gaussian 100 Myr around each pericentre, 5000/pericentre; LMC + MW reflex | mLCS; progenitor mass linear to 0; 120,000 particles/pericentre; LMC + reflex + dyn. friction | AGAMA Lagrange-point stripping (Bowden+15): r_prog +- 1.2 r_t, sigma 0.5 km/s; Sgr with mass loss per pericentre + MW reflex | JAX spray (Fardal+15, or Chen+25 in code), uniform-in-time release, Plummer progenitor, float32 leapfrog 100 steps |
| Ordering / "unwrapping" | phase chi = sum (r - r_prog) since release (Eq. 3): sign separates arms; keep first apocentre per arm; trailing: drop debris stripped after last apocentre | none: select particles by phi1 window | none; knots where the model kinks are masked (Fig. 14) | none: bin in phi1; KDE copes with secondary tracks | C26: unwrap each particle's angle from its birth angle (fails when not monotonic); C26b: G14 ordering -- code: argsort(xhi) then np.unwrap of host-centred polar angle (utils.py:101-117) |
| Model summary | binned centroid; Gaussian fit to each apocentre (3 numbers) | per data star: straight-line fit to particles within +-2.5 deg in phi1 -> mean and intrinsic width | per spline knot: straight line to particles within +-5 deg -> mean and error of the mean; require sigma_sim < sigma_data/5 | per 10-deg bin: quadratic trend + KDE of residuals, per observable | per 10-20 deg bin: median radius, 16-84% half-width, count |
| Data | 3 scalars (apocentre distances, angle) | individual RR Lyrae: phi2, d, pmra, pmdec (no v_los) | spline tracks with knots (mixture models, Stan): phi2, DM, PMs, v_GSR | members: phi2, PMs; 66 v_los; distances only as a check | projected radius per angle bin |
| Likelihood | Gaussian on 3 numbers, obs errors only (Eq. 7) | per star: N(m_obs - m_sim; sigma_obs^2 + sigma_sim^2) (Eq. 3) | per knot, Eq. 13: sigma_data^2 + sigma_sim^2 + sigma_nuis^2 (5 fitted floors; chi2/dof ~ 2.5 without) | per star: Gaussian error x model KDE, integrated; density not used | per bin Gaussian (+ free sigma_sys in C26b); model noise only a gate (sigma_model^2 <= sigma_data^2/9) |
| Free parameters | 3 potential + 10 nuisance (6 orbit, M_sat, a_sat, sigma, t_back) | 5 progenitor + 5 halo (+ LMC 5) | 25 (halo, LMC, progenitor) | 5 progenitor only; potential and Sgr fixed per run | 13-14 (halo, progenitor, t_int, sigma_sys) |
| Optimiser / sampler | emcee | simplex from 100 prior draws, then emcee (~3e5 models) | emcee (~1.9e6 models) | dual annealing + Nelder-Mead (likelihood discontinuous) | dynesty (2000 live points), ~10 h/stream CPU |
| Validation | 3 N-body mocks | none | none of the full inference | mock recovery of Sgr models | mocks (population study) |

## Lessons for omega Cen

1. **Nobody fits a fold by unwrapping in general.** Tracks are built in an along-stream coordinate in which the observed
   segment is single-valued; regions where the MODEL folds are masked (K23) or handled by a KDE (D22). Explicit ordering
   (G14 chi; release time) is used to pick one arm/wrap. For us: the observed northern arm is single-valued in b; on the
   model side select the trailing arm by the sign of chi (or by release side) and, if needed, by release time, then bin in b.
   The knee itself must NOT be masked: it is the signal. Mask only where the model is multi-valued in b.
2. **Model summary per bin/knot from a local linear fit to particles** (E19, K23), with the model's error of the mean in
   the likelihood and enough particles that sigma_sim < sigma_data/5 (K23; C26b uses /3).
3. **Likelihood:** Gaussian per bin and observable, sigma_data^2 + sigma_sim^2. Extra error floors (K23 sigma_nuis, C26b
   sigma_sys) are ad hoc systematics -> need the user's approval before use (project rule).
4. **Progenitor gravity in the spray is essential** (G14 Fig. 3); omega Cen is massive (3-4e6 Msun) -- use our model's mass
   profile as the moving progenitor potential (already done in run_prescribed.py).
5. **Release history matters:** G14 uniform in time, E19/K23 concentrated at pericentres; our tracer runs show pericentre-
   driven escape. Prefer pericentre-weighted release (or Chen+25 calibration) and validate against the tracer runs.
6. **A good track fit does not mean an unperturbed stream; the progenitor's present-day phase space absorbs perturbations**
   (D22: best fit with arms swapped by Sgr; optimiser shifted distance 7.47 -> 7.27 kpc). Keep Gaia/catalogue priors on
   omega Cen's 6D point and report any pull.
7. **Weak perturbations are degenerate with none** (D22 mocks): the DM test needs mock calibration (E19 and K23 had none).
8. **Likelihoods from particles are discontinuous** (bins gain/lose particles): derivative-free optimisers with loose
   tolerance (D22), simplex from many prior draws (E19), common random numbers.
9. **StreaMAX is not usable as is for us:** the rotating bar exists in potentials.py:297-333 but generate_stream calls the
   host acceleration without t (integrants.py:20,65,114), no composite host, 2D projected radius only, float32. AGAMA (our
   pipeline) already handles the time-dependent barred host; reuse the StreaMAX ideas (G14 ordering, model-noise gate), not
   the code.
