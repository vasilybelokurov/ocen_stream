# Codex review: direct per-angle profile by optimisation on sprays (2026-10-19)

Codex CLI, effort high, read-only.


**Proceed with direct profiles, but reject the premise that fixed seed and epoch offset make score differences “nearly noise-free.”** They preserve random draws; they do not remove discontinuities or simulation error.

1. **Validate CRN before committing the budget.** At fixed particle count, the epoch grid and age selection stay fixed as these parameters change; escape draws retain their identity. Chi-bin membership still changes discontinuously, and cutoff queries introduce additional boundaries. Thus determinism is plausible; smoothness and useful variance reduction remain unverified. Test paired parameter perturbations across several seed/offset pairs. ([grid4_common.py:58–69](<./bin/streams/grid4_common.py#L58>), [spray.py:55–61](<./src/ocen_dm/streams/spray.py#L55>), [spray.py:134–141](<./src/ocen_dm/streams/spray.py#L134>), [score.py:299–312](<./src/ocen_dm/streams/score.py#L299>), [score.py:379–402](<./src/ocen_dm/streams/score.py#L379>))

2. **Use bounded, scaled Nelder–Mead as the first baseline.** Start from two **distinct nuisance-parameter locations**, share CRN across both, and transfer successful solutions between neighbouring angles. Permit a third start when independent checks disagree. With 60–100 evaluations, CMA-ES at population ten gets only 6–10 generations; I would not expect a reliable eight-dimensional profile from that budget. Treat this as an empirical optimiser choice, not a convergence guarantee. Preserve the existing parameter transformation: physical bar size depends on other coordinates. ([emulator_design.py:33–40](<./bin/streams/emulator_design.py#L33>))

3. **Stop using the global GP to determine the reported profile.** One checked prediction is −21915, while its measured score is −23784—already enough to reject that prediction locally. A local GP may propose candidates, but require measured improvements before trusting it. Nearby off-angle runs are starting-point information, not observations of the fixed-angle objective. ([emax3_0.json:15–16,49](<./results/streams/search_T2c0_check3_20261019/runs/emax3_0.json#L15>))

4. **Budget adaptively.** A reasonable 400-spray allocation is approximately 20 diagnostics, 280 at 128k for search, 60 at 256k for promotion/restarts, and 40 final checks. Use 128k only after checking that candidate rankings survive promotion. Stop on exhausted budget or small search scale **plus** failure to improve on independent seeds; label unresolved angles unconverged. The proposed 5–6 hours excludes substantial validation work: a recorded 256k spray takes about 15 minutes. Keep fidelity explicit—the current observation loader discards `nrel`, seed and offset. ([w3_038.json:20,51](<./results/streams/search_T2c0_wave3_20261019/runs/w3_038.json#L20>), [search_wave.py:40–55](<./bin/streams/search_wave.py#L40>))

5. **Separate three uncertainties.** Report simulation standard error from independent seed/offset replicates; use paired differences when comparing angles. Report restart disagreement and further-search gains separately as unresolved optimisation error—not a symmetric confidence interval. Estimate statistical angle uncertainty from repeated mock catalogues and refitting. `var_jk` uses particle deletion with the catalogue fixed, so it cannot supply catalogue uncertainty or missed-basin error. ([score.py:377,418–422](<./src/ocen_dm/streams/score.py#L418>))

**Verified versus assumed**

I verified the mechanisms above and inspected all three waves and the check runs. Tests cover cutoff accuracy and jackknife equivalence; I did not run them. They do not establish optimiser convergence or profile coverage. ([test_score_cutoff.py:29–67](<./tests/test_score_cutoff.py#L29>))

The optimiser choice, allocation and stopping thresholds are recommendations. The −22174 benchmark remains an assumption: the plotting script hard-codes it. Acceptance at 27.5° against a 27° benchmark confounds optimisation error with angle displacement; include an exact-27° control and repeated independent truth evaluations. ([plot_search_T2.py:20](<./bin/streams/plot_search_T2.py#L20>), [mock_data.py:34](<./bin/streams/mock_data.py#L34>))

**Strongest argument against this answer**

Two local searches may repeatedly miss a narrow, curved basin. Direct evaluation removes surrogate prediction error; it does not establish that the nuisance maximum was found. Four hundred sprays may therefore produce a precise-looking underestimate of the profile.

**Checks needed to be sure**

- Paired CRN variance and step-size scans; repeatability under the intended worker configuration.
- 128k/256k ranking agreement and residual bias at finalists.
- Exact-27° recovery, additional starts, and finer angle sampling.
- Coverage on untouched catalogues with new generating sprays.
- Before reusing batches, verify score settings and simulation identities: the search loader checks catalogue identity but does not validate each run’s fidelity or manifest against the batch. ([search_wave.py:40–55](<./bin/streams/search_wave.py#L40>))