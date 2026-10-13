The premise needs one correction: the frame swap changes **\(z_\odot\)** as well as \(V_{\odot,y}\) and \(R_0\): 0 to 17 pc, 252.24 to 232.24 km/s, and 8.178 to 8.122 kpc ([frames.py:16–18](src/ocen_dm/streams/frames.py:16)). The T1 script supplies distance explicitly, so the frame’s default cluster distance is overridden ([sun_dist_bar_grids.py:18–20](bin/streams/sun_dist_bar_grids.py:18), [grid4_common.py:40–49](bin/streams/grid4_common.py:40)).

### Direct answers

**(a) T1 is not a projection-only test.** `make_spray` uses one `frame` both to convert the cluster’s observed state into the model-frame initial condition and to convert the resulting debris back to observables ([grid4_common.py:40–50](bin/streams/grid4_common.py:40)). Thus the swap changes the inferred Galactocentric orbit and the projection. The host setup is unchanged for these runs, but the cluster’s initial state in that host is not ([grid4_common.py:46–49](bin/streams/grid4_common.py:46); [restricted.py:68–84](src/ocen_dm/streams/restricted.py:68)).

For a **fixed** Galactocentric trajectory at 4 kpc, a 20 km/s transverse observer-velocity shift corresponds to about 1.05 mas/yr, so projection alone is large enough in scale to matter. But T1 does not hold that trajectory fixed: its frame change also alters the initial condition. The packet reports that at \(d=5.6\) kpc the W–R16 gap changes from −782 to −93 in the two frames, with W’s gain concentrated in the outer segment and losses near the cluster; those figures are reported results, not independently recomputed here ([packet:2025–2034](question packet)). They do not isolate reflex from orbit changes.

**(b) The proposed decomposition is the right idea, but implement it as two independent frame choices.** Cross initial-condition frame with projection frame: Baumgardt IC/Baumgardt projection, Baumgardt IC/Ibata projection, Ibata IC/Baumgardt projection, Ibata IC/Ibata projection. Run at a fixed distance first; then repeat at the alternative distance if needed. The current API cannot express this: `make_spray` has only one `frame` argument ([grid4_common.py:40–50](bin/streams/grid4_common.py:40)). Save the model-frame particle states and reproject those same trajectories under both projection frames; this avoids duplicate integrations for the projection-only comparison.

A host \(v_c(R_0)\) sweep is a **separate** experiment, not a prerequisite for this decomposition. The packet reports the host at 228.8 km/s, while the Ibata frame’s stated solar velocity is constructed using 220 km/s plus 12.24 km/s peculiar motion ([packet:2020–2025](question packet), [frames.py:7–8, 16–18](src/ocen_dm/streams/frames.py:7)). That mismatch matters when interpreting a physically self-consistent solar-motion model. Test it separately by varying host normalization and solar motion together under a stated relation, while keeping the observed heliocentric cluster data fixed.

**(c) Cost-aware order:**

1. Run the IC × projection factorial at one distance, preferably 5.6 kpc, then at 5.43 kpc if the result depends on IC frame. Decision rule: if the score gain follows projection frame with either IC, reflex/projection dominates; if it follows IC with either projection, orbit changes dominate; if there is a large interaction, the effects are coupled.
2. Separate the remaining frame ingredients with one-at-a-time controls: \(V_{\odot,y}\), \(R_0\), and \(z_\odot\). Then run the host-\(v_c\)/solar-motion-consistency sweep. Decision rule: call the 15–20 km/s shift an orbit/potential effect only if it survives the projection-only control and moves coherently with the host or initial-condition change.
3. Extend the bar grid at its current boundaries: test \(\Omega_b<34\) at 16° and amplitude \(>1.4\) at 28°/size 1.3, while retaining common seeds and scoring. Both optima are reported at explored edges, so the current grid cannot establish either as an optimum ([packet:2036–2046](question packet)). Decision rule: require an interior preference and a gain larger than the reported roughly ±36 lnL spray noise ([packet:2046–2048](question packet)).
4. Try LMC forcing after these controls. It adds a potential/reflex change and therefore cannot cleanly explain the shift until the simpler frame and host effects are separated.

**(d) Code audit:** The frame round trip is internally paired: `to_model` and `observables` use matching sign flips for model \(x\) and \(v_x\) ([frames.py:31–39, 48–56](src/ocen_dm/streams/frames.py:31)). The relevant gap is test coverage: I found no direct test of these stream-frame transforms or their round trip; the existing frame test covers a different progenitor-orbit module ([test_progenitor_orbits.py:8–22](tests/test_progenitor_orbits.py:8)). `score_conditional` has tests for background-only cases, density cancellation, bin duplication, and degraded kinematics ([test_score_conditional.py:24–65](tests/test_score_conditional.py:24)), but the supplied tests do not establish its behavior for invalid covariance matrices, empty data, or measured radial velocities.

One data-safety issue in the T1 runner: it rewrites the accumulated JSON result file directly after each run, without an atomic replace; interruption during the write can leave a damaged checkpoint ([sun_dist_bar_grids.py:25–26, 41–42](bin/streams/sun_dist_bar_grids.py:25)). The per-run NPZ is written first, which provides some recovery, but the runner reads the JSON before resuming ([sun_dist_bar_grids.py:33–38](bin/streams/sun_dist_bar_grids.py:33)).

### What I verified and what I assumed

I verified the frame definitions, the shared IC/projection `frame` parameter, the host construction path, the score implementation and its tests, and the result-writing order by reading the cited code. The reported likelihoods, segment shifts, host circular speed, and spray-noise estimate come from the packet; I did not independently rerun the simulations or verify those outputs ([packet:2019–2048](question packet)).

I assume the score and the reported results are comparable across frames, the saved/recomputed spray trajectories use identical physical inputs and seeds, and the packet’s approximate reflex estimate refers to a transverse velocity difference at about 4 kpc.

### Strongest argument against this answer

The reported outer-stream improvement is large and has the sign expected from a changed solar reflex, while the host is held fixed. That makes projection a plausible major contributor. But because each T1 run also changes the cluster’s inferred initial condition—and changes \(R_0\) and \(z_\odot\)—the table cannot show that projection is the dominant cause.

### What I would check to be sure

- Implement and inspect the crossed IC/projection comparison, including the proposed distances.
- Verify the reported lnL and segment values from the saved NPZ files and result table.
- Add direct transform round-trip tests for both frames, and score tests with finite radial velocities and malformed covariances.
- Check that the JSON checkpoint survives interruption and can resume from the saved NPZ files.