## Recommendation

The premise is broadly correct for `score_chi_kde`: it makes a 4D KDE in \((l,b,\mu_{\alpha*},\mu_\delta)\) within each \(\chi\) bin, then averages the bin densities with equal weights. The sky coordinates are part of each KDE, so the likelihood retains the model’s density along the stream. The module header describes a different scorer, `score()`, and is stale for `score_chi_kde`. [score.py:1–11](<src/ocen_dm/streams/score.py:1>), [score.py:151–160](<src/ocen_dm/streams/score.py:151>)

Use the proposed **ratio of the joint mixture to its along-coordinate marginal**, with an explicit conditional background. If \(f_k(u,w)\) is the normalized KDE for \(\chi\)-bin \(k\), the conditional model is

\[
p_{\rm mod}(w\mid u)=
\frac{\frac1K\sum_k f_k(u,w)}
     {\frac1K\sum_k f_k^u(u)}
=\sum_k \underbrace{\frac{f_k^u(u)}{\sum_j f_j^u(u)}}_{\Pr(k\mid u)}
  \frac{f_k(u,w)}{f_k^u(u)}.
\]

That weighting is consequential: at a given \(u\), overlapping \(\chi\) bins contribute according to their model support there. It cancels the mixture’s marginal density in \(u\), but it **does not** make the mixture weights independent of the \(\chi\!\to u\) mapping. That is the right result if the intended prior over bins is equal before conditioning. If instead the desired branch proportions should reflect particle counts or physical debris density, equal bin weights are the wrong prior. The current scorer explicitly uses equal bin weights. [score.py:159–160](<src/ocen_dm/streams/score.py:159>), [score.py:195–202](<src/ocen_dm/streams/score.py:195>)

I would retain the \(\chi\)-binned mixture for now, but compare it against a pooled, \(u\)-local KDE. The mixture can keep distinct branches visible; pooling all particles in a \(\phi_1\) bin is simpler and can be less noisy, but can blur branches or let the densest branch dominate. The mixture’s weakness is sparse bins: each retained bin gets equal prior weight even if it barely meets `nmin`, and the current bandwidth depends on that bin’s particle count. [score.py:166–176](<src/ocen_dm/streams/score.py:166>)

Make the background conditional in \(w\). A simple fixed mixture is
\[
L_i=(1-\epsilon)p_{\rm mod}(w_i\mid u_i)+\epsilon p_{\rm bg}(w_i),
\]
where \(p_{\rm bg}(w)\) is normalized over the scored \(w\)-dimensions. But fixed \(\epsilon\) is not the conditional form of a joint model/background mixture. For that, define a joint background \(g(u,w)=g_u(u)g_w(w)\), and use
\[
L_i =
\frac{(1-\epsilon)f(u_i,w_i)+\epsilon g_u(u_i)g_w(w_i)}
     {(1-\epsilon)f_u(u_i)+\epsilon g_u(u_i)}.
\]
This keeps the ratio defined where model support is tiny and makes the background fraction rise where the model has little support. Choose \(g_u\) over the actual analysis interval or selection window. Do not rescue a tiny denominator with an arbitrary floor and then call the result a normalized conditional density.

A concrete API would be:

```python
score_chi_kde_conditional(
    data, model, u_data, u_model, w_data, w_model,
    dchi=5., chi_max=None, nmin=20, age_max=700.,
    bandwidths=..., eps=0.05, bg_u=..., bg_w=...,
)
```

Have `u` and `w` supplied explicitly so the path projection and scored dimensions are reviewable. Compute each bin’s joint density and \(u\)-marginal with a consistent KDE; evaluate the final ratio and background sum in log space. The existing scorer builds the KDE from fixed `l,b,pmra,pmdec` columns and adds Gaia PM covariance, but does not project through either path or condition on \(u\). [score.py:161–165](<src/ocen_dm/streams/score.py:161>), [score.py:178–195](<src/ocen_dm/streams/score.py:178>)

Keep the sum over members as the default. More members in a region normally mean more evidence there; normalizing each \(u\)-bin’s data contribution would change the target from the observed sample likelihood to an approximately equal-bin scoring rule. If the concern is correlated members, calibrate uncertainty by resampling along-stream blocks rather than assigning every star equal independent information. The code sums per-star log likelihoods. [score.py:197–202](<src/ocen_dm/streams/score.py:197>)

For the coordinate, I would start with chord-GC \(\phi_1\) for the primary published comparison and treat spline \(s\) as a sensitivity analysis. That is a pragmatic choice, not a conclusion that GC is better conditioned near the cluster. The spline projection is nearest-point based, and its \(s\) is arc length along the path; the GC projection uses a rotated longitude and polynomial-subtracted cross-track coordinate. Both can make off-path or ambiguous projections problematic. [path.py:67–77](<src/ocen_dm/streams/path.py:67>), [path.py:127–130](<src/ocen_dm/streams/path.py:127>)

The packet’s near-cluster width and extent summaries favor the spline’s narrower projection there, but they are not independently verified here. In that broad, lumpy region, either single-path coordinate may assign \(u\) arbitrarily; test the score with a predeclared near-cluster exclusion or a separate broad component. Also, the provided path-building scripts derive the spline from member positions and build the GC frame from that ridge. Using those data-derived paths to score the same members risks using the data twice; freeze the path from independent data or use cross-fitting. [build_stream_path.py:14–17](<bin/streams/build_stream_path.py:14>), [build_gc_frame.py:15–21](<bin/streams/build_gc_frame.py:15>)

Handle finite \(u\)-range edges explicitly: define the analysis interval, use a boundary-corrected KDE or exclude a bandwidth-sized edge region, and report the support/background fraction by \(u\). GC longitude is angular and needs wrap-safe handling at its discontinuity; spline endpoint extension does not guarantee that distant off-path points project sensibly. [path.py:43–64](<src/ocen_dm/streams/path.py:43>), [path.py:103–105](<src/ocen_dm/streams/path.py:103>)

For calibration, first freeze bandwidths across model comparisons; the current implementation estimates robust bandwidths from each bin’s particle count and spread, with floors and caps. Then compare a small bandwidth grid, repeat the best candidates with independent spray seeds as well as common random numbers, and use along-stream block resampling of members to measure score variability. Report score differences alongside the spread across those perturbations, rather than treating raw \(\Delta\ln L\) as decisive. [score.py:168–176](<src/ocen_dm/streams/score.py:168>)

One concrete edge case to cover before relying on the result: if no \(\chi\) bin survives filtering, `K=0`; the subsequent mean, `argmax`, and indexing operations do not provide a valid background-only result. Add explicit handling and tests. [score.py:166–180](<src/ocen_dm/streams/score.py:166>), [score.py:195–202](<src/ocen_dm/streams/score.py:195>)

## Verified in code vs assumed

Verified: `score_chi_kde` uses \(\chi\) bins, equal bin weights, a robust option with per-bin bandwidth estimation, Gaia PM covariance, and a per-star sum of log likelihoods. The path code provides the stated spline and GC projection functions. The path-building scripts use member data to build those paths. [score.py:151–202](<src/ocen_dm/streams/score.py:151>), [path.py:43–77](<src/ocen_dm/streams/path.py:43>), [path.py:127–144](<src/ocen_dm/streams/path.py:127>)

Assumed from the packet: the sample size, stream-membership context, 29 radial velocities, path width/extent summaries, and observed noise/likelihood roughness. I have not independently inspected their input data or reproduced those measurements.

## Strongest argument against this recommendation

Conditioning on an estimated \(u\) can discard real information about the model’s along-stream structure, including the very pile-ups or gaps that may distinguish dynamical models. Equal \(\chi\)-bin priors also impose a deliberate weighting that may not represent the physical stream. If the observed member selection function and particle sampling are understood well enough, a calibrated joint likelihood may be more informative than conditioning away \(u\).

## What I would check to be sure

- Validate both \(u\) projections on held-out members and model particles, especially near the cluster, at both ends, and wherever the path turns or approaches itself.
- Compare conditional and pooled-\(u\) KDEs on mock sprays with known branch structure and known \(u\)-density.
- Test support handling, zero surviving bins, coordinate wrapping, and the background’s normalization and units.
- Verify how STREAMFINDER selection and member dependence affect the effective sample size.
- Confirm whether `score_chi_kde` has dedicated tests; I did not find matching tests in the inspected test search.