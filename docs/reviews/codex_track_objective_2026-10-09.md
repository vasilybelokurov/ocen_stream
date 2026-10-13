## Assessment

The code supports the central failure mode in the packet. The grid selects trailing particles with \(b>15\) and \(-75<l<-20\), then estimates a track in 2-degree \(b\) bins. The score drops model bins flagged as unreliable and sums chi-square only over the remaining bins; it does not penalise the masked bins. Thus a fold that makes a bin’s mixture fit unreliable can remove that bin’s contribution to the score. [spray_bar_grid.py:21-29](bin/streams/spray_bar_grid.py:21), [spray_bar_grid.py:43-48](bin/streams/spray_bar_grid.py:43), [track.py:15-16](src/ocen_dm/streams/track.py:15), [track.py:47-63](src/ocen_dm/streams/track.py:47)

There is one boundary to that claim: the track bins end at \(b=43\), and particles outside the longitude cut are discarded before fitting. Debris beyond those cuts cannot affect the current score through those bins. [track.py:15](src/ocen_dm/streams/track.py:15), [spray_bar_grid.py:44-45](bin/streams/spray_bar_grid.py:44)

## Recommended design

**(a) Extracting the model track.** I would not order particles by release age or progenitor-orbit arc length and treat that ordering as the stream coordinate. The code gives each particle a release age, but does not show that age is monotonic along the present-day stream; the progenitor orbit is also distinct from the paths of the released particles. [spray.py:58-63](src/ocen_dm/streams/spray.py:58), [spray.py:70-83](src/ocen_dm/streams/spray.py:70)

For this comparison, use a **local model distribution conditioned on sky position**, not a single globally ordered curve:

```python
model_conditionals(particles, sky_bandwidth, observable_bandwidths) -> LocalModel
```

At each observed star or track location \(s=(l,b)\), use nearby model particles to estimate the conditional mean and covariance of \(q=(\mathrm{pmra},\mathrm{pmdec},v_{\rm los},\log d)\). This can retain distinct kinematic branches where a fold overlaps the observed sky region. Keep release age attached as a diagnostic or for a separately justified young-debris cut; do not use it to force a branch assignment. A principal curve or nearest-neighbour chain could still be useful for plots, but branch crossings and particle noise make either a risky basis for the likelihood.

**(b) Data-model comparison and overshoot.** Prefer per-star comparisons in a local sky window over matching one summary point per \(b\) bin. The existing data product includes per-star \(v_{\rm los}\) values and errors for the selected stars, while the current track estimator fits only \(l\), `pmra`, and `pmdec` by \(b\) bin. [measure_track.py:25-30](bin/streams/measure_track.py:25), [measure_track.py:36-39](bin/streams/measure_track.py:36), [track.py:41-45](src/ocen_dm/streams/track.py:41)

A concrete initial objective is a local conditional likelihood with a broad background component:

```python
score_stars(data_stars, local_model, selection, weights) -> Score
```

\[
\log L_{\rm data} =
\sum_i \log\left[
  \sum_{j\in \mathcal N(s_i)}
  w_j\,K_h(s_i-s_j)\,
  \mathcal N(q_i\mid q_j,\ C_i+H)
  + \epsilon\,p_{\rm bg}(q_i)
\right],
\]

where \(C_i\) contains the star’s measurement covariance and \(H\) is a predeclared intrinsic smoothing covariance. Normalize or scale coordinates before choosing bandwidths; the observables have different units. Do not treat every spray particle as a measured star: spray density and release sampling are not established here as a survey selection model. The current grid score also uses only three observables, so adding \(v_{\rm los}\) and distance requires explicit error and covariance handling, not just extra terms. [spray_bar_grid.py:21-29](bin/streams/spray_bar_grid.py:21), [frames.py:48-56](src/ocen_dm/streams/frames.py:48)

I would keep “debris where the data have none” as a **separate, initially diagnostic penalty**, not silently fold it into the per-star likelihood. For example,

\[
P_{\rm out} =
\frac{\sum_j w_j\,\mathbf 1[\mathrm{age}_j<600]\,
                   \mathbf 1[d(s_j,\mathrm{tube})>r]}
     {\sum_j w_j\,\mathbf 1[\mathrm{age}_j<600]}.
\]

A term \(\lambda P_{\rm out}\) is easy to compute, but it is not a calibrated likelihood without a defined observed footprint, member-selection function, and a defensible \(\lambda\). The data script applies a stream-membership and sky cut; it does not establish completeness or a detection probability outside that selected sample. [measure_track.py:19-29](bin/streams/measure_track.py:19)

**(c) Smoothness and compute.** Keep the same release draws across candidate models, increase particle count until rankings stabilise, and use kernel estimates rather than hard bins and fit-success masks. The grid already passes `seed=1` for each model, and `spray` uses that seed for release times and offsets; that is a useful common-random-number starting point. [spray_bar_grid.py:40-45](bin/streams/spray_bar_grid.py:40), [spray.py:64-75](src/ocen_dm/streams/spray.py:64) The current mask thresholds and finite-bin mixture fits introduce discrete changes in which bins contribute. [spray_bar_grid.py:26-29](bin/streams/spray_bar_grid.py:26), [track.py:33-38](src/ocen_dm/streams/track.py:33)

The grid filters to the trailing arm only **after** calling `spray`; the spray creates paired leading and trailing particles. So “release only trailing” could save compute, but it would require changing the release/integration path, not merely moving the existing filter. [spray.py:42-55](src/ocen_dm/streams/spray.py:42), [spray.py:74-83](src/ocen_dm/streams/spray.py:74), [spray_bar_grid.py:43-45](bin/streams/spray_bar_grid.py:43)

**(d) Cheapest validation.** First freeze a visual ranking of the existing 45 grid models using the same plot limits and blinded parameter labels, then compare it with the new score’s ranking and inspect disagreements. But the grid JSON written by this script stores binned tracks and counts, not particle phase space, so it cannot evaluate the proposed fold-aware objective by itself. [spray_bar_grid.py:47-48](bin/streams/spray_bar_grid.py:47) The cheapest meaningful check is to regenerate the 45 candidates with fixed shared random draws at a moderate particle count, compute the new score, then rerun only the best, worst, and most discordant cases at higher counts. Treat rank stability across those counts as a required check before using the objective for optimization.

Concrete signatures:

```python
extract_local_model(particles, sky_bandwidth, observable_bandwidths) -> LocalModel
score_stars(data_stars, local_model, selection_model=None) -> Score
overshoot_fraction(particles, observed_tube, age_max_myr, tube_radius) -> float
score_model(data_stars, particles, local_model, overshoot_weight=None) -> Score
```

## Verified in code vs assumed

**Verified:** the spray returns arm labels and release ages; the grid selects trailing particles, applies the stated sky cuts, fits by \(b\) bins, and omits masked bins from chi-square; the observed-data script saves per-star \(v_{\rm los}\) values and errors. [spray.py:58-63](src/ocen_dm/streams/spray.py:58), [spray.py:81-84](src/ocen_dm/streams/spray.py:81), [spray_bar_grid.py:21-29](bin/streams/spray_bar_grid.py:21), [spray_bar_grid.py:43-48](bin/streams/spray_bar_grid.py:43), [measure_track.py:36-39](bin/streams/measure_track.py:36)

**Assumed from the packet, not verified by reading these files:** the visual fold’s location and age, the 10–40 particle counts in knee bins, the small data-error values, and the claim that the best current grid models visibly overshoot. I did not inspect model plots or rerun the grid.

## Strongest argument against this design

The local conditional likelihood can reward a model for placing *some* particles near every observed star while ignoring how much debris it predicts elsewhere. The overshoot term is meant to address that, but without a selection function its weight and tube definition are subjective. A single curve matched to binned data is easier to explain and may be more stable at low particle counts; if the folds are well separated in sky position and the scientific target is only the main arm, the additional distribution model may be unnecessary complexity.

## What I would check to be sure

- Inspect particle-level outputs and plots for representative good, bad, and borderline grid models; confirm the reported fold and quantify branch occupancy.
- Establish what the stream-member catalogue selection permits us to call “no members,” especially beyond the observed track end.
- Test whether the local likelihood’s rankings persist as particle count and smoothing bandwidth change.
- Check whether the 29 \(v_{\rm los}\) stars overlap the spatial range used for the track fit and whether their errors support a useful constraint.
- Compare new-objective rankings with independent visual ratings, and inspect every disagreement before trusting a best-fit model.