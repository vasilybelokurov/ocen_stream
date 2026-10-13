# Stream proper motions and the bar angle: renewed investigation

The task is to reproduce the remaining stream-54 proper-motion structure and
determine whether the apparent preference for a low bar angle survives joint
refitting. This note will record findings, including negative results.

## First experiment: complete the nuisance refinement

The previous broad six-dimensional search found comparable maxima at 16 and
28 degrees, but was not locally refined. Other angle comparisons fixed
distance, proper motions, or bar size. Those comparisons do not establish the
profiled angle preference.

`bin/streams/refine_bar_profiles.py` refines both angles with the original
conditional score. Pattern speed, amplitude, bar size, cluster distance, and
both cluster proper motions move together. Coordinates are scaled before
bounded COBYQA optimization. The initial candidates include the best saved
Sobol points and the best relevant grid models. Every evaluation and spray is
saved; the result file is replaced atomically. Existing output is preserved.

The distance measurement is 5.43 +/- 0.05 kpc, and each PM measurement has an
uncertainty of 0.027 mas/yr. These are separate penalty terms. Bar length is
bounded by corotation using the measured Hunter A2 half-maximum length of
4.25 kpc. The 5 kpc definition remains an option. Bounds and measurements are
recorded in the output. There is no new angular prior or error inflation.

The search uses 8,000 release epochs and seed 1. Candidate improvement will
be checked at 32,000 epochs with independent seeds 4, 5, and 6. Comparisons
are made only at equal particle count. This is a diagnostic refinement; an
optimization budget ending before convergence is reported as such, and
held-out seed checks do not substitute for mock calibration.

CPU limit approved by the user: two cores. Run each angle in a separate
single-thread process, with BLAS also limited to one thread. Expected first
refinement runtime: 15-30 minutes, to be updated from measured timings.

## Interpretation requirements

A better raw score is insufficient. Compare the actual sky and PM tracks,
the remaining PM residuals, the external distance/PM tension, the nuisance
boundaries, and scores from independent seeds. Keep the near-cluster and
distal segments visible. Do not claim that a prior forcing a normal bar
angle has explained the low-angle preference.

Possible further diagnostics include projecting PM residuals along and
across the measured sky track and testing whether the stars with radial
velocities can reach physically plausible escape locations in the barred
host. These are distinct from comparing a star orbit with the cluster orbit
using arbitrary position/velocity scales.

## PM geometry diagnostic

`bin/streams/diagnose_pm_profiles.py` predicts PMs with particles weighted
only by their agreement with each member's sky position. It retains the
existing chi-bin weights, and never uses an observed PM to select a model
particle. The diagnostic plot uses a declared two-degree tube around the
measured ridge; the fit still uses all members. Each bin records its member
count and model PM/distance percentiles. This avoids attributing a change
in broad distribution shape to a shift of the ridge.

The initial diagnostic, before the local fits finished, suggests two
different errors. Around phi1=11-17 degrees, the PM vector direction differs
by several degrees. At the distal knee, phi1=23-27, the vector direction is
much closer, but its magnitude is too small. A purely local distance
rescaling at fixed relative space velocity would remove most of that distal
PM residual. This is a decomposition of the error, not a measured distance
or a new freedom in the dynamical model: changing the actual cluster
distance also changes the orbit, and must be forward-modelled.

## Separate physical hypothesis: spiral arms

The stream host omits the spiral potential supplied with the local Hunter
model. The published model contains independently rotating bar and spiral
patterns. The reference spiral speed is 22.5 km/s/kpc and the reported test
range is 17.5-25. The original setup uses both components at phase -25
degrees today, in a frame with Sun at negative x.

Sources: [Hunter et al. 2024, Sections 2.1.6 and 2.2.2](https://arxiv.org/html/2403.18000v2),
and [the authors' AGAMA implementation](https://github.com/GalacticDynamics-Oxford/Agama/blob/master/py/example_mw_potential_hunter24.py).
The local exported potential and construction script are the numerical
inputs; they are not assumed to equal every parameter in the published text.

`src/ocen_dm/streams/spiral.py` reflects the exported CylSpline tables into
this project's x_model=-x_source frame by applying the exact real-Fourier
parity to potential and derivative coefficients. It then adds independent
positive pattern rotation. A sign change of pattern speed alone would
leave the spiral winding wrong. Original and transformed hashes are saved
under `results/streams/spiral_potential/`.

`bin/streams/diagnose_spiral_forcing.py check` verifies force/potential
reflection, zero-amplitude identity, force addition, and past/present phase.
The reflection force error is 9.5e-16 relative to the largest checked spiral
force. Phase/addition errors are below 4e-14. The source has a numerical
azimuthal-monopole remainder of 1.17e-7 of the spiral potential amplitude;
reflection preserves it to 8e-17. An initial absolute-zero threshold rejected
that source remainder; the check now verifies preservation of the source
monopole explicitly. No measured-data errors or likelihood widths changed.

The pilot is a paired on/off experiment at fixed cluster/bar parameters,
fixed present spiral phase 25 degrees, and a fresh spray seed. A source
snapshot is frozen before each pilot. A gain in one seed is only a lead:
check the PM tracks, independent seeds, and nuisance refitting before
interpreting it as an explanation for the angle preference.
