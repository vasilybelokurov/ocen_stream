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
