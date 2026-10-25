# DOP853 stragglers on the GPU: drop or hand off? Fable review (2026-10-25)

Test (bin/streams/gpu_straggler_test.py, 5 points, fixed per-slot banks 7/8/9): dropping particles unfinished after CAP attempts.
cap 800: 900-2100 dropped, along the stream (10-40 r_J), dchi2 -0.4..+1.3; cap 1600: 130-190 dropped, median 0.45-0.49 r_J today,
dchi2 0.00-0.03; cap 3200: ~30 inside 0.25 r_J, dchi2 0.00.

Fable (verified on the npz files; Claude re-checked item 1):
- Why ~0: omega Cen sits at (phi1, dphi2) = (0, -8.23 deg) in the chord stream frame (the ridge polynomial does not pass through the
  cluster); the sky window at phi1 = 0 is 0.03 +- 3 deg. No particle within 0.5 r_J today is inside any sky window or PM fit (Claude:
  572 such particles at c0, 0 inside the window). The cluster region is invisible to the score by construction.
- Cap-1600 set: 93-98% bound (E_rel < 0) and ~90% within ~1 r_J today; the dchi2 0.0008-0.027 comes from 2-7 unbound stream particles
  per point. Cap 2000: 0-1 such particles; 3200: 0.
- "Dropping the trapped particles" is a misdescription: ~1560-1590 particles (0.9%) are bound within 1 r_J today; the cap removes only
  the ~10% that made the deepest passages. Harmless only because the score cannot see the cluster region.
- Attempt count is a numerical criterion (scales with tolerance). A physical one is available at the cap time: drop r < r_J and
  E_rel < 0 w.r.t. the cluster, keep integrating the rest. Safeguard: log per model the number of capped particles with r > 2 r_J.
- Implementation: capped particles must be masked explicitly (out_y is initialised to zeros).
- Preferred: hand the stragglers to the CPU at the cap (no model change; ~0.3-2 s, saves 3-4.5 of the 5 s); measure on Colab's 2 vCPUs.
  If dropping is kept: cap 2000, with the r < r_J & E_rel < 0 test, and state that the cluster region is outside the score frame.
