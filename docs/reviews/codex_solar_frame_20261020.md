# Codex review: self-consistent solar frame (2026-10-20)

Question sent: frames.py + mw_variants.py head + the analysis below. Claude-verified against the papers: GRAVITY 2021 R0 = 8275 +- 9 stat +- 33 sys pc (eq. in sec. 5); Oyama+2024 mu_l = -6.391 +- 0.005 mas/yr; Poder+2023 v_c = 233 +- 7; Zhou+2023 234.04 +- 0.08 +- 1.36; Bennett & Bovy 2019 z_sun = 20.8 +- 0.3 pc. Feng+2026 not checked.

## Question
Context: we fit the omega Cen tidal stream ("stream 54", phi1 -5..30 deg, d ~5.4 kpc) with test-particle sprays in a
rotating-bar Milky Way (Hunter et al.-type potential: bar + discs + halo; free parameters bar angle, pattern speed, bar
amplitude/size, omega Cen distance, v_c(R0) via halo normalisation, halo flattening q). Observables: sky track, Gaia PMs, v_los.
The solar frame converts observables <-> Galactocentric. Current frame (code below): R0 = 8.178 kpc, z_sun = 0,
Vsun = (11.1, 252.24, 7.25) km/s held FIXED while v_c(R0) of the potential is a free parameter (220-240 in fits; best fits pile
at the lower edge 220-223). 252.24 = 240 + 12.24 (Schoenrich+2010 V). Problem: (1) Vsun is not tied to v_c, so v_c + V_pec is
inconsistent whenever vc != 240; (2) Vsun/R0 implies mu_SgrA* = 252.24/(4.74047*8.178) = 6.507 mas/yr vs measured 6.411 +- 0.008
(Reid & Brunthaler 2020, 2020ApJ...892...39R).

My proposal: constrain V_tot = vc + V_pec by the Sgr A* proper motion: V_tot = 4.74047 * 6.411 * R0. With R0 = 8.178 (GRAVITY 2019,
2019A&A...625L..10G) -> 248.5 km/s; with R0 ~8.275 (GRAVITY 2021, 2021A&A...647A..59G; value from memory, please verify) -> 251.5.
Then let v_c vary in a literature band (Eilers+2019 2019ApJ...871..120E: 229.0 +- 0.2 stat, 2-5% sys, at R0 = 8.122) and V_pec = V_tot - v_c.
U_sun = 11.1, W_sun = 7.25 (Schoenrich+2010). Candidate band v_c 229-237.

Questions (be critical; verify numbers against the papers, give arXiv/DOI):
1. What is the current best R0 (GRAVITY 2021/2022, Do+2019 Keck) with errors? Which should we adopt and why?
2. Is fixing V_tot from mu_SgrA* and floating v_c (with V_pec = V_tot - v_c) the right self-consistent parametrisation? Or
   should we instead fix V_pec (Schoenrich 12.24, or other values, e.g. Bland-Hawthorn & Gerhard 2016) and let V_tot = v_c + V_pec
   float, putting a prior from mu_SgrA*? Which is more defensible for a stream fit? Note: in a barred, non-axisymmetric
   potential the LSR / v_c(R0) is ill-defined; V_pec relative to "v_c" depends on the bar. What do people do (e.g. Ibata+2019
   omega Cen/Fimbulthul, Bonaca/Koposov stream fits, Vasiliev+2021 Sgr, Pal 5 bar fits Pearson+2017, Bonaca+2020)?
3. Eilers+2019 v_c is at R0 = 8.122 and has slope -1.7 km/s/kpc; how to transfer to R0 = 8.178 or 8.275? Is Eilers the right
   reference, vs Zhou+2023 / Ou+2024 / Gaia DR3-based curves? Give a defensible v_c(R0) band.
4. z_sun: we use 0; is ~20.8 pc (Bennett & Bovy 2019) worth including for a stream at 5 kpc? Impact estimate.
5. The potential (bar + discs) was built for R0 = 8.178 (v_c is set at that radius). If R0 changes, what else must change
   consistently (bar angle definition relative to Sun-GC line, v_c normalisation radius)?
6. Recommend one concrete self-consistent frame (R0, z_sun, U,V,W or V_tot, v_c band) with justification, and list failure modes.

Code (src/ocen_dm/streams/frames.py):

## Answer
Research mode used: ordinary Codex web research fallback
Native Deep Research: unavailable from this invocation path
Scientific source rules: yes
Model effort: high   Date: 2026-10-20

---

## Executive summary

- **Use the Sun’s total Galactocentric velocity as a frame parameter, independently of the potential’s circular speed.** Holding \(V_{\rm tot}\) fixed while varying \(v_c\) is consistent; it changes the implied \(V_{\rm pec}\). The problem is interpreting \(252.24\) as \(v_c+12.24\) throughout the fit.
- **Adopt \(R_0\simeq8.277\) kpc as the fiducial distance.** GRAVITY’s 2021 optical-aberration correction superseded its 2019 value of 8.178 kpc. Its published 2024 multi-star fit remains at \(8.2759\pm0.0086\) kpc formally; retain the approximately 0.033 kpc systematic floor. Keck’s lower distance remains a substantive disagreement. [GRAVITY 2021](https://arxiv.org/pdf/2101.12098), [GRAVITY 2024, Table B.1](https://inspirehep.net/files/3a65506b699f644dc3636bef2f048b40).
- **Your Sgr A* calculation is correct:** Reid & Brunthaler’s measurement gives \(V_{\rm tot}=248.539\) km/s at 8.178 kpc and \(251.548\) km/s at 8.277 kpc. However, a 2024 VERA/VLBA analysis gives a slightly different proper motion, corresponding to \(250.763\) km/s at 8.277 kpc. Use an astrometric prior or sensitivity analysis rather than treating either central value as exact. [Reid & Brunthaler 2020](https://arxiv.org/pdf/2001.04386), [Oyama et al. 2024](https://arxiv.org/pdf/2401.02312).
- **A hard \(229\!-\!237\) km/s interval is narrower than the defensible external uncertainty.** I recommend a working prior \(v_{c,0}(R_0)\sim\mathcal N(235,7^2)\) km/s, with approximately \(220\!-\!250\) km/s available for exploration. This is a synthesis for this application, not a published combined measurement.
- **Include \(z_\odot=20.8\) pc.** Its effect is small but relevant to precise stream modelling. Changing both \(R_0\) and \(V_{\rm tot}\) to the recommended values changes your present \(V_y\) by only \(-0.69\) km/s, so the preference for \(220\!-\!223\) km/s cannot be attributed to the solar velocity without controlled refits.

## Detailed findings

### 1. Distance to the Galactic centre

The relevant measurements are:

| Analysis | \(R_0\), kpc | Uncertainty and assumptions |
|---|---:|---|
| GRAVITY 2019 | \(8.178\pm0.013_{\rm stat}\pm0.022_{\rm sys}\) | S2 orbit; formal statistical errors approximately \(1\sigma\). This predates the optical-aberration correction. [Paper](https://arxiv.org/pdf/1904.05721). |
| GRAVITY 2021 | \(8.275\pm0.009_{\rm stat}\pm0.033_{\rm sys}\) | Corrected GRAVITY astrometry; systematic contributions combined in quadrature. Your remembered value is correct. [Equation 38](https://arxiv.org/pdf/2101.12098). |
| GRAVITY 2022 | \(8.277\pm0.009_{\rm stat}\) | Joint S2/S29/S38/S55 fit; paper quotes systematics of approximately 0.030 kpc, referring to the 2021 analysis. [Paper](https://arxiv.org/pdf/2112.07478). |
| GRAVITY 2024 | \(8.2759\pm0.0086_{\rm formal}\) | Published four-star fit, with Schwarzschild-precession parameter fitted. This is not a new 8.6-pc total-accuracy claim. [Table B.1](https://inspirehep.net/files/3a65506b699f644dc3636bef2f048b40). |
| Do et al. 2019, GR fixed | \(7.946\pm0.050_{\rm stat}\pm0.032_{\rm sys}\) | Marginalized 68% interval; systematic term from reference-frame jackknife. [Paper and supplement](https://arxiv.org/pdf/1907.10731). |
| Do et al. 2019, redshift parameter free | \(7.959\pm0.059_{\rm stat}\pm0.032_{\rm sys}\) | Table 1 posterior estimate. The main text also quotes the maximum-likelihood distance, 7.971 kpc, with these errors. These explain several apparently conflicting quotations of this paper. [Table 1](https://arxiv.org/pdf/1907.10731). |

**Recommendation:** use **8.277 kpc**, with a working distance prior of approximately
\[
R_0\sim\mathcal N(8.277,\;0.034^2)\ {\rm kpc},
\]
conditional on accepting the GRAVITY calibration and identifying Sgr A* with the model centre. The 0.034 kpc width approximates the quadrature combination of the 2021 statistical and systematic terms; it is not an uncertainty obtained by averaging successive GRAVITY papers.

The reason to prefer this scale is the corrected interferometric astrometry and its stability in subsequent multi-star fits. The 2021 paper attributes the earlier GRAVITY shift to optical aberrations and investigates the Keck discrepancy: radial velocities and astrometric reference-frame differences can explain substantial portions, but the authors do not establish a complete, independently demonstrated resolution. [GRAVITY 2021, §5 and Appendix C](https://arxiv.org/pdf/2101.12098).

The Keck–GRAVITY difference is approximately **4–5 nominal standard deviations**, depending on which Do fit is compared and treating systematic terms as Gaussian. Thus, the above narrow prior does **not** cover all credible cross-instrument uncertainty. An external check from bar-star kinematics gives \(8.23\pm0.12\) kpc, with bootstrap uncertainty and a deliberately broad solar-angular-speed uncertainty; it supports the general 8.2–8.3 kpc scale without resolving the instrumental disagreement. [Leung et al., 2023 journal volume; online December 2022](https://doi.org/10.1093/mnras/stac3529).

I found no later verified distance result that materially supersedes this GRAVITY scale as of 2026-10-20.

### 2. Parametrisation: separate the observer from the gravitational field

Define the circular-speed parameter explicitly through the **azimuthally averaged potential**:
\[
\Phi_0(R,z)\equiv\langle\Phi(R,\phi,z,t)\rangle_\phi,\qquad
v_{c,0}^2(R)=R\,\partial_R\Phi_0(R,0).
\]

This quantity is well defined in a barred model. What is generally unavailable is a circular orbit, and hence a unique circular-orbit LSR, in the **full** non-axisymmetric potential.

The frame transformation requires the actual solar velocity:
\[
\boldsymbol v_\odot=(U_\odot,V_{\rm tot},W_\odot).
\]
It does not require a choice of \(v_{c,0}\). Consequently:

> **Fixing—or tightly constraining—\(V_{\rm tot}\), while independently varying \(v_{c,0}\), is a consistent parametrisation.**

You can define
\[
V_{\rm pec,0}\equiv V_{\rm tot}-v_{c,0}(R_0)
\]
as a derived quantity. Its interpretation is motion relative to your axisymmetric reference potential.

For a local stellar LSR, a more informative decomposition is
\[
V_{\rm tot}=v_{c,0}+v_{\rm stream}+V_{\rm pec,local}.
\]
Here \(v_{\rm stream}\) represents the local departure from axisymmetric circular rotation. Setting \(v_{\rm stream}=0\) and fixing \(V_{\rm pec,local}\) is an additional physical assumption.

Schönrich, Binney & Dehnen report
\[
(U,V,W)_\odot=
(11.1^{+0.69}_{-0.75},\,12.24\pm0.47,\,
7.25^{+0.37}_{-0.36})\ {\rm km\,s^{-1}},
\]
with additional estimated systematics approximately \((1,2,0.5)\) km/s. These are model-based local-kinematic estimates, not a requirement that every Galactic potential satisfy \(V_{\rm tot}-v_{c,0}=12.24\). [Schönrich et al. 2010](https://arxiv.org/pdf/0912.3693).

Bland-Hawthorn & Gerhard’s review adopts approximately \((10,11,7)\) km/s for the local peculiar motion, with quoted uncertainties \((1,2,0.5)\) km/s, while discussing potentially appreciable streaming relative to circular rotation. Their adopted \(V\) is **11**, not 24 km/s. [Bland-Hawthorn & Gerhard 2016, §5.3.3](https://arxiv.org/pdf/1602.07702).

**For this stream fit, I favour \((R_0,V_{\rm tot},v_{c,0})\)** with external frame priors. Fixing \(V_{\rm pec}=12.24\) and imposing Sgr A* is mathematically valid, but it imports a stronger local-disc assumption. At \(R_0=8.277\) kpc, those two constraints force
\[
v_{c,0}\simeq251.548-12.24=239.31\ {\rm km\,s^{-1}}.
\]
A fit advertised as freely measuring \(v_c\) would then be strongly constrained externally.

#### The Sgr A* constraint

Using the magnitude of the longitudinal component, not the total two-dimensional proper motion,
\[
\Omega_\odot\equiv V_{\rm tot}/R_0
\simeq4.74047\,|\mu_{l*}|
=30.3912\pm0.0379\ {\rm km\,s^{-1}\,kpc^{-1}}.
\]
The measured motion is negative; the solar rotation speed is positive. The conversion assumes negligible intrinsic longitudinal motion of Sgr A* and the usual nearly aligned geometry. The radio uncertainty is \(1\sigma\). [Reid & Brunthaler 2020](https://arxiv.org/pdf/2001.04386).

My calculated values are:

| \(R_0\), kpc | \(V_{\rm tot}\), km/s |
|---:|---:|
| 8.178 | 248.539 |
| 8.275 | 251.487 |
| 8.277 | 251.548 |

At fixed 8.277 kpc, the formal radio contribution is only 0.314 km/s. Including a 0.034 kpc distance uncertainty gives approximately **1.08 km/s**, assuming independent, approximately Gaussian errors.

A relevant update is Oyama et al.’s combined VERA/VLBA result,
\[
\mu_{l*}=-6.391\pm0.005\ {\rm mas\,yr^{-1}},
\]
with formal \(1\sigma\) uncertainty. It gives 250.763 km/s at 8.277 kpc, **0.785 km/s below** the Reid–Brunthaler value. It reuses earlier VLBA data, so the two measurements should not be combined as independent constraints. [Oyama et al. 2024, §5.2](https://arxiv.org/pdf/2401.02312).

I would retain Reid–Brunthaler for a clearly documented fiducial frame, propagate its uncertainty, and repeat the inference with Oyama’s value. Alternatively, an explicitly conservative **additional 1 km/s frame-systematic nuisance term** covers this difference; that is my proposed modelling allowance, not a published measurement.

#### What stream papers actually do

There is no single convention across the examples.

| Study | Verified treatment | Implication |
|---|---|---|
| Ibata et al. 2019, Fimbulthul | \(R_0=8.122\) kpc, \(z_\odot=17\) pc, circular speed 220 km/s, plus Schönrich peculiar motion. [Methods](https://arxiv.org/pdf/1902.09544). | Your quoted solar parameters reproduce their stated choice. This is a historical adopted frame, not a modern Sgr A* calibration. |
| Koposov, Rix & Hogg 2010, GD-1 | Link the LSR speed to the fitted potential; add fixed Dehnen–Binney peculiar motion, including \(V_{\rm pec}=5.25\) km/s. [§5](https://arxiv.org/pdf/0907.1085). | An example of the alternative parametrisation, under older assumptions. |
| Pearson et al. 2017, barred Pal 5 | Fixed solar position \(R_0=8.0\) kpc and total tangential velocity 244 km/s; model circular speed approximately 220 km/s. [§1.3](https://arxiv.org/pdf/1703.04627). | Fixed total observer velocity in a bar experiment. |
| Bonaca et al. 2020, Pal 5 | \(R_0=8.1\) kpc; circular speed 220 plus azimuthal offset 24, giving total 244 km/s. [§4.1](https://arxiv.org/pdf/1910.00592). | Again, the offset is not Schönrich’s 12.24. |
| Vasiliev, Belokurov & Erkal 2021, Sgr | Fixed solar position \((-8.1,0,0.02)\) kpc and velocity \((12.9,245.6,7.8)\) km/s; halo normalization chosen to give circular speed near 235 km/s. [§§2, 3.3](https://arxiv.org/pdf/2009.10726). | Separates total solar velocity from potential normalization. |

These precedents establish that your proposed separation is normal practice; their numerical choices should not be treated as independent measurements of today’s preferred frame.

### 3. Circular-speed evidence and transferring Eilers

For a **fixed published physical curve**, Eilers’ linear fit gives
\[
v_c(R)=229.0-1.7(R-8.122)\ {\rm km\,s^{-1}},
\]
hence **228.905 km/s at 8.178 kpc** and **228.740 km/s at 8.275 kpc**. The fitted normalization and slope have formal errors 0.2 km/s and 0.1 km/s/kpc; the paper estimates 2–5% systematic effects on speed. Those percentages are not a calibrated confidence interval. [Eilers et al. 2019](https://arxiv.org/pdf/1810.09466).

But changing the assumed solar frame in their **inference** is a different operation. Eilers used \(R_0=8.122\) kpc and \(V_{\rm tot}\simeq245.8\) km/s, inferred from Sgr A*, without requiring a peculiar-motion decomposition. Moving to \(V_{\rm tot}=251.55\) changes inferred stellar azimuthal velocities and Jeans terms. It cannot be represented by the \(-0.26\) km/s radial correction alone. [Eilers et al., §3](https://arxiv.org/pdf/1810.09466).

The later evidence is:

| Source | Relevant result | Qualification |
|---|---|---|
| Zhou et al. 2023 | \(234.04\pm0.08_{\rm stat}\pm1.36_{\rm sys}\) km/s at \(R_0=8.122\) kpc | Formal fit error and assessed systematic shifts; axisymmetric Jeans analysis, \(V_{\rm tot}=245.6\). The scalar error’s coverage is not separately specified in the result passage. [Paper](https://arxiv.org/pdf/2212.10393.pdf). |
| Ou et al. 2024 | At \(R=8.19\) kpc, \(232.83^{+0.56}_{-0.55}\) km/s | Table 1 bootstrap statistical errors; additional systematic effects are several percent. Uses \(R_0=8.178\), \(V_y=247.3\). [Paper](https://arxiv.org/pdf/2303.12838). |
| Põder et al. 2023, Gaia DR3 | \(233\pm7\) km/s, using the 8.277-kpc scale | Bayesian propagation of distance and tracer-profile nuisance parameters; curve errors use the 16th–84th posterior percentiles. [Paper](https://arxiv.org/pdf/2309.02895). |
| Feng et al. 2026 | \(236.8\pm0.8\) km/s at \(R_0=8.275\) kpc | From an averaged curve combining their Cepheids with previous work after frame adjustments. The quoted scalar error has no explicit coverage statement; it is not a comprehensive systematic bound. Published 7 January 2026. [Publisher](https://doi.org/10.1093/mnras/stag011). |

**Eilers remains a useful reference, but not a sufficient narrow prior.** The later studies are not independent enough to justify a simple inverse-variance average. They share Gaia information, sometimes tracer samples, and assumptions about equilibrium, axisymmetry, density gradients and distances.

For this application, my recommendation is:

\[
\boxed{v_{c,0}(R_0)\sim\mathcal N(235,7^2)\ {\rm km\,s^{-1}}.}
\]

The width is a **working prior standard deviation**, deliberately broader than formal fit errors. Approximately \(220\!-\!250\) km/s is a useful exploration interval, not a hard literature confidence interval. Your \(229\!-\!237\) interval is reasonable for a focused sensitivity grid, but too restrictive as the sole allowed range.

This caution is supported by Koop et al.: their Gaia DR3 analysis finds an approximately 8 km/s difference between curves inferred above and below the plane, and tests show that equilibrium, axisymmetric Jeans reconstruction becomes less robust at large radii. This does not invalidate the solar-radius measurements; it limits how literally to interpret tiny formal errors. [Koop et al. 2024](https://doi.org/10.1051/0004-6361/202450911).

### 4. Solar height

Bennett & Bovy obtain **\(z_\odot=20.8\pm0.3\) pc** after accounting for asymmetry in the local vertical density distribution. The quoted uncertainty comes from their MCMC analysis; the result passage does not explicitly define its coverage. It is a local-midplane measurement, not a universal 0.3-pc uncertainty in the plane of the entire Galaxy. [Bennett & Bovy 2019](https://arxiv.org/pdf/1809.03507).

**Include it**, while retaining the distinction between a local measured midplane and the symmetry plane of your idealized discs.

My geometric estimates are:

- The displacement scale at 5.4 kpc is \(20.8/5400=0.00385\) rad, or **0.221°**.
- Astropy also tilts the axes by approximately \(z_\odot/R_0=\mathbf{0.144^\circ}\).
- Near ω Cen, using approximately \(l=309.1^\circ,b=15^\circ,d=5.4\) kpc, that transformation changes Galactocentric \(z\) by about **12.5 pc**, and \(x\) by about **3.5 pc**, relative to \(z_\odot=0\).
- Rotating a 200 km/s velocity through 0.0025 rad produces a component change of order **0.5 km/s**.

The 0.221° figure is a scale estimate, **not a predicted stream-track displacement**. Transforming identical observables into a frame and back leaves them unchanged. The scientific effect comes from changing their position and velocity relative to the model’s disc plane and then evolving them. Astropy documents the height-dependent tilt explicitly. [Astropy frame definition](https://docs.astropy.org/en/stable/coordinates/galactocentric.html).

### 5. What changes when \(R_0\) changes?

**A new observer distance does not require rescaling every component of a potential expressed in physical units.** Distinguish changing the observer from rebuilding a Galactic mass model.

For your supplied formulation:

1. **Recompute the progenitor’s Galactocentric phase space from the same observables.** Use the same frame for projecting every stream realization.

2. **Specify the normalization radius.** If the fitted parameter means \(v_{c,0}\) at the new solar radius, impose
   \[
   c=\frac{v_{c,\rm target}^2-v_{c,\rm baryon}^2(R_{\rm new})}
            {v_{c,\rm halo,unit}^2(R_{\rm new})}.
   \]
   Alternatively, retaining normalization at 8.178 kpc is valid, but then the parameter is \(v_{c,0}(8.178)\), and literature comparisons require evaluating the model at the actual solar radius.

3. **Retain the definition of bar angle relative to the present Sun–GC line.** Moving the Sun radially does not itself change that angle. A reflection of the \(x\)-axis changes azimuth and rotation-sign conventions, so literature angles and pattern-speed signs require explicit translation.

4. **Disc lengths, bar size and masses need not automatically change.** If their adopted values were inferred from angular observables under the old distance, recalibrating those observational constraints is a separate modelling exercise. Likewise, a physical pattern speed does not automatically rescale because the observer moved.

5. **Keep the axisymmetric reference identifiable when varying the bar.** If the added perturbation has exactly zero azimuthal mean, its amplitude leaves \(v_{c,0}\) unchanged. If size/amplitude changes alter the mean mass distribution, include that change in the normalization.

Hunter et al. themselves define circular speed through the axisymmetric potential and distinguish it from the Sun’s total speed, discussing peculiar motion and streaming as contributions to the difference. Their synthetic-observer convention is not a solar-frame recommendation for your data. [Hunter et al. 2024, §§3.3–4.1](https://arxiv.org/html/2403.18000v2).

A small precision point: Astropy’s `galcen_distance` is the Sun–centre distance; the cylindrical solar radius is \(\sqrt{D_{\rm GC}^2-z_\odot^2}\). Their difference here is only **0.026 pc**, negligible for the potential normalization. [Astropy API](https://docs.astropy.org/en/stable/api/astropy.coordinates.Galactocentric.html).

### 6. Concrete recommendation

For a fiducial realization, I recommend:

| Quantity | Adopt |
|---|---:|
| \(R_0\) | **8.277 kpc** |
| \(z_\odot\) | **20.8 pc** |
| Solar Cartesian \(v_x\) in the supplied Astropy convention | **+11.1 km/s** |
| Solar total \(v_y\) | **251.548 km/s** |
| Solar Cartesian \(v_z\) | **+7.25 km/s** |
| Potential parameter | \(v_{c,0}\) of the azimuthally averaged potential at the solar radius |
| Working circular-speed prior | **\(\mathcal N(235,7^2)\) km/s** |
| Derived azimuthal offset | \(V_{\rm pec,0}=251.548-v_{c,0}\) |

At \(v_{c,0}=235\), the derived offset is **16.55 km/s**. Across \(229\!-\!237\), it is **22.55–14.55 km/s**. Those values should be labelled relative to the axisymmetric reference, without claiming they directly measure the local stellar peculiar motion.

For inference, propagate \(R_0\) and the Sgr A* angular-speed uncertainty jointly, preserving
\[
V_{\rm tot}=\Omega_\odot R_0.
\]
Retain conventional \(U,W\) centrally, but allow uncertainties of order 1–2 km/s in sensitivity analyses rather than treating them as exact.

**One qualification about \(W\):** this recommendation imposes the longitudinal Sgr A* constraint. It does not exactly impose its latitude motion. Reid–Brunthaler measure \(\mu_b=-0.219\pm0.007\) mas/yr; if Sgr A* is fixed vertically, that corresponds to a projected solar vertical speed near **8.6 km/s** at 8.277 kpc, with approximately 0.3 km/s uncertainty after including distance. Strict use of both radio components requires projecting the complete velocity vector through the adopted geometry and constraining \(W\) accordingly. Keeping 7.25 is a conventional choice whose approximately 1 km/s discrepancy should be included in the frame sensitivity analysis. [Reid & Brunthaler 2020](https://arxiv.org/pdf/2001.04386).

## Disagreements, uncertainties and failure modes

- **The old frame is observationally questionable, not algebraically impossible.** At \(v_c=220\), its \(V_{\rm tot}=252.24\) implies an offset of 32.24 km/s. Whether that is plausible depends on the reference LSR and streaming assumptions.
- **The lower-edge preference may survive the correction.** At fixed \(R_0=8.178\), correcting \(V_y\) changes it by \(-3.70\) km/s—up to 0.145 mas/yr at 5.4 kpc. Adopting both the new distance and new velocity changes \(V_y\) by only \(-0.69\) km/s—up to 0.027 mas/yr. These are fixed-phase-space projection sensitivities, not predictions for refitted streams.
- **Frame changes can be absorbed by distance and dynamics.** Solar reflex proper motion scales as \(V_\perp/d\); ω Cen distance, halo normalization, flattening and bar parameters can compensate each other. A boundary optimum is not a secure circular-speed measurement.
- **Do not double-count external information.** Rotation curves already depend on adopted solar frames. A tiny-error circular-speed prior plus Sgr A* and a fixed peculiar-motion prior can impose incompatible assumptions or reuse information.
- **The local force does not determine the inner orbit.** Changing only halo normalization may drive \(v_c\) downward to compensate for the adopted baryonic mass, bar structure or radial force profile. Positive halo normalization also imposes a baryonic floor.
- **Keep density flattening distinct from potential flattening.** Your \(q\) is a halo-density axis ratio; constraints quoted for \(q_\Phi\) cannot be applied directly.
- **The distance and radio systematics remain unresolved at their formal precision.** GRAVITY versus Keck is the larger issue; the two recent radio reductions differ by about 0.8 km/s in inferred \(V_{\rm tot}\).
- **A single short stream segment may not separate all these freedoms.** Stripping history and time-dependent bar dynamics can affect the inference. The decisive scientific check is a set of controlled refits separating the \(R_0\), \(V_{\rm tot}\) and \(z_\odot\) changes, followed by fits with frame uncertainties marginalized.

## Source list

All papers below are peer-reviewed journal publications. Linked arXiv copies may predate the final journal version. Bland-Hawthorn & Gerhard is a review.

| Paper | Publication and identifiers |
|---|---|
| GRAVITY Collaboration, geometric distance | **2019**, A&A 625 L10; [arXiv:1904.05721](https://arxiv.org/abs/1904.05721); [DOI:10.1051/0004-6361/201935656](https://doi.org/10.1051/0004-6361/201935656) |
| GRAVITY Collaboration, optical aberrations | **2021**, A&A 647 A59; [arXiv:2101.12098](https://arxiv.org/abs/2101.12098); [DOI:10.1051/0004-6361/202040208](https://doi.org/10.1051/0004-6361/202040208) |
| GRAVITY Collaboration, multiple stellar orbits | **2022**, A&A 657 L12; [arXiv:2112.07478](https://arxiv.org/abs/2112.07478); [DOI:10.1051/0004-6361/202142465](https://doi.org/10.1051/0004-6361/202142465) |
| GRAVITY Collaboration, extended-mass constraints | **2024**, A&A 692 A242; [arXiv:2409.12261](https://arxiv.org/abs/2409.12261); [DOI:10.1051/0004-6361/202452274](https://doi.org/10.1051/0004-6361/202452274) |
| Do et al., S0-2 relativistic redshift | **2019**, Science 365 664; [arXiv:1907.10731](https://arxiv.org/abs/1907.10731); [DOI:10.1126/science.aav8137](https://doi.org/10.1126/science.aav8137) |
| Leung et al., distance from bar-star kinematics | **2023** volume, online **2022**; MNRAS 519 948; [DOI:10.1093/mnras/stac3529](https://doi.org/10.1093/mnras/stac3529) |
| Reid & Brunthaler, Sgr A* proper motion III | **2020**, ApJ 892 39; [arXiv:2001.04386](https://arxiv.org/abs/2001.04386); [DOI:10.3847/1538-4357/ab76cd](https://doi.org/10.3847/1538-4357/ab76cd) |
| Oyama et al., VERA Sgr A* astrometry | **2024**, PASJ 76 163; [arXiv:2401.02312](https://arxiv.org/abs/2401.02312); [DOI:10.1093/pasj/psad088](https://doi.org/10.1093/pasj/psad088) |
| Schönrich, Binney & Dehnen, local kinematics | **2010**, MNRAS 403 1829; [arXiv:0912.3693](https://arxiv.org/abs/0912.3693); [DOI:10.1111/j.1365-2966.2010.16253.x](https://doi.org/10.1111/j.1365-2966.2010.16253.x) |
| Bland-Hawthorn & Gerhard, *The Galaxy in Context* | **2016**, ARA&A 54 529; [arXiv:1602.07702](https://arxiv.org/abs/1602.07702); [DOI:10.1146/annurev-astro-081915-023441](https://doi.org/10.1146/annurev-astro-081915-023441) |
| Eilers et al., circular velocity from 5–25 kpc | **2019**, ApJ 871 120; [arXiv:1810.09466](https://arxiv.org/abs/1810.09466); [DOI:10.3847/1538-4357/aaf648](https://doi.org/10.3847/1538-4357/aaf648) |
| Zhou et al., luminous-RGB rotation curve | **2023**, ApJ 946 73; [arXiv:2212.10393](https://arxiv.org/abs/2212.10393); [DOI:10.3847/1538-4357/acadd9](https://doi.org/10.3847/1538-4357/acadd9) |
| Ou et al., circular curve and dark-matter profile | **2024**, MNRAS 528 693; [arXiv:2303.12838](https://arxiv.org/abs/2303.12838); [DOI:10.1093/mnras/stae034](https://doi.org/10.1093/mnras/stae034) |
| Põder et al., Bayesian Gaia DR3 curve | **2023**, A&A 676 A134; [arXiv:2309.02895](https://arxiv.org/abs/2309.02895); [DOI:10.1051/0004-6361/202346474](https://doi.org/10.1051/0004-6361/202346474) |
| Feng et al., Gaia DR3 Cepheid rotation curve | **2026**, MNRAS 546 stag011; [DOI:10.1093/mnras/stag011](https://doi.org/10.1093/mnras/stag011) |
| Koop et al., robustness of Jeans rotation curves | **2024**, A&A 692 A50; [DOI:10.1051/0004-6361/202450911](https://doi.org/10.1051/0004-6361/202450911) |
| Bennett & Bovy, vertical waves and solar height | **2019** volume, online **2018**; MNRAS 482 1417; [arXiv:1809.03507](https://arxiv.org/abs/1809.03507); [DOI:10.1093/mnras/sty2813](https://doi.org/10.1093/mnras/sty2813) |
| Ibata et al., ω Cen/Fimbulthul | **2019**, Nature Astronomy 3 667; [arXiv:1902.09544](https://arxiv.org/abs/1902.09544); [DOI:10.1038/s41550-019-0751-x](https://doi.org/10.1038/s41550-019-0751-x) |
| Koposov, Rix & Hogg, GD-1 potential fit | **2010**, ApJ 712 260; [arXiv:0907.1085](https://arxiv.org/abs/0907.1085); [DOI:10.1088/0004-637X/712/1/260](https://doi.org/10.1088/0004-637X/712/1/260) |
| Pearson et al., Galactic bar and Pal 5 | **2017**, Nature Astronomy 1 633; [arXiv:1703.04627](https://arxiv.org/abs/1703.04627); [DOI:10.1038/s41550-017-0220-3](https://doi.org/10.1038/s41550-017-0220-3) |
| Bonaca et al., Pal 5 tail variations | **2020**, ApJ 889 70; [arXiv:1910.00592](https://arxiv.org/abs/1910.00592); [DOI:10.3847/1538-4357/ab5afe](https://doi.org/10.3847/1538-4357/ab5afe) |
| Vasiliev, Belokurov & Erkal, *Tango for three* | **2021**, MNRAS 501 2279; [arXiv:2009.10726](https://arxiv.org/abs/2009.10726); [DOI:10.1093/mnras/staa3673](https://doi.org/10.1093/mnras/staa3673) |
| Hunter et al., realistic Galactic potential | **2024**, A&A 692 A216; [arXiv:2403.18000](https://arxiv.org/abs/2403.18000); [DOI:10.1051/0004-6361/202450000](https://doi.org/10.1051/0004-6361/202450000) |