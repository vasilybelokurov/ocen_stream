# CMD mask plan: Fable and Codex (2026-10-23)

User requirement: isochrone-based CMD masks broadened by photometric errors (vs magnitude) plus the cluster's intrinsic width from
error deconvolution. Plan (Claude): MIST ridge; per-star errors from WSDB flux_over_error; per-bin deconvolution with a field term;
mask |r| < k sqrt(s_int^2 + e^2).

Fable (checks on the cached cluster sample): MIST + 0.085 matches the MS but the cluster RGB is +0.045..0.055 redder and the SGB
>= 0.2 mag fainter -> correct the ridge on the cluster; the G0 ~19.6 "limit" is the raw G < 20 cut (A_G 0.31), so the faint MS
must be extrapolated; intrinsic width evaluated at the template-frame magnitude, photometric error at the star's own G; MS core
Gaussian (no resolvable bMS/rMS bimodality in Gaia; dBIC <= 0), mildly red-skewed (binaries); RGB narrow core + red tail and
20-30 per cent field at the cluster PM -> empirical band; no crowding gradient at 20-40'; HB excluded; k: S/sqrt(B) flat for
k 1.4-2 for a Gaussian on a flat background; default k = 2 and check S/N(k) on phi1 18-22. Against: the empirical density mask
already encodes most of this; the Neyman-Pearson optimum is a likelihood ratio P_stream/P_field.

Codex: fit unsmoothed measurements (the old template's smoothing adds ~0.06 mag in colour); sigma = 1.0857/flux_over_error, band
errors in quadrature; regularised mixture / extreme deconvolution with a contaminant term (single Gaussian as baseline); separate
segments or 2-D around the TO/SGB; reddening uncertainty correlated along (1.339, 2.740); population gradients make the annulus
mixture an assumption for the stream; evaluate intrinsic width at the cluster-equivalent magnitude; faint BP can be biased;
choose k by injections / completeness per magnitude and phase; existing-code issues: "95 per cent" mask is not 95 per cent
completeness, dm fit compares on a dm-dependent domain, dereddened cache lacks an ID-alignment check (now asserted), dm-relation
metadata vs execution mismatch.

Implemented (bin/streams/gaia_cmd_mask.py): segments MS (colour residual, MIST + quadratic offset), RGB (colour residual, MIST +
constant offset; RGB branch = steep post-SGB part of the isochrone), TO/SGB (empirical cluster ridge in colour bins joined to the
MS and RGB ridges; ellipse distance; colour width = MS intrinsic at 17.7, magnitude width calibrated to 95.4 per cent control-
subtracted completeness); per-star errors (cluster) / sky-median sigma(G) (stream stars without fetched errors); sigma_dm = 0.1.
