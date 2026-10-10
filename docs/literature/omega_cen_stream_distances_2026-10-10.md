# Distances to omega Cen stream (Fimbulthul) stars: literature (2026-10-10)

Research agent read full texts (arXiv PDFs); key quotes spot-checked by me in the extracted texts (Ibata+2019 Fig. 3a caption
"offset by 0.7 magnitudes ... ~1.5 kpc closer than omega Cen"; Methods "mean distance to these stars of 4.1 kpc, as predicted by
the STREAMFINDER algorithm"; Fig. 1 <parallax> = 0.265 +- 0.008 mas; Ibata+2021 table note "column 9 provides the distance to the
star dSF estimated by the algorithm").

- No standard-candle or spectroscopic distance gradient exists along Fimbulthul. All along-stream distances come from STREAMFINDER:
  PARSEC SSP template (12.5 Gyr, fixed [Fe/H] per run) fitted to Gaia photometry; weak parallax term from Ibata+2021
  (https://arxiv.org/abs/2012.05245, Eq. 2). Authors note a distance-metallicity degeneracy and multiple distance solutions.
- Ibata+2019 Nature Astronomy (https://arxiv.org/abs/1902.09544): stream ~0.7 mag brighter than omega Cen in the CMD (Fig. 3a),
  ~4.1 kpc (STREAMFINDER mean); <plx> = 0.265 +- 0.008 mas (DR2, no contamination correction); omega Cen 5630 +- 100 pc (Braga+2018).
- Ibata, Malhan & Martin 2019 (https://arxiv.org/abs/1901.07566): Fimbulthul anchor-point distance 4.22 +- 0.01 kpc (orbit fit).
  Extinction SFD98 + Schlafly & Finkbeiner 2011.
- Ibata+2021: per-star dSF (DR2 run; local streamfinder_ibata2021_edr3.fits, Fimbulthul = Stream 16). Agent's reduction: dSF falls
  from ~5.5 kpc (l ~ -57) to ~3.0 kpc (l ~ -29); knee box (b 30-42, l -55..-40) median 4.1 kpc (16-84%: 3.6-4.7), median EDR3 plx 0.248 mas.
- Ibata+2024 (https://arxiv.org/abs/2311.17202; our catalogue): no distance column (G0, (BP-RP)0 dereddened); distances only as the
  Fig. 3 map; stream 55 "~3 kpc".
- Simpson+2020 GALAH (https://arxiv.org/abs/1911.01548): assumed (m-M)0 = 12.9 (3.8 kpc) for the Ibata sample vs 13.7 for omega Cen.
- Kuzma+2021 (https://arxiv.org/abs/2108.02531): 3 RR Lyrae within 5 deg: 5.2 +- 0.6 kpc. Kuzma+2026 (https://arxiv.org/abs/2605.23474):
  omega Cen dynamical 5.29 +- 0.2 kpc; no tail distances.
- Malhan+2022 (https://arxiv.org/abs/2202.07660): weighted-mean parallax (Lindegren zero-point), no gradient.
- Youakim+2023 quotes "Fimbulthul spans 2.4-7.2 kpc (Malhan+2022)": UNVERIFIED (not found in Malhan+2022).
Full per-paper notes: the agent report (session); PDFs/text in the session scratchpad.

Implication for this project: the literature puts the knee at ~0.70-0.75 of omega Cen's distance; our CMD-shift track
(stream54_cmd_distance.py, relative to members at b 15-20) gives 0.88-0.92 and is the outlier.
