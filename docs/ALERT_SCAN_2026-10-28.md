# Alert assumptions: what the repos say (scan of 2026-10-28)

A read-only scan of every hdps branch (104 origin/*), the open PRs, the argus-hdps org repos (argus_alert_sim,
argus_alert_forecast, argus_sim, argus_camera_data, array_arrangement, argus_coordination, lab_notes, ota-test-monitor)
and ArgusSim, against the choices in our alert step (hdps alert-v4-local, `tasks/alerts.py`, steps 1-7 of 2026-10-27/28).
Six parallel searches; the key claims were re-checked by hand. Evidence notes and scripts:
`/private/tmp/claude-501/-Users-njm-MESA-ARGUS/cbed7a34-1bba-4573-a7c9-803221995c0c/scratchpad/scan/`.

Sources: avsc = the v4 `.avsc` files (the contract authority under ADR-0022); TN = TN-0004 alert schema; ADR30 = ADR-0030
alert schema v4; ICD = docs/.../alert-packet.md; all at PR #474 head 38cc23a3 (nothing newer exists; PR open, changes
requested 10-23, ADR "Proposed", TN "Draft").

## 1. Wrong in our packets (fixable in the alert step)

| Field | What the sources say | Ours | Fix |
|---|---|---|---|
| cutoutScience/Difference/Template | 32x32 float32 **electrons** (avsc, ADR30 sec. 4, TN) | segment units (x1/0.762 too high); our docstring says electrons, the code never scales | x `to_e` on all three planes |
| cameraIds, otaIds, filterIds, focuserIds | hdps ops-DB `hardware.py`: every device has an `asset_id` in ONE shared integer space; filters are physical (lot numbers); the band is the `filter` field | `[TELID]` in all four (ids collide across device types) | distinct ids per device type derived from the telescope, e.g. camera 100000+tel, OTA 200000+tel, filter 300000+tel, focuser 400000+tel |
| TELCONF (in the frame key, so epochId) | uint16 ops-DB code (ADR-0026); without it hdps uses CRC-16 of CAMID, whose value for an empty id is 0 | TELCONF = tel_id, and tel_id is 0-based (0..1189): telescope 0 gets 0 | offset (e.g. tel_id + 1) |
| numReferenceSources | "reference sources used in the WCS solution"; TPV_NMAT = 0 reads as unsolved | 0 | null (no stars: the WCS is the simulation's) |
| residualRMS | "WCS residual RMS (arcsec)" = sqrt(mean r^2) of matched stars (TPV_RMS) | SIMWCSR x scale: the true-WCS fit's LARGEST residual (~0.01") | the true-WCS fit's RMS, labelled as such |
| epochId fallback | frames without FRAMEKEY are skipped (ADR-0024 sec. 3); issue #289 calls the stem "filled incorrectly" | filename stem | skip the packet (all our frames have FRAMEKEY) |
| minipixIdx | block of the candidate's pixel | `int(x)//32` (floor; ~3% wrong at block edges; upstream the same) | `round(x)//32` |
| flags SATURATED | hdps never sets NONLINEAR (any branch) | checks SATURATED or NONLINEAR | harmless; keep |

## 2. Explained: the numbers we saw

- **beta magnitudes 0.16 mag brighter than truth_mag.** Not an error. truth_mag is AB through the Argus beta filter curve
  alone (no optics, QE, atmosphere); SIMZPTOT and truth_flux_now use the full response, so the two differ by a colour
  term by construction (beta, BP-RP 0.6..2.0: -0.09..-0.30 mag; f286's stars, BP-RP ~1.5-1.65: -0.25..-0.28; seen:
  -0.23). Validate against SIMZPTOT - 2.5 log10(truth_flux_now), not truth_mag.
- **PSF fluxes 5-6% low.** The constant Moffat (beta 3.5) on the engine's noiseless PSF gives 0.931-0.961 of the true
  flux: ~2.4 points from the PSF halo (3.5% of the light), 2-4 from the core shape. hdps's real zero points (ATLAS on
  main; Gaia XP in #479/#484) are fitted to the detector's measured r = 4 px aperture flux, so a real zero point
  absorbs such losses; our total-flux SIMZPTOT does not, and no hdps zero point is total-flux AB.
- **Correlated-noise factor below 1 (0.5-0.8).** An ideal resample with exact 1/W would give f ~ 2 (PSF) and ~1.8
  (r = FWHM). f < 1 means the variance fed in is 2.6-4x too large: hdps's variance floor is the detector's
  regional background rms^2 (256 px regions, 3-sigma clip) in crowded fields, and on the difference the sky is partly
  counted twice (science_sky_level is the tile mean with stars and zero-filled gaps). Our blank-position calibration
  absorbs it: on the captured tiles flux/error scatters by 0.73-1.25 at faint stars. hdps's own docs expect f > 1.
- **Saturation flags on everything brighter than ~12.5 mag.** Expected: full well 21657 e- = 58000 ADU (gain 0.3734);
  hdps flags raw ADU >= 58000. Read noise is added after the clip, so only ~56% of full pixels are flagged. Real IMX461
  frames saturate ~0.34 mag fainter.
- **No NEAR_BAD_PIXEL flags.** No bad, hot or cosmic-ray pixels are simulated, and there are no masters.
- **Sky.** coadd_background_variance is a sky LEVEL (median of the reduced frame, e- per detector pixel) used as a
  Poisson variance; beta 32-43 e- per detector pixel in 60 s on f286 (24-33 per tile pixel), rho 58-94; consistent with
  ArgusSim.
- **Positions ~2x the textbook error.** x, y are first-moment centroids, not PSF fits; hdps computes no centroid errors
  anywhere. Its native error would be the PSF fit's CRLB (psf/build.py fit_star, var_cx/var_cy) x f.

## 3. Not defined anywhere (our choice stands; to be written down)

- RA error with or without cos(Dec) (ours: RA cos Dec, as LSST); raDecCovariance unit (we send 0).
- Limiting-magnitude granularity (ours per tile; ZTF per image; Hank's sim per measurement); referenceDepth asks "at
  the location of the candidate" (ours per tile; the coadd's NPIX map would give it per position).
- Magnitude of a negative flux (ours null, as upstream; Hank's sim uses |flux|, ZTF style). Negative (fading) alerts:
  allowed (isFluxPositive is required) but not required; hdps sends positives only; ZOGY marks negatives `is_negative`.
- Aperture radius and the aperture zero point's convention: no radius anywhere; Hank's sim sets aperZeropoint =
  psfZeropoint (no correction) - a consumer assuming that misreads ours by the aperture correction (~0.31 mag).
- Duplicates in overlapping tiles: no rule. hdps has a home-tile function (`SegmentationScheme.query_tile_id`, nearest
  tile centre); ADR30 leaves linking to the broker (same eventId = same event).
- Template convolved or not (ours and upstream: the unconvolved reference); padding; the centre pixel of a 32x32 stamp.
- Hardware id values (no asset-tag values in any repo; TN-0003 in PR #431 mentions a "six-digit identifier").
- alertCadence for coadds (ours = window length, right for single frames; coadds need the cadence tier 60/900 s).

## 4. Confirmed (no change)

psfFlux*/psfMagnitude* (AB, electrons), psfChi2 (reduced), psfOverAp, fwhm/psfFwhm (pixels), psfBeta, ellipticity
(1 - b/a; upstream sends a/b, an upstream bug), nPixels, peak (electrons), flags bits 1/2/4/8, x/y, referenceId
(= TANSEGID: hdps has no reference version id), referenceStart/End, epochId (FRAMEKEY), eventId (HEALPix NESTED
order 16, "provisional"), filter codes, coadd (NCOMBINE > 1; upstream's rule is true on every frame), times (TT anchor
2025-01-01, MJD = 60676 + epoch/86400), realBogusConfidence (1 - 2 std of tree votes), skyBackground (per pixel).

## 5. Coming upstream (open PRs) that will change our packets

- **#474 docs vs schema:** the docs already require an envelope array `crossMatches` (may be empty) and drop the five
  candidate fields refcat*/nearestSrc*; the .avsc and models have not caught up (no crossMatch.avsc exists). Our
  packets follow the .avsc. crossMatch structure: {refCatName, refCatVersion, crossMatchProv, sources[] by srcSep},
  each source with exactly one synthetic argus_b/argus_rho AB magnitude; catalogue "Argus reference catalog (Gaia +
  Pan-STARRS)"; radius is producer policy; no implementation anywhere.
- **#488 flux-conserving reprojection (Hank):** tiles x0.755 (the fix for our x1.31 finding), FLUXCONS/PIXAREA,
  coadd_sky_level split from coadd_background_variance. Changes every electron value downstream, cutouts and absolute
  RB features: a retrain and a feature-spec bump; our CPU reprojector mirror needs the same Jacobian; skyBackground
  must then read coadd_sky_level.
- **#481 frame-key mode byte** (exposure and filter index in bits 56-61) and **#486** (FRAMEKEY re-derived, old kept
  as ACQFKEY): epochId values change. ADR numbering clash: #481 and #474 are both "ADR-0030"; 0031 and 0032 are taken.
- **#478 IMX455/461 assumptions:** saturation 60000 ADU; with no saturation_value given, an unconfigured camera
  (ARGUSSIM-...) raises - our runconfig must pass saturation_adu.
- **#460 SFFT DC over fit pixels:** fixes the GPU path's whole-tile mean subtraction (section 6).
- **#454:** references "camera-agnostic by intent" - conflicts with our local same-camera reference commit.
- **#479/#484/#487 Gaia XP zero points:** AB, e-/s, fitted to the r = 4 px detector aperture; error a "statistical lower
  bound" from a stored covariance.
- **#410 movers (asteroids):** open, changes requested; not connected to alerts (associator issue #293); would match
  nothing unless MESA_ARGUS injects the objects.
- Issues #289, #476, #477 overlap our local alert work (expect merge conflicts).
- Upstream bug: AlertVettingTask calls `AlertProducer.send_async` (an async def) without await, so upstream Kafka
  sends nothing. Our runs use a synchronous file writer and are unaffected. No schema registry exists; the wire format is
  JSON (an ADR-0022 "nonconformance"). For the broker team: schemaless Avro files + the .avsc set.

## 6. hdps's GPU route (why huepil had to hide cupy)

1. **HDPS_FORCE_CPU=1 with cupy importable:** DataCleanupTask and ImageReductionTask choose the GPU when cupy
   imports; the CPU source detector then reads the stale raw ADU frame (`source_detection_cpu.py:484`
   `np.asarray(image.data)`), so the variance floor is in ADU^2: 1/W and Var(D) x 7.17 (= 1/gain^2). CoaddTap also
   moves tiles to the GPU (the 37 TypeErrors). Fix: gate those tasks on `_on_cpu()`; read `image.ensure_cpu()`.
2. **Full GPU path, ~2% of the candidates:** the GPU SFFT subtracts the whole-tile mean of the difference, uncovered
   pixels included, lifting partly covered tiles by ~(uncovered share) x sky; the alert step's detection has no
   background of its own. Fix: PR #460 (effdb5a8), the mean over the fit pixels.
With both fixes huepil could run hdps on its GPU; it must then reproduce the CPU run (test: the f286 stride-10 night).

## 7. Simulation choices to decide

- **Array layout:** MESA_ARGUS renders thinB (1190 OTAs, tel_id 0..1189). The canonical layout since 2026-09-23 is
  A170r_1200 (1200 OTAs, different subarrays and numbering; org repo argus_sim). The layout name in exposures.ecsv is
  never checked against the one loaded at render time.
- **argus_sim (org, 2026-10-27) vs our ArgusSim:** delivered PSF replaces Zemax; beta = beam-averaged CT446 curve, rho =
  Chroma CT699; scalar optics (zero points -0.01 beta, -0.06 rho); site 31.0, -104.0, 2000 m; 16-min ratchets; band
  "beta" not "g". MESA_ARGUS cannot import it as it is. Our frames depend on ArgusSim feat/arbitrary-sed-input.
- camera.write_raw ignores the observation's [camera] gain/offset/saturation while hdps receives [camera] gain: a
  latent mismatch if those are ever changed.
