# Merging MESA_Argus with the argus-hdps work

Goal: a synthetic Argus catalogue and data stream that runs through hdps, so the pipeline can be tested and tuned on
data whose truth is known. MESA_Argus moves into the argus-hdps organisation, under a better name, with the work it
duplicates removed.

**Main goal (2026-10-16):** a repository of simulated Argus data lasting at least a month, as close to real as possible:
the night sky tiled as Argus sees it (minipixes, with the survey's gaps) for a whole month, from a leave-N-out
simulation of the minipixes; then alerts from it. Everything below serves that. First: proper source extraction on
the direct images, then real DIA (reference, subtraction, detection and photometry on the difference image).

Clones of the other repos: `~/argus_repos/` (synthref, hdps, argus_alert_sim). ArgusSim: `~/ArgusSim`.

## Checklist

### argus_alert_sim (Hank's simulated alert stream; uses MESA_Argus through `plugins/synthstar`)
- [x] Update the plugin to current MESA_Argus: PR https://github.com/argus-hdps/argus_alert_sim/pull/1 (open).
      It pinned MESA_Argus at 12031c7 and called `sources.get`, `sources.validate`, `sources.Field` and the `GRB`
      generator, none of which exist now.
- [x] `GRB_AFTERGLOW` dropped from the plugin; the forecast's ORPHAN_AFTERGLOW has no model for now.
- [ ] Re-add the GRB afterglow to MESA_Argus's transient models, then register it in the plugin again.
- [ ] Follow-ups: microlensing and supernova classes; one image/stamp implementation instead of two
      (`argus_alert_sim/stamps.py` is a port of `night_movie.Scene` and `difference_image`); CV_OUTBURSTS uses
      E(B-V) 0 (MESA_Argus files now record `distance_pc`, so sightline dust could be applied); `PeriodicCurve`
      reddens each template by its source star's own sightline E(B-V), which slightly changes the g/rho amplitude
      ratio when the shape is reused on another sightline.
- [ ] Build the star library (newton) as files `plugins/synthstar/library.py` reads; improve its matching (it gives
      each Gaia variable a random template of its class) rather than writing a second matcher.

### synthref (Hank's synthetic Gaia-XP subtraction references)
Each of these changes measurements: decide one at a time.
- [ ] System response: measured, beam-averaged rho filter (CT692-276), DR280 vendor optics, beta red edge. We use
      ArgusSim's idealised 560-825 nm rho box and a PlaneWave CDK optics curve. Route: synthref's
      `export_filter_curves.py` -> ArgusSim -> MESA_Argus. Ask: argus_survey_model ships `argus_rho_ct699.csv`
      (CT699-259), a different part from synthref's CT692-276.
- [ ] PSF: fitted misaligned-wavefront DR280 PSF and the static halo (3.6-4% of the flux within 60 px).
- [ ] Constant stars at their real positions from their Gaia XP spectra (dust already in them), with synthref's
      blend handling (Riello et al. 2021 excess). Our template + 3D dust stays for stars without XP and for
      synthetic or moved stars.
- [ ] Gaia positions propagated to the observation epoch with proper motions (we use the 2016 positions).
- Not to pull: catsfft / svsub (subtraction is hdps's), LRN / M31 work, `design_local.py` (Zemax numbers) and the
  IMX461 data (restricted).

### hdps
- [x] The measurement pipeline is hdps itself: raw camera frames through hdps's own worker (`GPUWorker`) and the
      production Task chain on the `hdps_full_cpu` branch (2026-10-21). The earlier MESA_Argus stages (extraction,
      calibration, psf, reference, ZOGY/SFFT, alerts) are deleted.
- [ ] Propose upstream: SFFT `mask_no_coverage_in_fit` (keep no-data pixels out of the kernel fit; hdps main fits
      across them). Report: AlertVettingTask reads `MAGZP`, which the chain never sets (it sets `coadd_magzp`), so
      alert magnitudes are empty; ffi-catalogs.md gives the rate-referenced recipe for `magzp`, the frame's
      `coadd_magzp` is electron-referenced.
- [ ] Report: with `mask_bright_stars` on, SFFTSubtractionTask's `_bright_star_mask` thresholds the science tile with
      its sky still in it (hdps's own convention: "science frames still carry sky", docker/demo/config.yaml), while
      its docstring assumes sky-subtracted inputs. Effective threshold ~1500-1700 e- above sky instead of 3000, moving
      with the sky; nothing brighter than about beta 15 can alert. Proposed fix: subtract the science tile's
      background before thresholding. hdps's production config leaves the mask off; MESA_Argus had it on until
      2026-10-25 and now leaves it off too (the alerts should be the ones Argus sends). Runs made before then
      carry `mask_bright_stars = true` in their resolved observation.toml.
      On a bright-sky night the whole tile is above the threshold: M67 beta (sky ~3550 e- per tile pixel) got no
      alert candidate in any frame, its differences being normal (noise 49 e-, ~400 detections per frame).
- [ ] Report: `LinearWCSSolver` accepts a false alignment in a crowded field. It takes the detections in a circle
      about the frame centre; on M3's core the quad match locks onto a wrong solution (LLR 63, above the threshold
      25) with the pixel scale wrong by 5-15x (0.06-0.20"/px against 1.007"). The TPV fit then contorts to it
      (higher-order PV up to 1e8), the WCS does not invert even inside the frame, and ReprojectionTask drops every
      tile ("non-finite WCS grid"). 13 of 228 beta and 52 of 186 rho frames of the M3 night came out at the right
      scale. Nothing checks the affine scale against `pixscale`, which the solver is given. Proposed fix: reject an
      alignment whose scale (or shear) differs from `pixscale` by more than a few percent and keep searching larger
      subsets. Not reproducible with our settings: LinearWCSSolverTask n_stars 150/600/1500, TPV terms 7/11/17 and
      no Chebyshev basis all leave the M3 frames without tiles.
- [ ] Write our exposures as L2 `.arg` segments with hdps's own `hdps.archive.data_store.encode_frames` (same
      0.879"/px tile grid as ours), and a reference in `CoaddReferenceProvider`'s layout (synthref `render.py`).
- [ ] SFFT fixes (kernel mirror, DC offset, spatially varying background) are on branches `origin/408/16..20`,
      not on main (checked 2026-10-15): needed before tuning hdps on our data.

### ArgusSim
- [ ] MESA_Argus imports the private `~/ArgusSim` (has `ZemaxPSF`); the public argus_sim in argus_survey_model
      (what argus_alert_sim installs) has no `ZemaxPSF`. Settle on one.

### The move and the rename
- [ ] Move to argus-hdps; new package name; argus_alert_sim's submodule URL and `import synthstar` follow.
- [ ] Remove what is duplicated (stamps/image code, throughput, PSF) once the above are settled.

## Log
- 2026-10-15: cloned synthref, hdps, argus_alert_sim to `~/argus_repos/`; reviewed synthref against MESA_Argus
  (findings above). argus_alert_sim's plugin is broken against MESA_Argus main (API change, GRB removed).
- 2026-10-15: argus_alert_sim branch `mesa-argus-current` (local, not pushed): plugin follows MESA_Argus dfd20ad;
  GRB_AFTERGLOW dropped from the plugin; 15 tests pass, 7 skip (need Hank's Gaia catalogue); `aas models` lists
  CV_OUTBURSTS; `aas check` not run (needs Hank's visit store). Library loads 49 templates from `stars/`, 31 from
  `runs/`. Waiting on: port the GRB afterglow into argus_alert_sim, or leave ORPHAN_AFTERGLOW without a class.
  Also fixed: the README's separate plugin install fails in uv; installing both packages in one command works.
- 2026-10-15: PR argus-hdps/argus_alert_sim#1 opened (branch `mesa-argus-current`). User's call: drop
  GRB_AFTERGLOW and leave the gap; the afterglow comes back later in MESA_Argus's transients.

- 2026-10-16: difference panel was display-only and drawn on a linear scale set by the brightest variable (looked
  blank); now asinh (cfac795). Source extraction bug found: `week_f325`'s V2 (G 13.9) measured ~18 mag in 16% of
  exposures. Cause: SEP windowed centroids (XWIN/YWIN, window from the reference FWHM 4.34 px) of a G 18.5 neighbour
  8.5 px away migrate onto V2's centre; the nearest-neighbour cross-match then takes the faint detection. Detection
  itself is fine. Also: `--start`/`--end` are ignored in survey mode (night_movie.py).
- 2026-10-16: plan: use hdps for extraction and DIA, with CPU mirrors of its GPU files (same classes and outputs)
  where no NVIDIA GPU exists (the Mac's Apple M5 GPU runs Metal, not CUDA). hdps is edited locally only, branch
  `cpu-mirror` in ~/argus_repos/hdps, no PRs. The simulator will write calibrated sensor frames (hdps
  `FullFrameImage`) so hdps runs detection, WCS solving, reprojection to tiles (`TANSegmentList`), subtraction,
  difference detection and alert vetting. Step 1 done: every hdps package imports without CuPy (4 files edited);
  hdps tests on CPU: astrometry 157, diffimg 140, skymap 171 passed (GPU-marked tests skipped).
  Next: photometry/background_cpu.py + source_detection_cpu.py mirroring background.py, source_detection.py, .cu.
- 2026-10-16: hdps branch `cpu-mirror` commit ecd73320: `photometry/source_detection_cpu.py` (CPUSourceDetector:
  the 4 CUDA kernels of source_detection.cu as numba loops, host steps copied). `hdps.photometry.SourceDetector`
  resolves to GPUSourceDetector on a CUDA host, else CPUSourceDetector (`HDPS_FORCE_CPU=1` forces CPU);
  SourceDetectionTask and DiffDetectionTask construct SourceDetector. CPU copies of hdps's detector tests: 96 pass
  (aperture sum vs hdps's CPU reference, sub-pixel phase, centroid iteration, error model, determinism,
  per-pixel-variance detection, de-duplication). On week_f325 exposure 18 (the SEP failure) it finds V2 at 0.21 px
  with 43519 e and keeps the G 18.5 neighbour separate (0.37 px, 846 e). Not yet compared with the GPU code on a
  GPU (none available).
- 2026-10-19: hdps packages installed into the system Python (pip --user -e; no existing package versions changed).
  MESA_Argus extraction is now hdps's own detector (78b0d94): production settings of SourceDetectionTask, one-to-one
  association with the simulated stars, calibrators need S/N >= 50, per-exposure catalogues (parquet). f325 survey
  night (210 rho exposures): V1 median -0.013, robust scatter 0.020 mag, 4 off by >0.5 mag; V2 +0.023, 0.016, 1 off
  (SEP had V2 off in 16% of exposures). The SEP code is kept as sep_photometry.py for the demo and the offset study
  (the offset study's capture of the engine's photometry object no longer applies; it needs rewriting or retiring).
- 2026-10-19: hdps detector defect found (same on GPU and CPU; the mirror reproduces the kernels): noise peaks ~6 px
  from a bright star (S/N ~3) pass detection; the 3-pass centroid iteration (window up to the 6 px matched-filter
  half-width) moves them onto the bright star; their re-centred apertures take part of its light; the de-dup grid pass
  (keep rule: a later source replaces the kept one if brighter than the previously visited one) can then keep a fake
  and drop the star. f325 exposure 40: V1 real 55258 e (S/N 184) dropped, fake 30999 e (S/N 3.4) kept. Also: hdps
  saturation is 0.9999 x 65534 (16-bit ADC units); simulated images are electrons with a 21657 e full well.
  Survey fields Dec < -22 (f244, f276, f279) never reach 3 h above the array's 32.84 deg minimum altitude.
- 2026-10-19: hdps branch pushed as `hdps_full_cpu` (argus-hdps/hdps, at ecd73320, local branch `cpu-mirror` tracks
  it). No PR. So far it holds the CUDA-free imports and the CPU source detector; ZOGY, the reprojector and the
  difference-detection kernel are not mirrored yet.
- 2026-10-19: hdps PR #471 (user; walked_off in source_detection.cu) merged into hdps_full_cpu with the CPU mirror
  change (0678cd5b). MESA_Argus f4393b5: exposures made on the sensor grid, resampled onto the hdps tile grid by
  Catmull-Rom (as hdps's reprojector). 701f88a: measurement pipeline as stages (synthstar/pipeline/: frames,
  extraction, calibration, truth), chosen by name from [pipeline]; calibration against ATLAS-Refcat2 with hdps's zero
  point (no simulation truth). f294 survey night, rho, 154 frames: plane zero point in every frame (25.786 +/- 0.021),
  V1 +0.053 / 0.031 mag, V2 +0.024 / 0.026, no outliers. Refcat2 at ~/atlas_refcat2 (linked from
  ~/.rico_cache/atlas_refcat2), still downloading (26 of 82 GB on 2026-10-19).

- 2026-10-19: Refcat2 download complete (83 GB, 192 files).
- 2026-10-19: difference imaging and alerts (synthstar/pipeline/: psf, reference, dia, alerts). observe.py first
  observes a sacrificial night (the night before, same stars, no transients); its per-pixel median is the reference,
  written in hdps's CoaddReferenceProvider format with hdps's PSF commissioning products (EPSF, PMERRKER, PMERRCAT).
  ZOGY: hdps's CPU subtractor; PSFs are hdps's Moffat field fit (select_stars + FramePSFBuilder, as TilePSFTask), not
  the stacked ePSF: hdps's own ZOGY task keeps prefer_measured_epsf off, and here the stacked ePSF carried a pedestal
  (17% of its flux at 12-20 px, true stars 2.5%) that left ~85 false alerts per frame around bright stars. The
  reference's PSF-model-error plane enters S_corr as in hdps's GPU path. hdps branch cpu-mirror ac88d60b (local, not
  pushed): ZOGYReferenceSubtractor takes extra_var_r / extra_var_n, as ZOGYGPUSubtractor. Departures, for want of a
  real/bogus model: threshold 8 sigma; no candidates within 10 px of saturation or 20 px of the edge (FFT wrap-around).
  Alerts: hdps Alert schema 3.0, orjson of to_dict(), JSON lines per frame. f294 survey night, rho, 154 frames, dwarf
  nova placed: 89 alerts, all the dwarf nova, no false alerts; it is alerted in 28/28 frames brighter than 17.5,
  51/63 at 17.5-18, 10/63 at 18-18.3. Reference 5 sigma depth 20.8 (rho).
- 2026-10-19: full observation, f294 survey night, beta and rho, movie on (outputs/pseudo_real/test_alerts_f294/).
  psf_min_stars 8 (FramePSFBuilder's floor; hdps TilePSFTask 12 per full tile): the cut-out is ~1/14 of a tile and
  the beta one has 10 clean PSF stars. No frame skipped. rho: 93 alerts, 89 the dwarf nova, 4 false (one frame).
  beta: 151 alerts, 79 the dwarf nova (14/14 frames brighter than 17.5, 51/58 at 17.5-18, 14/68 fainter); 72 false,
  all 3-6 px to the same side of two bright unsaturated stars (90k and 41k e-) in 58 and 14 frames: an asymmetric
  beta PSF the symmetric Moffat does not follow. The reference-side PSF-model-error plane covers them (V_err ~2e4
  there) but not enough; production uses the reference side only too (tasks.py extra_var_rs), with real/bogus
  vetting after it. Open: science-side V_err (the GPU seam extra_var_ns exists, unused in production), or the
  measured PSF asymmetry.
- 2026-10-20: the pipeline rebuilt as hdps itself. Each exposure: ArgusSim's survey (telescope, band, tracking angle)
  and geometry place every star on its telescope's sensor (synthstar/sensor.py), rendered in an N x N-minipix
  readout window (field.window, default 56) and written as a raw camera frame (synthstar/camera.py: real AEGIS
  header keys, toy gain = full well / 58000 e-/ADU, readout parity of the real frames). One observation file defines
  every parameter with default, limits and justification (synthstar/schema.py; template.toml generated from it).
  Real/bogus not run (every score 1). CPU paths for ReprojectionTask, CoaddTapTask, SFFTSubtractionTask and
  DiffDetectionTask on hdps cpu-mirror.
- 2026-10-21: hdps cpu-mirror rebased onto main after #425 (tasks.py split into a package). MESA_Argus's own runner
  replaced by hdps's GPUWorker (synthstar/hdps_worker.py: ManifestSource, its WCS cache, coadd finalize, product
  writer; fakeredis for valkey). LightweightFITSOutputTask (catalogue, WCS, zero point) and TileImageOutputTask (the
  movie's tile) added to the chain. Alert packets: hdps's own (AlertVettingTask), Avro with the public argus.alert
  schema (v3.0; v4 is on origin/feature/alert-schema-v4, unmerged) read from hdps at run time, one file per exposure
  in place of Kafka. The packets carry hdps's gaps: UUID alertId (ADR-0027's HEALPix id not implemented), no
  event/SSO association, alertCadence 0, no inline cutouts (stamps in alert_stamps), psfMagnitudeDif empty (MAGZP).
- 2026-10-21: squares around every star and a gradient in hdps's difference images: hdps's SFFT fits its kernel over
  the whole tile including where the reference has no data (one per-ratchet coadd covers ~38% of a tile at the
  56-minipix window); the kernel took a flat pedestal (sum 1.36 vs 1.10), a 13 x 13 px square around every star.
  hdps cpu-mirror 2093773c: mask_no_coverage_in_fit; 83a6bb1c: the CPU SFFT does no whole-tile median subtraction
  (the GPU code has none; on a part-covered tile it shifted the difference by the sky, +1000 e-). On one frame:
  level 0-2 e-, swing across a tile 1-4 e- (was up to 1905), squares 45 e- (was 11644). Bright stars leave ~20%
  residuals (0.2 px registration, PSF from another telescope's reference), which alert with real/bogus off: accepted.
- 2026-10-23: validating the live notebook's edited star against injected stars hdps measured
  (tools/check_edited_star.py) found two facts about hdps as our chain runs it. (1) Tiles carry surface brightness:
  a star's flux summed on a tile is its electrons x (1.007"/0.879")^2 = 1.31. (2) With `mask_bright_stars` on (our
  config; off in hdps's production config) the mask is applied to the science tile with its sky in, so an injected
  dwarf nova at beta 14.7 never became a candidate in 135 frames; reconstructed with the sky in, the mask covers it in
  every frame and none of hdps's 441 candidates, sky-subtracted it covers it in half the frames. An hdps issue (the
  mask's assumption against hdps's own sky convention); the simulation reproduces hdps as it is.
- 2026-10-24: the live notebook's Messier nights (newton). All six engines failed writing an empty photometry table
  (fixed 3b55487; tools/remeasure.py measures again from hdps's products). Through hdps: M45 normal (frames not
  processed 8% beta, 4% rho). M3: the linear WCS locks onto false solutions in the cluster core (see the hdps
  checklist): no tile in beta, 18 frames with tiles in rho. M67: the linear solver finds no alignment in 46% of beta
  and 32% of rho frames (LLR -5, "No alignable subset"; field angle and telescope do not explain it; the solved
  frames have the right scale), and beta has no alert candidates because the bright-star mask covers its whole tile
  (sky above the threshold). Training nights lose 2.1% of frames in hdps (20 band-nights, Mac).
