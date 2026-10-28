# Alert packet fields: review of groups 3 and 4 (2026-10-27)

A read-only review of the plan to fill the empty fields of the v4.0 alert packets that hdps's AlertVettingTask writes
for our simulated nights. Groups 1, 9 and 2 (zero point, seeing, electrons, coadd flag, ellipticity, per-source FWHM,
epochId, the +1 px detection shift, flux errors, aperture flux, sky, filter, cadence) are done or being tested; this
covers what is left.

Paths: H = ~/argus_repos/hdps_v4/packages (branch alert-v4-local), M = ~/MESA_ARGUS.
AL = H/hdps_orchestrate/src/hdps/orchestrate/tasks/alerts.py, FE = H/hdps_alerts/src/hdps/alerts/features.py,
FIT = H/hdps_photometry/src/hdps/photometry/psf/fit.py, DI = H/hdps_orchestrate/src/hdps/orchestrate/tasks/diffimg.py,
CO = H/hdps_orchestrate/src/hdps/orchestrate/coadd.py, TC = M/mesa_argus/true_calibration.py, NM = M/night_movie.py.
Line numbers are as of the review and move with edits.

## Already acted on (in group 2)

- Fields passed explicitly to Candidate must leave ALERT_UNPRODUCED_FIELDS in the same change (a duplicate keyword
  raises TypeError, caught per alert: every packet silently dropped). Done for filter and psfFluxDifErr; compare packet
  counts before and after any such change.
- Correlated noise: Var(D) treats segment pixels as independent; the resample correlates them, so errors and S/N are
  too small (estimate ~1.4x in sigma). Now: hdps's calibrate_blank_scatter (FIT ~978) per tile, Var(flux) x f, in the
  packet's psfFluxDifErr, psfMagnitudeDifErr and psfSnr; the feature snr_psf unchanged (the model is trained on it).
  forced_photometry on D would double-count the source Poisson term (Var(D) has it, DI ~408-414).
- The sky is read from presub_science independently of emit_stamps (sfft_science_sky_level is the tile mean, stars and
  zero-filled gaps included).

## Group 3

**psfFluxSci/Err, psfMagnitudeSci/Err.** presub_science (DI ~980-985) is a host copy with the sky in and NaN -> 0 at
gaps; only with retain_planes (off in production; ~16 MB a tile). The science variance is carried: 1/W from
seg.uncertainty.output_inverse_variance (SFFT reads it, DI ~383-397). The diagnostic fit weighted by Var(D) is wrong
(Var(R) x K^2 in it; inf at reference gaps). Use hdps's forced_photometry (FIT ~502) on a stand-in (data, W, dq_flags),
noise_inflation="blank": local constant sky, source Poisson, SATURATED masked, TOO_MASKED / EDGE_TRUNCATED flags.
x to_e; same zero point as the difference (D is on the science scale). Caveats: saturated variables (wings only),
blends, forced at the difference centroid (wrong for dipoles), no halo in the Moffat (biased low like psfFluxDif); SSO
packets require these fields (models.py ~268-272). Validate against truth_flux_now in electrons (the AB colour term is
there by construction). 0.5-1 day.

**aperFluxDifErr, aperSnr, aperMagnitudeDif/Err, aperZeropoint.** sep.sum_circle with var=Var(D), x sqrt(f_ap) from
blank apertures (measure_noise_correlation(D).f, psf/noise.py ~75, is the large-aperture limit: too high at r ~ 2.5-3.5
px; the analytic correlation kernel misses K x R noise). aperZeropoint = psfZeropoint + 2.5 log10(EF), EF the
aperture's share of the light. SIMAPFR is not it (the source detector's flux_radius aperture on the raw frame, TC
~70-107). Options: (a) Moffat beta 3.5 analytic (EF ~0.79 at r = FWHM); (b) the simulation's true share; (c) a per-tile
curve of growth on isolated stars (what a production pipeline does). Recommend (c) or (a); with (b) aperture mags are
truer than PSF mags and their offset shows the PSF model's bias. A 0.3 px centring error costs a few % at r = FWHM.
DECISION: magnitude of a negative flux: null (now) or of |flux| (ZTF style), the same for PSF and aperture. 0.5 day
once f exists.

**psfLimMag, aperLimMag.** zp - 2.5 log10(5 sigma_blank to_e) from the same calibrate_blank_scatter call; aperture
from the blank-aperture scatter and aperZeropoint. DECISION: per tile (ZTF diffmaglim is per image), local at the
candidate (no own Poisson), or per 256 px DiffDetection region. Detection is 3 sigma, so many alerts are fainter than
psfLimMag: expected, document it.

**xErr, yErr, raErr, decErr, raDecCovariance.** x, y are not fitted: SEP barycentres (FE ~201) on a fixed 4 px Gaussian
S/N map; the PSF fit holds the position. DECISION: (a) the Fisher bound from render_moffat_stamp_grad (reference.py
~223) x sqrt(f), scaled empirically (alone it understates SEP's scatter); (b) fit the centroid with the PSF (changes x,
y, ra, dec, eventId: no longer what production sends); (c) an empirical sigma(SNR). Sky: Sigma_sky = J Sigma J^T, J =
seg.wcs.pixel_scale_matrix x 3600. DECISION: conventions the schema leaves open (raErr in RA or RA cos(dec); covariance
unit): propose on-sky RA cos(dec) and arcsec^2, ask hdps to document. DECISION: a WCS floor in quadrature (true WCS
~0.01 px; production TPV_RMS). Validation needs truth_ra/truth_dec columns (M/mesa_argus/pipeline/truth.py ~55-71 has
only truth_sep_arcsec); sep/sigma should be Rayleigh(1). ~1 day.

**peak.** Max of D within ~1 px of (x, y) on valid pixels, x to_e (SEP's peak is S/N-map units). The cutouts are
documented as electrons (AL ~234) but carry raw segment values (no to_e): a broker summing cutoutDifference does not get
psfFluxDif. DECISION: scale peak and cutouts by to_e together (matches PR #488's flux-conserving tiles) or document both
as surface brightness.

**flags.** Saturated: dq_flags & SATURATED within ~2 FWHM (set at image_reducer.py ~392, ~527-529 for raw >= 58000 ADU;
the simulation clips at full well = exactly 58000 ADU, camera.py ~20: works on equality, fragile); the reprojector ORs it
over the 4x4 footprint (cpu_reprojector.py ~100); saturated pixels get W = 0. CoaddReferenceProvider._load (DI
~1384-1480) does not read the coadd's DQ: DECISION: add that or leave reference-only saturation unflagged.
Near-bad-pixel: DEFAULT_REJECT_MASK (psf/select.py ~83-92) minus SATURATED; DECISION: include NO_REFERENCE or not
(always 0 in the simulation apart from gaps). Near-edge: edge_dist (FE ~257) is the distance to any invalid pixel incl.
holes and ignores the array border; candidates within 12 px of the border never become packets (the 25 px stamp does not
fit, PSF flux non-finite, skipped): in an older run 2006 of 40826 rows; so near-edge = min(distance to the border,
distance to W <= 0 or NO_REFERENCE). Astrometry-suspect: production would use TPV_MED (astrometry/chain.py ~397-400);
always 0 here. 0.5 day.

## Group 4

**referenceId, referenceStart/End JdTt and ArgusEpoch.** SFFT stamps reference_filename on each segment (DI ~701-703).
DECISION: read that file's primary header in the alert step (cached per path; no provider change) or keep it in the
provider's cache and stamp the cards (cleaner). Cards (CO ~1087-1094): TANSEGID, NCOMBINE, EXPTIME (total), TSTART,
TSTOP (UTC isot: starts of the first and last frames processed). Start = isot_to_argus_epoch(TSTART) (UTC -> TT); End =
isot_to_argus_epoch(TSTOP) + EXPTIME/NCOMBINE (assumes equal exposures); JD = anchor + epoch/86400. referenceId always
equals tansegId (references matched by tile id): DECISION: accept or ask hdps for a reference product id. The camera
fallback can pick another camera's coadd: record which. Validate against the reference pass's exposures. 0.5 day; best
value (required fields now 0.0 and -1).

**referenceDepth.** sigma^2 = BKGVAR/NPIX at (x, y) (CO ~1104; DI ~1443-1456) x sum(CORRKERN) if written, the noise
equivalent area from the reference's SEEING, x to_e. Trap: the coadd's MAGZP is the median of the frames' coadd_magzp =
SIMZPAP, an aperture zero point (TC ~190; tasks/coadd.py ~219): off by 2.5 log10(1/SIMAPFR) for a PSF depth. hdps-native:
ZP_ref = psfZeropoint - 2.5 log10(sfft_kernel_sum) (DI ~1007; D = S - K x R). Stash a reference to the provider's cached
variance plane at DI ~701. Check: depth ~ psfLimMag + 1.25 log10(NCOMBINE), allowing for seeing. 1 day.

**numReferenceSources, residualRMS.** Production: TPV_NMAT and TPV_RMS (radial, arcsec), astrometry/distortion.py
~612-614: wire to these keys (null under TrueCalibrationTask). SIMWCSR is a different statistic (largest residual of the
true WCS fit, px): not residualRMS. DECISION: null, or emulate in TrueCalibrationTask (match detected stars to true
positions through the true WCS; stamp TPV_RMS, TPV_NMAT, TPV_MED; no hdps change). Recommended: emulate. Raw header cards
(SIMWCSR, SIMZPTOT, SIMAPFR, FILTER) already reach the alert step (images.py ~898-901, skymap.py ~95).

**psfZpErr, aperZpErr.** Production: magzp_rms, magzp_nstars (tasks/astrometry.py ~763-764); production's zero point is
an aperture zero point, "QA" only (CO ~1121-1131), so production PSF magnitudes would be off by the aperture correction
(an hdps issue). DECISION: 0.0 (exact, but implies perfect calibration; the colour term remains), null, emulated rms/sqrt(n),
or the per-star rms. Avoid 0.0 if brokers fold it into magnitude errors. aperZpErr should carry the aperture-correction
error.

## Cross-cutting

1. f first: every Err, Snr, LimMag, position error and depth depends on it (done in group 2).
2. Duplicate alerts: tiles overlap by 32 px (hdps_skymap tiling_scheme.py ~126-128). In an older run
   (interactive_f294_g, every candidate accepted) 309 same-exposure pairs within 1.5" on different tiles both became
   packets (618 of 38820, 1.6%), sometimes with different eventIds. DECISION: flag, or dedupe by tile ownership.
3. No negative-going alerts: only positive peaks are detected (FE ~299-300); a fading source never alerts as fading.
   In that run 7687 of 40826 rows were positive peaks whose PSF fit came out negative (magnitude null). ZTF/LSST
   brokers expect negative-subtraction alerts. Recheck in a current run.
4. Truth: validate in electrons, on transients (reference flux 0) and constant stars (difference 0); for variables
   truth_flux_reference is the cycle-mean over the science night (NM ~1354-1361, ~1582), not the reference ratchet. Add
   truth_ra, truth_dec, truth_saturated, the reference time span.
5. New fields go in the alert_catalog rows too; the truth join is on alertId (M/tools/alert_sample.py ~206-215).
6. The even kernel at hdps_orchestrate tasks/photometry.py ~356 only sets the regional rms: harmless.

**What a broker team checks first:** sentinels (JD 0, error -1, depth 0); internal consistency (mag = zp - 2.5 log10
flux, S/N = flux/err, magErr = 1.0857 err/flux, cutout sum vs flux, peak vs cutout max); pulls of repeat detections of
one eventId; duplicates; detected vs limiting magnitudes; missing negative alerts; time scales (TT here vs ZTF's UTC
jd); cutout orientation (no WCS for the cutouts).

**Recommended order:** reference provenance; aperture errors, S/N, limits, magnitudes; science photometry; flags, peak
and cutout units; position errors with the new truth columns; zero-point error and WCS statistics; referenceDepth.
