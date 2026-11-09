"""Toy: each frame's astrometric and photometric calibration from the simulation, in place of hdps's solve (user,
2026-10-26: "remove the astrometric solving, its not needed and a lot of them fail"; the zero point: the simulation's
own).

hdps's production chain calibrates a frame with ReferenceRetrievalTask -> LinearWCSSolverTask -> SourceMatchingTask ->
TPVWCSSolverTask: a WCS solved against ATLAS-Refcat2 and, from the same crossmatch, the frame's zero point
(coadd_magzp, TPVWCSSolverTask's _harvest_photometric_qa). In the simulation both are known. When the observation's
chain has TrueCalibrationTask in place of those four Tasks, the engine writes them into each raw frame (cards()) and
the Task hands them to hdps as the solve would:

- the WCS: the frame's primary WCS cards, a TAN projection about the telescope's optical axis fitted to the
  simulation's own geometry (sensor.Projector.wcs: exact in the apparent frame, within 0.01 px after CIRS -> ICRS).
  It becomes the frame's linear_wcs and full_wcs, its cards in the frame's metadata (as TPVWCSSolverTask writes them,
  so hdps's solved-frame product carries them);
- the zero point, for hdps's aperture flux (SourceDetectionTask's flux, the one the crossmatch calibrates): SIMZPTOT,
  the throughput's AB zero point at the exposure's airmass (ArgusSim ABPhot.electrons_from_sed of a flat-spectrum
  AB 0 source, x the exposure time; the stars' own electrons come from the same call), plus SIMAPFR, 2.5 log10 of the
  share of a star's electrons hdps's aperture measures (hdps's own source detector on the exposure's PSF: its seeing,
  the field's tabulated field angle): SIMZPAP = SIMZPTOT + 2.5 log10(SIMAPFR), set as coadd_magzp. As in any
  photometric system a colour term remains: a star whose spectrum is not flat in f_nu is off by the difference
  between its AB magnitude (ABPhot's filter-only response) and its electrons (the full response).

Also set, for hdps's alert step (AlertVettingTask, alert-v4-local; before 2026-10-27 it read only the coadd header's
MAGZP and SEEING, absent on frames, so every alert had no magnitude and a 4.5 px PSF):
- psf_magzp: SIMZPTOT, the zero point of a PSF flux (a PSF fit counts the star's whole light; the aperture's
  SIMZPAP would make every alert 2.5 log10(1/SIMAPFR) too bright);
- coadd_seeing: the frame's stellar FWHM in arcsec, measured on the image (stellar_fwhm), as TPVWCSSolverTask's QA
  harvest measures it (there by second moments) when the solve runs.

Not set, with the crossmatch gone: magzp_rms, magzp_nstars.
"""
import numpy as np
from hdps.core.images import FullFrameImage
from hdps.orchestrate.tasks.base import Task

TYPE = "TrueCalibrationTask"
REPLACES = ("ReferenceRetrievalTask", "LinearWCSSolverTask", "SourceMatchingTask", "TPVWCSSolverTask")
_WCS_KEYS = ("WCSAXES", "CRPIX", "PC", "CD", "CDELT", "CUNIT", "CTYPE", "CRVAL", "LONPOLE", "LATPOLE", "RADESYS",
             "EQUINOX")
MAX_WCS_RESIDUAL = 0.05  # px: the true WCS reproduces the stars' placement to ~0.01 px over the whole sensor
_STAR_E = 1.0e5  # electrons of the star the aperture share is measured on: peak ~1e4 e-, below saturation
_POSITIONS = 4  # sub-pixel positions per axis the aperture share is averaged over
_FWHM_STARS = 60  # the brightest usable detected stars a frame's stellar FWHM is fitted to
_ISOLATION_PX = 8.0  # a star used for the FWHM has no other detection this close (px)


def wanted(obs):
    """Whether the observation's chain calibrates frames from the simulation (the engine then writes the cards)."""
    return any(t["type"] == TYPE for t in obs["hdps"]["tasks"])


def wcs_cards(w):
    """The WCS's cards to carry in a header: the projection only (no MJDREF or other time cards)."""
    return {k: v for k, v in w.to_header().items() if k.startswith(_WCS_KEYS)}


def ab_zero_point(ab, band, airmass_grid):
    """Electrons/s of a flat-spectrum (f_nu) AB 0 source on airmass_grid, as ArgusSim's ABPhot gives a star's."""
    import astropy.units as u

    wave = ab.tp.wav
    fnu = np.full(len(wave), 3631.0) * u.Jy
    return np.array([ab.electrons_from_sed(band, wave, fnu, airmass=float(x)).value for x in airmass_grid])


_detector = None
_aperture = {}


def aperture_fraction(psfs, band, seeing, angle_index, detection):
    """The share of a star's electrons hdps's source detector puts in its aperture (flux_radius, its annulus
    background), for the PSF at this seeing and tabulated field angle: hdps's detector (SourceDetector, the settings of
    the run's SourceDetectionTask) on the star as the frames draw it (sensor.expected, with the run's PSF wing if it has
    one) on a flat sky, averaged over sub-pixel positions."""
    global _detector
    key = (band, round(float(seeing), 3), int(angle_index))
    if key in _aperture:
        return _aperture[key]
    import time

    from hdps.core.telemetry import Telemetry
    from hdps.photometry import SourceDetector

    from mesa_argus import sensor

    if _detector is None:
        _detector = SourceDetector(detect_sigma=detection["detect_sigma"], bg_region_size=64, max_sources=1000,
                                   flux_radius=detection["flux_radius"], matched_filter_width=detection["matched_filter_width"],
                                   centroid_n_iter=detection["centroid_n_iter"], cr_detect_contrast=detection["cr_detect_contrast"])
    rng = np.random.default_rng(0)
    angle = np.array([psfs.angles(band)[angle_index]])
    shares = []
    for fy in np.arange(_POSITIONS) / _POSITIONS:
        for fx in np.arange(_POSITIONS) / _POSITIONS:
            x, y = np.array([96.0 + fx]), np.array([96.0 + fy])
            image = sensor.expected(x, y, np.array([_STAR_E]), angle, band, seeing, 1000.0, (0, 0, 192, 192), psfs)
            image = image + rng.normal(0.0, 1.0, image.shape)  # a flat sky; the 1 e- noise sets the detection threshold
            frame = FullFrameImage(data=image.astype(np.float32), metadata={}, timestamp=time.time(),
                                   telemetry=Telemetry(output_enabled=False))
            table = _detector.detect_sources(frame).sources.table
            d = np.hypot(np.asarray(table["xcentroid"]) - x[0], np.asarray(table["ycentroid"]) - y[0])
            if len(d) and d.min() < 3:
                shares.append(float(table["flux"][int(np.argmin(d))]) / _STAR_E)
    if not shares:
        raise RuntimeError(f"hdps's detector did not find the PSF star ({band}, seeing {seeing}, angle {angle_index})")
    _aperture[key] = float(np.mean(shares))
    return _aperture[key]


def cards(w, residual_px, residual_rms_px, zp_total, aperture_share):
    """The raw frame's calibration cards: the WCS, the zero points, the aperture share and the WCS fit's residuals,
    largest and RMS (a largest over MAX_WCS_RESIDUAL px stops the run: a wrong WCS loses the frame's tiles in hdps's
    reprojection)."""
    if not residual_px <= MAX_WCS_RESIDUAL:
        raise RuntimeError(f"true WCS: residual {residual_px:.3g} px over the sensor (at most {MAX_WCS_RESIDUAL})")
    out = wcs_cards(w)
    out.update(SIMZPTOT=(float(zp_total), "simulation: AB zero point, total electrons (mag)"),
               SIMAPFR=(float(aperture_share), "simulation: share of a star's e- in hdps's aperture"),
               SIMZPAP=(float(zp_total + 2.5 * np.log10(aperture_share)), "simulation: zero point, hdps aperture flux"),
               SIMWCSR=(float(residual_px), "simulation: true WCS fit, largest residual (px)"),
               SIMWCSM=(float(residual_rms_px), "simulation: true WCS fit, RMS residual (px)"))
    return out


def stellar_fwhm(image, scale_arcsec):
    """The frame's stellar FWHM [arcsec] as hdps's alert step models a star (hdps.alerts.features: a Moffat of beta
    MOFFAT_BETA plus a constant background, inverse-variance weighted, on its 25 px stamp): the FWHM whose summed
    chi-square over the frame's brightest usable detected stars is least (best_fwhm), times the plate scale. Usable:
    not flagged saturated, its brightest pixel (sky included) under half the full well, no other detection within
    _ISOLATION_PX, its stamp on the frame. Measured on the image, as TPVWCSSolverTask's QA harvest measures
    coadd_seeing; that one takes second moments, which read 11-43% wider than the Moffat that best fits the
    simulation's PSF, and the alert step's PSF fluxes would then come out 10-26% high (2026-10-27, Zemax PSF at
    0.8-2" seeing and three field angles). None with fewer than 5 usable stars, or a fit at the search's edge."""
    from hdps.alerts.features import best_fwhm, psf_stamp

    from mesa_argus import camera

    table = getattr(getattr(image, "sources", None), "table", None)
    if table is None or len(table) < 5:
        return None
    # After a GPU-resident reduction the calibrated frame is the device copy and .data the stale raw one (hdps's
    # TPVWCSSolverTask harvest reads it the same way).
    data = image.data
    if getattr(image, "_is_gpu_current", False) and not getattr(image, "_is_cpu_current", True) \
            and getattr(image, "_gpu_data", None) is not None:
        import cupy

        data = cupy.asnumpy(image._gpu_data)
    data = np.asarray(data, dtype=np.float64)
    var = np.maximum(data, 1.0)  # electrons: Poisson, sky included
    valid = np.isfinite(data)
    x, y = np.asarray(table["xcentroid"], float), np.asarray(table["ycentroid"], float)
    snr = np.asarray(table["snr"], float)
    saturated = np.asarray(table["is_saturated"], bool)
    full_well = camera.GAIN * camera.SATURATION_ADU
    stamps = []
    for k in np.argsort(-np.where(np.isfinite(snr), snr, -np.inf)):
        if len(stamps) == _FWHM_STARS:
            break
        if saturated[k]:
            continue
        d2 = (x - x[k]) ** 2 + (y - y[k]) ** 2
        d2[k] = np.inf
        if d2.min() < _ISOLATION_PX ** 2:
            continue
        stamp = psf_stamp(data, var, valid, x[k], y[k])
        if stamp is None or stamp[0].max() >= 0.5 * full_well:
            continue
        stamps.append(stamp)
    if len(stamps) < 5:
        return None
    lo, hi = 0.8, 8.0
    fwhm = best_fwhm(stamps, lo, hi)
    if not lo + 0.05 < fwhm < hi - 0.05:
        return None
    return float(fwhm * scale_arcsec)


class TrueCalibrationTask(Task):
    """The frame's WCS and zero point from the simulation (the raw frame's cards), in place of ReferenceRetrievalTask,
    LinearWCSSolverTask, SourceMatchingTask and TPVWCSSolverTask (module docstring)."""

    def __init__(self):
        super().__init__(takes=FullFrameImage, gives=FullFrameImage, requirements={"CPU": 1})

    def _run_impl(self, input_param: FullFrameImage) -> FullFrameImage:
        from astropy.io import fits
        from astropy.wcs import WCS

        meta = input_param.metadata
        header = fits.getheader(meta["filepath"])
        if "SIMZPAP" not in header or "CTYPE1" not in header:
            raise RuntimeError(f"{meta.get('filename')}: no simulation calibration cards (the engine writes them when "
                               f"the chain has {TYPE}; frames made without it must be rendered again)")
        w = WCS(header)
        input_param.linear_wcs = w.deepcopy()
        input_param.full_wcs = w
        for card, value in wcs_cards(w).items():
            meta[card] = value
        meta["coadd_magzp"] = float(header["SIMZPAP"])
        meta["magzp_band"] = str(header.get("FILTER", ""))
        meta["psf_magzp"] = float(header["SIMZPTOT"])
        scale = float(np.sqrt(abs(np.linalg.det(w.pixel_scale_matrix)))) * 3600.0
        # Calibration quality under the keys hdps's own fits write: the zero point is the simulation's exact one
        # (error 0); the WCS is fitted to the simulation's true mapping, not to stars (no TPV_NMAT: the packet's
        # numReferenceSources stays empty), its RMS residual as TPV_RMS (arcsec).
        meta["psf_magzp_err"] = 0.0
        if "SIMWCSM" in header:
            meta["TPV_RMS"] = float(header["SIMWCSM"]) * scale
        seeing = stellar_fwhm(input_param, scale)
        if seeing is not None:
            meta["coadd_seeing"] = seeing
        return input_param

    def configure(self, config=None) -> None:
        if config:
            raise ValueError(f"{TYPE} takes no settings: {sorted(config)}")


def register():
    """Make the Task known to hdps's build_task_list (the worker builds the chain by type name)."""
    from hdps.orchestrate import pipeline

    pipeline._ensure_registry()
    pipeline.TASK_REGISTRY[TYPE] = TrueCalibrationTask
