"""The edited star measured on each composed frame, with hdps's own code:

- photometry: hdps's source detector (SourceDetector, CPU mirror) on the composed science cut-out, the detection
  within 2" of the star; flux -> magnitude by the frame's cut-out zero point (the field pack calibrates it against
  hdps's own catalogue magnitudes of the field's stars in that frame, so the star is on hdps's scale);
- alerts: hdps's alert detection and features (hdps.alerts.features.detect_and_featurize) on the composed difference
  image around the star, every frame's candidate scored in one call by the real/bogus bundle
  (hdps.alerts.classifier.RFBundleClassifier) at its operating point; with no bundle, every candidate an alert.

The cut-outs carry no variance plane, so one is made to reproduce hdps's features: the difference image's robust noise
plus 0.3 of the science image's counts above sky, scaled so its normalised residual rms equals the one hdps recorded
for that frame (plane_rms_norm), on a 256 px window around the star (hdps calibrates its detection statistic per
256 px region), with the FWHM hdps used for the frame. On hdps's own candidates in a real run (186 of them, re-found
on the run's cut-outs) this gives the same accept or reject as hdps for 96%, median score difference 0.016; edge_dist
(hdps's masks are not in the cut-outs) and d_neg differ most.
"""
import threading
import time
from functools import lru_cache

import numpy as np

MATCH_ARCSEC = 2.0
FLUX_FLOOR = 1e-6  # hdps's detector writes a net-negative aperture sum as 1e-12; hdps drops <= 1e-6 (TransientOutputTask)
HALF_WINDOW = 128  # the window the alert features run on: 257 x 257 tile px, as hdps's 256 px calibration regions
SOURCE_VARIANCE = 0.3  # the share of the science image's counts in the variance plane (fitted to hdps's features)


_local = threading.local()


def detector():
    """hdps's source detector, one per thread (its buffers are its own)."""
    if not hasattr(_local, "detector"):
        from hdps.photometry import SourceDetector

        _local.detector = SourceDetector(detect_sigma=3.0, bg_region_size=64, max_sources=100000, flux_radius=4.0,
                                         matched_filter_width=4.0, centroid_n_iter=3)
    return _local.detector


@lru_cache(maxsize=4)
def classifier(bundle):
    from hdps.alerts.classifier import RFBundleClassifier

    return RFBundleClassifier(bundle) if bundle else None


def detect(science):
    """hdps's source detector on a cut-out: its source table (xcentroid, ycentroid, flux, flux_err, snr, ...)."""
    from hdps.core.images import FullFrameImage
    from hdps.core.telemetry import Telemetry

    data = np.nan_to_num(np.asarray(science, np.float32))
    image = FullFrameImage(data=data, metadata={}, timestamp=time.time(), telemetry=Telemetry(output_enabled=False))
    return detector().detect_sources(image).sources.table


def to_offsets(extent, shape, x, y):
    """Cut-out pixel positions to arcsec from the block centre."""
    ny, nx = shape
    return (extent[0] + (np.asarray(x) + 0.5) * (extent[1] - extent[0]) / nx,
            extent[2] + (np.asarray(y) + 0.5) * (extent[3] - extent[2]) / ny)


def _pixel(extent, shape, dx, dy):
    ny, nx = shape
    return ((dx - extent[0]) / (extent[1] - extent[0]) * nx - 0.5, (dy - extent[2]) / (extent[3] - extent[2]) * ny - 0.5,
            (extent[1] - extent[0]) / nx)


def photometry(composed, dx, dy, zp):
    """(mag, err, snr, saturated) of the star on the composed science cut-out; nan where it is not detected."""
    tab = detect(composed["science"])
    x, y, px = _pixel(composed["extent"], np.shape(composed["science"]), dx, dy)
    if len(tab) == 0 or zp is None or not np.isfinite(zp):
        return np.nan, np.nan, np.nan, False
    sep = np.hypot(np.asarray(tab["xcentroid"]) - x, np.asarray(tab["ycentroid"]) - y) * px
    k = int(np.argmin(sep))
    if sep[k] > MATCH_ARCSEC or not tab["flux"][k] > FLUX_FLOOR:
        return np.nan, np.nan, np.nan, False
    flux, err = float(tab["flux"][k]), float(tab["flux_err"][k])
    return zp - 2.5 * np.log10(flux), 1.0857 * err / flux, float(tab["snr"][k]), bool(tab["is_saturated"][k])


def window(extent, shape, dx, dy):
    """The (rows, cols) slices of the alert features' window around (dx, dy) arcsec, kept inside the cut-out."""
    x, y, _ = _pixel(extent, shape, dx, dy)
    ny, nx = shape
    size = min(2 * HALF_WINDOW + 1, ny, nx)
    x0 = int(np.clip(round(x) - size // 2, 0, nx - size))
    y0 = int(np.clip(round(y) - size // 2, 0, ny - size))
    return slice(y0, y0 + size), slice(x0, x0 + size)


def window_stats(cut, dx, dy):
    """(noise variance of the difference, sky of the science) in the window around (dx, dy) on a frame's own
    cut-outs: properties of hdps's frame, not of a star added to it."""
    ys, xs = window(cut["extent"], np.shape(cut["difference"]), dx, dy)
    D = np.asarray(cut["difference"], np.float64)[ys, xs]
    D = D[np.isfinite(D)]
    if D.size < 600:
        return np.nan, np.nan
    med = np.median(D)
    return float((1.4826 * np.median(np.abs(D - med))) ** 2), float(np.nanmedian(np.asarray(cut["science"])[ys, xs]))


def bright_star_mask(science, reference, threshold, base, growth):
    """hdps's bright-star detection mask (SFFTSubtractionTask._bright_star_mask, with mask_bright_stars): True where
    detection is suppressed. Pixels above threshold in the science or the reference, grown by base px; then each
    magnitude brighter in the reference grows its pixels' mask by growth px more."""
    from scipy.ndimage import maximum_filter

    def dilate(m, r):
        return maximum_filter(m, size=2 * int(r) + 1) if r > 0 else m

    bad = dilate((reference > threshold) | (science > threshold), base)
    step, k, peak = 10 ** 0.4, 1, float(np.nanmax(reference)) if np.isfinite(reference).any() else 0.0
    while growth > 0 and threshold * step ** k < peak and k <= 12:
        bad |= dilate(reference > threshold * step ** k, base + growth * k)
        k += 1
    return bad


def candidate(composed, dx, dy, fwhm_px, plane_rms=None, stats=None, mask=None):
    """hdps's alert features of the candidate within 2" of the star on the composed difference image, or None.
    fwhm_px, plane_rms: what hdps used and recorded for the frame (the field's candidates' fwhm_est and
    plane_rms_norm); stats: window_stats of the frame (computed here if not given); mask: (threshold, base, growth)
    of hdps's bright-star detection mask when the run used it. hdps compares the tiles as they reach it, sky
    included (its docstring says sky-subtracted; fitted on a real run: this way the mask covers an injected dwarf nova
    hdps never detected in every frame, and none of the 441 places hdps did make a candidate; sky-subtracted, the
    dwarf nova in only half the frames). The reference is taken as science - difference (the PSF-matched reference
    the difference was made with)."""
    from hdps.alerts.features import detect_candidates, extract_features, moffat_score_map, prod_score_map

    diff = np.asarray(composed["difference"], np.float64)
    sci = np.asarray(composed["science"], np.float64)
    x, y, px = _pixel(composed["extent"], diff.shape, dx, dy)
    ys, xs = window(composed["extent"], diff.shape, dx, dy)
    D, S = diff[ys, xs], sci[ys, xs]
    valid = np.isfinite(D)
    noise2, sky = stats if stats is not None else window_stats(composed, dx, dy)
    if valid.sum() < 600 or not np.isfinite(noise2):
        return None
    var = noise2 + SOURCE_VARIANCE * np.clip(np.nan_to_num(S - sky), 0.0, None)
    if mask is not None:  # hdps gives these pixels infinite variance: no detection there
        science = np.nan_to_num(S)
        valid &= ~bright_star_mask(science, science - np.nan_to_num(D), *mask)
        if valid.sum() < 600:
            return None
    if plane_rms is not None and np.isfinite(plane_rms) and plane_rms > 0:
        zn = np.where(valid, D, np.nan) / np.sqrt(var)
        var = var * (float(np.nanstd(zn[np.abs(zn) < 10])) / plane_rms) ** 2
    # hdps.alerts.features.detect_and_featurize, step by step, with the features of the star's candidate only (each
    # candidate's are computed on its own; the plane's statistics and the negative detections are the same)
    D32, var32 = np.where(valid, D, 0.0).astype(np.float32), var.astype(np.float32)
    SP = prod_score_map(D32, valid)
    SM = moffat_score_map(D32, var32, valid, fwhm_px)
    pos = detect_candidates(SP, valid, 3.0)
    if len(pos) == 0:
        return None
    sep = np.hypot(np.asarray(pos["x"]) + xs.start - x, np.asarray(pos["y"]) + ys.start - y) * px
    k = int(np.argmin(sep))
    if sep[k] > MATCH_ARCSEC:
        return None
    feats = extract_features(pos[k:k + 1], detect_candidates(-SP, valid, 3.0), D32, var32, valid, SP, SM, fwhm_px)
    return {name: float(np.asarray(v)[0]) for name, v in feats.items()}


def alerts(candidates, bundle, cut):
    """For each frame's candidate (or None): (score, alerted). One classifier call for all of them."""
    clf = classifier(bundle)
    rows = [c for c in candidates if c is not None]
    scores = np.ones(len(rows))
    threshold = 0.5
    if clf is not None and rows:
        scores = clf.score_features({n: np.array([c[n] for c in rows]) for n in clf.feature_names})
        threshold = clf.cut_for(cut)
    out, i = [], 0
    for c in candidates:
        if c is None:
            out.append((np.nan, False))
        else:
            out.append((float(scores[i]), bool(scores[i] >= threshold)))
            i += 1
    return out
