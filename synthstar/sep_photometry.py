"""SExtractor (SEP) photometry: what the engine measured with until 2026-10-19 (it now uses hdps's own source
detector, synthstar/photometry.py). Kept for the demo page's SEP port (demo/validate/) and the photometry offset
study (studies/photometry_offset/), which are about this method.

SEP (Barbary 2016; SExtractor's algorithms, Bertin & Arnouts 1996) on every exposure, with SExtractor's defaults:
- background: mesh 64 x 64 px, 3 x 3 median filter (BACK_SIZE 64, BACK_FILTERSIZE 3);
- detection: 1.5 sigma above background over >= 5 connected pixels after the default 3 x 3 convolution
  (DETECT_THRESH 1.5, DETECT_MINAREA 5), deblending 32 thresholds / 0.005 contrast;
- neighbours: other objects' pixels masked from each Kron radius and aperture via the segmentation map (as
  SExtractor's MASK_TYPE does for MAG_AUTO);
- positions: windowed centroids (XWIN_IMAGE / YWIN_IMAGE);
- photometry: MAG_AUTO, a Kron elliptical aperture of 2.5 Kron radii, at least 3.5 (PHOT_AUTOPARAMS 2.5, 3.5).
Saturated pixels are not masked (SExtractor keeps saturated stars; they are flagged).

Each exposure is calibrated on its own: catalogue stars (least-blended, unsaturated, with Gaia colours) matched to
their detections give the zero point with a linear colour term in BP-RP (3-sigma clipped). Catalogue stars
(variables included) are cross-matched to the nearest detection within 0.5 FWHM; none: no measurement in that
exposure.
"""

import numpy as np
import sep

# every setting an observation can change ([measure] in observe.py's file); these defaults are SExtractor's own
SETTINGS = dict(
    back_size=64,          # BACK_SIZE: background mesh (pixels)
    back_filtersize=3,     # BACK_FILTERSIZE: median filter of the mesh
    detect_thresh=1.5,     # DETECT_THRESH (sigma above the background rms)
    detect_minarea=5,      # DETECT_MINAREA (pixels)
    deblend_nthresh=32,    # DEBLEND_NTHRESH
    deblend_cont=0.005,    # DEBLEND_MINCONT
    kron_fact=2.5,         # PHOT_AUTOPARAMS: aperture = max(kron_fact x Kron radius, min_radius) in units of (a, b)
    min_radius=3.5,
    match_fwhm=0.5,        # cross-match radius (FWHM) between catalogue stars and detections
    n_calibrators=300,     # calibration stars per exposure (the least blended of the brightest 4x this many)
    colour_term=True,      # zero point a + b (BP-RP); false: a only
    clip_sigma=3.0,        # zero-point fit: reject calibrators beyond this many (MAD) sigma, refit
)


def settings(given=None):
    """SETTINGS with `given` applied; unknown keys refused."""
    given = dict(given or {})
    bad = sorted(set(given) - set(SETTINGS))
    if bad:
        raise SystemExit(f"[measure]: unknown keys {bad}; it takes {sorted(SETTINGS)}")
    return {**SETTINGS, **given}


def calibration_stars(x, y, cat_flux, n, k=300, edge=16, radius=8.0):
    """The k bright stars least contaminated by neighbours, away from the edge.

    Contamination of a star: catalogue flux of its neighbours, Gaussian-weighted by distance (sigma = radius/2
    pixels), over its own flux.
    """
    candidates = np.nonzero((x > edge) & (x < n - edge) & (y > edge) & (y < n - edge) & np.isfinite(cat_flux))[0]
    candidates = candidates[np.argsort(-cat_flux[candidates])][: 4 * k]  # the brightest
    contamination = []
    for i in candidates:
        d = np.hypot(x - x[i], y - y[i])
        near = (d > 0) & (d < 3 * radius)
        contamination.append(np.sum(cat_flux[near] * np.exp(-0.5 * (d[near] / (radius / 2)) ** 2)) / cat_flux[i])
    return candidates[np.argsort(contamination)[:k]]


def extract(image, fwhm, cfg=None):
    """SExtractor's defaults with MAG_AUTO as SExtractor measures it (see the module docstring); cfg: settings().

    Returns x, y (windowed), flux, flux error and peak of every detection."""
    c = cfg or SETTINGS
    data = np.ascontiguousarray(image, dtype=np.float64)
    bkg = sep.Background(data, bw=c["back_size"], bh=c["back_size"], fw=c["back_filtersize"], fh=c["back_filtersize"])
    sub = data - bkg.back()
    sep.set_extract_pixstack(10_000_000)
    sep.set_sub_object_limit(8192)
    obj, seg = sep.extract(sub, c["detect_thresh"], err=bkg.globalrms, minarea=c["detect_minarea"],
                           deblend_nthresh=c["deblend_nthresh"], deblend_cont=c["deblend_cont"], segmentation_map=True)
    ids = np.arange(1, len(obj) + 1, dtype=np.int32)
    x, y = np.asarray(obj["x"], float), np.asarray(obj["y"], float)
    xw, yw, fl = sep.winpos(sub, x, y, fwhm / 2.3548)
    good = fl == 0
    a, b, th = np.asarray(obj["a"], float), np.asarray(obj["b"], float), np.asarray(obj["theta"], float)
    th = np.clip(th, -np.pi / 2, np.pi / 2)
    kr, kf = sep.kron_radius(sub, x, y, a, b, th, 6.0, seg_id=ids, segmap=seg)
    # PHOT_AUTOPARAMS 2.5, 3.5: the elliptical aperture is max(2.5 x Kron radius, 3.5) in units of (a, b)
    r = np.maximum(c["kron_fact"] * np.nan_to_num(kr, nan=0.0), c["min_radius"])
    flux, ferr, ff = sep.sum_ellipse(sub, x, y, a, b, th, r, err=bkg.globalrms, subpix=1, seg_id=ids, segmap=seg)
    return np.where(good, xw, x), np.where(good, yw, y), np.asarray(flux, float), np.asarray(ferr, float), obj["peak"]


class SExtractorPhotometry:
    """measure(image) -> the targets' flux, sigma, mag, mag_err (nan where not detected) and saturation flags."""

    def __init__(self, x, y, cat_mag, cat_flux, n, fullwell, targets, colour, fwhm, cfg=None):
        self.cfg = settings(cfg)
        self.x, self.y = np.asarray(x, float), np.asarray(y, float)
        self.cat_mag, self.colour = np.asarray(cat_mag, float), np.asarray(colour, float)
        self.fullwell, self.targets, self.fwhm = fullwell, list(targets), fwhm
        picks = calibration_stars(self.x, self.y, np.asarray(cat_flux, float), n, k=self.cfg["n_calibrators"])
        self.cal = np.setdiff1d(picks[np.isfinite(self.colour[picks])], self.targets)

    def measure(self, image):
        x, y, flux, ferr, peak = extract(image, self.fwhm, self.cfg)
        sat_level = 0.98 * self.fullwell

        def nearest(i):  # standard cross-match: the nearest detection within match_fwhm x FWHM
            d = np.hypot(x - self.x[i], y - self.y[i])
            j = int(np.argmin(d))
            return j if d[j] <= self.cfg["match_fwhm"] * self.fwhm else -1

        # zero point with a colour term: mag + 2.5 log10(flux) = a + b (BP-RP), 3-sigma clipped
        cal_y, cal_c = [], []
        for i in self.cal:
            j = nearest(i)
            if j >= 0 and flux[j] > 0 and peak[j] < sat_level:
                cal_y.append(self.cat_mag[i] + 2.5 * np.log10(flux[j]))
                cal_c.append(self.colour[i])
        cal_y, cal_c = np.array(cal_y), np.array(cal_c)
        use = np.ones(cal_y.size, bool)
        a, b = np.median(cal_y), 0.0  # fewer than 5 calibration stars: no colour term
        for _ in range(10):
            if use.sum() < 5:
                break
            if self.cfg["colour_term"]:
                b, a = np.polyfit(cal_c[use], cal_y[use], 1)
            else:
                a = float(np.median(cal_y[use]))
            r = cal_y - (a + b * cal_c)
            new = np.abs(r) <= self.cfg["clip_sigma"] * 1.4826 * np.median(np.abs(r[use] - np.median(r[use])))
            if np.array_equal(new, use):
                break
            use = new
        c_t = self.colour[self.targets]
        zp = a + b * np.where(np.isfinite(c_t), c_t, np.median(cal_c[use]))

        j = np.array([nearest(t) for t in self.targets])
        f = np.where((j >= 0) & (flux[j] > 0), flux[j], np.nan)
        s = np.where(j >= 0, ferr[j], np.nan)
        return dict(flux=f, sigma=s, mag=zp - 2.5 * np.log10(f), mag_err=2.5 / np.log(10) * s / f,
                    saturated=(j >= 0) & (peak[j] >= sat_level))
