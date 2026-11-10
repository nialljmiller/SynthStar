"""An exposure on a telescope's sensor, as ArgusSim defines the array: where each star falls, and the image.

Geometry (all ArgusSim, argus_sim.coverage_exact): a position goes to the array frame as covering(apparent=True) puts
it (ICRS -> CIRS once per night at the night's mean exposure time; array-frame hour angle = Earth rotation angle at the
site + the exposure's tracking angle theta - RA; alt/az at the site latitude), then into the telescope's focal plane
as covering's _members tests membership: gnomonic coordinates along the camera's long axis (et, along the ring) and
short axis (er, radial) of Layout.basis. Sensor pixel = the sensor centre + the gnomonic coordinate x focal length /
pixel size (ArgusSim telescope.focal_length, pixel_size, n_pixels_x, n_pixels_y): x along the long axis, y along the
short. The field angle (the PSF depends on it) is the angle from the telescope's axis.

The image (ours: ArgusSim has no image renderer): each star's ArgusSim PSF stamp (telescope.psf_model: "zemax", the
ZemaxPSF at its field angle; "gaussian", the GaussianPSF, the same everywhere) at the exposure's seeing, moved so its
light centroid lands on the star, scaled by its electrons; plus sky + dark per pixel; Poisson; read noise; clipped at
the full well. With [image] psf_box above ArgusSim's stamp (28 px), the stamp is ArgusSim's same PSF made in a wider box
(Zemax: its array padded before the seeing, PSFs.fine; Gaussian: its stamp in the box's centre), with psf_halo the halo
measured on sky added, each star drawn out to where its light falls below psf_box_floor. With psf_model "gaussian" no
Zemax file is opened and argus_sim's ZemaxPSF is not imported (the public argus_sim has none).
"""
import numpy as np

SHIFTS = 16  # a stamp is moved to its star in steps of 1/SHIFTS px: the light centroid within 1/32 px of the star
CORE = 49  # px: with psf_box, the part of the PSF moved on the oversampled grid (PSFs.fine); the halo is drawn in place
TILE_PX = 900.0 / 1024.0  # hdps tile pixel (arcsec): the unit of the measured halo (psf_halo_<band>)
# arcsec: GaussianPSF's sample (its own default). ArgusSim's Survey passes survey.stamp_resolution (0.01), a 2800 x 2800
# grid and ~4 s per stamp; at 0.1 a stamp takes ~5 ms and its half-flux diameter differs from 0.05's by 0.06%.
GAUSSIAN_SAMPLE = 0.1


def array_frame(ra, dec, times):
    """(alt, az) in ArgusSim's array frame for positions ra, dec (deg) at each time, before the tracking angle:
    returns a function of theta_deg -> (alt, az) arrays of shape (n_times, n_positions)."""
    import astropy.units as u
    from argus_sim import c
    from argus_sim.coverage_exact import _hadec_to_altaz
    from astropy.coordinates import CIRS, SkyCoord
    from astropy.time import Time

    times = Time(times)
    nights = np.floor(times.mjd + c.observatory.longitude / 360.0 - 0.5)
    ra_c = np.empty((len(times), len(ra)))
    dec_c = np.empty_like(ra_c)
    for n in np.unique(nights):  # ICRS -> CIRS once per night, at the night's mean time (covering, apparent=True)
        these = nights == n
        mid = Time(np.mean(times.mjd[these]), format="mjd", scale="utc")
        cirs = SkyCoord(np.asarray(ra) * u.deg, np.asarray(dec) * u.deg).transform_to(CIRS(obstime=mid))
        ra_c[these], dec_c[these] = cirs.ra.deg, cirs.dec.deg
    era = times.earth_rotation_angle(longitude=c.observatory.longitude * u.deg).deg

    def at(theta_deg):
        ha = ((era + np.asarray(theta_deg))[:, None] - ra_c + 180.0) % 360.0 - 180.0
        alt, az = _hadec_to_altaz(ha, dec_c, c.observatory.latitude, np)
        return alt, az

    return at


def focal_plane(alt, az, ota, layout):
    """Gnomonic coordinates (u along the long axis, v along the short) in telescope `ota`'s focal plane, and the
    field angle (deg), for array-frame positions alt, az (any shape)."""
    from argus_sim.coverage_exact import _unit

    C, et, er = layout.basis()
    P = _unit(np.ravel(alt), np.ravel(az), np)
    d = P @ C[ota]
    u = (P @ et[ota]) / d
    v = (P @ er[ota]) / d
    angle = np.degrees(np.arccos(np.clip(d, -1.0, 1.0)))
    shape = np.shape(alt)
    return u.reshape(shape), v.reshape(shape), angle.reshape(shape)


def to_pixels(u, v):
    """Sensor pixel (x along the long axis, y along the short; 0-based, centres at integers) of gnomonic u, v; the
    camera's readout orientation (camera.READOUT_Y_SIGN: ArgusSim does not define it) sets the direction of y."""
    from argus_sim import c

    from synthstar import camera

    t = c.telescope
    f_px = t.focal_length / (t.pixel_size / 1000.0)
    return (t.n_pixels_x - 1) / 2.0 + u * f_px, (t.n_pixels_y - 1) / 2.0 + camera.READOUT_Y_SIGN * v * f_px


def _cirs_at(x, y, era, theta_deg, ota, layout):
    """(RA, Dec) CIRS (deg) at sensor pixel (x, y), for Earth rotation angle era (deg): the inverse of the array-frame
    transform -> focal_plane -> to_pixels."""
    from argus_sim import c

    from synthstar import camera

    t = c.telescope
    f_px = t.focal_length / (t.pixel_size / 1000.0)
    gu = (np.asarray(x, float) - (t.n_pixels_x - 1) / 2.0) / f_px
    gv = (np.asarray(y, float) - (t.n_pixels_y - 1) / 2.0) / (camera.READOUT_Y_SIGN * f_px)
    C, et, er = layout.basis()
    P = C[ota] + gu[..., None] * et[ota] + gv[..., None] * er[ota]
    P = P / np.linalg.norm(P, axis=-1, keepdims=True)
    alt, az = np.arcsin(np.clip(P[..., 2], -1, 1)), np.arctan2(P[..., 1], P[..., 0])
    lat = np.radians(c.observatory.latitude)
    dec = np.arcsin(np.clip(np.sin(lat) * np.sin(alt) + np.cos(lat) * np.cos(alt) * np.cos(az), -1, 1))
    ha = np.degrees(np.arctan2(-np.cos(alt) * np.sin(az), np.sin(alt) * np.cos(lat) - np.cos(alt) * np.sin(lat) * np.cos(az)))
    return (era + float(theta_deg) - ha) % 360.0, np.degrees(dec)


def sky_at(x, y, time, theta_deg, ota, layout):
    """(RA, Dec) ICRS (deg) at sensor pixel (x, y) in an exposure: the inverse of array_frame -> focal_plane ->
    to_pixels (ArgusSim's geometry; CIRS -> ICRS at the exposure time)."""
    import astropy.units as u
    from argus_sim import c
    from astropy.coordinates import CIRS, SkyCoord
    from astropy.time import Time

    when = Time(time)
    era = when.earth_rotation_angle(longitude=c.observatory.longitude * u.deg).deg
    ra_cirs, dec_cirs = _cirs_at(x, y, era, theta_deg, ota, layout)
    icrs = SkyCoord(ra_cirs * u.deg, dec_cirs * u.deg, frame=CIRS(obstime=when)).icrs
    return icrs.ra.deg, icrs.dec.deg


def window(x0, y0, side_px):
    """The readout window (x_origin, y_origin, width, height) of side side_px centred on sensor pixel (x0, y0) and
    kept on the sensor; side_px 0: the whole sensor."""
    from argus_sim import c

    nx, ny = c.telescope.n_pixels_x, c.telescope.n_pixels_y
    if not side_px:
        return 0, 0, nx, ny
    side = int(min(side_px, nx, ny))
    xo = int(np.clip(round(x0 - side / 2), 0, nx - side))
    yo = int(np.clip(round(y0 - side / 2), 0, ny - side))
    return xo, yo, side, side


def gaussian_psf(cfg):
    """ArgusSim's GaussianPSF for cfg, with the parameters ArgusSim's Survey gives it (psf_model "gaussian"): an optical
    Gaussian of telescope.spot_size (um, rms) in a survey.stamp_size stamp at telescope.plate_scale, sampled at
    GAUSSIAN_SAMPLE (not Survey's survey.stamp_resolution: see GAUSSIAN_SAMPLE). Imports only argus_sim.psf.GaussianPSF:
    no Zemax file is read."""
    from argus_sim.psf import GaussianPSF

    t = cfg.telescope
    return GaussianPSF(stamp_size=cfg.survey.stamp_size, stamp_resolution=GAUSSIAN_SAMPLE, plate_scale=t.plate_scale,
                       pixel_size=t.pixel_size, rms_spot_size=t.spot_size)


def gaussian_stamp(model, seeing, dy, dx):
    """GaussianPSF model's with_seeing (its Gaussian convolved with ArgusSim's Moffat seeing kernel, summed into stamp
    pixels, unit flux) with the Gaussian's centre moved by (dy, dx) stamp pixels from the stamp's centre ((n - 1) / 2:
    a pixel corner for ArgusSim's even stamp). The move is exact (the model is drawn about the centre it is given), not
    a Fourier shift: a Fourier shift of this ~2 px FWHM stamp misses ArgusSim's own moved stamp by up to 1.5% of the
    peak at 0.8" seeing. GaussianPSF's mx, my are in its grid's arcsec, and a stamp pixel is `downscale` grid samples of
    stamp_size / (N - 1) each (its linspace; it bins as if they were stamp_size / N), so its pixel is computed here as
    with_seeing bins: 1.0036 grid arcsec at the 1.0072" plate scale (its whole stamp, spot and seeing are 0.36% larger in
    pixels than at the plate scale: ArgusSim's model as it is)."""
    n = model.x.shape[0]
    pixel = int(np.round(model.plate_scale / (model.stamp_size / n))) * float(model.x[0, 1] - model.x[0, 0])
    return model.with_seeing(float(seeing), mx=float(dx) * pixel, my=float(dy) * pixel, return_fwhm=False)


class PSFs:
    """ArgusSim PSF stamps (unit flux) by (band, seeing, tabulated field angle, sub-pixel phase), made as needed and
    kept. The model is cfg.telescope.psf_model's:

    "zemax" (ArgusSim's default): ZemaxPSF.from_config(cfg), the Zemax PSFs at their tabulated field angles with the
    delivered-EE50 blur and the Moffat seeing; it opens telescope.psf_fits_path (zemax_psf_*.fits) and
    telescope.ee50_profile_path (ee50_delivered_profile_*.csv).

    "gaussian": ArgusSim's GaussianPSF (gaussian_psf): an optical Gaussian of telescope.spot_size um rms convolved with
    the Moffat seeing (GaussianPSF.with_seeing), the same at every field angle and in both bands (angles(): one angle,
    0). Only argus_sim.psf.GaussianPSF is imported and no file is opened, so it runs with the public argus_sim (no
    ZemaxPSF). Stamps are moved to their star by the Gaussian's own centre (gaussian_stamp: exact), not by a Fourier
    shift. The [image] options work as with Zemax, on the Gaussian stamp: psf_box above 28 puts the 28 px stamp in the
    box's centre (at the stamp's edge the Gaussian with Moffat seeing is 4e-7 of the star per pixel at 3" seeing,
    2e-8 at 2", 5e-18 at 1": no visible square), psf_halo adds the halo measured on sky about its centre, psf_wing fits its own outer profile (steep:
    the Moffat's r^-9.5) as for Zemax, psf_seeing_step and psf_box_floor as for Zemax. psf_halo_taper is not used: it
    rounds Zemax's square optics array, and the Gaussian has none. A new seeing costs ~5 ms per sub-pixel shift used
    (up to SHIFTS^2 = 256: ~1.3 s in a crowded window); psf_seeing_step lets exposures share them.

    wing: None, ArgusSim's stamp as it is (it ends at ArgusSim's Zemax grid, 28 x 28 px: a bright star sits in a
    visible square); or the [image] settings, the stamp continued by a power law (ours: ArgusSim has none). Between
    the psf_wing_fit radii the stamp's mean radial profile is a power law, A r^-alpha (slope -2.7 to -3.6 at 1-2"
    seeing, near diffraction's -3); a star is drawn as the stamp inside psf_wing_join[0], the law beyond
    psf_wing_join[1], a linear blend between, out to psf_wing_radius, the whole renormalised to unit flux there
    (law(): the stamp's share drops by the wing's). A star's wing is drawn only where it exceeds psf_wing_floor
    electrons per pixel (fainter wings are left out: below any noise).

    box: the [image] psf_box. Above ArgusSim's stamp_size the stamps are box x box px, from fine(): moved on ArgusSim's
    oversampled grid, then summed into detector pixels (a Fourier shift of the summed, undersampled stamp rings: in
    a wide box the ringing holds several % of the light); a star is drawn in the part of its stamp outside which its
    light is below psf_box_floor (half_widths). With [image] psf_halo, the delivered PSF's halo as measured on sky
    (psf_halo_<band>: A, r0, beta of A [1 + (r/r0)^2]^-beta per tile pixel, a share of the core's light) is added
    after the seeing; ArgusSim's optics stop at 14.5 px, the measured PSF does not."""

    def __init__(self, cfg, wing=None):
        self.gaussian = getattr(cfg.telescope, "psf_model", "zemax") == "gaussian"
        if self.gaussian:  # ArgusSim's GaussianPSF: no Zemax file, no ZemaxPSF import
            self.model = gaussian_psf(cfg)
        else:
            from argus_sim import ZemaxPSF

            self.model = ZemaxPSF.from_config(cfg)
        self.cache = {}
        self.seeings = []  # the seeings whose stamps the cache holds, oldest first
        self.centroids = {}
        self.wing = wing if wing and wing.get("psf_wing") else None
        self.laws = {}
        self.box = int((wing or {}).get("psf_box", self.model.stamp_size))
        self.floor = float((wing or {}).get("psf_box_floor", 0.5))
        self.halo = ({b: (wing or {})[f"psf_halo_{b}"] for b in ("beta", "rho")} if (wing or {}).get("psf_halo")
                     else None)
        self.taper = (wing or {}).get("psf_halo_taper", [10, 14])
        self.seeing_step = float((wing or {}).get("psf_seeing_step", 0.0))
        self.fines = {}
        self.envelopes = {}

    @property
    def wide(self):
        return self.box > self.model.stamp_size

    def quantized(self, seeing):
        """The seeing the PSF is made for: rounded to psf_seeing_step (0: as given), so exposures whose seeing differs
        by less reuse one set of stamps."""
        return round(float(seeing) / self.seeing_step) * self.seeing_step if self.seeing_step > 0 else float(seeing)

    def fine(self, band, seeing, angle_index):
        """The PSF's core: ArgusSim's ZemaxPSF.with_seeing at pixel phase 0, step for step up to its sum into detector
        pixels (the delivered-EE50 match, the Moffat seeing), on ArgusSim's oversampled grid (3-4 samples per pixel),
        with its Zemax array zero-padded first by whole detector pixels: the seeing and the EE50 blur then carry light
        past the array's 29 px instead of stopping (or reflecting) at its edge. Made in CORE px (or the box, if
        smaller): the optics end 14.5 px out (with psf_halo, tapered to psf_halo_taper's outer radius), and blurred they
        fall below 1e-13 of the star per pixel by 24 px. Returns (that array's FFT, the oversampling, the first
        detector pixel (y, x) of the core box about the phase-0 peak, as with_seeing cuts its stamp_size, and the
        core's light centroid (y, x) in it)."""
        key = (band, round(float(seeing), 3), int(angle_index))
        if key not in self.fines:
            from argus_sim.psf import _BAND_MAP, _match_ee50, _moffat_kernel_2d
            from scipy.signal import fftconvolve

            from synthstar.argus import BANDS

            m = self.model
            if not m._compact:
                raise ValueError("psf_box needs ArgusSim's compact Zemax PSF file (Array Size, Oversample columns)")
            fits_band = _BAND_MAP.get(BANDS.get(band, band), BANDS.get(band, band))
            angles, rows = m._band_index[fits_band]
            row = rows[angle_index]
            size, ds, spacing = int(m._array_sizes[row]), int(m._oversamples[row]), float(m._spacings[row])
            psf = np.array(m._hdul[1].data["PSF Array"][row][: size * size], dtype=np.float64).reshape(size, size)
            psf /= psf.sum()
            if self.halo is not None:  # the optics tapered to 0 on a circle inside their square; the halo carries on
                inner, outer = self.taper
                yy, xx = np.indices(psf.shape)
                r = np.hypot(yy - (psf * yy).sum(), xx - (psf * xx).sum()) / ds
                psf *= np.cos(0.5 * np.pi * np.clip((r - inner) / (outer - inner), 0.0, 1.0)) ** 2
                psf /= psf.sum()
            n = min(self.box, CORE)
            psf = np.pad(psf, ds * ((n + 16 - size // ds) // 2 + 1))
            profile = m.ee50_profile.get(fits_band) if m.ee50_profile else None
            if profile is not None and m._ee50_radii is not None:
                ee50_lower = 2.0 * float(m._ee50_radii[row])
                target_hfd = max(float(np.interp(float(angles[angle_index]), profile[0], profile[1])), ee50_lower)
                if m.ee50_max_px is not None:
                    target_hfd = min(target_hfd, m.ee50_max_px * m.pixel_size)
                if target_hfd > ee50_lower:
                    psf = _match_ee50(psf, spacing, target_hfd=target_hfd)
            if seeing > 0:
                kernel = _moffat_kernel_2d(seeing * (m.pixel_size / m.plate_scale) / spacing)
                psf = np.maximum(fftconvolve(psf, kernel, mode="same"), 0.0)
            psf /= psf.sum()
            binned = _binned(psf, ds)
            y0, x0 = (v - n // 2 for v in np.unravel_index(np.argmax(binned), binned.shape))
            core = binned[y0:y0 + n, x0:x0 + n]
            yy, xx = np.indices(core.shape)
            centroid = (float((core * yy).sum() / core.sum()), float((core * xx).sum() / core.sum()))
            self.fines[key] = (np.fft.fft2(psf), ds, (y0, x0), centroid)
        return self.fines[key]

    def half_widths(self, band, seeing, angle_index, electrons):
        """For stars of these electrons, the half-width (px, about the stamp's centre pixel) of the square each is drawn
        in: the smallest beyond which the PSF (phase 0) is below psf_box_floor / electrons everywhere, at least
        ArgusSim's own stamp (half of stamp_size) and at most the box."""
        key = (band, round(float(seeing), 3), int(angle_index))
        if key not in self.envelopes:
            p = self.stamp(band, seeing, angle_index, 0, 0)
            c = p.shape[0] // 2
            yy, xx = np.indices(p.shape)
            ring = np.maximum(np.abs(yy - c), np.abs(xx - c))
            peak = np.array([p[ring == k].max() for k in range(c + 1)])
            beyond = np.append(np.maximum.accumulate(peak[::-1])[::-1][1:], 0.0)  # brightest pixel outside half-width k
            self.envelopes[key] = beyond
        beyond = self.envelopes[key]
        k = np.searchsorted(-beyond, -self.floor / np.maximum(np.asarray(electrons, float), 1e-30), side="right")
        return np.clip(k, self.model.stamp_size // 2, len(beyond) - 1)

    def angles(self, band):
        if self.gaussian:  # the Gaussian is the same at every field angle: one, the axis
            return np.zeros(1)
        from synthstar.argus import BANDS

        return self.model.field_angles(BANDS.get(band, band))  # ArgusSim's band names (beta: "g")

    def law(self, band, seeing, angle_index):
        """(A, alpha, scale) of the wing at this seeing and field angle: the power law fitted to the stamp's mean
        radial profile (1 px annuli about its centroid) between the psf_wing_fit radii, and the factor that makes the
        blended stamp plus the wing out to psf_wing_radius unit flux."""
        key = (band, round(float(seeing), 3), int(angle_index))
        if key not in self.laws:
            p = self.stamp(band, seeing, angle_index, 0, 0).astype(float)
            yy, xx = np.indices(p.shape)
            r = np.hypot(yy - (p * yy).sum(), xx - (p * xx).sum())
            lo, hi = self.wing["psf_wing_fit"]
            radii = np.arange(lo, hi + 1)
            prof = np.array([p[(r >= a - 0.5) & (r < a + 0.5)].mean() for a in radii])
            slope, icept = np.polyfit(np.log(radii), np.log(prof), 1)
            A, alpha = float(np.exp(icept)), float(-slope)
            if alpha <= 2.0:
                raise ValueError(f"PSF wing ({band}, seeing {seeing}, angle {angle_index}): slope {-alpha:.2f}; its "
                                 f"light diverges unless it falls faster than r^-2")
            r1, r2 = self.wing["psf_wing_join"]
            R = float(self.wing["psf_wing_radius"])
            w = np.clip((r2 - r) / (r2 - r1), 0.0, 1.0)
            law = A * np.maximum(r, r1) ** -alpha
            square = (w * p + (1 - w) * law).sum()
            beyond = 2 * np.pi * A * (r2 ** (2 - alpha) - R ** (2 - alpha)) / (alpha - 2)  # r2 < r < R, a disc
            beyond -= law[r >= r2].sum()  # less its part inside the square, counted above
            self.laws[key] = (A, alpha, float(1.0 / (square + beyond)))
        return self.laws[key]

    def wing_radius(self, band, seeing, angle_index, electrons):
        """How far out a star of these electrons is drawn: where its wing falls to psf_wing_floor (e-/px), at most
        psf_wing_radius; 0 when it is below the floor already at the join."""
        A, alpha, scale = self.law(band, seeing, angle_index)
        r = (float(electrons) * scale * A / float(self.wing["psf_wing_floor"])) ** (1.0 / alpha)
        return 0.0 if r <= self.wing["psf_wing_join"][0] else min(r, float(self.wing["psf_wing_radius"]))

    def stamp(self, band, seeing, angle_index, iy, ix):
        """ArgusSim's stamp at pixel phase 0, moved by (iy, ix) / SHIFTS px, unit flux (with a wing, the PSF inside the
        join radius is this x law()'s scale). Zemax: moved by a Fourier shift, not by ArgusSim's pixel_phase: that steps
        by its 1/3 px oversampling (int(phase x 3)) and re-cuts the stamp about its peak, so the light does not move by
        the phase asked for (phases 0 and 0.25 give the same stamp). Gaussian: the Gaussian's centre moved
        (gaussian_stamp), the light centroid at (13.5, 13.5) + (iy, ix) / SHIFTS in its 28 px stamp. The stamps of the
        last few seeings are kept."""
        seeing = round(float(seeing), 3)
        key = (band, seeing, int(angle_index), int(iy), int(ix))
        if key not in self.cache:
            if seeing not in self.seeings:
                self.seeings.append(seeing)
                if len(self.seeings) > 3:
                    old = self.seeings.pop(0)
                    self.cache = {k: v for k, v in self.cache.items() if k[1] != old}
                    self.fines = {k: v for k, v in self.fines.items() if k[1] != old}
            if self.wide:  # the core moved on the oversampled grid, summed into pixels; the halo drawn where it lands
                if self.gaussian:  # the Gaussian stamp moved by its centre; its phase-0 light centroid is its centre
                    core = gaussian_stamp(self.model, seeing, iy / SHIFTS, ix / SHIFTS)
                    n = core.shape[0]
                    cy = cx = (n - 1) / 2.0
                else:
                    from scipy.ndimage import fourier_shift

                    f, ds, (by, bx), (cy, cx) = self.fine(band, seeing, angle_index)
                    if iy or ix:
                        f = fourier_shift(f, (iy * ds / SHIFTS, ix * ds / SHIFTS))
                    n = min(self.box, CORE)
                    core = _binned(np.maximum(np.fft.ifft2(f).real, 0.0), ds)[by:by + n, bx:bx + n]
                o = (self.box - n) // 2
                p = np.zeros((self.box, self.box))
                p[o:o + n, o:o + n] = core / core.sum()
                if self.halo is not None:  # per tile pixel (a share of the core's light) -> per detector pixel
                    A, r0, beta = self.halo[band]
                    tile = self.model.plate_scale / TILE_PX  # tile pixels per detector pixel
                    yy, xx = np.indices(p.shape)
                    r = np.hypot(yy - (o + cy + iy / SHIFTS), xx - (o + cx + ix / SHIFTS)) * tile
                    p += A * (1.0 + (r / r0) ** 2) ** -beta * tile ** 2
            elif self.gaussian:
                p = gaussian_stamp(self.model, seeing, iy / SHIFTS, ix / SHIFTS)
            elif iy or ix:
                from scipy.ndimage import fourier_shift

                p = self.stamp(band, seeing, angle_index, 0, 0).astype(float)
                p = np.fft.ifft2(fourier_shift(np.fft.fft2(p), (iy / SHIFTS, ix / SHIFTS))).real
            else:
                from synthstar.argus import BANDS

                angle = self.angles(band)[angle_index]
                p, _ = self.model.with_seeing(seeing=seeing, band=BANDS.get(band, band), field_angle_deg=float(angle),
                                              pixel_phase=(0.0, 0.0))
            p = np.maximum(p, 0.0)
            self.cache[key] = (p / p.sum()).astype(np.float32)
        return self.cache[key]

    def centroid(self, band, seeing, angle_index):
        """(y, x) of the light centroid of stamp(band, seeing, angle_index, 0, 0) (px from its first pixel): where a
        star is, the stamp's light centroid lands. It moves with the field angle (ArgusSim's ZemaxPSF: -0.16 to +0.26 px
        along x between 0 and 2 deg), so each angle's own."""
        key = ("centroid", band, round(float(seeing), 3), int(angle_index))
        if key not in self.centroids:
            p = self.stamp(band, seeing, angle_index, 0, 0).astype(float)
            yy, xx = np.indices(p.shape)
            self.centroids[key] = (float((p * yy).sum()), float((p * xx).sum()))
        return self.centroids[key]


def _binned(a, ds):
    """An oversampled array summed into ds x ds blocks (detector pixels), as ArgusSim's with_seeing does."""
    ny, nx = a.shape[0] // ds, a.shape[1] // ds
    return a[: ny * ds, : nx * ds].reshape(ny, ds, nx, ds).sum(axis=(1, 3))


def expected(x, y, electrons, field_angle, band, seeing, background, win, psfs):
    """The mean image of one exposure in the readout window win = (x_origin, y_origin, width, height): stars at
    sensor pixels x, y with `electrons` each (their PSFs at their field angles, with psfs' wing if it has one), plus
    `background` electrons per pixel (sky + dark). Electrons, float32, height x width."""
    seeing = psfs.quantized(seeing)
    xo, yo, w, h = win
    mean = np.full((h, w), float(background), np.float32)
    angles = psfs.angles(band)
    ai = np.abs(field_angle[:, None] - angles[None, :]).argmin(axis=1)  # nearest tabulated angle (as ZemaxPSF)
    lx, ly = x - xo, y - yo
    n0 = psfs.stamp(band, seeing, 0, 0, 0).shape[0]
    reach = np.full(len(lx), float(n0))  # how far from the window a star still lands light in it
    if psfs.wing is not None:
        radius = np.array([psfs.wing_radius(band, seeing, a, e) if e > 0 else 0.0 for a, e in zip(ai, electrons)])
        reach = np.maximum(reach, radius)
    if psfs.wide:  # each star's square: out to where its light falls below psf_box_floor
        half = np.zeros(len(lx), int)
        for a in np.unique(ai):
            half[ai == a] = psfs.half_widths(band, seeing, a, electrons[ai == a])
        reach = half + 2.0
    inside = (lx > -reach) & (lx < w + reach) & (ly > -reach) & (ly < h + reach) & (electrons > 0)
    for k in np.nonzero(inside)[0]:
        # the stamp's origin and its shift, so the stamp's light centroid lands on the star
        cy, cx = psfs.centroid(band, seeing, ai[k])
        tx, ty = lx[k] - cx, ly[k] - cy
        ox, oy = int(np.floor(tx)), int(np.floor(ty))
        ix, iy = int(round((tx - ox) * SHIFTS)), int(round((ty - oy) * SHIFTS))
        if ix == SHIFTS:
            ox, ix = ox + 1, 0
        if iy == SHIFTS:
            oy, iy = oy + 1, 0
        p = psfs.stamp(band, seeing, ai[k], iy, ix)
        if psfs.wide:
            c, hk = p.shape[0] // 2, int(half[k])
            p, ox, oy = p[c - hk:c + hk + 1, c - hk:c + hk + 1], ox + c - hk, oy + c - hk
        n = p.shape[0]
        scale = 1.0 if psfs.wing is None else psfs.law(band, seeing, ai[k])[2]
        if psfs.wing is None or not radius[k]:  # the stamp alone (a wing below the floor is left out)
            x0, x1, y0, y1 = max(ox, 0), min(ox + n, w), max(oy, 0), min(oy + n, h)
            if x1 > x0 and y1 > y0:
                mean[y0:y1, x0:x1] += electrons[k] * scale * p[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
            continue
        # the stamp blended into the wing, both about the star, out to its radius
        A, alpha, _ = psfs.law(band, seeing, ai[k])
        r1, r2 = psfs.wing["psf_wing_join"]
        R = radius[k]
        x0, x1 = max(min(ox, int(np.floor(lx[k] - R))), 0), min(max(ox + n, int(np.ceil(lx[k] + R)) + 1), w)
        y0, y1 = max(min(oy, int(np.floor(ly[k] - R))), 0), min(max(oy + n, int(np.ceil(ly[k] + R)) + 1), h)
        if x1 <= x0 or y1 <= y0:
            continue
        r = np.hypot(np.arange(y0, y1)[:, None] - ly[k], np.arange(x0, x1)[None, :] - lx[k])
        weight = np.clip((r2 - r) / (r2 - r1), 0.0, 1.0)
        patch = (1.0 - weight) * A * np.maximum(r, r1) ** -alpha * (r <= R)
        sy0, sy1, sx0, sx1 = max(oy, y0), min(oy + n, y1), max(ox, x0), min(ox + n, x1)
        if sx1 > sx0 and sy1 > sy0:
            patch[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] += (weight[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
                                                           * p[sy0 - oy:sy1 - oy, sx0 - ox:sx1 - ox])
        mean[y0:y1, x0:x1] += (electrons[k] * scale * patch).astype(np.float32)
    return mean


def render(x, y, electrons, field_angle, band, seeing, background, win, psfs, rng, readnoise, fullwell):
    """One exposure in the readout window win = (x_origin, y_origin, width, height): expected(), then Poisson, read
    noise, clipped at the full well. Returns the image (electrons, float32, height x width)."""
    mean = expected(x, y, electrons, field_angle, band, seeing, background, win, psfs)
    image = rng.poisson(mean).astype(np.float32)
    np.minimum(image, fullwell, out=image)
    image += rng.normal(0.0, readnoise, image.shape).astype(np.float32)
    return image


class Projector:
    """Every star's sensor position in every exposure, as ArgusSim's geometry places it (array_frame -> focal_plane
    -> to_pixels): ICRS -> CIRS once per night (at the night's mean exposure time, as covering), then per exposure
    the array-frame transform with its tracking angle and its telescope's axes."""

    def __init__(self, ra, dec, times, layout):
        import astropy.units as u
        from argus_sim import c
        from astropy.coordinates import CIRS, SkyCoord
        from astropy.time import Time

        self.layout = layout
        self.times = Time(times)
        nights = np.floor(self.times.mjd + c.observatory.longitude / 360.0 - 0.5)
        self.night = np.unique(nights, return_inverse=True)[1]
        self.mid = [Time(np.mean(self.times.mjd[self.night == n]), format="mjd", scale="utc")
                    for n in range(self.night.max() + 1)]
        self.cirs = []
        for mid in self.mid:
            cirs = SkyCoord(np.asarray(ra) * u.deg, np.asarray(dec) * u.deg).transform_to(CIRS(obstime=mid))
            self.cirs.append((cirs.ra.deg, cirs.dec.deg))
        self.era = self.times.earth_rotation_angle(longitude=c.observatory.longitude * u.deg).deg

    def _place(self, ra_c, dec_c, j, theta_deg, ota):
        from argus_sim import c
        from argus_sim.coverage_exact import _hadec_to_altaz

        ha = (self.era[j] + float(theta_deg) - ra_c + 180.0) % 360.0 - 180.0
        alt, az = _hadec_to_altaz(ha, dec_c, c.observatory.latitude, np)
        u, v, angle = focal_plane(alt, az, int(ota), self.layout)
        x, y = to_pixels(u, v)
        return x, y, angle

    def pixels(self, j, theta_deg, ota):
        """Sensor x, y (px) and field angle (deg) of every star in exposure j."""
        ra_c, dec_c = self.cirs[self.night[j]]
        return self._place(ra_c, dec_c, j, theta_deg, ota)

    def wcs(self, j, theta_deg, ota, origin, grid=(17, 11)):
        """Exposure j's true WCS in the pixels of a readout window whose first pixel is sensor pixel origin (x0, y0):
        a TAN projection about the telescope's optical axis. Sky points over the whole sensor are placed by the same
        transform as the stars (CIRS at the same epoch); the tangent point is the axis, on the sky and on the sensor
        (CRVAL, CRPIX), and the CD matrix is the linear least-squares map from their pixel offsets to their gnomonic
        coordinates about it (no iterative fit: astropy's fit_wcs_from_points converged to wrong solutions for some
        telescopes, 2026-10-26). The geometry is gnomonic about the axis in the apparent (CIRS) frame, so TAN is exact
        there; CIRS -> ICRS adds differential aberration, a few milli-pixels across the sensor. Returns (astropy WCS,
        largest residual over the points, their RMS residual; px)."""
        import astropy.units as u
        from argus_sim import c
        from astropy.coordinates import CIRS, SkyCoord
        from astropy.wcs import WCS

        t = c.telescope
        frame = CIRS(obstime=self.mid[self.night[j]])
        gx, gy = np.meshgrid(np.linspace(0, t.n_pixels_x - 1, grid[0]), np.linspace(0, t.n_pixels_y - 1, grid[1]))
        gx = np.r_[(t.n_pixels_x - 1) / 2.0, gx.ravel()]  # first: the optical axis, the tangent point
        gy = np.r_[(t.n_pixels_y - 1) / 2.0, gy.ravel()]
        ra_c, dec_c = _cirs_at(gx, gy, self.era[j], theta_deg, int(ota), self.layout)
        sky = SkyCoord(ra_c * u.deg, dec_c * u.deg, frame=frame).icrs
        back = sky.transform_to(frame)  # the stars' path: ICRS -> CIRS -> the sensor
        x, y, _ = self._place(back.ra.deg, back.dec.deg, j, theta_deg, ota)
        x, y = x - origin[0], y - origin[1]
        ra, dec = sky.ra.rad, sky.dec.rad
        ra0, dec0 = ra[0], dec[0]
        cos_c = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0)
        xi = np.degrees(np.cos(dec) * np.sin(ra - ra0) / cos_c)  # gnomonic, east
        eta = np.degrees((np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)) / cos_c)  # north
        offsets = np.c_[x - x[0], y - y[0]]
        cd = np.linalg.lstsq(offsets, np.c_[xi, eta], rcond=None)[0].T
        w = WCS(naxis=2)
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.crval = [np.degrees(ra0), np.degrees(dec0)]
        w.wcs.crpix = [x[0] + 1.0, y[0] + 1.0]  # FITS pixels count from 1
        w.wcs.cd = cd
        w.wcs.radesys = "ICRS"
        w.wcs.set()
        px, py = w.world_to_pixel(sky)
        r = np.hypot(px - x, py - y)
        return w, float(np.max(r)), float(np.sqrt(np.mean(r**2)))
