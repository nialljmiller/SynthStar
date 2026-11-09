"""Stage 2 — the image maker: a scene (stars, night, dust, placements; written by observe.py) rendered to raw Argus
camera frames. Holds the scene and its stars, their per-exposure electron rates and the observing conditions, the
detector Scene, and the raw-frame writer. Reads the star library (sed.h5) and ArgusSim's instrument model; writes raw
FITS. This is the unit destined for argus_sim/argus_image. Nothing here imports hdps or the measurement/movie layer."""
import copy
import csv
import json
import warnings
from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
import numpy as np
from scipy import sparse

# (ZemaxPSF is imported only where psf_model is "zemax" (Scene): the public argus_sim has none)
from argus_sim import ABPhot, SpectralSky, SystemThroughput
from argus_sim import c as ARGUS_CONFIG
from argus_sim import skymap_shim as sh
from argus_sim.observatory import Observatory, seasonal_seeing_params, seeing_at_airmass

from mesa_argus import dust
from mesa_argus.argus import BANDS, FLAM, system_response
from mesa_argus.seds import Templates
from mesa_argus.engine import state
from mesa_argus.engine.state import MINIPIX_PX, SKY_PX, MARGIN, AIRMASS_GRID, PSF_PHASES, FIELDS, use_field, set_block, block_label
from mesa_argus.engine.models import RSPStar, TransientSource

_REPO = Path(__file__).resolve().parents[2]  # repo root


def block_offsets(ra, dec):
    """Offsets (arcsec) from the block centre in the HDPS tile's tangent-plane pixel frame."""
    tile_ra, tile_dec = (float(v[0]) for v in sh.tile_center(np.array([state.TILE])))
    cd, sd = np.cos(np.radians(dec)), np.sin(np.radians(dec))
    ctd, std = np.cos(np.radians(tile_dec)), np.sin(np.radians(tile_dec))
    dra = np.radians(ra - tile_ra)
    den = std * sd + ctd * cd * np.cos(dra)
    xi = np.degrees(cd * np.sin(dra) / den)
    eta = np.degrees((ctd * sd - std * cd * np.cos(dra)) / den)
    crpix = sh.SEGMENT_PIXEL_SIZE / 2.0 + 0.5
    px, py = -xi / sh.PIXEL_SCALE_DEG + crpix - 0.5, eta / sh.PIXEL_SCALE_DEG + crpix - 0.5
    return (px - (state.BLOCK[0] + 0.5) * MINIPIX_PX) * SKY_PX, (py - (state.BLOCK[1] + 0.5) * MINIPIX_PX) * SKY_PX


def column(t, name):
    """A table column as floats, masked entries as nan (np.asarray would make them 0)."""
    c = t[name]
    return np.asarray(c.filled(np.nan) if hasattr(c, "filled") else c, float)


def block_centre_label():
    """The block centre as display text, e.g. 13ʰ42ᵐ11.5ˢ +28°22′30″."""
    tile_ra, tile_dec = (float(v[0]) for v in sh.tile_center(np.array([state.TILE])))
    crpix = sh.SEGMENT_PIXEL_SIZE / 2.0 + 0.5
    px, py = (state.BLOCK[0] + 0.5) * MINIPIX_PX, (state.BLOCK[1] + 0.5) * MINIPIX_PX
    xi, eta = -(px - crpix + 0.5) * sh.PIXEL_SCALE_DEG, (py - crpix + 0.5) * sh.PIXEL_SCALE_DEG
    centre = SkyCoord(tile_ra * u.deg, tile_dec * u.deg).spherical_offsets_by(xi * u.deg, eta * u.deg)
    h, m, s = centre.ra.hms
    sign, d, dm, ds = centre.dec.signed_dms
    return f"{int(h):02d}ʰ{int(m):02d}ᵐ{s:04.1f}ˢ {'+' if sign > 0 else '−'}{int(d):02d}°{int(dm):02d}′{ds:02.0f}″"


def star_catalogue():
    """The field's star list: Gaia DR3, or a synthetic field's (a real field's stars at random positions)."""
    return state.DATA / ("synthetic_stars.ecsv" if FIELDS[state.FIELD].get("synthetic") else "gaia_dr3_core.ecsv")


def load_followed():
    """A hand-made field's followed stars (data/<field>/followed.json); {} for a scene."""
    path = state.DATA / "followed.json"
    return json.loads(path.read_text()) if path.exists() else {}


def load_placed():
    """The stars placed by observe.py (sources.json), in order: real Gaia stars (positive ids) and placed stars and
    transients (-1, -2, ...), each with its SED file, position, timing and dust."""
    path = state.DATA / "sources.json"
    return json.loads(path.read_text()) if path.exists() else []


def engine_of(v):
    """What computed a followed star's light (a load_pulsators entry)."""
    if "transient" in v:
        return v["transient"].get("engine", v["transient"].get("generator", "sncosmo"))
    if "placed" in v:
        return v["placed"]["engine"]
    return "PHOEBE" if v["kind"] == "EB" else "MESA RSP"


def load_scene():
    """Every star near the block: the catalogue, plus the placed stars (appended at their own offsets). Columns: Source,
    RA, Dec, Gmag, BP-RP, dx, dy (arcsec from the block centre), in_block, class (Gaia's variability class)."""
    stars = Table.read(star_catalogue())
    placed = [v for v in load_placed() if v["source"] < 0]  # real Gaia stars are in the catalogue already
    if placed:
        extra = Table({"Source": [v["source"] for v in placed], "RA_ICRS": [np.nan] * len(placed),
                       "DE_ICRS": [np.nan] * len(placed)})
        if any(v["timing"] == "periodic" for v in placed):  # periodic: the catalogue G and colour of its mean; transient: none
            extra["Gmag"] = [v.get("G", np.nan) for v in placed]
            extra["BP-RP"] = [v.get("bp_rp", np.nan) for v in placed]
        keep = [c for c in ("Source", "RA_ICRS", "DE_ICRS", "Gmag", "BP-RP", "distance_pc", "ebv_own", "ebv") if c in stars.colnames]
        stars = vstack([stars[keep], extra])
    dx, dy = block_offsets(column(stars, "RA_ICRS"), column(stars, "DE_ICRS"))
    if placed:
        dx[-len(placed):] = [v["dx"] for v in placed]
        dy[-len(placed):] = [v["dy"] for v in placed]
    near = (np.abs(dx) <= state.AREA_HALF + MARGIN) & (np.abs(dy) <= state.AREA_HALF + MARGIN)
    has_light = np.isfinite(column(stars, "Gmag")) | (np.asarray(stars["Source"]).astype(np.int64) < 0)
    stars = stars[near & has_light]
    stars["dx"], stars["dy"] = dx[near & has_light], dy[near & has_light]
    stars["in_block"] = (np.abs(stars["dx"]) <= state.HALF) & (np.abs(stars["dy"]) <= state.HALF)
    if FIELDS[state.FIELD].get("synthetic"):
        stars["class"] = [""] * len(stars)
        return stars
    classifier = Table.read(state.DATA / "gaia_classifier.ecsv")
    gaia_class = dict(zip(np.asarray(classifier["Source"]).astype(np.int64), np.asarray(classifier["Class"])))
    stars["class"] = [str(gaia_class.get(int(s), "")) for s in np.asarray(stars["Source"]).astype(np.int64)]
    return stars


def load_pulsators():
    """{source: dict(kind, period, epoch (BJD of maximum), amp, mean mags)}: Gaia's RR Lyrae and Cepheids, the
    ephemerides observe.py adds (HADS, binaries), and the placed stars. Used to phase the models, never as light curves."""
    out = {}
    for name, family in (("gaia_rrlyrae.ecsv", "RR"), ("gaia_cepheid.ecsv", "CEP")):
        if not (state.DATA / name).exists():
            continue
        t = Table.read(state.DATA / name)
        for r in t:
            pf = float(r["PF"]) if np.isfinite(float(r["PF"])) and float(r["PF"]) > 0 else np.nan
            p1 = float(r["P1O"]) if "P1O" in t.colnames and np.isfinite(float(r["P1O"])) and float(r["P1O"]) > 0 else np.nan
            period = pf if np.isfinite(pf) else p1
            if not np.isfinite(period) or not np.isfinite(float(r["EpochG"])):
                continue
            kind = "RRab" if family == "RR" and np.isfinite(pf) else ("RRc" if family == "RR" else "Type II Cepheid")
            if family == "CEP" and "Class" in t.colnames:  # Gaia's own classification
                klass, subclass, mode = str(r["Class"]), str(r["SubClass"]), str(r["ModeClass"])
                if klass == "DCEP" and mode in ("FUNDAMENTAL", "FIRST_OVERTONE"):
                    kind = "DCEP F" if mode == "FUNDAMENTAL" else "DCEP 1O"
                    period = pf if mode == "FUNDAMENTAL" else p1
                elif klass == "T2CEP":
                    kind = {"BL_HER": "BL Her", "W_VIR": "W Vir", "RV_TAU": "RV Tau"}.get(subclass, "Type II Cepheid")
                else:
                    kind = f"{klass} {mode}"
            if family == "RR" and np.isfinite(pf) and np.isfinite(p1):
                kind = "RRd"
            amp = {b: float(r[f"ptp{b}"]) for b in ("G", "BP", "RP")}
            if "BPmagavg" in t.colnames:
                mean = {b: float(r[f"{b}magavg"]) for b in ("G", "BP", "RP")}
            else:
                mean = {"G": float(r["Gmagavg"])}
            out[int(r["Source"])] = dict(period=period, epoch=float(r["EpochG"]) + 2455197.5, kind=kind, amp=amp, mean=mean)
    if (state.DATA / "extra_pulsators.json").exists():  # ephemerides from other catalogues (HADS from ASAS-SN; binaries)
        for source, v in json.loads((state.DATA / "extra_pulsators.json").read_text()).items():
            out[int(source)] = dict(period=v["period"], epoch=v["epoch_bjd"], kind=v["kind"], amp={"G": v["amp"]},
                                    mean={"G": v["G"]})
    for v in load_placed():  # placed stars: -1, -2, ...; a transient's epoch is its t0
        if v["source"] > 0:  # a real Gaia star: phased from the Gaia tables above
            continue
        if v["timing"] == "transient":
            out[v["source"]] = dict(period=np.nan, epoch=v["t0_bjd"], kind=v["kind"], amp={"G": np.nan}, mean={}, transient=v)
        else:
            out[v["source"]] = dict(period=v["period"], epoch=v["epoch_bjd"], kind=v["kind"], amp={"G": v["amp_G"]},
                                    mean={"G": v["G"]}, placed=v)
    return out


def pulsator_title(follow, pulsators, source):
    """What is followed, for the movie's title: e.g. '1 BL Her: MESA RSP · 1 eclipsing binary: PHOEBE'."""
    entries = [pulsators[int(source[k])] for k in follow]
    catalogued = [v["kind"] for v in entries if "transient" not in v and "placed" not in v]
    n_binaries = sum(kind == "EB" for kind in catalogued)
    pulsating = [kind for kind in catalogued if kind != "EB"]
    parts = []
    if pulsating:
        if all(kind.startswith("RR") for kind in pulsating):
            parts.append(f"{len(pulsating)} RR Lyrae: MESA RSP")
        else:
            parts.append(f"{len(pulsating)} pulsator{'s' if len(pulsating) != 1 else ''} "
                         f"({', '.join(dict.fromkeys(pulsating))}): MESA RSP")
    if n_binaries:
        parts.append(f"{n_binaries} eclipsing binar{'ies' if n_binaries != 1 else 'y'}: PHOEBE")
    placed = [v for v in entries if "placed" in v]
    if placed:
        groups = {}
        for v in placed:
            groups[(v["kind"], engine_of(v))] = groups.get((v["kind"], engine_of(v)), 0) + 1
        parts.append(", ".join(f"{n} {kind}: {engine}" for (kind, engine), n in groups.items()))
    transients = [v for v in entries if "transient" in v]
    if transients:
        parts.append(f"{len(transients)} transient{'s' if len(transients) != 1 else ''}: "
                     + ", ".join(dict.fromkeys(f"{v['kind']} ({engine_of(v)})" for v in transients)))
    return " · ".join(parts) or "no modelled variables"


def choose_followed(stars, pulsators, n):
    """Row indices of the followed stars. A scene: exactly the stars placed (real ones by amplitude, then the placed
    ones in order). A hand-made field: two RRc, then the largest-amplitude RRab, up to n."""
    source = np.asarray(stars["Source"]).astype(np.int64)
    in_block = np.asarray(stars["in_block"])
    if FIELDS[state.FIELD].get("followed_only"):
        chosen = {v["source"] for v in load_followed().values()} | {v["source"] for v in load_placed() if v["source"] > 0}
        real = [k for k in range(len(stars)) if int(source[k]) in chosen]
        placed = sorted((k for k in range(len(stars)) if int(source[k]) < 0), key=lambda k: -int(source[k]))
        return sorted(real, key=lambda k: -pulsators[int(source[k])]["amp"]["G"]) + placed
    exclude = FIELDS[state.FIELD].get("exclude", set())
    candidates = [k for k in range(len(stars)) if in_block[k] and int(source[k]) in pulsators and int(source[k]) not in exclude]
    by_amplitude = sorted(candidates, key=lambda k: -pulsators[int(source[k])]["amp"]["G"])
    kind = {k: pulsators[int(source[k])]["kind"] for k in candidates}
    follow = [k for k in by_amplitude if kind[k] == "RRc"][:2]
    follow += [k for k in by_amplitude if kind[k] == "RRab"][: max(0, n - len(follow))]
    return sorted(follow, key=lambda k: -pulsators[int(source[k])]["amp"]["G"])


def night_schedule(date, max_airmass, cfg, start=None, end=None):
    """Every exposure of the night at the field, as ArgusSim schedules it (mesa_argus.night.exposures), in the UT
    window: mid-exposure UTC times, BJD_TDB, airmass, altitude, azimuth."""
    from mesa_argus import night

    times, airmass, alt, az = night.exposures(state.CENTRE.ra.deg, state.CENTRE.dec.deg, date, start=start, end=end,
                                              max_airmass=max_airmass, cfg=cfg)
    return times, night.bjd_tdb(times, state.CENTRE.ra.deg, state.CENTRE.dec.deg, cfg=cfg), airmass, alt, az


def survey_schedule(band, cfg):
    """The survey's exposures of the field in one band (scene/exposures.ecsv, written by observe.py from
    mesa_argus.survey): mid-exposure UTC times, BJD_TDB, airmass, altitude, azimuth, and the table's rows (telescope,
    field angle, the ratchet's zenith seeing)."""
    from astropy.time import Time

    from mesa_argus import night

    rows = Table.read(state.DATA / "exposures.ecsv")
    rows = rows[rows["band"] == band]
    if len(rows) == 0:
        raise SystemExit(f"the survey takes no {band} exposures of this field in the window")
    times = Time(list(rows["time"]), scale="utc")
    airmass, alt, az = night.sky_position(times, state.CENTRE.ra.deg, state.CENTRE.dec.deg, night.location(cfg))
    return times, night.bjd_tdb(times, state.CENTRE.ra.deg, state.CENTRE.dec.deg, cfg=cfg), airmass, alt, az, rows


def phases_at(var, bjd):
    return ((bjd - var["epoch"]) / var["period"]) % 1.0


def catmull_rom_matrix(n_out, n_in, start, step):
    """Resampling matrix (n_out x n_in): output pixel i is the Catmull-Rom interpolation of the input at
    start + i * step (input pixel coordinates), normalised to unit sum, as hdps's reprojector (tan_reprojection.cu,
    flux_weight 0) resamples sensor pixels onto a tile."""
    u = start + np.arange(n_out) * step
    j0 = np.floor(u).astype(int)
    t = u - j0
    weights = np.stack([-0.5 * t**3 + t**2 - 0.5 * t, 1.5 * t**3 - 2.5 * t**2 + 1.0,
                        -1.5 * t**3 + 2.0 * t**2 + 0.5 * t, 0.5 * t**3 - 0.5 * t**2], axis=1)
    m = np.zeros((n_out, n_in))
    for k in range(4):
        cols = np.clip(j0 - 1 + k, 0, n_in - 1)
        np.add.at(m, (np.arange(n_out), cols), weights[:, k])
    return m / m.sum(axis=1, keepdims=True)


class Scene:
    """Every star's ArgusSim PSF, for each PSF set (a seeing, at a field angle: --field-angle unless given per set);
    sky and dark per exposure; observe() makes an exposure.

    An exposure is made on the sensor's pixel grid (ArgusSim's PSF is in sensor pixels, 1.007"), then resampled onto the
    hdps sky-tile grid (0.879") by Catmull-Rom interpolation, as hdps's reprojector does. Everything outside the class
    sees the tile: n, plate, centre and the stars' x, y are tile pixels; native_* are the sensor grid."""

    def __init__(self, stars, args, cfg, ab, seeings, field_angles=None):
        self.native_plate = cfg.telescope.plate_scale
        self.native_n = int(np.ceil(2 * state.AREA_HALF / self.native_plate)) + 30
        self.native_n += self.native_n % 2
        self.native_centre = self.native_n / 2.0
        self.plate = SKY_PX
        self.n = int(np.ceil(2 * state.AREA_HALF / self.plate))
        self.n += self.n % 2
        self.centre = self.n / 2.0
        # tile pixels on the hdps tile grid (astropy convention, pixel centres at integers): ArgusSim's tile code puts
        # them at i + 0.5, so the tile coordinate is half a pixel less than the block offsets give
        self.x = self.centre + np.asarray(stars["dx"]) / self.plate - 0.5
        self.y = self.centre + np.asarray(stars["dy"]) / self.plate - 0.5
        step = self.plate / self.native_plate  # tile pixel i is at native_centre + (i + 0.5 - centre) * step
        self._resample_args = (self.n, self.native_n, self.native_centre + (0.5 - self.centre) * step, step)
        self._resample = None
        if cfg.telescope.psf_model == "gaussian":  # ArgusSim's GaussianPSF: no Zemax file read, no ZemaxPSF import
            from mesa_argus import sensor

            gaussian, made = sensor.gaussian_psf(cfg), {}

            def psf_at(seeing, band, angle, iy, ix):
                """The Gaussian's centre moved by -phase, as ZemaxPSF's pixel_phase moves the light (the stamp's centroid
                moves by -phase, below); the same in both bands and at every field angle, so made once per seeing."""
                key = (float(seeing), iy, ix)
                if key not in made:
                    made[key] = sensor.gaussian_stamp(gaussian, seeing, -iy / PSF_PHASES, -ix / PSF_PHASES)
                return made[key].copy()
        else:
            from argus_sim import ZemaxPSF

            psf_model = ZemaxPSF.from_config(cfg)

            def psf_at(seeing, band, angle, iy, ix):
                p, _ = psf_model.with_seeing(seeing=seeing, band=band, field_angle_deg=angle,
                                             pixel_phase=(iy / PSF_PHASES, ix / PSF_PHASES))
                return p
        self.seeings = list(seeings)
        self.field_angles = list(field_angles) if field_angles is not None else [args.field_angle] * len(self.seeings)
        self.lookup = {}  # (band, PSF set index, y phase, x phase) -> unit-flux PSF stamp
        for si, (seeing, angle) in enumerate(zip(self.seeings, self.field_angles)):
            for name, band in BANDS.items():
                for iy in range(PSF_PHASES):
                    for ix in range(PSF_PHASES):
                        p = psf_at(seeing, band, angle, iy, ix)
                        p = np.maximum(p, 0.0)
                        self.lookup[name, si, iy, ix] = p / p.sum()
        p0 = self.lookup["rho", 0, 0, 0]
        yy, xx = np.indices(p0.shape)
        self.c0 = ((p0 * yy).sum(), (p0 * xx).sum())  # the stamp's centroid
        native_x = self.native_centre + np.asarray(stars["dx"]) / self.native_plate
        native_y = self.native_centre + np.asarray(stars["dy"]) / self.native_plate
        # each star's stamp origin and sub-pixel phase (the stamp centroid moves by -phase)
        self.oy = np.ceil(native_y - self.c0[0]).astype(int)
        self.ox = np.ceil(native_x - self.c0[1]).astype(int)
        self.py = np.clip(np.round((self.oy + self.c0[0] - native_y) * PSF_PHASES).astype(int), 0, PSF_PHASES - 1)
        self.px = np.clip(np.round((self.ox + self.c0[1] - native_x) * PSF_PHASES).astype(int), 0, PSF_PHASES - 1)
        self._designs = {}  # (band, seeing index) -> PSF matrix, built on demand, the last few kept
        self.seeing_index = None  # per exposure (set_seeing)
        self.sky = None  # electrons/s/pixel per band per exposure (set_sky)
        self.dark, self.readnoise, self.fullwell = cfg.telescope.dark_current, cfg.telescope.readnoise, cfg.telescope.fullwell
        self.exptime = args.exptime

    @property
    def resample(self):
        """The sensor-to-tile Catmull-Rom matrix, made when first needed (the raw-frame path never resamples)."""
        if self._resample is None:
            self._resample = catmull_rom_matrix(*self._resample_args)
        return self._resample

    @property
    def touches(self):
        return self.resample != 0

    def _design(self, name, si):
        """Sparse (pixels x stars) matrix of unit-flux PSFs at seeing self.seeings[si]."""
        stack = np.stack([self.lookup[name, si, iy, ix] for iy in range(PSF_PHASES) for ix in range(PSF_PHASES)])
        p = stack[self.py * PSF_PHASES + self.px]  # (stars, stamp, stamp)
        yy, xx = np.indices(p.shape[1:])
        y = yy[None] + self.oy[:, None, None]
        x = xx[None] + self.ox[:, None, None]
        n = self.native_n
        on = (y >= 0) & (y < n) & (x >= 0) & (x < n) & (p > 0)
        cols = np.broadcast_to(np.arange(len(self.x))[:, None, None], p.shape)
        return sparse.csc_matrix((p[on].astype(np.float32), ((y * n + x)[on], cols[on])), shape=(n * n, len(self.x)))

    def design(self, name, si):
        key = (name, int(si))
        if key not in self._designs:
            if len(self._designs) >= 6:
                self._designs.pop(next(iter(self._designs)))
            self._designs[key] = self._design(name, si)
        return self._designs[key]

    def set_sky(self, sky_rate):
        self.sky = sky_rate

    def set_seeing(self, index):
        self.seeing_index = np.asarray(index)

    def design_at(self, name, j):
        """The PSF matrix for exposure j's seeing."""
        return self.design(name, self.seeing_index[j])

    def background(self, name, j):
        """Sky + dark electrons per pixel in exposure j."""
        return (self.sky[name][j] + self.dark) * self.exptime

    def to_tile(self, native):
        """A sensor-grid image resampled onto the tile grid."""
        return self.resample @ native.reshape(self.native_n, self.native_n) @ self.resample.T

    def model(self, name, rates, j):
        """Exposure j without noise, on the tile grid (sky + dark + stars, electrons per pixel)."""
        return self.to_tile(self.background(name, j) + self.design_at(name, j) @ (rates * self.exptime))

    def observe(self, name, rates, j, rng):
        """One exposure: Poisson draw of sky + dark + stars per sensor pixel, clipped at full well, plus read noise;
        resampled onto the tile grid. A tile pixel is full when any sensor pixel it interpolates is."""
        bg = self.background(name, j)
        mean = bg + self.design_at(name, j) @ (rates * self.exptime)
        charge = np.minimum(rng.poisson(mean), self.fullwell)
        image = charge + rng.normal(0.0, self.readnoise, mean.shape)
        full = (charge >= self.fullwell).reshape(self.native_n, self.native_n).astype(float)
        return dict(image=self.to_tile(image), full=(self.touches @ full @ self.touches.T) > 0, background=bg,
                    sensor=image.reshape(self.native_n, self.native_n))


def constant_star_rates(stars, pulsators, ab, workers=1):
    """Every star as a constant. Its Gaia G and BP-RP were measured through the dust in front of it (column ebv_own):
    that dust is taken out (mesa_argus.dust.deredden: the MESA template whose reddened colour is the measured one, and
    the dust-free G), and the dust where it is observed (column ebv) put back into the template's SED (CCM89). At its
    real position the two are the same, and the star is as Gaia measured it; scattered (a synthetic field), its G and
    BP-RP become those seen through the new sightline. Returns electrons/s on AIRMASS_GRID and above-atmosphere
    magnitudes, per band; the templates; each star's colour as measured (filled where Gaia has none). workers: the
    templates' photometry in that many forked processes ([stars] photometry_workers)."""
    templates = Templates()
    G = column(stars, "Gmag")
    colour = column(stars, "BP-RP")
    # no (or a blended) BP-RP, common in crowded cores: the median colour, at the same G, of stars > 2.5' from the centre
    everyone = Table.read(star_catalogue())
    r_arcmin = 60 * np.hypot((column(everyone, "RA_ICRS") - state.CENTRE.ra.deg) * np.cos(state.CENTRE.dec.rad),
                             column(everyone, "DE_ICRS") - state.CENTRE.dec.deg)
    g_all, c_all = column(everyone, "Gmag"), column(everyone, "BP-RP")
    good = (r_arcmin > 2.5) & np.isfinite(g_all) & np.isfinite(c_all)
    bins = np.arange(11.5, 20.51, 0.5)
    centres = 0.5 * (bins[1:] + bins[:-1])
    in_bin = [good & (g_all >= lo) & (g_all < hi) for lo, hi in zip(bins[:-1], bins[1:])]
    median_colour = np.array([np.median(c_all[b]) if np.sum(b) >= 5 else np.nan for b in in_bin])
    known = np.isfinite(median_colour)
    missing = ~(np.isfinite(colour) & (colour > -0.5) & (colour < 3.0))
    colour = np.where(missing, np.interp(G, centres[known], median_colour[known]), colour)
    source = np.asarray(stars["Source"]).astype(np.int64)
    for k in np.nonzero([int(s) in pulsators for s in source])[0]:
        # a Gaia-classified variable: its intensity-mean colour; a blended one a typical colour for its kind
        v = pulsators[int(source[k])]
        mean_colour = v["mean"].get("BP", np.nan) - v["mean"].get("RP", np.nan)
        if v["kind"].startswith("RR"):
            low, high, typical = 0.2, 0.9, (0.42 if v["kind"] == "RRc" else 0.55)
        else:
            low, high, typical = 0.3, 1.3, 0.7
        colour[k] = mean_colour if np.isfinite(mean_colour) and low <= mean_colour <= high else typical
    print(f"{int(missing.sum())} sources without usable BP-RP take the median colour, at their G, of stars > 2.5' from the centre")

    # the dust: in front of the star where Gaia measured it, and where it is observed (none in hand-made fields)
    ebv_own = np.nan_to_num(column(stars, "ebv_own")) if "ebv_own" in stars.colnames else np.zeros(len(stars))
    ebv = np.nan_to_num(column(stars, "ebv")) if "ebv" in stars.colnames else np.zeros(len(stars))
    intrinsic, G0 = colour.copy(), G.copy()
    dusty = ebv_own > 0
    intrinsic[dusty], G0[dusty] = dust.deredden(G[dusty], colour[dusty], ebv_own[dusty])
    moved = ebv != ebv_own  # a synthetic field: seen through other dust than Gaia saw it through
    if moved.any():
        seen_G, seen_colour = dust.redden_gaia(G0[moved], intrinsic[moved], ebv[moved])
        stars["Gmag"][moved], stars["BP-RP"][moved] = seen_G, seen_colour
    print(f"dust: E(B-V) {np.min(ebv):.3f}-{np.max(ebv):.3f} where observed (median {np.median(ebv):.3f}); "
          f"{int(moved.sum())} stars seen through other dust than Gaia saw them through", flush=True)

    print(f"ArgusSim photometry of every star ({workers} process{'es' if workers > 1 else ''}) …", flush=True)
    rate = {b: np.zeros((len(stars), len(AIRMASS_GRID))) for b in BANDS}
    mag = {b: np.zeros(len(stars)) for b in BANDS}
    template = np.array([templates.index(c) for c in intrinsic])
    scale = 10 ** (-0.4 * (G0 - templates.g[template]))
    step = 0.05  # ArgusSim once per template at E(B-V) on this grid, linear in between (in magnitudes)
    grid = {}
    for t in np.unique(template):
        e = ebv[template == t]
        grid[t] = np.arange(np.floor(e.min() / step), np.ceil(e.max() / step) + 0.5) * step  # covers every star's E(B-V)
    state._PHOTOMETRY = (templates, ab, grid)
    if workers > 1:  # forked: the workers share the templates and ArgusSim's throughputs with this process
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork")) as pool:
            light = dict(zip(grid, pool.map(template_photometry, grid, chunksize=4)))
    else:
        light = {t: template_photometry(t) for t in grid}
    state._PHOTOMETRY = None
    for t, nodes in grid.items():
        these = template == t
        e = ebv[these]
        for b in BANDS:
            node_mag, node_rate = light[t][b]
            if len(nodes) == 1:
                rate[b][these] = np.outer(scale[these], node_rate[0])
                mag[b][these] = node_mag[0] - 2.5 * np.log10(scale[these])
                continue
            log_rate = np.array([np.interp(e, nodes, np.log(node_rate[:, i])) for i in range(len(AIRMASS_GRID))]).T
            rate[b][these] = scale[these, None] * np.exp(log_rate)
            mag[b][these] = np.interp(e, nodes, node_mag) - 2.5 * np.log10(scale[these])
    return rate, mag, templates, colour


def template_photometry(t):
    """Template t through ArgusSim at each of its E(B-V) nodes: per band, the magnitude above the atmosphere and the
    electrons/s on AIRMASS_GRID. Each node's reddened SED is made once, for both bands and every airmass."""
    templates, ab, grid = state._PHOTOMETRY
    seds = [dust.redden(templates.wave.value, templates.flux[t], x) * FLAM for x in grid[t]]
    return {b: (np.array([float(ab.mag_from_sed(band, templates.wave, f)) for f in seds]),
                np.array([[ab.electrons_from_sed(band, templates.wave, f, airmass=a).value for a in AIRMASS_GRID]
                          for f in seds]))
            for b, band in BANDS.items()}


def followed_star_light(k, v, args, ab, bjd, airmass, model_file, model_ebv, model_distance):
    """A followed star's model and its SED, electrons/s and magnitude at every exposure, and its cycle (or, for a
    transient, its track over the night) for the CMD."""
    if "transient" in v:
        tr = v["transient"]
        model = TransientSource(_REPO / tr["file"], tr.get("t0_bjd"), tr.get("ebv"))
        seds = np.array([model.sed_exposure(b, args.exptime) for b in bjd])
        mag = {b: np.array([float(ab.mag_from_sed(band, model.wave, f * FLAM)) if f.any() else np.inf for f in seds])
               for b, band in BANDS.items()}
        cycle = {b: np.where(np.isfinite(mag[b]), mag[b], np.nan) for b in BANDS}
    else:
        model = RSPStar(model_file, ab, model_ebv, model_distance)
        seds = np.array([model.sed(ph) for ph in phases_at(v, bjd)])
        mag = {b: np.array([float(ab.mag_from_sed(band, model.wave, f * FLAM)) for f in seds]) for b, band in BANDS.items()}
        grid = np.linspace(0, 1, 120, endpoint=False)
        cycle = {b: np.array([float(ab.mag_from_sed(band, model.wave, model.sed(ph) * FLAM)) for ph in grid])
                 for b, band in BANDS.items()}
    rate = {b: np.array([ab.electrons_from_sed(band, model.wave, f * FLAM, airmass=x).value for f, x in zip(seds, airmass)])
            for b, band in BANDS.items()}
    return model, seds, rate, mag, cycle


def conditions(times, airmass, alt, az, args, stars, cfg, ab, survey=None):
    """Seeing per exposure, the detector scene (PSFs), and the sky per exposure, all from ArgusSim. survey: the survey's
    exposure rows (survey_schedule), each with its ratchet's zenith seeing and its field angle."""
    if survey is None:
        # seeing: the season's median zenith seeing at the site (DIMMs, Barker et al. 2003) x airmass^0.6; PSFs on a 0.05" grid
        zenith_seeing = args.seeing if args.seeing else seasonal_seeing_params(times[0])[0]
        seeing = seeing_at_airmass(zenith_seeing, airmass)
        seeing_grid = np.round(np.arange(np.floor(seeing.min() * 20) / 20, seeing.max() + 0.05, 0.05), 2)
        print(f"seeing: {zenith_seeing:.2f}\" at zenith -> {seeing.min():.2f}-{seeing.max():.2f}\" (ArgusSim site model); "
              f"{len(seeing_grid)} PSF sets", flush=True)
        print("building the detector scene …", flush=True)
        scene = Scene(stars, args, cfg, ab, seeing_grid)
        scene.set_seeing(np.abs(seeing[:, None] - seeing_grid[None, :]).argmin(axis=1))
    else:
        # seeing: each ratchet's zenith draw x airmass^0.6; PSFs at 0.05" in seeing and 0.01 deg in field angle
        seeing = seeing_at_airmass(np.asarray(survey["seeing_zenith"], float), airmass)
        sets = np.c_[np.round(seeing * 20) / 20, np.round(np.asarray(survey["field_angle"], float), 2)]
        unique, index = np.unique(sets, axis=0, return_inverse=True)
        print(f"seeing (survey: per ratchet): {seeing.min():.2f}-{seeing.max():.2f}\"; field angles "
              f"{np.min(survey['field_angle']):.2f}-{np.max(survey['field_angle']):.2f} deg; {len(unique)} PSF sets", flush=True)
        print("building the detector scene …", flush=True)
        scene = Scene(stars, args, cfg, ab, unique[:, 0], unique[:, 1])
        scene.set_seeing(np.ravel(index))
    # sky: airglow (solar activity, airmass), zodiacal light, integrated starlight, Moon (Krisciunas & Schaefer 1991),
    # site light
    observatory = Observatory(cfg)
    per_exposure = [observatory.sky_components_at(float(a), float(z), t) for a, z, t in zip(alt, az, times)]
    component = {key: np.array([float(np.ravel(c[key])[0]) for c in per_exposure]) for key in per_exposure[0]}
    sky_rate = {}
    for name, band in BANDS.items():
        sky = SpectralSky(ab.tp, ab, band)
        sky_rate[name] = np.asarray(sky.sky_electrons(component["airglow_s10"], component["zodiacal_s10"],
                                                      component["starlight_s10"], component["moon_s10"],
                                                      component["artif_s10"], transparency=1.0)) * scene.native_plate**2
    scene.set_sky(sky_rate)
    print(f"sky (ArgusSim): V {np.min(component['v_mag']):.2f}-{np.max(component['v_mag']):.2f} mag/arcsec2; Moon "
          f"{np.mean(component['illum_frac']) * 100:.0f}% illuminated, altitude {np.min(component['moon_alt']):.0f} to "
          f"{np.max(component['moon_alt']):.0f} deg; rho sky {sky_rate['rho'].min():.1f}-{sky_rate['rho'].max():.1f} e/s/pixel")
    return seeing, scene


def block_radec(dx, dy):
    """RA, Dec (deg) of offsets dx, dy (arcsec) from the block centre in the tile's tangent plane: the inverse of
    block_offsets."""
    tile_ra, tile_dec = (float(v[0]) for v in sh.tile_center(np.array([state.TILE])))
    crpix = sh.SEGMENT_PIXEL_SIZE / 2.0 + 0.5
    px = (state.BLOCK[0] + 0.5) * MINIPIX_PX + np.asarray(dx, float) / SKY_PX
    py = (state.BLOCK[1] + 0.5) * MINIPIX_PX + np.asarray(dy, float) / SKY_PX
    xi = np.radians(-(px - crpix + 0.5) * sh.PIXEL_SCALE_DEG)
    eta = np.radians((py - crpix + 0.5) * sh.PIXEL_SCALE_DEG)
    d0 = np.radians(tile_dec)
    ra = tile_ra + np.degrees(np.arctan2(xi, np.cos(d0) - eta * np.sin(d0)))
    dec = np.degrees(np.arctan2(np.sin(d0) + eta * np.cos(d0), np.hypot(xi, np.cos(d0) - eta * np.sin(d0))))
    return ra % 360.0, dec


def binned(image, k):
    """Block means of k x k pixels (float16; the edges that do not fill a block dropped)."""
    h, w = (image.shape[0] // k) * k, (image.shape[1] // k) * k
    return image[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3)).astype(np.float16)


def raw_frame_path(folder, row, args):
    """Where an exposure's raw camera frame is written: (<folder>/<CAMID>_<UTC start>.fits, start, end)."""
    from datetime import timedelta

    from astropy.time import Time

    from mesa_argus import camera

    mid = Time(row["time"])
    start, end = mid - timedelta(seconds=args.exptime / 2), mid + timedelta(seconds=args.exptime / 2)
    return folder / f"{camera.camera_id(int(row['tel_id']))}_{start.datetime:%Y%m%d_%H%M%S}.fits", start, end


def write_raw_frame(folder, image, win, row, args, layout, cards=None):
    """The exposure as a raw camera frame (mesa_argus.camera): <folder>/<CAMID>_<UTC start>.fits. The header's
    pointing is the readout window's centre (the camera's pointing over what it reads out); layout: ArgusSim's ring
    layout the exposures were found with (row's ota is its row); cards: more header cards (the simulation's
    calibration, mesa_argus.true_calibration)."""
    from mesa_argus import camera, sensor

    path, start, end = raw_frame_path(folder, row, args)
    ra, dec = sensor.sky_at(win[0] + (win[2] - 1) / 2, win[1] + (win[3] - 1) / 2, row["time"], row["theta_deg"],
                            int(row["ota"]), layout)
    folder.mkdir(parents=True, exist_ok=True)
    camera.write_raw(path, image, dict(utc_start=start.isot, utc_end=end.isot, exptime=args.exptime,
                                       ratchet=f"{start.datetime:%Y%m%d}_{int(row['ratchet'])}", tel_id=int(row["tel_id"]),
                                       subarray=int(layout.subarray[int(row["ota"])]),
                                       target_ra=float(ra), target_dec=float(dec), band=row["band"], x_origin=win[0],
                                       y_origin=win[1], cards=cards or {}))
    return path


def window_footprint(win, row, layout, n=4):
    """A readout window's outline on the sky: (m, 2) arcsec from the block centre (block_offsets), n points along each
    side of its outer pixel edges, closed, as ArgusSim's geometry places them (mesa_argus.sensor.sky_at)."""
    from mesa_argus import sensor

    x0, y0, w, h = float(win[0]) - 0.5, float(win[1]) - 0.5, float(win[2]), float(win[3])  # (pixel centres at integers)
    t = np.arange(n) / n
    x = np.r_[x0 + t * w, np.full(n, x0 + w), x0 + w - t * w, np.full(n, x0), x0]
    y = np.r_[np.full(n, y0), y0 + t * h, np.full(n, y0 + h), y0 + h - t * h, y0]
    ra, dec = sensor.sky_at(x, y, row["time"], row["theta_deg"], int(row["ota"]), layout)
    return np.c_[block_offsets(np.asarray(ra), np.asarray(dec))].astype(np.float32)
