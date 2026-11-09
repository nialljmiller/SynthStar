#!/usr/bin/env python3
"""The engine: one night of Argus 60 s exposures of a scene, simulated and measured; the movie (CMD, colour exposure,
difference image, light curves, SEDs) and the photometry CSV. observe.py writes the scene and runs this, once per band.

    python night_movie.py --scene outputs/pseudo_real/f294/scene --band rho --stride 10   # by hand
    python night_movie.py --field m3                                                                    # a hand-made field

The night, step by step (main):
  1. the stars: the catalogue (constant stars) and the followed stars (real variables, placed stars, transients)
  2. the night: ArgusSim's exposure schedule at the field (mesa_argus.night), airmass
  3. constant stars: the MESA SED matched to each star's Gaia BP-RP, scaled to its G -> electrons/s per band (ArgusSim)
  4. followed stars: their SED at every exposure (pulsation / orbit phase, or time since t0), through the observation's
     dust -> electrons/s
  5. conditions per exposure (ArgusSim): seeing (seasonal median x airmass^0.6), ArgusSim's PSF with that seeing
     ([argussim.telescope] psf_model: the Zemax PSF, or the Gaussian spot with no Zemax file read), sky
  6. the difference-image reference: the median of the best-seeing exposures
  7. every exposure: PSFs + sky -> Poisson, full well, read noise -> hdps source detector, matched to the simulated
     stars, per-exposure zero point (mesa_argus.pipeline)
  8. the movie, frame by frame (the same exposures again, from the same seeds)

Assumptions (field angle, coverage, seeing, transparency, ...): docs/ASSUMPTIONS.md.
"""
import argparse
import copy
import csv
import json
import shutil
import textwrap
import warnings
from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astropy.utils import iers
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.axis as maxis  # noqa: E402
import matplotlib.text as mtext  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.colors import AsinhNorm  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402

# (ZemaxPSF is imported only where psf_model is "zemax" (Scene): the public argus_sim has none)
from argus_sim import ABPhot, SpectralSky, SystemThroughput  # noqa: E402
from argus_sim import c as ARGUS_CONFIG  # noqa: E402
from argus_sim import skymap_shim as sh  # noqa: E402
from argus_sim.observatory import Observatory, seasonal_seeing_params, seeing_at_airmass  # noqa: E402

from mesa_argus import dust  # noqa: E402
from mesa_argus.argus import BANDS, BLUE, COLORS, FLAM, NAVY, RED, SYMBOLS, add_logo, apply_style, system_response  # noqa: E402
from mesa_argus.seds import Templates  # noqa: E402

iers.conf.auto_download = False
iers.conf.auto_max_age = None  # the bundled Earth-orientation table at any age: sub-second UT1, sub-arcsec polar motion
_REPO = Path(__file__).resolve().parent  # data/, runs/, outputs/ are in the repo, wherever this is run from

from mesa_argus.engine import state  # noqa: E402  the shared mutable field/block state (keystone of the stage split)
from mesa_argus.engine.state import (MINIPIX_PX, SKY_PX, MARGIN, AIRMASS_GRID, PSF_PHASES, FIELDS,  # noqa: E402
                                     use_field, set_block, block_label)


def __getattr__(name):  # forward the shared mutable state (night_movie.HALF, .TILE, .FIELD, ...) to engine.state, live
    try:
        return getattr(state, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


# ---------------------------------------------------------------- the stars
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


from mesa_argus.engine.models import RSPStar, TransientSource  # noqa: E402  (extracted to mesa_argus/engine/models.py)


# ---------------------------------------------------------------- the night
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


# ---------------------------------------------------------------- the detector
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


# ---------------------------------------------------------------- the steps of main
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


PREVIEW_BIN = 4  # the pipeline movie's raw-frame preview: block means of 4 x 4 sensor pixels
PHOTOMETRY_COLUMNS = ("exposure", "ut_hours", "airmass", "seeing", "star", "true_mag", "saturated", "measured_mag",
                      "measured_err", "band", "tel_id", "ratchet", "field_angle", "time_utc")  # <name>_night_<band>.photometry.csv


def binned(image, k):
    """Block means of k x k pixels (float16; the edges that do not fill a block dropped)."""
    h, w = (image.shape[0] // k) * k, (image.shape[1] // k) * k
    return image[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3)).astype(np.float16)


def slim_record(r):
    """A frame's record as the movies need it, picklable: no paths or open objects (detections as arrays, the WCS as
    its header text)."""
    cat = r.get("catalogue")
    return dict(window=r["window"], field_xy=r["field_xy"], j=r["j"], failed=r.get("failed"), truth=r["truth"],
                tel_id=r["tel_id"], ratchet=r["ratchet"], field_angle=r["field_angle"], zp=r.get("zp"),
                preview=r["preview"], preview_bin=r["preview_bin"],
                det_x=None if cat is None else np.asarray(cat["xcentroid"], np.float32),
                det_y=None if cat is None else np.asarray(cat["ycentroid"], np.float32),
                wcs=None if r.get("wcs") is None else r["wcs"].to_header_string(relax=True))


def movie_records(slim):
    """Slim records back into what pipeline_movie.render reads (catalogue x/y, an astropy WCS)."""
    from astropy.io import fits
    from astropy.table import Table
    from astropy.wcs import WCS

    out = []
    for r in slim:
        r = dict(r)
        r["catalogue"] = None if r["det_x"] is None else Table(dict(xcentroid=r["det_x"], ycentroid=r["det_y"]))
        r["wcs"] = None if r["wcs"] is None else WCS(fits.Header.fromstring(r["wcs"]))
        out.append(r)
    return out


def alert_table(work, stem):
    """Every row of hdps's alert catalogue for a frame (AlertVettingTask's alert_catalog: position, PSF flux, score,
    accepted, the feat_* features its real/bogus model reads), with each accepted row's alertId from the frame's Avro
    packets (the packet's candidate ra/dec are the row's, written from the same values). None if hdps wrote none."""
    import pandas as pd

    from mesa_argus import alert_files

    paths = sorted(p for p in work.glob(f"alerts_*_{stem}.parquet") if not p.name.startswith("alerts_stamps_"))
    if not paths:
        return None
    parts = []
    for p in paths:  # each row's place in hdps's files: its catalogue row, and its stamps (the npz's row_index)
        part = pd.read_parquet(p)
        part.insert(0, "catalog_file", p.name)
        part.insert(1, "catalog_row", range(len(part)))
        stamps = p.with_name(p.name.replace("alerts_", "alerts_stamps_", 1)).with_suffix(".npz")
        part.insert(2, "stamps_file", stamps.name if stamps.exists() else "")
        parts.append(part)
    table = pd.concat(parts, ignore_index=True)
    ids = {}
    packets = work / "alerts" / f"{stem}.avro"
    if packets.exists():
        for packet in alert_files.read(packets):
            ids[(packet["candidate"]["ra"], packet["candidate"]["dec"])] = str(packet["alertId"])  # (v4: a UUID)
    table.insert(0, "alertId", [ids.get((float(r), float(d)), "") for r, d in zip(table["ra_deg"], table["dec_deg"])])
    return table


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


def measure_frame(work, stem, band, obs, star_x, star_y, expected_flux):
    """hdps's solved frame (LightweightFITSOutputTask's fits_image: the SOURCES catalogue, the solved WCS, hdps's zero
    point from TPVWCSSolverTask's zeropoint_for_image), calibrated: mag = coadd_magzp - 2.5 log10(flux), mag_err =
    1.0857 flux_err / flux. coadd_magzp on the frame is referenced to electrons (TPVWCSSolverTask's
    _harvest_photometric_qa: "Electron-referenced, like every in-pipeline consumer"); the rate-referenced recipe in
    hdps's ffi-catalogs.md is for the store row (ffi_row_from_image), so no 1/exptime here. Each detection's simulated
    star for scoring (truth.label). (None, None, None, None) when hdps did not solve the frame."""
    from astropy.io import fits
    from astropy.table import Table
    from astropy.wcs import WCS

    from mesa_argus.pipeline import truth

    path = work / "solved" / f"{stem}_solved.fits"
    if not path.exists():
        return None, None, None, None
    with fits.open(path) as hdul:
        header = hdul[0].header
        catalogue = Table(hdul["SOURCES"].data)
    wcs = WCS(header)
    x, y = np.asarray(catalogue["xcentroid"], float), np.asarray(catalogue["ycentroid"], float)
    catalogue["ra"], catalogue["dec"] = wcs.all_pix2world(x, y, 0)
    catalogue["saturated"] = np.asarray(catalogue["is_saturated"], bool)
    flux, flux_err = np.asarray(catalogue["flux"], float), np.asarray(catalogue["flux_err"], float)
    zp = None
    if "coadd_magzp" in header:
        zp = dict(zp=float(header["coadd_magzp"]), rms=float(header.get("magzp_rms", np.nan)),
                  n=int(header.get("magzp_nstars", 0)), band=str(header.get("magzp_band", "")))
        with np.errstate(divide="ignore", invalid="ignore"):
            catalogue["mag_instrumental"] = catalogue["mag"]
            catalogue["mag"] = np.where(flux > 0, zp["zp"] - 2.5 * np.log10(flux), np.nan)
            catalogue["mag_err"] = np.where(flux > 0, 1.0857 * flux_err / flux, np.nan)
    else:  # hdps: no usable crossmatch, instrumental magnitudes only
        catalogue["mag_instrumental"] = catalogue["mag"]
        catalogue["mag"] = np.nan
        catalogue["mag_err"] = np.nan
    star_of = truth.label(catalogue, star_x, star_y, expected_flux, {"match_radius": obs["truth"]["match_radius"]})
    return catalogue, zp, star_of, wcs


def block_cut():
    """The block cut-out's slice of hdps's tile grid, and (cx, cy, x0, y0): the block centre and the cut's corner in
    tile pixels (block_cutouts, ReferenceMask)."""
    cx, cy = (state.BLOCK[0] + 0.5) * MINIPIX_PX - 0.5, (state.BLOCK[1] + 0.5) * MINIPIX_PX - 0.5  # hdps tile grid
    h = int(np.ceil((state.HALF + 2) / SKY_PX)) + 1
    x0, y0 = int(round(cx)) - h, int(round(cy)) - h
    return (slice(max(y0, 0), y0 + 2 * h + 1), slice(max(x0, 0), x0 + 2 * h + 1)), (cx, cy, x0, y0)


def block_cutouts(work, stem):
    """hdps's tile holding the block (TileImageOutputTask's science_tile and difference_tile), cut around it:
    (science before subtraction, difference, extent in arcsec from the block centre), or None when hdps made no
    difference image of that tile."""
    from astropy.io import fits

    sci_path = work / f"epoch_{stem}_tile_{state.TILE:06d}_sci.fits"
    diff_path = work / f"epoch_{stem}_tile_{state.TILE:06d}_diff.fits"
    if not (sci_path.exists() and diff_path.exists()):
        return None
    sci, diff = fits.getdata(sci_path, "SCI"), fits.getdata(diff_path, "DIFF")
    cut, (cx, cy, x0, y0) = block_cut()
    sci_cut, diff_cut = np.asarray(sci)[cut].astype(np.float32), np.asarray(diff)[cut].astype(np.float32)
    extent = ((max(x0, 0) - 0.5 - cx) * SKY_PX, (max(x0, 0) + sci_cut.shape[1] - 0.5 - cx) * SKY_PX,
              (max(y0, 0) - 0.5 - cy) * SKY_PX, (max(y0, 0) + sci_cut.shape[0] - 0.5 - cy) * SKY_PX)
    return dict(science=sci_cut, difference=diff_cut, extent=extent)


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


class WindowMosaic:
    """The field movie's whole-window images: hdps's tile images (TileImageOutputTask's science_tile and
    difference_tile, every tile the readout window touches) on one display grid, and hdps's minipix grid to draw over
    them.

    The panels' view is a square about the union of the frames' readout windows (footprints: window_footprint), or
    with zoom > 1 its central 1/zoom about the block. The display grid is the main tile's (TILE's) pixel grid, north
    up, in blocks of `bin` x `bin` tile pixels (a power of two, so up to 32 the main tile's minipix edges are
    display-pixel edges), over the frames' windows inside the view, `bin` the smallest that brings it to at most
    target_px across. Each tile pixel goes to the display pixel holding its centre (its tile's
    WCS -> sky -> the main tile's tangent plane), and only from the tile hdps gives that sky position to: neighbouring
    tiles overlap by one minipix (28.125"), and each position lies in one tile's minipix (skymap_shim.
    radec_to_composite_id). A display pixel is the mean of its covered tile pixels (hdps's tiles are 0 where the frame
    does not reach); blank (nan) where fewer than half of them are covered or hdps wrote no image of the tile."""

    def __init__(self, footprints, target_px, zoom=1.0):
        points = np.concatenate([f for f in footprints if f is not None])
        x0, x1 = (points[:, 0].min(), points[:, 0].max())
        y0, y1 = (points[:, 1].min(), points[:, 1].max())
        half = 0.51 * max(x1 - x0, y1 - y0) / zoom  # the panels' view: square, about every frame's window, or about
        cx, cy = (0.5 * (x0 + x1), 0.5 * (y0 + y1)) if zoom == 1 else (0.0, 0.0)  # the block (zoom > 1)
        self.view = (cx - half, cx + half, cy - half, cy + half)
        x0, x1, y0, y1 = max(x0, self.view[0]), min(x1, self.view[1]), max(y0, self.view[2]), min(y1, self.view[3])
        # u, v: the main tile's 0-based pixel + 0.5 (pixel edges at integers, minipix edges at multiples of 32)
        self.cu, self.cv = (state.BLOCK[0] + 0.5) * MINIPIX_PX, (state.BLOCK[1] + 0.5) * MINIPIX_PX  # the block centre
        u, v = np.array([x0, x1]) / SKY_PX + self.cu, np.array([y0, y1]) / SKY_PX + self.cv
        span = max(np.ptp(u), np.ptp(v))
        self.bin = 1
        while span / self.bin > target_px:
            self.bin *= 2
        b = self.bin
        self.u0, self.v0 = np.floor(u.min() / b) * b, np.floor(v.min() / b) * b
        self.shape = (int(np.ceil((v.max() - self.v0) / b)), int(np.ceil((u.max() - self.u0) / b)))  # (rows, columns)
        self.index, self.owned, self.wcs = {}, {}, {}  # per tile: each pixel's display pixel (-1: none); counts; WCS
        self.ever = np.zeros(self.shape, bool)  # display pixels with an image in any frame

    def extent(self, r0, r1, c0, c1):
        """Display rows r0:r1, columns c0:c1 in arcsec from the block centre (imshow's extent)."""
        b = self.bin
        return ((self.u0 + c0 * b - self.cu) * SKY_PX, (self.u0 + c1 * b - self.cu) * SKY_PX,
                (self.v0 + r0 * b - self.cv) * SKY_PX, (self.v0 + r1 * b - self.cv) * SKY_PX)

    def tiles(self):
        """The tiles whose sky the display grid reaches (skymap_shim's tessellation at every display pixel)."""
        x0, x1, y0, y1 = self.extent(0, self.shape[0], 0, self.shape[1])
        x, y = np.meshgrid(np.linspace(x0, x1, self.shape[1] + 1), np.linspace(y0, y1, self.shape[0] + 1))
        return sorted(int(t) for t in np.unique(sh.radec_to_composite_id(*block_radec(x.ravel(), y.ravel()))[0]))

    def pixel_of(self, dx, dy):
        """Display (row, column) of offsets dx, dy (arcsec; floats: their floors are the pixel)."""
        return ((np.asarray(dy) / SKY_PX + self.cv - self.v0) / self.bin,
                (np.asarray(dx) / SKY_PX + self.cu - self.u0) / self.bin)

    def _tile(self, tile, header):
        """Tile `tile`'s pixels' display pixels (flat; -1 outside the grid or in another tile's sky), made once."""
        from astropy.wcs import WCS

        if tile not in self.index:
            w, n, rows = WCS(header), sh.SEGMENT_PIXEL_SIZE, 256
            index = np.full(n * n, -1, np.int32)
            for r0 in range(0, n, rows):  # (in strips: 4 M pixels at once would take ~0.5 GB)
                y, x = np.mgrid[r0:r0 + rows, 0:n]
                ra, dec = w.all_pix2world(x.ravel(), y.ravel(), 0)
                mine = sh.radec_to_composite_id(ra, dec)[0] == tile
                r, c = (np.floor(a).astype(np.int64) for a in self.pixel_of(*block_offsets(ra, dec)))
                ok = mine & (r >= 0) & (r < self.shape[0]) & (c >= 0) & (c < self.shape[1])
                index[r0 * n:(r0 + rows) * n] = np.where(ok, r * self.shape[1] + c, -1)
            self.index[tile], self.wcs[tile] = index, w
            self.owned[tile] = np.bincount(index[index >= 0], minlength=self.ever.size)
        return self.index[tile]

    def frame(self, work, stem):
        """A frame's images from hdps's tile files: dict(science (before subtraction), difference, extent), float32,
        cut to the display pixels it covers; None when hdps wrote no tile image of the frame."""
        from astropy.io import fits

        size = self.ever.size
        owned, covered, science, difference = (np.zeros(size) for _ in range(4))
        for sci_path in sorted(work.glob(f"epoch_{stem}_tile_*_sci.fits")):
            diff_path = sci_path.with_name(sci_path.name[:-len("_sci.fits")] + "_diff.fits")
            if not diff_path.exists():
                continue
            tile = int(sci_path.name[:-len("_sci.fits")].rsplit("_", 1)[1])
            with fits.open(sci_path) as hdul:
                index = self._tile(tile, hdul["SCI"].header)
                sci = np.asarray(hdul["SCI"].data, float).ravel()
            diff = np.asarray(fits.getdata(diff_path, "DIFF"), float).ravel()
            use = (index >= 0) & (sci != 0) & np.isfinite(sci)
            k = index[use]
            owned += self.owned[tile]
            covered += np.bincount(k, minlength=size)
            science += np.bincount(k, weights=sci[use], minlength=size)
            difference += np.bincount(k, weights=diff[use], minlength=size)  # (a nan pixel: the display pixel nan)
        shown = ((covered > 0) & (covered >= 0.5 * owned)).reshape(self.shape)
        if not shown.any():
            return None
        self.ever |= shown
        rows, cols = np.nonzero(shown)
        cut = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
        with np.errstate(invalid="ignore", divide="ignore"):
            sci_m, diff_m = (np.where(shown, (a / covered).reshape(self.shape), np.nan)[cut].astype(np.float32)
                             for a in (science, difference))
        return dict(science=sci_m, difference=diff_m,
                    extent=self.extent(rows.min(), rows.max() + 1, cols.min(), cols.max() + 1))

    def _runs(self, dx, dy, keep):
        """The stretches of a sampled line (dx, dy: arcsec, along its last axis) where keep holds and some frame has an
        image: [(m, 2) arrays]."""
        r, c = (np.floor(a).astype(np.int64) for a in self.pixel_of(dx, dy))
        inside = (r >= 0) & (r < self.shape[0]) & (c >= 0) & (c < self.shape[1])
        keep = keep & inside
        keep[inside] &= self.ever[r[inside], c[inside]]
        out = []
        for x, y, k in zip(np.atleast_2d(dx), np.atleast_2d(dy), np.atleast_2d(keep)):
            edges = np.flatnonzero(np.diff(np.r_[0, k.astype(np.int8), 0]))
            out += [np.c_[x[a:b], y[a:b]].astype(np.float32) for a, b in zip(edges[::2], edges[1::2]) if b - a > 1]
        return out

    def grid(self):
        """hdps's minipix grid where any frame has an image, as lists of (m, 2) arrays (arcsec from the block centre):
        each tile's minipix edges (0-based tile pixel 32 m - 0.5) inside the sky the tile holds, and the edges between
        tiles' skies (the Dec bands' edges and, within each band, its RA slots' edges: skymap_shim's tessellation),
        which are minipix edges too. (A line straight in one tile's plane is straight in the main tile's: both are
        gnomonic; it is sampled every tile pixel only to find where it leaves its tile's sky.)"""
        n = sh.SEGMENT_PIXEL_SIZE
        along = np.arange(-0.5, n, 1.0)
        edges = np.arange(0, n + 1, MINIPIX_PX) - 0.5
        across, along = np.meshgrid(edges, along, indexing="ij")  # (line, point along it)
        lines = []
        for tile, w in self.wcs.items():
            for x, y in ((across, along), (along, across)):  # the lines of constant x, then of constant y
                ra, dec = w.all_pix2world(x.ravel(), y.ravel(), 0)
                mine = (sh.radec_to_composite_id(ra, dec)[0] == tile).reshape(x.shape)
                dx, dy = (a.reshape(x.shape) for a in block_offsets(ra, dec))
                lines += [np.r_[s[:1], s[-1:]] for s in self._runs(dx, dy, mine)]  # straight: its two ends
        # the edges between tiles' skies, over the display grid's sky (its outline's RA, Dec range)
        x0, x1, y0, y1 = self.extent(0, self.shape[0], 0, self.shape[1])
        t = np.linspace(0.0, 1.0, 400)
        ra, dec = block_radec(np.r_[x0 + t * (x1 - x0), np.full(400, x1), x1 - t * (x1 - x0), np.full(400, x0)],
                              np.r_[np.full(400, y0), y0 + t * (y1 - y0), np.full(400, y1), y1 - t * (y1 - y0)])
        tile_ra = float(sh.tile_center(np.array([state.TILE]))[0][0])
        rel = (ra - tile_ra + 180.0) % 360.0 - 180.0  # RA about the main tile's centre (the grid may cross RA 0)
        ra_lo, ra_hi, dec_lo, dec_hi = tile_ra + rel.min(), tile_ra + rel.max(), dec.min(), dec.max()
        seams = []
        sample = np.linspace(0.0, 1.0, 4000)
        for band in np.arange(np.floor((dec_lo - sh.DEC_MIN) / sh.SPACING_DEG),
                              np.ceil((dec_hi - sh.DEC_MIN) / sh.SPACING_DEG) + 1):
            lo = sh.DEC_MIN + band * sh.SPACING_DEG  # the band's lower edge (constant Dec: curved in the plane)
            seams += [np.r_[s[::10], s[-1:]] for s in self._runs(*block_offsets(ra_lo + sample * (ra_hi - ra_lo),
                                                                                    np.full(len(sample), lo)),
                                                                  np.ones(len(sample), bool))]
            spacing = sh.SPACING_DEG / np.cos(np.radians(lo + 0.5 * sh.SPACING_DEG))  # its RA slots' width
            for k in np.arange(np.floor(ra_lo / spacing), np.ceil(ra_hi / spacing) + 1):
                edge_dec = lo + np.linspace(0.0, sh.SPACING_DEG, 400)  # a slot's edge (constant RA: straight)
                seams += [np.r_[s[:1], s[-1:]] for s in self._runs(*block_offsets(np.full(400, (k * spacing) % 360.0),
                                                                                  edge_dec), np.ones(400, bool))]
        return lines, seams

    def summary(self, mosaics, footprints, label):
        """What the field movie draws from (movie data's "window"); label: what the window is, for the title."""
        lines, seams = self.grid()
        return dict(mosaics=mosaics, footprints=footprints, view=self.view, grid=lines, seams=seams, bin=self.bin,
                    tiles=sorted(int(t) for t in self.wcs), label=label)


class ReferenceMask:
    """The field movies' grey in the Difference panel ([output] movie_mask_reference_e, movie_mask_radius_px,
    movie_mask_growth_px): the pixels about the stars brighter than the threshold in a frame's own reference coadd,
    the disc growing by `growth` px per magnitude above it (as hdps's bright-star mask grows). The coadd is the one
    hdps's CoaddReferenceProvider subtracted: the frame's camera's of the tile, else any camera's; the deepest
    (NCOMBINE), then by name. Display only; made from the run's coadds, so a finished run's movies get it too."""

    def __init__(self, coadds, threshold, radius, growth):
        self.coadds, self.threshold, self.radius, self.growth = Path(coadds), float(threshold), radius, growth
        self.tiles, self.display = {}, {}

    def _coadd(self, camera, tile):
        from astropy.io import fits

        everyone = sorted(self.coadds.glob(f"*_tile_{tile:06d}_*.fits"))
        mine = [p for p in everyone if f"_{camera}_tile_" in p.name] or everyone

        def depth(p):
            try:
                with fits.open(p) as h:
                    return int(h[0].header.get("NCOMBINE", 0))
            except Exception:
                return 0

        return max(mine, key=lambda p: (depth(p), p.name)) if mine else None

    def tile(self, camera, tile):
        """(mask over the tile's pixels, the tile's header), or (None, None) with no coadd of the tile."""
        from astropy.io import fits
        from scipy.ndimage import distance_transform_edt

        key = (camera, tile)
        if key not in self.tiles:
            path = self._coadd(camera, tile)
            if path is None:
                self.tiles[key] = (None, None)
                return self.tiles[key]
            with fits.open(path) as h:
                hdu = h["SCI"] if "SCI" in h else h[0]
                ref, header = np.nan_to_num(np.asarray(hdu.data, np.float32)), hdu.header.copy()
            masked = np.zeros(ref.shape, bool)
            peak, k = float(ref.max()), 0
            while self.threshold * 10 ** (0.4 * k) < peak and k <= 12:  # each magnitude brighter: a wider disc
                level = ref > self.threshold * 10 ** (0.4 * k)
                masked |= distance_transform_edt(~level) <= self.radius + self.growth * k
                k += 1
            self.tiles[key] = (masked, header)
        return self.tiles[key]

    def block(self, camera, cutout):
        """The block cut-out with its masked difference pixels nan (a copy), or as it was with no coadd."""
        masked = self.tile(camera, state.TILE)[0]
        if cutout is None or masked is None:
            return cutout
        m = masked[block_cut()[0]]
        return dict(cutout, difference=np.where(m, np.nan, cutout["difference"]).astype(np.float32))

    def window(self, camera, image, mosaic, tiles):
        """A whole-window image (WindowMosaic.frame) with its masked difference pixels nan (a copy): a display pixel
        is masked when a quarter of the tile pixels it holds are."""
        if image is None:
            return image
        if camera not in self.display:
            size = mosaic.ever.size
            hit, held = np.zeros(size), np.zeros(size)
            for tile in tiles:
                masked, header = self.tile(camera, tile)
                if masked is None:
                    continue
                index = mosaic._tile(tile, header)
                held += mosaic.owned[tile]
                hit += np.bincount(index[masked.ravel() & (index >= 0)], minlength=size)
            self.display[camera] = ((held > 0) & (hit >= 0.25 * held)).reshape(mosaic.shape)
        x0, x1, y0, y1 = image["extent"]  # back to the display rows and columns WindowMosaic.frame cut
        r0, c0 = (int(round(a)) for a in mosaic.pixel_of(x0, y0))
        h, w = image["difference"].shape
        m = self.display[camera][r0:r0 + h, c0:c0 + w]
        return dict(image, difference=np.where(m, np.nan, image["difference"]).astype(np.float32))


def alerts_of(work, stem, follow, star_ra, star_dec):
    """hdps's accepted alerts in a frame (AlertVettingTask's alert_catalog): [dict(key, dx, dy)], dx, dy in arcsec
    from the block centre; key: the followed star within 2" (truth, for the movie), else the alert's position."""
    import pandas as pd

    out = []
    for path in sorted(work.glob(f"alerts_*_{stem}.parquet")):
        table = pd.read_parquet(path)
        table = table[table["accepted"]]
        if not len(table):
            continue
        dx, dy = block_offsets(table["ra_deg"].to_numpy(), table["dec_deg"].to_numpy())
        for i in range(len(table)):
            sep = np.hypot((star_ra[follow] - table["ra_deg"].iloc[i]) * np.cos(np.radians(star_dec[follow])),
                           star_dec[follow] - table["dec_deg"].iloc[i]) * 3600 if len(follow) else np.array([])
            near = np.nonzero(sep <= 2.0)[0]
            key = follow[near[np.argmin(sep[near])]] if len(near) else ("pos", round(float(dx[i]) / 3), round(float(dy[i]) / 3))
            out.append(dict(key=key, dx=float(dx[i]), dy=float(dy[i])))
    return out


# ---------------------------------------------------------------- main
def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", type=Path, help="a scene folder written by observe.py")
    p.add_argument("--field", help="a hand-made field (m3, m15, rr308108, rr416445); with --scene, its name")
    p.add_argument("--catalogues", type=Path, help="also write every measured exposure's detections (parquet) here")
    p.add_argument("--observation", type=Path, help="the resolved observation file (observe.py): hdps chain, window, camera")
    p.add_argument("--hdps-pass", choices=["reference", "science"], default="science",
                   help="reference: the reference nights, hdps coadds only; science: the whole chain against them")
    p.add_argument("--work", type=Path, help="hdps's working folder (raw frames, products, coadds), per band below it")
    p.add_argument("--pipeline-movie", action="store_true", help="also draw the pipeline movie (mesa_argus.pipeline_movie)")
    p.add_argument("--reference", type=Path, help="hdps's reference coadds (the reference pass's coadds/): SFFT's reference_dir")
    p.add_argument("--no-render", action="store_true", help="measure (and write FITS) but draw no movie")
    p.add_argument("--measure-only", action="store_true",
                   help="measure again from the raw frames and hdps's products a run left in --work (a run that stopped "
                        "after hdps): no rendering, no hdps; then the outputs as usual")
    p.add_argument("--stop-after-render", action="store_true",
                   help="render the raw frame stack to --work and stop before hdps (the image-maker half of a decoupled "
                        "run); the fused run (no flag) renders and runs hdps together as before")
    p.add_argument("--hdps-from-stack", action="store_true",
                   help="run hdps on the raw frame stack already in --work, then measure as usual (the hdps half of a "
                        "decoupled run): no rendering")
    p.add_argument("--render-only", action="store_true",
                   help="draw the movies from a finished run's <output>.movie.joblib (output.movie_data): no "
                        "rendering, no hdps; the scene is rebuilt as in the run")
    p.add_argument("-o", "--output", type=Path,
                   help="default: next to the scene; a hand-made field: outputs/pseudo_real/<field>/<field>_night_<band>.mp4")
    p.add_argument("--date", help="local date the night starts (default: the field's own)")
    p.add_argument("--max-airmass", type=float, default=None, help="a further airmass limit (the array's minimum altitude always applies)")
    p.add_argument("--start", help="UT HH:MM to start (default: when the field is first observable)")
    p.add_argument("--end", help="UT HH:MM to stop (default: when the field is last observable)")
    p.add_argument("--exptime", type=float, default=60.0)
    p.add_argument("--seeing", type=float, default=None, help="zenith seeing FWHM (arcsec); default: ArgusSim's seasonal median")
    p.add_argument("--field-angle", type=float, default=1.44, help="deg from the optical axis; 1.44 = area-weighted median")
    p.add_argument("--stride", type=int, default=1, help="render and measure every Nth exposure")
    p.add_argument("--fps", type=float, default=24)
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--highlight", type=int, default=6, help="hand-made fields: pulsators to follow")
    p.add_argument("--seed", type=int, default=3, help="noise")
    p.add_argument("--area", type=float, default=8.0, help="side (arcmin) of the simulated area (kernel and zero point)")
    p.add_argument("--block", type=int, default=None, help="minipixes across the block shown (default: the field's)")
    p.add_argument("--band", choices=list(BANDS), default="rho", help="band shown in the images, light curves and CMD")
    return p


def main(argv=None):
    import os

    args = parser().parse_args(argv)
    if args.scene is not None:  # relative paths are relative to where the command was run
        args.scene = (Path.cwd() / args.scene) if not args.scene.is_absolute() else args.scene
        args.field = args.field or args.scene.resolve().parent.name
    for key in ("catalogues", "reference", "observation", "work"):
        if getattr(args, key) is not None and not getattr(args, key).is_absolute():
            setattr(args, key, Path.cwd() / getattr(args, key))
    if args.output is None:
        folder = args.scene.parent if args.scene is not None else _REPO / "outputs" / "pseudo_real" / args.field
        args.output = folder / f"{args.field}_night_{args.band}.mp4"
    if not args.output.is_absolute():
        args.output = Path.cwd() / args.output
    os.chdir(_REPO)
    use_field(args.field, args.scene)
    set_block(args.block or FIELDS[args.field].get("block_n", 3))
    if args.observation is not None:  # the stars kept: out to a readout window's far corner, wherever it falls
        from mesa_argus import runconfig

        resolved = runconfig.resolve(runconfig.load(args.observation))
        window = resolved["field"]["window"]
        side = (window or ARGUS_CONFIG.telescope.n_pixels_x) * MINIPIX_PX * SKY_PX if window else \
            ARGUS_CONFIG.telescope.n_pixels_x * ARGUS_CONFIG.telescope.plate_scale
        state.AREA_HALF = max(side * np.sqrt(2), state.HALF + MARGIN)
    else:
        state.AREA_HALF = max(30.0 * args.area, state.HALF + MARGIN)
    args.date = args.date or FIELDS[args.field]["date"]
    B = args.band  # the band shown in the images, light curves and CMD
    cfg = copy.deepcopy(ARGUS_CONFIG)
    for key, value in (FIELDS[args.field]["site"] or {}).items():  # the instrument stays Argus; only the site moves
        setattr(cfg.observatory, key, value)
    if args.observation is not None:  # the observation's PSF ([argussim.telescope] psf_model, spot_size) for the Scene
        # and the sensor's PSFs (mesa_argus.sensor.PSFs); the other [argussim] values are not applied to ArgusSim here
        telescope = resolved["argussim"]["telescope"]
        cfg.telescope.psf_model, cfg.telescope.spot_size = telescope["psf_model"], telescope["spot_size"]
    print(f"PSF: ArgusSim's {cfg.telescope.psf_model} model"
          + (f" (GaussianPSF: optical spot {cfg.telescope.spot_size} um rms with the Moffat seeing; no Zemax file read)"
             if cfg.telescope.psf_model == "gaussian" else ""), flush=True)
    ab = ABPhot(throughput=SystemThroughput(filters={"g": True, "rho": True}, throughput_loss=cfg.telescope.throughput_loss),
                collecting_area=cfg.telescope.collecting_area)

    # 1. the stars
    stars = load_scene()
    pulsators = load_pulsators()
    source = np.asarray(stars["Source"]).astype(np.int64)  # Gaia id, or -1, -2, ... for placed stars
    in_block = np.asarray(stars["in_block"])
    G = column(stars, "Gmag")
    gaia_var = np.array([int(s) in pulsators for s in source])  # Gaia-classified: constant unless followed
    follow = choose_followed(stars, pulsators, args.highlight)  # row indices; k below is always a row index
    placed_label = {v["source"]: v["label"] for v in load_placed()}
    labels = {k: placed_label[int(source[k])] if source[k] < 0 else f"V{i + 1}" for i, k in enumerate(follow)}
    n_placed = int(np.sum(source < 0))
    if args.catalogues is not None:  # what each simulated star is, for the truth columns (pipeline.lightcurves)
        args.catalogues.mkdir(parents=True, exist_ok=True)
        (args.catalogues / "truth_stars.json").write_text(json.dumps(
            {"label": {str(k): v for k, v in labels.items()}, "source": [int(s) for s in source]}))
    print(f"{len(stars) - n_placed}{'synthetic stars' if FIELDS[args.field].get('synthetic') else 'Gaia sources'} "
          f"({int(in_block.sum()) - n_placed} in the block)" + (f"; {n_placed} transients" if n_placed else "")
          + f"; following {len(follow) - n_placed} pulsators with MESA RSP models; the other "
          f"{int((gaia_var & in_block).sum()) - len(follow)} Gaia-classified variables in the block are held constant")

    # 2. the night
    if args.exptime > cfg.survey.base_cadence_s:
        raise SystemExit(f"--exptime {args.exptime} s is longer than ArgusSim's cadence ({cfg.survey.base_cadence_s} s)")
    survey = None  # observing "survey": the exposures ArgusSim's array takes of the field, each in one band
    if (state.DATA / "exposures.ecsv").exists():
        times, bjd, airmass, alt, az, survey = survey_schedule(B, cfg)
        if args.observation is not None and not args.render_only:  # here, before the photometry: see 6. below
            from mesa_argus.survey import check_layout

            check_layout(survey, resolved["argussim"]["packing_strategy"]["layout_file"])
        print(f"survey ({B}): {len(times)} exposures through {len(np.unique(survey['tel_id']))} telescopes in "
              f"{len(np.unique(survey['ratchet']))} ratchets, UT {times[0].iso[11:16]}-{times[-1].iso[11:16]}, "
              f"airmass {airmass.min():.2f}-{airmass.max():.2f}")
    else:
        times, bjd, airmass, alt, az = night_schedule(args.date, args.max_airmass, cfg, args.start, args.end)
        print(f"night of {args.date}: {len(times)} exposures, UT {times[0].iso[11:16]}-{times[-1].iso[11:16]}, "
              f"airmass {airmass.min():.2f}-{airmass.max():.2f}")
    bands = [B] if survey is not None else list(BANDS)  # the bands each exposure is taken in
    ut = (times.jd - np.floor(times[0].jd - 0.5) - 0.5) * 24  # UT hours since 0h on the first date

    # 3. constant stars
    workers = (runconfig.resolve(runconfig.load(args.observation))["stars"]["photometry_workers"]
               if args.observation is not None else 1)
    static_rate, static_mag, templates, _ = constant_star_rates(stars, pulsators, ab, workers)
    G = column(stars, "Gmag")  # as seen where observed (a synthetic field: through its own sightline)

    # 4. followed stars: real variables and placed stars from their model files, with the observation's dust
    from mesa_argus import rsp

    model_file = {v["source"]: _REPO / "runs" / f"rsp_{rsp.PREFIX}{name}" / "pulsation.h5"
                  for name, v in load_followed().items()}
    model_file.update({v["source"]: _REPO / v["file"] for v in load_placed() if v["timing"] == "periodic"})
    model_ebv = {v["source"]: v["ebv"] for v in load_placed() if v["timing"] == "periodic"}
    model_distance = {v["source"]: v.get("distance_pc") for v in load_placed() if v["timing"] == "periodic"}
    models, follow_sed, follow_rate, follow_mag, follow_cycle = {}, {}, {}, {}, {}
    transient = {k for k in follow if "transient" in pulsators[int(source[k])]}
    for k in follow:
        v = pulsators[int(source[k])]
        models[k], follow_sed[k], follow_rate[k], follow_mag[k], follow_cycle[k] = followed_star_light(
            k, v, args, ab, bjd, airmass, model_file.get(int(source[k])), model_ebv.get(int(source[k])),
            model_distance.get(int(source[k])))
        cyc = follow_cycle[k]["rho"]
        if k in transient:
            tr = v["transient"]
            print(f"  {labels[k]}: {tr['kind']} {tr.get('engine', tr.get('generator', 'sncosmo'))} {tr['model']}, "
                  f"z {tr['z']:g}, t0 BJD {tr['t0_bjd']:.4f} (night starts {bjd[0] - tr['t0_bjd']:+.3f} d after t0); "
                  + (f"rho {np.nanmin(cyc):.2f}-{np.nanmax(cyc):.2f} over the night" if np.isfinite(cyc).any()
                     else "no light during the night"))
        else:
            print(f"  {labels[k]}: {v['kind']} {engine_of(v)}, period {models[k].period:.5f} d "
                  f"({'given' if 'placed' in v else 'Gaia'} {v['period']:.5f}); rho {cyc.min():.2f}-{cyc.max():.2f} "
                  f"(amplitude {np.ptp(cyc):.2f}); Gaia mean G {G[k]:.2f}")
    # the catalogue / reference brightness of a periodic star: its cycle-mean SED; a transient is not in the reference
    ref_rate = {}
    for k in follow:
        if k in transient:
            ref_rate[k] = {b: np.zeros(len(airmass)) for b in BANDS}
            continue
        mean_sed = np.mean([models[k].sed(ph) for ph in np.linspace(0, 1, 200, endpoint=False)], axis=0) * FLAM
        ref_rate[k] = {b: np.array([ab.electrons_from_sed(band, models[k].wave, mean_sed, airmass=x).value for x in airmass])
                       for b, band in BANDS.items()}
    # the reference a candidate is subtracted against: its telescope's coadd of the reference exposures that telescope
    # kept ([night] reference_ratchets: its best-seeing ratchets). A periodic star is in that coadd as it was at those
    # exposures, not at its cycle mean; SFFT matches the coadd's photometric scale to the science frame, so the star's
    # flux there is its mean SED over those exposures at the science exposure's airmass (electrons/s on AIRMASS_GRID,
    # {star: {telescope: rates}}). Before 2026-10-30 the truth used the cycle mean: off by up to ~20% per telescope.
    ref_by_tel = {}
    ref_scene = state.DATA.parent / "reference" / "scene" / "exposures.ecsv"
    if survey is not None and ref_scene.exists() and state.DATA.parent.name != "reference":
        from astropy.table import Table
        from astropy.time import Time

        from mesa_argus import night

        ref = Table.read(ref_scene)
        ref = ref[ref["band"] == B]
        ref_tel = np.asarray(ref["tel_id"], int)
        ref_bjd = night.bjd_tdb(Time(list(ref["time"]), scale="utc"), state.CENTRE.ra.deg, state.CENTRE.dec.deg, cfg=cfg)
        for k in follow:
            if k in transient:
                continue
            v = pulsators[int(source[k])]
            ref_by_tel[k] = {}
            for tel in np.unique(ref_tel):
                mean_sed = np.mean([models[k].sed(ph) for ph in phases_at(v, ref_bjd[ref_tel == tel])], axis=0) * FLAM
                ref_by_tel[k][int(tel)] = np.array([ab.electrons_from_sed(BANDS[B], models[k].wave, mean_sed, airmass=x).value
                                                    for x in AIRMASS_GRID])
        print(f"truth: each periodic star's reference flux from its telescope's reference exposures ({len(ref)} in {B}, "
              f"{len(np.unique(ref_tel))} telescopes)", flush=True)

    def rates_at(b, j, reference=False):
        """Electrons/s of every star in exposure j (reference: as the difference reference sees it, no transients)."""
        x = float(np.clip(airmass[j], AIRMASS_GRID[0], AIRMASS_GRID[-1]))
        i1 = int(np.clip(np.searchsorted(AIRMASS_GRID, x), 1, len(AIRMASS_GRID) - 1))
        f = (x - AIRMASS_GRID[i1 - 1]) / (AIRMASS_GRID[i1] - AIRMASS_GRID[i1 - 1])
        out = static_rate[b][:, i1 - 1] * (1 - f) + static_rate[b][:, i1] * f  # linear in airmass
        for k in follow:
            out[k] = 0.0 if (reference and k in transient) else follow_rate[k][b][j]
        return out

    def true_mag(b, k, j):
        return float(follow_mag[k][b][j]) if k in follow_mag else static_mag[b][k]

    # 5. conditions per exposure
    seeing, scene = conditions(times, airmass, alt, az, args, stars, cfg, ab, survey)

    # 6. the camera: every exposure on its telescope's sensor (ArgusSim's geometry, mesa_argus.sensor), in the
    # observation's readout window, written as a raw camera frame (mesa_argus.camera); 7. hdps's worker processes
    # them (mesa_argus.hdps_worker). The reference pass makes hdps's coadds (transients left out); the science pass
    # runs the whole chain against them, then the measurements from hdps's products (truth labels) and the movies.
    if survey is None:
        raise SystemExit("the engine observes as ArgusSim's survey does: observe.py writes scene/exposures.ecsv")
    from mesa_argus import hdps_worker, runconfig, sensor
    from mesa_argus.survey import array_layout

    obs = runconfig.resolve(runconfig.load(args.observation))
    # ArgusSim's ring layout ([argussim.packing_strategy] layout_file): each exposure's ota is a row of the layout
    # observe.py found the exposures with (exposures.ecsv meta), so the two must be the same (checked in 2. above, with
    # survey.check_layout). --render-only draws from the movie data alone (the frames' positions saved with it) and
    # places nothing with the layout, so it is not checked there: finished runs of either layout draw again.
    layout = array_layout(obs["argussim"]["packing_strategy"]["layout_file"])
    # the field movie's images: the whole readout window (output.movie_window_px; hdps's images of every tile it
    # touches, WindowMosaic) when the movie or its data is made, else the block (block_cutouts: the tile it is in)
    whole = obs["output"]["movie_window_px"] > 0 and (not args.no_render or obs["output"]["movie_data"])
    for t in obs["hdps"]["tasks"]:  # hdps's tile images: those the movie shows (None: every tile hdps differenced)
        if t["type"] == "TileImageOutputTask":
            t["config"]["tile_ids"] = None if whole else [int(state.TILE)]
    reference_pass = args.hdps_pass == "reference"
    window_px = int(round(obs["field"]["window"] * MINIPIX_PX * SKY_PX / ARGUS_CONFIG.telescope.plate_scale))
    star_ra, star_dec = block_radec(np.asarray(stars["dx"]), np.asarray(stars["dy"]))
    centre_ra, centre_dec = block_radec(np.zeros(1), np.zeros(1))
    projector = sensor.Projector(np.r_[centre_ra, star_ra], np.r_[centre_dec, star_dec], times, layout)
    east, north = obs["field"]["window_offset"]  # where the readout window's centre lies from the field (arcmin)
    window_ra, window_dec = block_radec(np.array([-60.0 * east]), np.array([60.0 * north]))  # the block's +x is west
    window_projector = sensor.Projector(window_ra, window_dec, times, layout)
    psfs = sensor.PSFs(cfg, wing=obs["image"])  # the observation's psf_model, with its [image] PSF options
    if psfs.gaussian and obs["image"]["psf_halo"]:
        print("PSF: [image] psf_halo_taper not used (it rounds Zemax's square optics array; the Gaussian has none)",
              flush=True)
    work = args.work / B
    expected_flux = static_rate[B][:, 0] * args.exptime
    for k in follow:
        expected_flux[k] = 0.0 if k in transient else np.mean(ref_rate[k][B]) * args.exptime
    frames = range(0, len(times), args.stride)
    seeds = np.random.SeedSequence(args.seed).spawn(len(times))
    calibration = None  # the simulation's WCS and zero point in each raw frame, when the chain takes them from it
    if any(t["type"] == "TrueCalibrationTask" for t in obs["hdps"]["tasks"]):
        from mesa_argus import true_calibration

        detection = next(t["config"] for t in obs["hdps"]["tasks"] if t["type"] == "SourceDetectionTask")
        calibration = dict(zp_rate=true_calibration.ab_zero_point(ab, BANDS[B], AIRMASS_GRID), detection=detection,
                           zp=[], share=[], residual=[])
    movie_file = args.output.with_suffix(".movie.joblib")

    def movies(frames, measured, alerted, cutouts, records, window=None):
        """8. the movies: the field movie (the panels as before, its images hdps's tile and difference image of the
        whole readout window (window: WindowMosaic.summary), or around the block) and the pipeline movie (each raw
        frame's preview, where it falls on the sensor, hdps's detections and WCS)."""
        if not args.no_render:
            o = obs["output"]
            # the field movies asked for ([output] movie_views): the window view (<name>_night_<band>.mp4) and the
            # block's own tile (_block.mp4); without whole-window images, the block alone under the first name
            views = list(dict.fromkeys(o["movie_views"])) if window is not None else ["block"]
            coadds = Path(args.reference) if args.reference else args.scene.parent / "reference" / "hdps" / B / "coadds"
            mask = (ReferenceMask(coadds, o["movie_mask_reference_e"], o["movie_mask_radius_px"], o["movie_mask_growth_px"])
                    if o["movie_mask_reference_e"] > 0 and coadds.is_dir() else None)
            cameras = [f"ARGUSSIM-{int(r['tel_id']):04d}" for r in records]
            cuts = [mask.block(c, k) for c, k in zip(cameras, cutouts)] if mask is not None else cutouts
            for view in views:
                win = window if view == "window" else None
                if win is not None and mask is not None:
                    mosaic = WindowMosaic(window["footprints"], o["movie_window_px"], o["movie_window_zoom"])
                    win = dict(window, mosaics=[mask.window(c, m, mosaic, window["tiles"])
                                                for c, m in zip(cameras, window["mosaics"])])
                draw_data = movie_data(args, stars, scene, follow, transient, models, follow_sed, follow_rate,
                                       follow_cycle, ref_rate, static_mag, rates_at, true_mag, measured, frames, times,
                                       bjd, ut, airmass, seeing, G, in_block, ab, cuts, alerted=alerted, window=win)
                draw_data.update(alerted=alerted, frame_j=list(frames), labels=labels, pulsators=pulsators,
                                 source=source, gaia_var=gaia_var, templates=templates, survey=survey, cutouts=cuts,
                                 masked=mask is not None)
                one = copy.copy(args)
                if view == "block" and len(views) > 1:
                    one.output = args.output.with_name(args.output.stem + "_block.mp4")
                render(one, draw_data, frames)
                print(f"field movie ({view}) -> {one.output.name}", flush=True)
        if args.pipeline_movie:
            from mesa_argus import pipeline_movie

            pipeline_movie.render(records, args.output.with_name(args.output.stem + "_pipeline.mp4"), state.FIELD, B,
                                  args.fps, args.dpi, ut, airmass, seeing)

    if args.render_only:  # a finished run's movie data: the movies only
        import joblib

        if not movie_file.exists():
            raise SystemExit(f"{movie_file}: no movie data (run with output.movie_data = true)")
        saved = joblib.load(movie_file)
        movies(saved["frames"], saved["measured"], saved["alerted"], saved["cutouts"], movie_records(saved["records"]),
               saved.get("window"))  # (movie data from before the whole-window images: the block)
        print(f"saved {args.output} (from {movie_file.name})")
        return

    print(f"{'reference pass' if reference_pass else 'observing'}: {len(frames)} exposures ({B}) as raw frames "
          f"({window_px} px window) -> {work / 'raw'}" + (" (already there: measuring them again)"
                                                           if args.measure_only or args.hdps_from_stack
                                                           else ""), flush=True)
    if not (args.measure_only or args.hdps_from_stack):  # a fresh stack; hdps ingests every raw/*.fits, so start clean
        shutil.rmtree(work / "raw", ignore_errors=True)
    records = []
    for n, j in enumerate(frames):  # 6. the camera: each exposure rendered and written as a raw frame
        row = survey[j]
        x, y, angle = projector.pixels(j, row["theta_deg"], row["ota"])
        wx, wy, _ = window_projector.pixels(j, row["theta_deg"], row["ota"])
        win = sensor.window(wx[0], wy[0], window_px)
        if args.measure_only or args.hdps_from_stack:  # the frame already on disk: read it, don't render
            from astropy.io import fits

            path = raw_frame_path(work / "raw", row, args)[0]
            image = fits.getdata(path).astype(np.float32)
        else:
            electrons = rates_at(B, j, reference=reference_pass) * args.exptime
            image = sensor.render(x[1:], y[1:], electrons, angle[1:], B, seeing[j], scene.background(B, j), win, psfs,
                                  np.random.default_rng(seeds[j]), ARGUS_CONFIG.telescope.readnoise,
                                  ARGUS_CONFIG.telescope.fullwell)
            cards = None
            if calibration is not None:
                w, residual, residual_rms = projector.wcs(j, row["theta_deg"], row["ota"], win[:2])
                zp_total = 2.5 * np.log10(np.interp(airmass[j], AIRMASS_GRID, calibration["zp_rate"]) * args.exptime)
                ai = int(np.abs(psfs.angles(B) - angle[0]).argmin())  # the field's tabulated field angle
                share = true_calibration.aperture_fraction(psfs, B, seeing[j], ai, calibration["detection"])
                cards = true_calibration.cards(w, residual, residual_rms, zp_total, share)
                for k, v in (("zp", zp_total), ("share", share), ("residual", residual)):
                    calibration[k].append(v)
            path = write_raw_frame(work / "raw", image, win, row, args, layout, cards)
        sx, sy = x[1:] - win[0], y[1:] - win[1]  # every star in the window's pixels (truth)
        bright = (G < 17) & (sx > 20) & (sx < win[2] - 20) & (sy > 20) & (sy < win[3] - 20)  # for the WCS check
        records.append(dict(path=path, window=win, field_xy=(x[0], y[0]), j=j, sx=sx, sy=sy,
                            preview=binned(image, PREVIEW_BIN), preview_bin=PREVIEW_BIN,
                            truth=(sx[bright], sy[bright], star_ra[bright], star_dec[bright]), tel_id=int(row["tel_id"]),
                            ratchet=int(row["ratchet"]), field_angle=float(row["field_angle"]),
                            footprint=window_footprint(win, row, layout) if whole else None))
        if (n + 1) % max(1, len(frames) // 10) == 0:
            print(f"  rendered {n + 1}/{len(frames)}", flush=True)
    if calibration is not None and calibration["zp"]:
        print(f"calibration from the simulation (TrueCalibrationTask): AB zero point {min(calibration['zp']):.3f}-"
              f"{max(calibration['zp']):.3f} mag (total e-), hdps's aperture holds {min(calibration['share']):.3f}-"
              f"{max(calibration['share']):.3f} of a star's e-; true WCS fit within "
              f"{max(calibration['residual']) * 1e3:.1f} mpx", flush=True)
    mosaic = None  # the field movie's whole-window images (WindowMosaic), made from hdps's tile images below
    if whole and not reference_pass:
        zoom = obs["output"]["movie_window_zoom"]
        mosaic = WindowMosaic([r["footprint"] for r in records], obs["output"]["movie_window_px"], zoom)
        if zoom > 1:  # hdps's tile images: only the tiles the zoomed view reaches
            for t in obs["hdps"]["tasks"]:
                if t["type"] == "TileImageOutputTask":
                    t["config"]["tile_ids"] = mosaic.tiles()

    if args.stop_after_render:  # the image-maker half of a decoupled run: raw stack on disk, stop before hdps
        print(f"rendered {len(frames)} frames -> {work / 'raw'}; stopping before hdps (--stop-after-render)", flush=True)
        return

    # 7. hdps's worker over the raw frames (mesa_argus.hdps_worker). The reference pass makes hdps's coadds
    # (transients left out); the science pass runs the whole chain against them.
    if reference_pass:
        result = hdps_worker.run(obs, work / "raw", work / "coadds", stop_after="CoaddTapTask",
                                 worker_id=f"{state.FIELD}_{B}_reference")
        shutil.rmtree(work / "coadds" / "solved", ignore_errors=True)
        print(f"hdps worker: {result}; reference coadds -> {work / 'coadds'}", flush=True)
        if not obs["output"]["raw"]:
            shutil.rmtree(work / "raw", ignore_errors=True)
        return
    if not args.measure_only:
        result = hdps_worker.run(obs, work / "raw", work, reference_dir=args.reference, worker_id=f"{state.FIELD}_{B}")
        print(f"hdps worker: {result}", flush=True)
    tile_bytes = sum(p.stat().st_size for p in work.glob("epoch_*_tile_*.fits"))
    print(f"hdps's tile images: {tile_bytes / 1e9:.2f} GB ({tile_bytes / 1e6 / max(1, len(records)):.0f} MB per frame)"
          + ("" if obs["output"]["raw"] else ", deleted at the end (output.raw = false)"), flush=True)

    # the measurements, from hdps's products: the solved frame's catalogue and zero point, the tile images, alerts
    rows, measured, alerted, cutouts, alert_rows = [], [], [], [], []
    mosaics = []
    n_alerts = n_skipped = 0
    for record in records:
        j, stem = record["j"], record["path"].stem
        row = survey[j]
        catalogue, zp, star_of, wcs = measure_frame(work, stem, B, obs, record["sx"], record["sy"], expected_flux)
        record.update(catalogue=catalogue, zp=zp, wcs=wcs, failed=None if catalogue is not None else "not solved by hdps")
        cutouts.append(block_cutouts(work, stem))
        mosaics.append(mosaic.frame(work, stem) if mosaic is not None else None)
        n_skipped += catalogue is None
        frame_alerts = alerts_of(work, stem, follow, star_ra, star_dec)
        table = alert_table(work, stem)
        if table is not None and len(table):  # every alert, labelled with what the simulation put there
            from hdps.alerts.features import FEATURE_SPEC_VERSION

            from mesa_argus.pipeline import truth

            for at, (key, value) in enumerate(dict(observation=state.FIELD, band=B, exposure=int(j), time_utc=str(row["time"]),
                                                   exptime=args.exptime, airmass=float(airmass[j]),
                                                   seeing=float(seeing[j]), tel_id=int(row["tel_id"]),
                                                   ratchet=int(row["ratchet"]),
                                                   field_angle=float(row["field_angle"]),
                                                   feature_spec=FEATURE_SPEC_VERSION).items()):
                table.insert(1 + at, key, value)
            for key, values in truth.label_alerts(table["ra_deg"], table["dec_deg"], star_ra, star_dec, expected_flux,
                                                  set(follow), transient, labels, obs["truth"]["alert_radius"]).items():
                table[key] = values
            table["truth_mag"] = [true_mag(B, int(k), j) if k >= 0 else np.nan for k in table["truth_star"]]
            # the change the difference image should show at the star (electrons): its flux now minus its flux in
            # the reference (a transient: none; a followed variable: as in its telescope's reference coadd, ref_by_tel,
            # else its cycle mean; any other star: unchanged). What decides real or bogus for a real/bogus training
            # set (tools/rb_training_set.py).
            now = rates_at(B, j) * args.exptime
            k = table["truth_star"].to_numpy()
            flux_now = np.where(k >= 0, now[np.maximum(k, 0)], np.nan)
            flux_ref = np.where(k >= 0, flux_now, np.nan)
            tel = int(row["tel_id"])
            for i, kk in enumerate(k):
                if kk in transient:
                    flux_ref[i] = 0.0
                elif kk in follow:
                    grid = ref_by_tel.get(kk, {}).get(tel)
                    flux_ref[i] = (float(np.interp(airmass[j], AIRMASS_GRID, grid)) if grid is not None
                                   else float(np.mean(ref_rate[kk][B]))) * args.exptime
            table["truth_flux_now"], table["truth_flux_reference"] = flux_now, flux_ref
            table["truth_diff_flux"] = flux_now - flux_ref
            alert_rows.append(table)
        alerted.append([(a["key"], a["dx"], a["dy"]) for a in frame_alerts])
        n_alerts += len(frame_alerts)
        if args.catalogues is not None and catalogue is not None:
            args.catalogues.mkdir(parents=True, exist_ok=True)
            star = np.full(len(catalogue), -1)
            star[star_of[star_of >= 0]] = np.nonzero(star_of >= 0)[0]
            catalogue["star"] = star
            meta = dict(field=state.FIELD, band=B, exposure=int(j), date_obs=times[j].isot, bjd_tdb=float(bjd[j]),
                        airmass=float(airmass[j]), seeing=float(seeing[j]), exptime=args.exptime,
                        tel_id=int(row["tel_id"]), ratchet=int(row["ratchet"]), fieldang=float(row["field_angle"]),
                        **{f"zp_{key}": value for key, value in (zp or {}).items()})
            catalogue.meta = meta
            catalogue.write(args.catalogues / f"{state.FIELD}_{B}_{j:04d}.parquet", overwrite=True)
        this_frame = {}
        for k in follow:
            d = int(star_of[k]) if star_of is not None else -1
            ok = d >= 0 and np.isfinite(catalogue["mag"][d]) and catalogue["snr"][d] >= 5  # S/N >= 5
            mag, err = (float(catalogue["mag"][d]), float(catalogue["mag_err"][d])) if ok else (np.nan, np.nan)
            this_frame[k] = (mag, err)
            rows.append(dict(exposure=j, ut_hours=ut[j], airmass=airmass[j], seeing=seeing[j], star=labels[k],
                             true_mag=true_mag(B, k, j), saturated=bool(d >= 0 and catalogue["saturated"][d]),
                             measured_mag=mag, measured_err=err, band=B, tel_id=int(row["tel_id"]),
                             ratchet=int(row["ratchet"]), field_angle=float(row["field_angle"]), time_utc=row["time"]))
        measured.append(this_frame)
    print(f"hdps: {n_alerts} alerts from {len(frames)} frames ({n_skipped} frames hdps could not process)", flush=True)
    window_images = None
    if mosaic is not None:
        side, zoom = obs["field"]["window"], obs["output"]["movie_window_zoom"]
        window_images = mosaic.summary(mosaics, [r["footprint"] for r in records],
                                       (f"a {side} × {side}-minipix readout window" if side else "the whole sensor")
                                       + (f" (×{zoom:g} about the block)" if zoom > 1 else ""))
        print(f"field movie: {window_images['label']}, hdps's tiles {window_images['tiles']} in {mosaic.bin} x "
              f"{mosaic.bin} tile pixels per image pixel ({mosaic.shape[1]} x {mosaic.shape[0]}); "
              f"{sum(m is not None for m in mosaics)}/{len(mosaics)} frames with an image", flush=True)
    if alert_rows:  # <name>_night_<band>.alerts.csv: every hdps alert row, its features and its truth
        import pandas as pd

        alerts = pd.concat(alert_rows, ignore_index=True)
        alerts.to_csv(args.output.with_suffix(".alerts.csv"), index=False)
        accepted = alerts[alerts["accepted"]]
        print("alert truth (accepted): " + ", ".join(f"{c} {n}" for c, n in accepted["truth_class"].value_counts().items()),
              flush=True)
    with open(args.output.with_suffix(".photometry.csv"), "w", newline="") as handle:  # (no rows: no star followed)
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else PHOTOMETRY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    if obs["output"]["movie_data"]:  # what the movies need, so they can be drawn later (--render-only)
        import joblib

        joblib.dump(dict(frames=list(frames), measured=measured, alerted=alerted, cutouts=cutouts,
                         records=[slim_record(r) for r in records], window=window_images), movie_file, compress=3)
        print(f"movie data -> {movie_file.name} ({movie_file.stat().st_size / 1e6:.0f} MB)", flush=True)
    movies(frames, measured, alerted, cutouts, movie_records([slim_record(r) for r in records]), window_images)
    if not obs["output"]["raw"]:  # the raw frames and hdps's bulky per-frame products (read above)
        shutil.rmtree(work / "raw", ignore_errors=True)
        shutil.rmtree(work / "solved", ignore_errors=True)
        for tile in work.glob("epoch_*_tile_*.fits"):
            tile.unlink()
    print(f"saved {args.output.with_suffix('.photometry.csv') if args.no_render else args.output}")


# ---------------------------------------------------------------- the movie
def alert_marks(d, n, fps):
    """The alert marks shown on movie frame n: [(dx, dy, alpha, scale)]. A mark appears when its object alerts, pops
    (scale 1.8, 1.3, then 1) at the start of a run of consecutive alerts, and fades over half a second after the last."""
    fade = max(1, int(fps) // 2)
    marks = {}
    for back in range(min(n, fade) + 1):
        for key, dx, dy in d["alerted"][n - back]:
            if key not in marks:
                run = 0
                while n - back - run - 1 >= 0 and any(m[0] == key for m in d["alerted"][n - back - run - 1]):
                    run += 1
                scale = (1.8, 1.3)[run] if back == 0 and run < 2 else 1.0
                marks[key] = (dx, dy, 1.0 - back / (fade + 1), scale)
    return list(marks.values())


def draw_alert_marks(ax, marks, size=24, lift=4.5, points=0.0):
    """Metal Gear Solid style: a red exclamation mark, white-outlined, above the object (clipped to the panel: the
    cut-out alerts are made on is larger than the block shown). size: its font size (points) at scale 1; it stands
    lift arcsec and `points` points above the object. The marks' Text artists."""
    from matplotlib import patheffects
    from matplotlib.transforms import offset_copy

    where = offset_copy(ax.transData, fig=ax.figure, y=points, units="points") if points else ax.transData
    return [ax.text(dx, dy + lift, "!", ha="center", va="bottom", fontsize=size * scale, weight="heavy", color=RED,
                    alpha=alpha, zorder=10, clip_on=True, transform=where,
                    path_effects=[patheffects.withStroke(linewidth=3.5 * size / 24, foreground="white", alpha=alpha)])
            for dx, dy, alpha, scale in marks]


def set_alert_mark(text, dx, dy, alpha, scale, size=24, lift=4.5):
    """An alert mark (draw_alert_marks, with the same size and lift) moved and faded in place."""
    from matplotlib import patheffects

    text.set(position=(dx, dy + lift), fontsize=size * scale, alpha=alpha, visible=True,
             path_effects=[patheffects.withStroke(linewidth=3.5 * size / 24, foreground="white", alpha=alpha)])


def alert_spans(d, k, n):
    """The light-curve spans (UT hours) of followed star k's alerts up to movie frame n: each alerted exposure +- half
    the cadence, consecutive ones merged."""
    t = np.array([d["ut"][j] for j in d["frame_j"]])
    half = 0.5 * float(np.median(np.diff(t))) if len(t) > 1 else 0.01
    spans = []
    for m in range(n + 1):
        if any(key == k for key, _, _ in d["alerted"][m]):
            lo, hi = t[m] - half, t[m] + half
            if spans and lo <= spans[-1][1] + 1e-9:
                spans[-1][1] = hi
            else:
                spans.append([lo, hi])
    return spans


def model_hours(ut):
    """Where the light-curve panel draws the models: 800 points over a night; every 3 minutes over a longer span."""
    n = 800 if ut[-1] - ut[0] <= 24 else int((ut[-1] - ut[0]) * 20)
    return np.linspace(ut[0], ut[-1], n)


MAX_SHOWN = 6  # followed stars the movie draws (an injected run follows hundreds; all are measured and labelled)
WINDOW_EDGE = "#f0a202"  # the frame's readout window, outlined over the whole-window images


def choose_shown(follow, stars, rel, alerted):
    """The followed stars the movie draws: those inside the block shown (all, if none is) that hdps measured (all, if
    none was), at most MAX_SHOWN: those that alerted first, then by how much they change over the night; in their
    followed order."""
    dx, dy = np.asarray(stars["dx"], float), np.asarray(stars["dy"], float)
    in_view = [k for k in follow if abs(dx[k]) <= state.HALF and abs(dy[k]) <= state.HALF]
    pool = in_view or list(follow)
    alerted_keys = {key for frame in (alerted or []) for key, _, _ in frame}

    def change(k):
        lo, hi = rel[k]["span"]
        return float(hi - lo) if np.isfinite(lo) and np.isfinite(hi) else 0.0

    measured = [k for k in pool if np.isfinite(rel[k]["measured"])]  # one never measured has no light curve to show
    pool = measured or pool
    keep = set(sorted(pool, key=lambda k: (k not in alerted_keys, -change(k)))[:MAX_SHOWN])
    return [k for k in follow if k in keep]


def movie_data(args, stars, scene, follow, transient, models, follow_sed, follow_rate, follow_cycle, ref_rate, static_mag,
               rates_at, true_mag, measured, frames, times, bjd, ut, airmass, seeing, G, in_block, ab, cutouts, alerted=None,
               window=None):
    """Everything draw() needs that stays fixed through the movie: models, display limits, colours, each star's
    mean over the night; which followed stars it draws (choose_shown). window: the whole readout window's images
    (WindowMosaic.summary), drawn in place of the block's, with their own image scales."""
    B = args.band
    # transients' model light curves on the light-curve panel's time grid (periodic models are drawn from their cycle)
    model_curve = {}
    for k in transient:
        when = bjd[0] + (model_hours(ut) - ut[0]) / 24.0
        model_curve[k] = np.array([float(ab.mag_from_sed(BANDS[B], models[k].wave, f * FLAM)) if f.any() else np.nan
                                   for f in (models[k].sed_at(b) for b in when)])
    # the image scales, from hdps's frames around the block (the movie shows hdps's tile and difference image):
    # brightness on an asinh stretch, linear out to the sky noise (Lupton et al. 2004); the difference scale out to its
    # 99.9th percentile
    yy, xx = np.indices((scene.n, scene.n))
    above = [c["science"] - np.nanmedian(c["science"]) for c in cutouts if c is not None]
    diffs = [c["difference"] for c in cutouts if c is not None]
    values = np.concatenate([a[np.isfinite(a)].ravel() for a in above]) if above else np.array([1.0])
    noise = 1.4826 * np.nanmedian(np.abs(values - np.nanmedian(values)))
    rgb = dict(w_beta=1.0, vmax=float(np.nanpercentile(values, 99.95)), lw=float(max(noise, 1.0)))
    dvals = np.concatenate([np.abs(a[np.isfinite(a)]).ravel() for a in diffs]) if diffs else np.array([1.0])
    dlim = float(max(np.nanpercentile(dvals, 99.9), 3 * noise, 1.0))
    if window is not None:  # the same, from the window's images (binned: lower noise and peaks than the tile's)
        images = [m for m in window["mosaics"] if m is not None]
        every = max(1, len(images) // 16)  # (every few rows of each frame: a night's images hold ~10^8 pixels)
        above = [(m["science"] - np.nanmedian(m["science"]))[::every] for m in images]
        wvals = np.concatenate([a[np.isfinite(a)].ravel() for a in above]) if above else np.array([1.0])
        wnoise = 1.4826 * np.nanmedian(np.abs(wvals - np.nanmedian(wvals)))
        wd = np.concatenate([np.abs(m["difference"][::every][np.isfinite(m["difference"][::every])]) for m in images]
                            ) if images else np.array([1.0])
        window = dict(window, rgb=dict(w_beta=1.0, vmax=float(np.nanpercentile(wvals, 99.95)), lw=float(max(wnoise, 1.0))),
                      dlim=float(max(np.nanpercentile(wd, 99.9), 3 * wnoise, 1.0)))
    # each star's mean over the night: the measurements' own, and the model's (a transient: where it is lit); the
    # light-curve axes cover 99.5% of the measurements (a rare bad cross-match must not flatten the rest; it is drawn
    # at the edge)
    rel = {}
    for k in follow:
        meas = np.array([f[k][0] for f in measured], float)
        model = np.array([true_mag(B, k, j) for j in frames], float)
        if k in transient:
            model = np.where(np.isfinite(model), model, np.nan)
        mean = np.nanmean if k in transient else np.mean
        lit = np.isfinite(meas).any()
        meas_rel = np.nanpercentile(meas - np.nanmean(meas), [0.25, 99.75]) if lit else []
        rel[k] = dict(measured=float(np.nanmean(meas)) if lit else np.nan, model=float(mean(model)),
                      span=(np.nanmin(np.r_[meas_rel, model - mean(model)]), np.nanmax(np.r_[meas_rel, model - mean(model)])),
                      resid=float(np.nanpercentile(np.abs(meas - model), 99.75)) if lit else np.nan)
    shown = choose_shown(follow, stars, rel, alerted)
    fixed = np.nonzero(in_block & (G < 20) & ~np.isin(np.arange(len(stars)), shown))[0]
    cmd_c = np.r_[[static_mag["beta"][k] - static_mag["rho"][k] for k in fixed],
                  np.concatenate([follow_cycle[k]["beta"] - follow_cycle[k]["rho"] for k in shown] or [np.array([])])]
    cmd_m = np.r_[[static_mag[B][k] for k in fixed], np.concatenate([follow_cycle[k][B] for k in shown] or [np.array([])])]
    if transient:  # a transient has no light (nan) before its model starts
        lit = np.isfinite(cmd_c) & np.isfinite(cmd_m)
        cmd_c, cmd_m = cmd_c[lit], cmd_m[lit]
    # the SED axis: every shown star's SED (300-1000 nm) over the night
    sed_values = np.concatenate([follow_sed[k][:, (models[k].wave.to_value(u.nm) >= 300) & (models[k].wave.to_value(u.nm) <= 1000)].ravel()
                                 for k in shown]) if shown else np.array([1e-16, 1e-14])
    sed_values = sed_values[sed_values > 0]
    if sed_values.size == 0:  # nothing lit all night (e.g. a transient after the window)
        sed_values = np.array([1e-16, 1e-14])
    palette = [BLUE, RED, "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#17becf"]  # the brand's dots first
    out = dict(stars=stars, scene=scene, follow=follow, shown=shown, transient=transient, models=models, follow_sed=follow_sed,
               follow_cycle=follow_cycle, model_curve=model_curve, static_mag=static_mag, true_mag=true_mag,
               times=times, bjd=bjd, ut=ut, airmass=airmass, seeing=seeing, G=G, in_block=in_block, ab=ab,
               colour_of={k: palette[i % len(palette)] for i, k in enumerate(shown)}, rgb=rgb, dlim=dlim, cmd_c=cmd_c, cmd_m=cmd_m,
               rel=rel, sed_ylim=(float(np.percentile(sed_values, 3)) / 1.5, float(sed_values.max()) * 1.5),
               centre=block_centre_label(), measured=measured,
               history={k: {"t": [], "m": [], "e": [], "true": []} for k in shown})
    if window is not None:
        out["window"] = window
    return out


class _FrameWriter(FFMpegWriter):
    """matplotlib's ffmpeg writer, fed frames already drawn (MovieFrame.render): the same ffmpeg command and encoding
    as grab_frame, without drawing the figure again."""

    def write_rgba(self, rgba):
        self._proc.stdin.write(rgba)


def render(args, d, frames):
    """The field movie: per frame, hdps's tile and difference images of the readout window (d["window"]) or around
    the block (saved by the engine), and the panels (MovieFrame: the figure built once, each frame drawn over it)."""
    B = args.band
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = _FrameWriter(fps=args.fps, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p",
                                                     "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])
    apply_style()
    fig = plt.figure(figsize=(16, 10), dpi=args.dpi)
    movie = MovieFrame(fig, args, d)
    with writer.saving(fig, str(args.output), dpi=args.dpi):
        for n, j in enumerate(frames):
            c = d["window"]["mosaics"][n] if "window" in d else d["cutouts"][n]
            exposures = {B: None if c is None else dict(image=c["science"], difference=c["difference"],
                                                       background=float(np.nanmedian(c["science"])), extent=c["extent"])}
            for k in d["shown"]:
                h = d["history"][k]
                h["t"].append(d["ut"][j])
                h["true"].append(d["true_mag"](B, k, j))
                h["m"].append(d["measured"][n][k][0])
                h["e"].append(d["measured"][n][k][1])
            d["n"] = n
            movie.update(exposures, j)
            writer.write_rgba(movie.render())
            if (n + 1) % max(1, len(frames) // 20) == 0:
                print(f"  {n + 1}/{len(frames)}", flush=True)
    plt.close(fig)


def draw(fig, args, exposures, j, d):
    """One movie frame: CMD, light curves, colour exposure, difference image, SEDs (all of it built on fig, for a
    caller that then saves or shows the figure; a movie builds its MovieFrame once and renders each frame)."""
    MovieFrame(fig, args, d).update(exposures, j)


class MovieFrame:
    """The field movie's figure for one movie's draw data (movie_data): the panels, axes, labels, colour bars and
    everything that stays the same through the movie made once; update() sets what changes from frame to frame, in
    place (the images, where each followed star is now, the light curves so far, the alerts, the SEDs, the time).

    render() draws a frame without drawing what stays the same again: the figure without the changing parts is kept
    as an image (the background, made on the first render), and each frame is drawn over it panel by panel, from the
    panel's lowest changing layer up (FLOORS), every artist there in matplotlib's own order (zorder, then the order
    they were made in, the order draw() always made them in). Tick marks and labels, titles and colour bars lie
    outside the panels' data areas and stay in the background; grid lines are redrawn where something changing lies
    beneath them. A rendered frame is the image a full redraw gives."""

    def __init__(self, fig, args, d):
        self.fig, self.args, self.d = fig, args, d
        B = args.band
        follow, colour_of, labels, stars = d["shown"], d["colour_of"], d["labels"], d["stars"]
        pulsators, source = d["pulsators"], d["source"]
        self.follow = follow
        grid = fig.add_gridspec(2, 2, width_ratios=[5.4, 4.6], height_ratios=[6, 4], wspace=0.16, hspace=0.3,
                                left=0.055, right=0.975, bottom=0.06, top=0.88)
        cmd = fig.add_subplot(grid[0, 0])
        pair = grid[0, 1].subgridspec(1, 2, wspace=0.06)
        image_ax = fig.add_subplot(pair[0])
        diff_ax = fig.add_subplot(pair[1], sharex=image_ax, sharey=image_ax)
        sed_ax = fig.add_subplot(grid[1, 1])
        light = grid[1, 0].subgridspec(2, 1, height_ratios=[2, 1], hspace=0.08)
        lc = fig.add_subplot(light[0])
        resid_ax = fig.add_subplot(light[1], sharex=lc)
        self.cmd, self.image_ax, self.diff_ax, self.sed_ax, self.lc, self.resid_ax = cmd, image_ax, diff_ax, sed_ax, lc, resid_ax
        # each panel's titles (the only text a new Axes holds): outside the panel, so in the background
        self.titles = [a for ax in (cmd, image_ax, diff_ax, sed_ax, lc, resid_ax) for a in ax.get_children()
                       if isinstance(a, mtext.Text)]

        # CMD: the constant stars in the block, each followed star's cycle (a transient: its track) and where it is now
        others = np.nonzero(d["in_block"] & (d["G"] < 20) & ~np.isin(np.arange(len(d["G"])), follow))[0]
        static_mag = d["static_mag"]
        cmd.scatter([static_mag["beta"][k] - static_mag["rho"][k] for k in others], [static_mag[B][k] for k in others],
                    s=4, color="#b9c4cc", zorder=1, label="Constant stars")
        self.now = {}
        for k in follow:
            cyc = d["follow_cycle"][k]
            if k in d["model_curve"]:
                cmd.plot(cyc["beta"] - cyc["rho"], cyc[B], color=colour_of[k], lw=1.3, alpha=0.9, zorder=3)
            else:  # a closed cycle
                cmd.plot(np.r_[cyc["beta"] - cyc["rho"], cyc["beta"][0] - cyc["rho"][0]], np.r_[cyc[B], cyc[B][0]],
                         color=colour_of[k], lw=1.3, alpha=0.9, zorder=3)
            self.now[k] = cmd.scatter([np.nan], [np.nan], s=60, color=colour_of[k], edgecolor=NAVY, linewidth=0.6,
                                      zorder=5)
        cmd.set(xlim=(np.percentile(d["cmd_c"], 0.5) - 0.15, np.percentile(d["cmd_c"], 99.5) + 0.15),
                ylim=(min(20.2, np.max(d["cmd_m"]) + 0.3), np.min(d["cmd_m"]) - 0.5),
                xlabel=r"$\beta-\rho$ (AB mag)", ylabel=rf"$m_{{\{B}}}$ (AB mag)")
        cmd.set_title("CMD", loc="left", weight="bold")
        cmd.legend(loc="lower left", fontsize=9, framealpha=0.85, edgecolor="none").set_zorder(2)
        cmd.grid(alpha=0.15)

        # light curves: measured points (hdps source detector, 4 px aperture) and the model behind, each relative to its own mean over the night
        ut = d["ut"]
        model_t = model_hours(ut)
        model_bjd = d["bjd"][0] + (model_t - ut[0]) / 24.0
        self.points, self.resid = {}, {}
        for k in follow:
            v = pulsators[int(source[k])]
            if k in d["model_curve"]:
                model = d["model_curve"][k]
            else:
                cyc = d["follow_cycle"][k][B]
                model = np.interp(((model_bjd - v["epoch"]) / v["period"]) % 1.0, np.linspace(0, 1, len(cyc) + 1), np.r_[cyc, cyc[0]])
            lc.plot(model_t, model - d["rel"][k]["model"], color=colour_of[k], lw=1.0, alpha=0.35)
            if k in d["model_curve"]:
                label = f"{labels[k]} {v['kind']}" + (f" z {v['transient']['z']:g}" if np.isfinite(v["transient"]["z"]) else "")
            else:
                label = f"{labels[k]} {v['kind']} {v['period']:.3f} d"
            self.points[k] = lc.errorbar([], [], yerr=[], fmt="o", ms=2.5, lw=0.5, color=colour_of[k], label=label)
            self.resid[k] = resid_ax.plot([], [], color=colour_of[k], lw=0.8)[0]
        engines = [engine_of(pulsators[int(source[k])]) for k in follow]
        model_name = " / ".join([e for e in ("MESA RSP", "PHOEBE") if e in engines]
                                + list(dict.fromkeys(e for e in engines if e not in ("MESA RSP", "PHOEBE"))))
        lc.plot([], [], color="0.5", lw=1.0, alpha=0.5, label=f"{model_name} model")
        self.alerts_shown = any(any(key in follow for key, _, _ in frame) for frame in d["alerted"])
        if self.alerts_shown:  # each alert: a band in the star's colour (made each frame: update)
            lc.fill_between([], [], color="0.5", alpha=0.25, lw=0, label="alert")
        self.spans = []
        self.now_line = []
        for ax in (lc, resid_ax):
            self.now_line.append(ax.axvline(ut[0], color=NAVY, lw=0.8, alpha=0.35))
            ax.grid(alpha=0.15)
            ax.set_xlim(ut[0] - 0.1, ut[-1] + 0.1)
        lo = np.nanmin([d["rel"][k]["span"][0] for k in follow] or [np.nan])
        hi = np.nanmax([d["rel"][k]["span"][1] for k in follow] or [np.nan])
        if not np.isfinite(lo):  # nothing lit all night
            lo, hi = -0.5, 0.5
        pad = 0.08 * max(hi - lo, 0.05)
        lc.set(ylim=(hi + pad, lo - 2.5 * pad), ylabel=rf"$\Delta m_{{\{B}}}$ (mag)")
        resid_lim = max(0.05, 1.3 * np.nanmax([d["rel"][k]["resid"] for k in follow] + [0.0]))
        resid_ax.set_ylim(-resid_lim, resid_lim)
        self.beyond = []  # points beyond the axes: a triangle at the edge
        for k in follow:
            for ax, series, lo_edge, hi_edge in ((lc, "m", lo - 2.5 * pad, hi + pad), (resid_ax, "r", -resid_lim, resid_lim)):
                for side, edge, marker in ((1, hi_edge, "v"), (-1, lo_edge, "^")):
                    line = ax.plot([], [], marker, ms=6, mfc="none", color=colour_of[k], clip_on=False)[0]
                    self.beyond.append((k, series, side, edge, line))
        lc.legend(loc="upper left", ncol=4, fontsize=7.5, frameon=True, framealpha=0.85, edgecolor="none", columnspacing=0.8,
                  handletextpad=0.3)
        lc.set_title("Light curves (hdps source detector)", loc="left", weight="bold")
        self.when = lc.set_title("", loc="right", fontsize=11, color=NAVY)
        plt.setp(lc.get_xticklabels(), visible=False)
        resid_ax.axhline(0, color="0.5", lw=0.6, ls=":")
        span_days = (d["times"][-1].jd - d["times"][0].jd) > 1
        resid_ax.set(xlabel=f"UT hours since {d['times'][0].iso[:10]} 0h" if span_days else f"UT hours ({args.date} night)",
                     ylabel="measured − true\n(mag)")
        lc.yaxis.set_label_coords(-0.075, 0.5)
        resid_ax.yaxis.set_label_coords(-0.075, 0.5)
        resid_ax.invert_yaxis()

        # the exposure as hdps sees it: its reprojected tiles (sky level removed), on an asinh stretch: the whole readout
        # window (d["window"], WindowMosaic: hdps's tiles on one grid, hdps's minipix grid over them) or around the block
        self.window = window = d.get("window")
        rgb = window["rgb"] if window else d["rgb"]
        self.norm = norm = AsinhNorm(linear_width=rgb["lw"], vmin=0.0, vmax=rgb["vmax"])
        self.face = image_ax.get_facecolor()
        whole = (-state.HALF, state.HALF, -state.HALF, state.HALF)
        self.image = image_ax.imshow(np.zeros((2, 2)), origin="lower", extent=whole, cmap="gray", vmin=0, vmax=1,
                                     interpolation="antialiased" if window else "nearest")  # (the window: > the panel)
        self.missing = image_ax.text(0, 0, "hdps could not process\nthis frame", color="white", ha="center", va="center",
                                     fontsize=11, visible=False)
        step = MINIPIX_PX * SKY_PX
        if window:  # hdps's minipix grid, the edges between its tiles' skies (WindowMosaic.grid), the block outlined
            self.grid_lines = []  # (each frame shows them inside its own readout window: update)
            for ax, colour, alpha in ((image_ax, "white", 0.3), (diff_ax, NAVY, 0.15)):
                self.grid_lines += [ax.add_collection(LineCollection(window["grid"], colors=colour, linewidths=0.3,
                                                                     alpha=alpha)),
                                    ax.add_collection(LineCollection(window["seams"], colors=colour, linewidths=0.6,
                                                                     alpha=2 * alpha))]
                ax.add_patch(Rectangle((-state.HALF, -state.HALF), 2 * state.HALF, 2 * state.HALF, fill=False, ec=BLUE, lw=1.0))
            self.outline = [ax.plot([], [], color=WINDOW_EDGE, lw=0.8)[0] for ax in (image_ax, diff_ax)]  # (update)
        else:
            for v in np.arange(-state.HALF, state.HALF + 1e-6, step):  # the minipix grid
                image_ax.plot([v, v], [-state.HALF, state.HALF], color="white", lw=0.5, alpha=0.35)
                image_ax.plot([-state.HALF, state.HALF], [v, v], color="white", lw=0.5, alpha=0.35)
        image_ax.add_patch(Rectangle((-step / 2, -step / 2), step, step, fill=False, ec=BLUE, lw=1.2))  # the central minipix
        self.star_marks(image_ax, follow, stars, labels, colour_of, 1.2)
        view = window["view"] if window else (-state.HALF - 2, state.HALF + 2, -state.HALF - 2, state.HALF + 2)
        image_ax.set(xlim=view[:2], ylim=view[2:], xlabel="Offset (arcsec)", ylabel="Offset (arcsec)")
        self.telescope = None
        if d["survey"] is None:
            image_ax.set_title(f"Exposure ({SYMBOLS['beta']} blue, {SYMBOLS['rho']} red)", loc="left", weight="bold")
        else:
            image_ax.set_title(f"Exposure ({SYMBOLS[B]})", loc="left", weight="bold")
            self.telescope = image_ax.text(0.97, 0.97, "", transform=image_ax.transAxes, va="top", ha="right", fontsize=8,
                                           color="white", bbox={"facecolor": NAVY, "alpha": 0.65, "edgecolor": "none", "pad": 3})
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="gray"), ax=image_ax, orientation="horizontal", fraction=0.05,
                           pad=0.13, label=f"{SYMBOLS[B]} electrons / pixel above sky")
        bar.set_ticks([v for v in (0, 10, 100, 1000, 10000) if v <= rgb["vmax"]])
        bar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        bar.ax.tick_params(labelsize=7)
        self.exposure = image_ax.text(0.03, 0.97, "", transform=image_ax.transAxes, va="top", fontsize=8, color="white",
                                      bbox={"facecolor": NAVY, "alpha": 0.65, "edgecolor": "none", "pad": 3})

        # the difference image (saturated pixels grey)
        diverging = plt.get_cmap("RdBu_r").copy()
        diverging.set_bad("#bfbfbf")
        # asinh stretch, linear out to the sky noise: the noise, faint residuals and the large variables all show
        dlim = window["dlim"] if window else d["dlim"]
        dnorm = AsinhNorm(linear_width=rgb["lw"], vmin=-dlim, vmax=dlim)
        self.blank = np.full((2, 2), np.nan)
        self.diff = diff = diff_ax.imshow(self.blank, origin="lower", interpolation="antialiased" if window else "nearest",
                                          cmap=diverging, norm=dnorm, extent=whole)
        self.star_marks(diff_ax, follow, stars, labels, colour_of, 1.0)
        self.marks = {image_ax: [], diff_ax: []}  # each image's alert marks (Text artists kept from frame to frame)
        # the marks' size and height above their object (draw_alert_marks): in the window, smaller, in points
        self.mark_style = dict(size=9, lift=0.0, points=2.5) if window else dict(size=24, lift=4.5, points=0.0)
        diff_ax.set_title("Difference (PSF-matched)", loc="left", weight="bold")
        diff_ax.text(0.03, 0.97, "hdps SFFT against its coadd reference" + ("\n(transients left out)" if d["model_curve"] else "")
                     + ("\ngrey: stars bright in the reference" if d.get("masked") else ""),
                     transform=diff_ax.transAxes, va="top", fontsize=7.5, color=NAVY,
                     bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 2})
        diff_ax.set(xlim=view[:2], ylim=view[2:], xlabel="Offset (arcsec)")
        plt.setp(diff_ax.get_yticklabels(), visible=False)
        dbar = fig.colorbar(diff, ax=diff_ax, orientation="horizontal", fraction=0.05, pad=0.13,
                            label="difference (electrons / pixel)")
        ticks = [v for v in (10, 100, 1000, 10000) if v <= dlim]
        dbar.set_ticks([-v for v in ticks[::-1]] + [0] + ticks)
        dbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        dbar.ax.tick_params(labelsize=7)

        # SEDs of the followed stars now, and as each Argus band records them (x filter x optics x QE x atmosphere^airmass)
        ab = d["ab"]
        self.response_wave = response_wave = ab.tp.wav.to_value(u.nm)
        self.seds = {}
        for k in follow:
            line = sed_ax.plot([], [], color=colour_of[k], lw=0.9)[0]
            self.seds[k] = (line, [sed_ax.plot([], [], color=colour_of[k], lw=0.9, ls=":")[0] for _ in BANDS])
        sed_ax.set(yscale="log", ylim=d["sed_ylim"], xlim=(300, 1000),
                   xlabel="Wavelength (nm)", ylabel=r"$f_\lambda$ at Argus (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)")
        sed_ax.set_title("SED", loc="left", weight="bold")
        self.response_ax = response_ax = sed_ax.twinx()
        self.titles += [a for a in response_ax.get_children() if isinstance(a, mtext.Text)]
        response_ax.spines["right"].set_visible(True)
        self.response = [response_ax.fill_between(response_wave, np.zeros(len(response_wave)), color=COLORS[name], alpha=0.10)
                         for name in BANDS]
        response_ax.set(ylim=(0, 1), ylabel="Detected response")
        sed_ax.plot([], [], color="0.4", lw=0.9, label="SED")
        sed_ax.plot([], [], color="0.4", lw=0.9, ls=":", label=r"SED × $\beta$, $\rho$ system response")
        sed_ax.legend(loc="upper right", fontsize=8, frameon=False)
        sed_ax.grid(alpha=0.12)

        # the title: where and when on one line; what is shown on a second (the shown stars, and how many more are simulated)
        what = pulsator_title(follow, pulsators, source) if follow else "no modelled variables in the block"
        more = len(d["follow"]) - len(follow)
        if more:
            what += f" · {more} more simulated outside this view or not shown"
        add_logo(fig, (0.012, 0.925, 0.15, 0.065))
        fig.suptitle(f"{d['centre']} ({FIELDS[args.field]['name']}) — {block_label()}"
                     + (f" in {window['label']}" if window else "") + f" — {args.date} — "
                     f"{args.exptime:g} s exposures — {SYMBOLS[B]}", fontsize=13.5, x=0.58, y=0.985)
        fig.text(0.58, 0.935, "\n".join(textwrap.wrap(what, 120, break_long_words=False)), ha="center", va="top",
                 fontsize=10.5, color=NAVY)

        # each panel's lowest changing layer (zorder): the images and alert bands lie at the bottom of theirs
        self.floors = {cmd: 5, image_ax: 0, diff_ax: 0, lc: 0, resid_ax: 0, sed_ax: 2, response_ax: 1}
        self.changing_titles = [self.when]  # the time above the light curves: drawn each frame
        self._background = self._size = None
        self._grids = {}  # each axis's grid lines drawn (as the background was drawn)

    def star_marks(self, ax, follow, stars, labels, colour_of, lw):
        """Each shown followed star's circle (3" about it) and label in an image panel; in the whole window, where
        arcsec are too small to see, sized in points."""
        for k in follow:
            x, y = stars["dx"][k], stars["dy"][k]
            if self.window:
                ax.plot(x, y, "o", ms=7, mfc="none", mec=colour_of[k], mew=lw)
                ax.annotate(labels[k], (x, y), xytext=(3, 3), textcoords="offset points", fontsize=8, color=colour_of[k],
                            weight="bold", clip_on=True)
            else:
                ax.add_patch(Circle((x, y), 3.0, fill=False, ec=colour_of[k], lw=lw))
                ax.text(x + 3.2, y + 3.2, labels[k], fontsize=8, color=colour_of[k], weight="bold", clip_on=True)

    def update(self, exposures, j):
        """Set frame j (survey row j; the frame's number among those shown is d["n"], and the light curves so far are
        d["history"])."""
        d, args, B = self.d, self.args, self.args.band
        n, true_mag, colour_of = d["n"], d["true_mag"], d["colour_of"]
        for k in self.follow:
            self.now[k].set_offsets([[true_mag("beta", k, j) - true_mag("rho", k, j), true_mag(B, k, j)]])
        shown = {}
        for k in self.follow:
            h = d["history"][k]
            t = np.array(h["t"], float)
            m, e = np.array(h["m"], float), np.array(h["e"], float)
            ok = np.isfinite(m)
            y = m[ok] - d["rel"][k]["measured"]
            data, _, bars = self.points[k].lines
            data.set_data(t[ok], y)
            bars[0].set_segments([[(x, a - s), (x, a + s)] for x, a, s in zip(t[ok], y, e[ok])])
            resid = m - np.array(h["true"], float)
            if d["survey"] is not None:  # not across the gaps between ratchets
                gap = np.r_[False, np.diff(t) > 0.1]
                t, resid = np.insert(t, np.nonzero(gap)[0], np.nan), np.insert(resid, np.nonzero(gap)[0], np.nan)
            self.resid[k].set_data(t, resid)
            shown[k, "m"], shown[k, "r"] = (np.array(h["t"], float)[ok], y), (t, resid)
        for span in self.spans:
            span.remove()
        self.spans = []
        if self.alerts_shown:
            for k in self.follow:
                for lo_t, hi_t in alert_spans(d, k, n):
                    for ax in (self.lc, self.resid_ax):
                        self.spans.append(ax.axvspan(lo_t, hi_t, color=colour_of[k], alpha=0.16, lw=0, zorder=0))
        for line in self.now_line:
            line.set_xdata([d["ut"][j], d["ut"][j]])
        for k, series, side, edge, line in self.beyond:
            t, y = shown[k, series]
            beyond = (y > edge) if side > 0 else (y < edge)
            line.set_data(t[beyond], np.full(int(np.sum(beyond)), edge))
        self.when.set_text(f"UT {d['times'][j].iso[11:16]} · airmass {d['airmass'][j]:.2f} · seeing {d['seeing'][j]:.2f}″")

        e = exposures[B]
        extent = e["extent"] if e is not None else (-state.HALF, state.HALF, -state.HALF, state.HALF)
        if e is not None:
            self.image.set_data(np.clip(self.norm(np.maximum(e["image"] - e["background"], 0.0)), 0, 1))
            self.image.set_extent(extent)
        self.image.set_visible(e is not None)
        self.missing.set_visible(e is None)
        self.image_ax.set_facecolor(self.face if e is not None else NAVY)
        self.diff.set_data(e["difference"] if e is not None else self.blank)
        self.diff.set_extent(extent)
        if self.window:  # the window's own outline; the difference blank where the exposure is (no tile image)
            # (always an array: matplotlib cannot set an image's alpha from an array back to None)
            self.diff.set_alpha(np.isfinite(e["image"]).astype(float) if e is not None else np.ones(self.blank.shape))
            footprint = self.window["footprints"][n]
            for line in self.outline:
                line.set_data(footprint[:, 0], footprint[:, 1])
            for lines in self.grid_lines:
                lines.set_clip_path(MPath(footprint), lines.axes.transData)
        if self.telescope is not None:
            self.telescope.set_text(f"telescope {int(d['survey']['tel_id'][j])}\n"
                                    f"{float(d['survey']['field_angle'][j]):.2f}° off-axis")
        self.exposure.set_text(f"{args.exptime:g} s · exposure {j + 1}")
        # the alert marks inside the images (around the block, most alerts lie elsewhere in hdps's readout window:
        # drawn, a mark there would be clipped away entirely; a mark stands on its point, at most ~35" tall)
        if self.window:
            x0, x1, y0, y1 = self.window["view"]
            marks = [m for m in alert_marks(d, n, args.fps) if x0 < m[0] < x1 and y0 < m[1] < y1]
        else:
            edge, margin = state.HALF + 2, 50.0
            marks = [m for m in alert_marks(d, n, args.fps)
                     if -edge - margin < m[0] < edge + margin and -edge - margin < m[1] + 4.5 < edge]
        style = self.mark_style
        for ax, pool in self.marks.items():
            if len(pool) < len(marks):  # the marks made as draw_alert_marks makes them, the same in both images
                pool += draw_alert_marks(ax, marks[len(pool):], **style)
            for text, mark in zip(pool, marks):
                set_alert_mark(text, *mark, size=style["size"], lift=style["lift"])
            for text in pool[len(marks):]:
                text.set_visible(False)

        tp, airmass = d["ab"].tp, d["airmass"][j]
        for k in self.follow:
            wave, sed = d["models"][k].wave.to_value(u.nm), d["follow_sed"][k][j]
            inside = (wave >= 300) & (wave <= 1000)
            line, recorded_lines = self.seds[k]
            line.set_data(wave[inside], sed[inside])
            on_grid = np.interp(self.response_wave, wave, sed, left=0.0, right=0.0)
            for (name, band), rec in zip(BANDS.items(), recorded_lines):
                recorded = on_grid * system_response(tp, band, airmass)
                rec.set_data(self.response_wave[recorded > 0], recorded[recorded > 0])
        for (name, band), fill in zip(BANDS.items(), self.response):
            fill.set_data(self.response_wave, system_response(tp, band, airmass), 0)

    def _redrawn(self, ax):
        """What a frame draws in panel ax: the artists from its floor up, in the order Axes.draw draws them, the axes'
        grid lines in place of the axes (their tick marks and labels stay in the background)."""
        floor = self.floors[ax]
        out = []
        for a in sorted(ax.get_children(), key=lambda a: a.get_zorder()):  # stable: equal zorders in the order made
            if a is ax.patch or a.get_zorder() < floor or not a.get_visible():
                continue
            if any(a is t for t in self.titles) and not any(a is t for t in self.changing_titles):
                continue
            if isinstance(a, maxis.Axis):
                if a not in self._grids:
                    self._grids[a] = self._grid_lines(a)
                out += self._grids[a]
            else:
                out.append(a)
        return out

    @staticmethod
    def _grid_lines(axis):
        """The visible grid lines an Axis draws (those of its ticks inside the view, as Axis._update_ticks picks
        them)."""
        lines = []
        low, high = sorted(axis.get_transform().transform(np.asarray(axis.get_view_interval(), float)))
        tolerance = 1e-10 * (high - low)
        for ticks, locs in ((axis.get_major_ticks(len(axis.get_majorticklocs())), axis.get_majorticklocs()),
                            (axis.get_minor_ticks(len(axis.get_minorticklocs())), axis.get_minorticklocs())):
            for tick, loc in zip(ticks, locs):
                at = float(axis.get_transform().transform(np.asarray([loc], float))[0])
                if tick.gridline.get_visible() and low - tolerance <= at <= high + tolerance:
                    lines.append(tick.gridline)
        return lines

    def _capture(self):
        """The background: the figure drawn without what frames draw (Agg's draw, so an interactive canvas does not
        show it)."""
        canvas = self.fig.canvas
        self._grids = {}
        for ax, floor in self.floors.items():  # the ticks of each axis whose grid is redrawn, made now to hide their lines
            for axis in (ax.xaxis, ax.yaxis):
                if axis.get_zorder() >= floor:
                    axis.get_gridlines()
        # hidden while the background is drawn; a changing title is emptied instead (a hidden title would move the
        # panel's other titles: Axes._update_title_position)
        hidden = [a for ax in self.floors for a in self._redrawn(ax) if not any(a is t for t in self.changing_titles)]
        hidden += [ax.patch for ax, floor in self.floors.items() if floor <= 0]  # redrawn first: its colour may change
        was = [a.get_visible() for a in hidden]
        texts = [t.get_text() for t in self.changing_titles]
        for a in hidden:
            a.set_visible(False)
        for t in self.changing_titles:
            t.set_text("")
        FigureCanvasAgg.draw(canvas)
        self._background = canvas.copy_from_bbox(self.fig.bbox)
        for a, visible in zip(hidden, was):
            a.set_visible(visible)
        for t, text in zip(self.changing_titles, texts):
            t.set_text(text)

    def render(self):
        """Draw the frame set by update(): the RGBA image of the whole figure (a memoryview of the canvas's buffer)."""
        canvas = self.fig.canvas
        size = (self.fig.dpi, *canvas.get_width_height(physical=True))
        if self._background is None or size != self._size:
            self._capture()
            self._size = size
        canvas.restore_region(self._background)
        renderer = canvas.get_renderer()
        for ax in sorted((a for a in self.fig.get_children() if a in self.floors), key=lambda a: a.get_zorder()):
            if self.floors[ax] <= 0:  # (the figure's order: equal zorders as made)
                ax.patch.draw(renderer)
            for a in self._redrawn(ax):
                a.draw(renderer)
        return canvas.buffer_rgba()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        main()
