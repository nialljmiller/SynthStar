"""Stage 4 — measurement and the movie: hdps's data products (solved frames, difference tiles, alert tables) joined
to the scene truth -> photometry, labelled alerts and cut-outs, and the movie (CMD, colour exposure, difference
image, light curves, SEDs). Reads hdps's OUTPUT FILES and the scene; it never runs hdps's executor. Imports the
stage-2 image maker (engine.imaging) for the scene geometry and labels it shares."""
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

from synthstar import dust  # noqa: E402
from synthstar.argus import BANDS, BLUE, COLORS, FLAM, NAVY, RED, SYMBOLS, add_logo, apply_style, system_response  # noqa: E402
from synthstar.seds import Templates  # noqa: E402
from synthstar.engine import state  # noqa: E402
from synthstar.engine.state import MINIPIX_PX, SKY_PX, MARGIN, AIRMASS_GRID, PSF_PHASES, FIELDS, use_field, set_block, block_label  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]  # repo root
import textwrap  # noqa: E402
from argus_sim import skymap_shim as sh  # noqa: E402
from synthstar.engine.imaging import (block_centre_label, block_offsets, block_radec, engine_of,  # noqa: E402
                                       pulsator_title)  # stage-2 scene geometry and labels this layer reuses

MAX_SHOWN = 6  # followed stars the movie draws (an injected run follows hundreds; all are measured and labelled)
WINDOW_EDGE = "#f0a202"  # the frame's readout window, outlined over the whole-window images


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

    from synthstar import alert_files

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

    from synthstar.pipeline import truth

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
