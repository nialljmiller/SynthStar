"""The edited star added into hdps's own frames (the field pack's block cut-outs of hdps's science tile and difference
image), and the airmass offset applied to them.

Stand-ins, each stated on screen:
- the star's PSF: ArgusSim's (ZemaxPSF at the frame's seeing and field angle) resampled onto hdps's tile grid
  (0.879"), without the small rotation between the sensor and the tile; when the pack's run drew its stars otherwise
  (the PSF's outer wing, [image] psf_wing; or a box wider than ArgusSim's 28 px with the on-sky halo, psf_box and
  psf_halo), the star drawn as the run drew its stars (synthstar.sensor.expected) over the cut-out, then resampled;
- its brightness on hdps's tile: hdps's reprojection keeps surface brightness (electrons per sensor-pixel area), so a
  star's light summed over the tile is its electrons x (sensor pixel / tile pixel)^2 = 1.31; found by comparing an
  injected star hdps rendered with the same light added here;
- its noise: photon noise with the variance-to-level ratio of the frame's own sky (hdps's interpolation smooths the
  noise: about 0.67), the same draw in the science and the difference image (the reference is deep);
- in the difference image it adds (now - reference) x PSF: SFFT's match taken as perfect for it;
- saturation: a tile pixel (surface brightness, so on the sensor's scale) holds at most the sensor's full well; the
  light cut off a saturated core is missing from the difference image too;
- airmass offset dX >= 0 (observing it lower): every frame dimmed by the band's extinction, blurred to the seeing at
  the new airmass (X^0.6), and given the extra sky noise (sky ~ X); the field's own alerts and photometry stay as
  observed.
"""
import os
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

SKY_PX = 900.0 / 1024.0  # hdps tile pixel (arcsec)
PSF_CACHE = Path(__file__).resolve().parents[2] / "data" / "interactive" / "psf"  # stamps, per ArgusSim version


@lru_cache(maxsize=1)
def _psfs():
    from argus_sim import c as cfg

    from synthstar import sensor

    return sensor.PSFs(cfg)


@lru_cache(maxsize=4)
def _wing_psfs(wing):
    """sensor.PSFs with a run's [image] settings (wing: their items, as a tuple)."""
    from argus_sim import c as cfg

    from synthstar import sensor

    return sensor.PSFs(cfg, wing=dict(wing))


def wing_key(image):
    """A pack's [image] settings as _wing_psfs takes them, when its stars were drawn other than with ArgusSim's 28 px
    stamp (the PSF's outer wing, or a wider psf_box with or without psf_halo); None otherwise (packs made before
    2026-10-26 have none)."""
    if not image or not (image.get("psf_wing") or int(image.get("psf_box", 28)) > 28):
        return None
    return tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in image.items()))


def plate():
    from argus_sim import c as cfg

    return float(cfg.telescope.plate_scale)


def fullwell():
    from argus_sim import c as cfg

    return float(cfg.telescope.fullwell)


def _cached(name, make):
    """An array from the PSF cache (data/interactive/psf/argus_sim-<version>/<name>.npy), made and saved when
    missing: building ArgusSim's PSF set takes seconds in every process, a stamp tens of milliseconds."""
    import argus_sim

    folder = PSF_CACHE / f"argus_sim-{argus_sim.__version__}"
    path = folder / f"{name}.npy"
    if path.exists():
        return np.load(path)
    value = np.asarray(make(), float)
    folder.mkdir(parents=True, exist_ok=True)
    part = folder / f"{name}.{os.getpid()}.{threading.get_ident()}.part.npy"  # (made by several processes or threads
    # at once: each writes its own file, moved into place whole)
    np.save(part, value)
    os.replace(part, path)
    return value


@lru_cache(maxsize=4)
def _angles(band):
    """The field angles ArgusSim's PSFs are tabulated at (sensor.PSFs.angles)."""
    return _cached(f"angles_{band}", lambda: _psfs().angles(band))


@lru_cache(maxsize=4096)
def _stamp(band, seeing, angle_index):
    seeing = round(float(seeing), 2)
    return _cached(f"stamp_{band}_{seeing:.2f}_{int(angle_index)}",
                   lambda: _psfs().stamp(band, seeing, int(angle_index), 0, 0))


def psf_on_tile(band, seeing, field_angle, dx, dy, extent, shape, half_arcsec=20.0, wing=None, electrons=0.0):
    """Unit-flux PSF of a star at (dx, dy) arcsec on a cut-out with this extent (x0, x1, y0, y1 arcsec) and shape:
    (rows slice, cols slice, values) of the patch around it. wing (wing_key) and the star's electrons: with the run's
    PSF wing, out to where it was drawn for a star this bright (the part on the cut-out)."""
    from scipy.ndimage import map_coordinates

    angles = _angles(band)
    if wing is not None:
        return _winged_on_tile(band, seeing, int(np.abs(angles - field_angle).argmin()), dx, dy, extent, shape,
                               half_arcsec, wing, electrons)
    stamp = _stamp(band, seeing, int(np.abs(angles - field_angle).argmin()))
    yy, xx = np.indices(stamp.shape)
    cy, cx = float((stamp * yy).sum()), float((stamp * xx).sum())
    ny, nx = shape
    px_x = (extent[1] - extent[0]) / nx
    px_y = (extent[3] - extent[2]) / ny
    i0 = max(int((dx - half_arcsec - extent[0]) / px_x), 0)
    i1 = min(int((dx + half_arcsec - extent[0]) / px_x) + 1, nx)
    j0 = max(int((dy - half_arcsec - extent[2]) / px_y), 0)
    j1 = min(int((dy + half_arcsec - extent[2]) / px_y) + 1, ny)
    if i1 <= i0 or j1 <= j0:
        return slice(0, 0), slice(0, 0), np.zeros((0, 0))
    x = extent[0] + (np.arange(i0, i1) + 0.5) * px_x  # tile pixel centres (arcsec)
    y = extent[2] + (np.arange(j0, j1) + 0.5) * px_y
    u = cx + (x - dx) / plate()  # where they fall on the sensor-pixel stamp
    v = cy + (y - dy) / plate()
    vv, uu = np.meshgrid(v, u, indexing="ij")
    values = np.clip(map_coordinates(stamp, [vv, uu], order=1, mode="constant", cval=0.0), 0.0, None)
    total = values.sum()
    return slice(j0, j1), slice(i0, i1), (values / total if total > 0 else values)


def _winged_on_tile(band, seeing, angle_index, dx, dy, extent, shape, half_arcsec, wing, electrons):
    """psf_on_tile with the run's PSF (its wing, or its wide box and halo): the star drawn as the run's frames draw it
    (sensor.expected: the stamp blended into the wing out to the radius its brightness reaches; in a wide box, the
    square outside which its light is below psf_box_floor), on sensor pixels over the patch, resampled onto the tile
    grid; values sum to the drawn light on the patch (the core's resampling loss restored), not to 1."""
    from scipy.ndimage import map_coordinates

    from synthstar import sensor

    psfs = _wing_psfs(wing)
    electrons = max(float(electrons), 1.0)
    if psfs.wing is not None:
        radius = psfs.wing_radius(band, seeing, angle_index, electrons)
    else:  # sensor.expected's reach for a star in a wide box
        radius = float(psfs.half_widths(band, seeing, angle_index, electrons)) + 2.0
    reach = max(half_arcsec, radius * plate())
    ny, nx = shape
    px_x = (extent[1] - extent[0]) / nx
    px_y = (extent[3] - extent[2]) / ny
    i0 = max(int((dx - reach - extent[0]) / px_x), 0)
    i1 = min(int((dx + reach - extent[0]) / px_x) + 1, nx)
    j0 = max(int((dy - reach - extent[2]) / px_y), 0)
    j1 = min(int((dy + reach - extent[2]) / px_y) + 1, ny)
    if i1 <= i0 or j1 <= j0:
        return slice(0, 0), slice(0, 0), np.zeros((0, 0))
    x = extent[0] + (np.arange(i0, i1) + 0.5) * px_x  # tile pixel centres (arcsec)
    y = extent[2] + (np.arange(j0, j1) + 0.5) * px_y
    u, v = (x - dx) / plate(), (y - dy) / plate()  # sensor pixels from the star
    u0, v0 = int(np.floor(u.min())) - 2, int(np.floor(v.min())) - 2
    win = (0, 0, int(np.ceil(u.max())) + 3 - u0, int(np.ceil(v.max())) + 3 - v0)
    angle = np.array([float(psfs.angles(band)[angle_index])])
    image = sensor.expected(np.array([-u0 + 0.0]), np.array([-v0 + 0.0]), np.array([electrons]), angle, band, seeing,
                            0.0, win, psfs) / electrons
    vv, uu = np.meshgrid(v - v0, u - u0, indexing="ij")
    values = np.clip(map_coordinates(image, [vv, uu], order=1, mode="constant", cval=0.0), 0.0, None)
    total = values.sum()
    drawn = float(image.sum()) * area_ratio()  # the light on the patch, on the tile's surface-brightness scale
    return slice(j0, j1), slice(i0, i1), (values * drawn / total / area_ratio() if total > 0 else values)


def extinction_ratio(band, x0, x1):
    """The fraction of a star's light left at airmass x1 against x0 in a band (ArgusSim's atmosphere, a flat
    spectrum); airmasses to 0.01."""
    return _extinction_ratio(band, round(float(x0), 2), round(float(x1), 2))


@lru_cache(maxsize=4096)
def _extinction_ratio(band, x0, x1):
    import astropy.units as u

    import night_movie as nm
    from synthstar.interactive.templates import photometry

    ab = photometry()
    wave = np.linspace(3000.0, 10500.0, 400) * u.AA
    flat = np.full(400, 1e-16) * nm.FLAM
    e0 = ab.electrons_from_sed(nm.BANDS[band], wave, flat, airmass=x0).value
    e1 = ab.electrons_from_sed(nm.BANDS[band], wave, flat, airmass=x1).value
    return float(e1 / e0)


def area_ratio():
    """Sensor pixel area over tile pixel area: a star of N electrons sums to N x this on hdps's tile."""
    return (plate() / SKY_PX) ** 2


def noise_ratio(cut):
    """(sky level, its noise variance / level) of a frame's science cut-out: what the photon noise of light added
    to it must have."""
    S = np.asarray(cut["science"], np.float64)
    S = S[np.isfinite(S)]
    sky = float(np.median(S))
    return sky, float((1.4826 * np.median(np.abs(S - sky))) ** 2 / max(sky, 1e-9))


def frame(cut, band, seeing, field_angle, airmass, star_e, ref_e, dx, dy, d_airmass=0.0, seed=0, sky=None, q=None,
          wing=None):
    """One frame's cut-outs with the edited star added (and the airmass offset): dict(science, difference, extent,
    psf (rows, cols, values), added (star electrons per pixel), seeing (now), keep (the extinction factor applied),
    saturated (the star's core reached the full well)). star_e / ref_e: the star's electrons in this exposure
    and in the reference (already at the offset airmass); sky, q: the science cut-out's sky level and its noise
    variance / level (noise_ratio, computed if not given); wing: the pack's PSF wing (wing_key), None for none."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    if sky is None or q is None:
        sky, q = noise_ratio(cut)
    sci = np.array(cut["science"], dtype=np.float64)
    diff = np.array(cut["difference"], dtype=np.float64)
    extent, shape = cut["extent"], sci.shape
    seeing_now, keep = seeing, 1.0
    if d_airmass > 0:  # observed lower: extinction, broader seeing, more sky
        x1 = airmass + d_airmass
        keep = extinction_ratio(band, airmass, x1)
        seeing_now = seeing * (x1 / airmass) ** 0.6
        sigma = np.sqrt(max(seeing_now ** 2 - seeing ** 2, 0.0)) / 2.3548 / SKY_PX
        stars = np.nan_to_num(sci - sky)
        sci = gaussian_filter(stars, sigma) * keep + sky * (x1 / airmass)
        diff = gaussian_filter(np.nan_to_num(diff), sigma) * keep
        extra = rng.normal(0.0, np.sqrt(max(q * sky * (x1 / airmass - 1.0), 0.0)), shape)
        sci += extra
        diff += extra
    rows, cols, psf = psf_on_tile(band, seeing_now, field_angle, dx, dy, extent, shape, wing=wing,
                                  electrons=max(star_e, ref_e))
    area = area_ratio()
    added = star_e * area * psf
    noise = rng.normal(0.0, 1.0, psf.shape) * np.sqrt(q * np.clip(added, 0.0, None))
    sci[rows, cols] += added + noise
    diff[rows, cols] += (star_e - ref_e) * area * psf + noise
    excess = np.clip(sci[rows, cols] - fullwell(), 0.0, None)  # the sensor's full well: a saturated core is cut off
    sci[rows, cols] -= excess
    diff[rows, cols] -= excess
    return dict(science=sci, difference=diff, extent=extent, psf=(rows, cols, psf), added=added,
                seeing=seeing_now, keep=keep, saturated=bool(excess.any()))
