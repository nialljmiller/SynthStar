"""Observing as the survey does (observe.py's [night] observing = "survey"): which exposures Argus actually takes of a
position, through which telescope, in which band, where in that telescope's field, and the seeing of each ratchet.

All from ArgusSim:
- coverage: coverage_exact.covering, the observation's ring layout ([argussim.packing_strategy] layout_file, default
  A170r_1200: 1200 telescopes, each in g = beta or rho by the layout's band table; thinB, 1190 telescopes, before
  2026-10-28) and ratchet schedule (60 s exposures, 14 min tracking + 1 min reset, Sun below -18 deg); a position is
  observed in an exposure when it falls inside a telescope's field. Where two telescopes overlap it is observed by
  both (two records, often in both bands); between telescopes and outside the array's footprint, not at all.
- field angle: the position's angle from that telescope's optical axis, in the array frame (the PSF depends on it).
- seeing: one zenith seeing per ratchet, drawn from the site's seasonal model (survey.draw_seasonal_seeing, Barker et
  al. 2003), x airmass^0.6 per exposure (as in ArgusSim); seeded, so a seed is one realisation.
Not included: weather (lost nights), clouds (transparency 1).
"""
from functools import lru_cache

import astropy.units as u
from astropy.coordinates import CIRS, SkyCoord
from astropy.table import Table
from astropy.time import Time
import numpy as np

from argus_sim import c as ARGUS_CONFIG
from argus_sim.coverage_exact import Layout, _hadec_to_altaz, _unit, covering
from argus_sim.survey import draw_seasonal_seeing

BAND_NAMES = {"g": "beta", "rho": "rho"}  # ArgusSim's filter names -> ours


def ut_hours(text):
    """'HH:MM' as hours, 20-24 h as -4-0 h (across midnight), as argus_image.night."""
    h = int(text[:2]) + int(text[3:5]) / 60
    return h - 24 if h >= 20 else h


@lru_cache(maxsize=4)
def array_layout(name):
    """ArgusSim's ring layout `name` ([argussim.packing_strategy] layout_file), with its bands (Layout.from_config)."""
    return Layout.from_config(layout=name)


def check_layout(rows, name):
    """Stop unless the exposures `rows` (scene/exposures.ecsv, from exposures below) were found with layout `name`: their
    ota and tel_id are that layout's rows, so with another layout every frame would be another telescope's."""
    made = rows.meta.get("layout")
    if made != name:
        raise SystemExit(f"scene/exposures.ecsv was made with the ring layout {made!r}, the observation asks for {name!r} "
                         f"([argussim.packing_strategy] layout_file): make the scene again (observe.py), or, to use it "
                         f"as it is, set layout_file = \"{made}\" in the observation file")


def exposures(ra, dec, date, nights=1, seed=0, start=None, end=None, *, layout):
    """Every exposure of the position in the nights starting on local `date` (UTC 23:00 + 14 h each), within the UT
    window start-end, through ArgusSim's ring layout named `layout` ([argussim.packing_strategy] layout_file): one row
    per (exposure, telescope). Columns: time (UTC isot, mid-exposure), ratchet, tel_id, ota (the telescope's row in
    that layout), theta_deg (the exposure's tracking angle), band (beta / rho), field_angle (deg), seeing_zenith
    (arcsec, the ratchet's). meta layout: the layout's name."""
    t0 = Time(f"{date} 23:00:00", scale="utc")
    t1 = t0 + (nights - 1) * u.day + 14 * u.hour
    layout = array_layout(layout)
    cover = covering(ra, dec, t0, t1, layout=layout, apparent=True)
    if len(cover.point) == 0:
        return Table(names=["time", "ratchet", "tel_id", "ota", "theta_deg", "band", "field_angle", "seeing_zenith"],
                     dtype=[str, int, int, int, float, str, float, float], meta=dict(layout=layout.name))
    j = cover.exposure
    times = cover.times[j]

    # the field angle: the position in the array frame (as covering places it) against the telescope's axis
    cirs = SkyCoord(ra * u.deg, dec * u.deg).transform_to(CIRS(obstime=times))
    era = times.earth_rotation_angle(longitude=ARGUS_CONFIG.observatory.longitude * u.deg).deg
    ha = (era + cover.theta_deg[j] - cirs.ra.deg + 180.0) % 360.0 - 180.0
    alt, az = _hadec_to_altaz(ha, cirs.dec.deg, ARGUS_CONFIG.observatory.latitude, np)
    axis = _unit(layout.alt_deg[cover.ota], layout.az_deg[cover.ota], np)
    field_angle = np.degrees(np.arccos(np.clip(np.sum(_unit(alt, az, np) * axis, axis=1), -1, 1)))

    # one zenith seeing per ratchet, in time order
    rng = np.random.default_rng(seed)
    ratchet = cover.ratchet[j]
    first = {}
    for k, t in zip(ratchet, times):
        first.setdefault(int(k), t)
    seeing = {k: draw_seasonal_seeing(t, rng) for k, t in sorted(first.items())}

    out = Table(dict(time=times.isot, ratchet=ratchet, tel_id=cover.tel_id, ota=cover.ota, theta_deg=cover.theta_deg[j],
                     band=[BAND_NAMES[b] for b in cover.band],
                     field_angle=field_angle, seeing_zenith=[seeing[int(k)] for k in ratchet]))
    clock = (times.datetime64.astype("datetime64[s]").astype(np.int64) % 86400) / 3600
    clock = np.where(clock >= 20, clock - 24, clock)
    keep = np.ones(len(out), bool)
    if start:
        keep &= clock >= ut_hours(start)
    if end:
        keep &= clock <= ut_hours(end)
    out = out[keep]
    out.meta.update(dict(layout=layout.name, seed=seed, nights=nights, date=date, ra=ra, dec=dec,
                         note="ArgusSim coverage_exact.covering (apparent=True); seeing per ratchet "
                              "survey.draw_seasonal_seeing; no weather, transparency 1"))
    return out
