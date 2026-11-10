"""When a pointing is observed, from ArgusSim's own survey rules: the night's exposures, the nearest night it is up, the
night its variables change most, and BJD_TDB at the site. Used by observe.py and night_movie.py.

The exposures are ArgusSim's schedule (coverage_exact.Schedule, built as Survey builds it from config.survey): a
base_cadence_s cadence in ratchets of ratchet_len minutes on UTC boundaries, the last reset_min of each a reset, and
the Sun below survey.min_sun_alt. Of those, the field is observed while it is at or above observatory.min_alt (the
lowest the array points); [night] max_airmass, if given, is a further limit of the observation's own. Airmass is
ArgusSim's (Pickering 2002), as its sky model and survey use.
"""
import copy
from datetime import date as Date, timedelta

import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
from astropy.time import Time
import numpy as np

from argus_sim import c as ARGUS_CONFIG
from argus_sim.coverage_exact import Schedule
from argus_sim.observatory import pickering_airmass

SITES = {"argus": None,  # ArgusSim's configured site
         "sro": dict(latitude=37.07, longitude=-119.41, altitude=1400.0)}  # Sierra Remote Observatories


def config(site):
    """ArgusSim's configuration, with the site moved if asked (the instrument stays Argus)."""
    cfg = copy.deepcopy(ARGUS_CONFIG)
    for k, v in (site or {}).items():
        setattr(cfg.observatory, k, v)
    return cfg


def location(cfg):
    o = cfg.observatory
    return EarthLocation(lat=o.latitude * u.deg, lon=o.longitude * u.deg, height=o.altitude * u.m)


def sky_position(t, ra, dec, site_location):
    """Airmass (ArgusSim's, Pickering 2002; inf below the horizon), altitude and azimuth of the field at times t."""
    here = SkyCoord(ra * u.deg, dec * u.deg).transform_to(AltAz(obstime=t, location=site_location))
    alt, az = here.alt.deg, here.az.deg
    return np.where(alt > 0, pickering_airmass(np.clip(alt, 1e-3, 90)), np.inf), alt, az


def ut_hours(h):
    """UT hours across midnight: 20-24 h become -4-0 h."""
    return np.where(h >= 20, h - 24, h)


def exposures(ra, dec, date, site=None, start=None, end=None, max_airmass=None, cfg=None, coarse=False):
    """The exposures of the night starting on local `date` (UTC 23:00 + 14 h), as ArgusSim schedules them (module
    docstring), with the field at or above min_alt (and at or below max_airmass), within the UT window start-end.
    coarse: every 5th minute only, by the same rules (for searches over many nights; ~5x faster).
    Returns (mid-exposure times, airmass, alt, az)."""
    cfg = cfg or config(site)
    site_location = location(cfg)
    sv = cfg.survey
    t0 = Time(f"{date} 23:00:00", scale="utc")
    if coarse:  # the schedule's own rules on a 5-minute grid: ratchet tracking minutes, the Sun below the limit
        t = t0 + (np.arange(0, 14 * 60, 5) + 0.5 * sv.base_cadence_s / 60) * u.min
        minute = t.datetime64.astype("datetime64[s]").astype(np.int64) / 60
        tracking = minute % sv.ratchet_len < sv.ratchet_len - sv.reset_min
        sun = get_sun(t).transform_to(AltAz(obstime=t, location=site_location)).alt.deg
        t = t[tracking & (sun < sv.min_sun_alt)]
    else:
        schedule = Schedule(exptime_s=sv.base_cadence_s, track_min=sv.ratchet_len - sv.reset_min, reset_min=sv.reset_min,
                            sun_alt_max_deg=sv.min_sun_alt)
        t = schedule.exposures(t0, t0 + 14 * u.hour, site_location)["time"]
    if len(t) == 0:
        return t, np.array([]), np.array([]), np.array([])
    airmass, alt, az = sky_position(t, ra, dec, site_location)
    ok = alt >= cfg.observatory.min_alt
    if max_airmass:
        ok &= airmass <= max_airmass
    clock = ut_hours((t.datetime64.astype("datetime64[s]").astype(np.int64) % 86400) / 3600)
    if start:
        ok &= clock >= ut_hours(int(start[:2]) + int(start[3:5]) / 60)
    if end:
        ok &= clock <= ut_hours(int(end[:2]) + int(end[3:5]) / 60)
    return t[ok], airmass[ok], alt[ok], az[ok]


def variability_score(jd, picks):
    """How visibly each chosen variable varies during a night, from its ephemeris, as a fraction 0-1 summed over the
    stars (so every star counts equally); ties broken by amplitude. Eclipsing binaries: 1 if the night covers a Gaia
    eclipse from -1 to +1 Gaussian sigma, 0.75 if only its centre, 0.5 if it comes within 1.5 sigma. Pulsators: half
    for the fraction of the cycle covered, half if maximum light or the rise to it (phase 0.85-1.05) is covered."""
    if len(jd) == 0:
        return 0.0
    score, amp = 0.0, 0.0
    for r in picks:
        if r["kind"] == "EB":
            v, p = r["veb"], r["period"]
            best = 0.0
            for k in (1, 2):
                ph, sig = v.get(f"PhaseGauss{k}", np.nan), v.get(f"sigPhaseGauss{k}", np.nan)
                if not np.isfinite(ph):
                    continue
                t_eclipse = v["TimeRef"] + 2455197.5 + ph * p
                centre = t_eclipse + np.ceil((jd[0] - t_eclipse) / p) * p  # the first eclipse centre after the start
                half_width = (sig if np.isfinite(sig) else 0.02) * p
                if centre - half_width >= jd[0] and centre + half_width <= jd[-1]:  # ingress to egress in the night
                    best = 1.0
                elif centre <= jd[-1]:
                    best = max(best, 0.75)
                elif np.isfinite(sig) and np.any(np.abs(((jd - t_eclipse) / p + 0.5) % 1.0 - 0.5) <= 1.5 * sig):
                    best = max(best, 0.5)
            score += best
            amp += r["amp_G"] * best
        else:
            if not np.isfinite(r.get("epoch_bjd", np.nan)):
                continue
            ph = ((jd - r["epoch_bjd"]) / r["period"]) % 1.0
            this = 0.5 * min(1.0, (jd[-1] - jd[0]) / r["period"]) + (0.5 if np.any((ph >= 0.85) | (ph <= 0.05)) else 0.0)
            score += this
            amp += r["amp_G"] * this
    return score + 1e-3 * amp


def best_night(ra, dec, date, site, start, end, max_airmass, picks, min_minutes=180):
    """The night within half a year of `date` (up >= min_minutes) on which the chosen variables change most."""
    d0 = Date.fromisoformat(date)
    best_score, best_date = -1.0, None
    for offset in range(-182, 183):
        d = (d0 + timedelta(days=offset)).isoformat()
        jd = exposures(ra, dec, d, site, start, end, max_airmass, coarse=True)[0].jd
        if 5 * len(jd) < min_minutes:
            continue
        score = variability_score(jd, picks)
        if score > best_score or (score == best_score and abs(offset) < abs((Date.fromisoformat(best_date) - d0).days)):
            best_score, best_date = score, d
    if best_date is None:
        raise SystemExit("never observable for 3 h from this site (dark, above the array's minimum altitude)")
    return best_date, best_score


def nearest_night(ra, dec, date, site, start, end, max_airmass, min_minutes=180):
    """The night nearest `date` (up to half a year away, in weekly steps) when the field is up for >= min_minutes."""
    d0 = Date.fromisoformat(date)
    for offset in sorted(range(-182, 183, 7), key=abs):
        d = (d0 + timedelta(days=offset)).isoformat()
        minutes = 5.0 * len(exposures(ra, dec, d, site, start, end, max_airmass, coarse=True)[0])
        if minutes >= min_minutes:
            return d, minutes
    raise SystemExit("never observable for 3 h from this site (dark, above the array's minimum altitude)")


def bjd_tdb(t, ra, dec, site=None, cfg=None):
    """BJD_TDB of astropy Time(s) t (UTC) for the field seen from the site."""
    site_location = location(cfg or config(site))
    return (t.tdb + t.light_travel_time(SkyCoord(ra * u.deg, dec * u.deg), location=site_location)).jd


def resolve_time(value, start, ra, dec, site):
    """BJD_TDB of an observation-file time: a number (BJD_TDB), "start+D" / "start-D" (days from the night's first
    exposure), or an ISO UTC time."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    v = str(value).strip()
    if v.startswith("start"):
        return float(bjd_tdb(start, ra, dec, site)) + (float(v[5:]) if v[5:] else 0.0)
    return float(bjd_tdb(Time(v, scale="utc"), ra, dec, site))
