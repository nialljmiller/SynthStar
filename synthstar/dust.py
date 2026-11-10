"""Interstellar dust: how much lies in front of a star (a 3D map), and what it does to light (CCM89). The same law
adds dust to a model at observation and takes it out of a measured Gaia star.

How much: E(B-V) at the star's distance along its sightline from Bayestar19 (Green et al. 2019; dec > -30 deg, median
of the samples), but never more than the whole column from SFD (Schlegel et al. 1998): beyond a sightline's reliable
distances Bayestar19 follows its prior, which far out of the plane can exceed all the dust there is. With no distance,
or where Bayestar19 has no data, the whole column (SFD). Both maps are in SFD units: Bayestar19's E(g-r) = 0.901 x its value (Green et al. 2019, eq. 30), as SFD's
(Schlafly & Finkbeiner 2011, Table 6: A_g - A_r = 3.172 - 2.271). E(B-V) = 0.884 x the map value: with CCM89 at
R_V = 3.1 that gives A_V = 2.742 per SFD unit, Schlafly & Finkbeiner's (2011, Table 6) value.

The maps are in ~/dustmaps_data (tools/fetch_dust.sh downloads them).
"""
from functools import cache
import os

import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np

RV = 3.1
MAP_TO_EBV = 0.884  # E(B-V) per SFD / Bayestar19 unit (module docstring)
DATA_DIR = os.path.expanduser("~/dustmaps_data")


def ccm89(wavelength_angstrom, rv=RV):
    """A_lambda / A_V, Cardelli, Clayton & Mathis (1989), 0.3-10 um^-1 (IR + optical/NIR)."""
    x = 1e4 / np.asarray(wavelength_angstrom, float)
    a, b = np.zeros_like(x), np.zeros_like(x)
    ir = (x >= 0.3) & (x < 1.1)
    a[ir], b[ir] = 0.574 * x[ir] ** 1.61, -0.527 * x[ir] ** 1.61
    opt = (x >= 1.1) & (x <= 3.3)
    y = x[opt] - 1.82
    a[opt] = 1 + 0.17699 * y - 0.50447 * y**2 - 0.02427 * y**3 + 0.72085 * y**4 + 0.01979 * y**5 - 0.77530 * y**6 + 0.32999 * y**7
    b[opt] = 1.41338 * y + 2.28305 * y**2 + 1.07233 * y**3 - 5.38434 * y**4 - 0.62251 * y**5 + 5.30260 * y**6 - 2.09002 * y**7
    out = a + b / rv
    out[(x < 0.3) | (x > 3.3)] = np.nan  # outside the law; the Argus bands lie well inside
    return out


def redden(wavelength_angstrom, flux, ebv, rv=RV):
    """flux seen through E(B-V) ebv of CCM89 dust (no extinction outside the law's range, beyond the Argus bands)."""
    alam = ccm89(wavelength_angstrom, rv) * rv * ebv
    return flux * 10 ** (-0.4 * np.nan_to_num(alam, nan=0.0))


# ---------------------------------------------------------------- how much dust
@cache
def maps():
    from dustmaps.config import config

    config["data_dir"] = DATA_DIR
    from dustmaps.bayestar import BayestarQuery
    from dustmaps.sfd import SFDQuery

    return BayestarQuery(version="bayestar2019"), SFDQuery()


def ebv(ra, dec, distance_pc=None):
    """E(B-V) in front of stars at ra, dec (deg) and distance_pc (nan or None: the whole column). Returns E(B-V) and
    which map gave it ("bayestar19" or "sfd")."""
    ra, dec = np.atleast_1d(np.asarray(ra, float)), np.atleast_1d(np.asarray(dec, float))
    distance = np.full(ra.shape, np.nan) if distance_pc is None else np.broadcast_to(np.asarray(distance_pc, float), ra.shape)
    distance = np.where(np.isfinite(distance) & (distance > 0), distance, 1e6)  # no distance: beyond all the dust
    bayestar, sfd = maps()
    value = np.asarray(bayestar(SkyCoord(ra * u.deg, dec * u.deg, distance=distance * u.pc), mode="median"), float)
    column = np.asarray(sfd(SkyCoord(ra * u.deg, dec * u.deg)), float)
    outside = ~np.isfinite(value)  # outside Bayestar19's footprint
    value = np.where(outside, column, np.minimum(value, column))
    return MAP_TO_EBV * value, np.where(outside, "sfd", "bayestar19")


# ---------------------------------------------------------------- taking it out of Gaia photometry
def gaia_mags(wave, flam, ebv=0.0):
    """Vega G and BP-RP of f_lambda seen through E(B-V) ebv (MESA colors' Gaia filter curves)."""
    from .binary import vega_mag

    seen = redden(wave, flam, ebv)
    return vega_mag(wave, seen, "G"), vega_mag(wave, seen, "Gbp") - vega_mag(wave, seen, "Grp")


INTRINSIC = np.round(np.arange(-0.40, 2.461, 0.02), 2)  # intrinsic BP-RP grid (the MESA templates' range)
EBV_GRID = np.round(np.arange(0.0, 4.001, 0.05), 2)


@cache
def template_grid():
    """For the MESA template of each intrinsic BP-RP (Templates.index, as every star's SED is chosen): its BP-RP and
    A_G seen through each E(B-V) of EBV_GRID. Arrays (intrinsic, E(B-V))."""
    from .seds import Templates

    templates = Templates()
    wave = templates.wave.value
    colour, a_g = np.zeros((len(INTRINSIC), len(EBV_GRID))), np.zeros((len(INTRINSIC), len(EBV_GRID)))
    for i, c in enumerate(INTRINSIC):
        flux = templates.flux[templates.index(c)]
        g0, _ = gaia_mags(wave, flux)
        for j, e in enumerate(EBV_GRID):
            g, colour[i, j] = gaia_mags(wave, flux, e)
            a_g[i, j] = g - g0
    return colour, a_g


def on_ebv(table, ebv):
    """table (intrinsic, E(B-V)) at each star's E(B-V): (stars, intrinsic), linear in E(B-V)."""
    j = np.clip(np.searchsorted(EBV_GRID, ebv) - 1, 0, len(EBV_GRID) - 2)
    w = (np.clip(ebv, 0, EBV_GRID[-1]) - EBV_GRID[j]) / (EBV_GRID[j + 1] - EBV_GRID[j])
    return table[:, j].T * (1 - w[:, None]) + table[:, j + 1].T * w[:, None]


def deredden(G, bp_rp, ebv):
    """Intrinsic BP-RP and dust-free G of stars measured at G, BP-RP through E(B-V) ebv: the template whose colour,
    seen through that dust, is the measured one; its A_G. Colours beyond the templates' take the end template."""
    G, bp_rp, ebv = (np.atleast_1d(np.asarray(x, float)) for x in (G, bp_rp, ebv))
    colour, a_g = template_grid()
    seen, extinction = on_ebv(colour, ebv), on_ebv(a_g, ebv)
    # the reddened colour rises with the intrinsic one, except where neighbouring templates (other metallicities)
    # redden differently: above E(B-V) ~ 0.9, by up to 0.05 mag at E(B-V) = 4; their running maximum is used
    seen = np.maximum.accumulate(seen, axis=1)
    intrinsic, a_G = np.zeros(len(G)), np.zeros(len(G))
    for k in range(len(G)):
        intrinsic[k] = np.interp(bp_rp[k], seen[k], INTRINSIC)
        a_G[k] = np.interp(intrinsic[k], INTRINSIC, extinction[k])
    return intrinsic, G - a_G


def redden_gaia(G0, intrinsic, ebv):
    """The inverse of deredden: G and BP-RP of stars of dust-free G0 and intrinsic BP-RP seen through E(B-V) ebv."""
    G0, intrinsic, ebv = (np.atleast_1d(np.asarray(x, float)) for x in (G0, intrinsic, ebv))
    colour, a_g = template_grid()
    seen, extinction = on_ebv(colour, ebv), on_ebv(a_g, ebv)
    a_G = np.array([np.interp(c, INTRINSIC, row) for c, row in zip(intrinsic, extinction)])
    bp_rp = np.array([np.interp(c, INTRINSIC, row) for c, row in zip(intrinsic, seen)])
    return G0 + a_G, bp_rp
