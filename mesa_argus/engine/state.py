"""Shared engine state: the field being observed and the block shown, and the fixed geometry constants.

use_field() and set_block() set the mutable module attributes (FIELD, DATA, CENTRE, TILE, BLOCK, BLOCK_N, HALF);
AREA_HALF is set by the run (night_movie.main), _PHOTOMETRY by constant_star_rates. Every engine module READS these
as `state.X` so a change is seen live (binding `from state import TILE` would freeze the value). Writes from other
modules are plain attribute assignments (`state.AREA_HALF = ...`), no `global` needed. night_movie.py forwards its
own dynamic attributes (night_movie.HALF, .TILE, ...) here through a module __getattr__, so existing importers keep
working. FIELDS and the constants are safe to import by value (the dict is mutated in place, the constants never
change). Nothing here imports hdps or the heavy engine modules."""
import json
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

_REPO = Path(__file__).resolve().parents[2]  # repo root (mesa_argus/engine/state.py -> repo)

# hand-made fields (before observe.py): Gaia data in data/<field>/, centre, HDPS tile and block centre, default night,
# site override
FIELDS = {
    "m3": dict(name="M3", centre=(205.5484, 28.3773), tile=345967, block=(34, 32), date="2027-05-05", site=None),
    # M15 for a comparison with ADDEM at Sierra Remote Observatories (approximate SRO coordinates; Argus instrument)
    "m15": dict(name="M15", centre=(322.49304, 12.16700), tile=298720, block=(26, 36), date="2026-10-05",
                site=dict(latitude=37.07, longitude=-119.41, altitude=1400.0),
                # an RRc whose first overtone RSP finds stable even at 0.70 x the cluster L: not followed
                exclude={1745948328028784640}),
    # one RRab (Gaia DR3 308108245007332224, P 0.382 d, G 13.9) and a few neighbours, b = -34 deg
    "rr308108": dict(name="field RRab", centre=(17.588538, 28.550090), tile=345631, block=(35, 54), date="2026-10-06",
                     site=dict(latitude=37.07, longitude=-119.41, altitude=1400.0)),
    # RRab Gaia DR3 416445908838640384 (P 0.404 d, G 14.4) with 21 stars G < 20 in the block, b = -10 deg
    "rr416445": dict(name="field RRab", centre=(8.841137, 52.998087), tile=417610, block=(12, 34), date="2026-10-06",
                     site=dict(latitude=37.07, longitude=-119.41, altitude=1400.0)),
}
for _f in sorted((_REPO / "data").glob("*/field.json")):  # any other data/<field>/field.json (older argus_field.py fields)
    _c = json.loads(_f.read_text())
    FIELDS[_f.parent.name] = dict(name=_c["name"], centre=tuple(_c["centre"]), tile=_c["tile"], block=tuple(_c["block"]),
                                  date=_c["date"], site=_c.get("site"), followed_only=True, block_n=_c.get("block_n", 3),
                                  synthetic=_c.get("synthetic", False))

# the field being observed (use_field sets these)
FIELD = "m3"
DATA = Path("data/m3")
CENTRE = SkyCoord(205.5484 * u.deg, 28.3773 * u.deg)
TILE, BLOCK = 345967, (34, 32)

MINIPIX_PX, SKY_PX = 32, 900.0 / 1024.0  # HDPS minipix (pixels) and tessellation pixel (arcsec)
BLOCK_N = 3  # the block shown: BLOCK_N x BLOCK_N minipixes centred on the block minipix (set_block)
HALF = BLOCK_N / 2 * MINIPIX_PX * SKY_PX  # its half-width (arcsec)
MARGIN = 15.0  # arcsec: stars this far outside the simulated area still spill light in
AREA_HALF = 240.0  # half-width (arcsec) of the simulated area (--area): the kernel and zero point use all of it
AIRMASS_GRID = np.round(np.arange(1.0, 2.61, 0.2), 2)  # constant stars' rates are tabulated on it
PSF_PHASES = 8  # sub-pixel phases per axis in the PSF lookup


def use_field(field, scene=None):
    """Point the module (and mesa_argus.rsp) at a scene folder (observe.py) or a hand-made field's data/<field>/."""
    global FIELD, DATA, CENTRE, TILE, BLOCK
    from mesa_argus import rsp

    if scene is not None:
        scene = Path(scene).resolve()
        c = json.loads((scene / "field.json").read_text())
        FIELDS[field] = dict(name=c["name"], centre=tuple(c["centre"]), tile=c["tile"], block=tuple(c["block"]),
                             date=c["date"], site=c.get("site"), followed_only=True, block_n=c.get("block_n", 3),
                             synthetic=c.get("synthetic", False))
    f = FIELDS[field]
    FIELD, DATA = field, (scene if scene is not None else _REPO / "data" / field)
    CENTRE = SkyCoord(f["centre"][0] * u.deg, f["centre"][1] * u.deg)
    TILE, BLOCK = f["tile"], f["block"]
    if scene is None:
        rsp.use(field)


def set_block(n):
    """Show an n x n block of minipixes (n odd)."""
    global BLOCK_N, HALF
    BLOCK_N, HALF = n, n / 2 * MINIPIX_PX * SKY_PX


def block_label():
    return f"{BLOCK_N} × {BLOCK_N} minipixes ({2 * HALF:.0f}″)"


_PHOTOMETRY = None  # (templates, ab, E(B-V) nodes per template) while constant_star_rates runs
