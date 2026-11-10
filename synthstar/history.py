"""Read MESA history files and derive per-point quantities shared by the outputs."""

import csv
from functools import cache

import numpy as np

from . import config as C
from .composition import ZX_SUN

PHASES = ("Main sequence", "H-shell burning", "Core He burning")


def read(name):
    return np.genfromtxt(C.RUNS / name / "LOGS" / "history.data", skip_header=5, names=True)


def trim_pre_ms(h):
    """Drop the pre-main sequence: start where hydrogen burning supplies >= 99% of L."""
    frac = 10.0 ** (h["log_LH"] - h["log_L"])
    start = int(np.argmax(frac >= 0.99)) if np.any(frac >= 0.99) else 0
    return h[start:]


def phases(h):
    """0 = main sequence, 1 = H-shell burning (SGB/RGB/Hertzsprung gap), 2 = core He burning."""
    ph = np.zeros(len(h), int)
    post_ms = h["center_h1"] < 1e-4
    he_peak = np.maximum.accumulate(h["center_he4"])
    cheb = np.maximum.accumulate(post_ms & (h["center_he4"] < he_peak - 0.03))
    ph[post_ms] = 1
    ph[cheb] = 2
    return ph


def photospheric_mh(h):
    """[M/H] handed to the atmosphere grid (photospheric Z/X relative to GS98)."""
    return np.log10((10.0 ** h["log_surf_cell_z"] / h["surface_h1"]) / ZX_SUN)


@cache
def _kurucz_nodes():
    path = C.colors_path("stellar_atm").rstrip("/") + "/lookup_table.csv"
    rows = list(csv.DictReader(open(path)))
    teff = np.array([float(r["teff"]) for r in rows])
    logg = np.array([float(r["logg"]) for r in rows])
    meta = np.array([float(r["metallicity"]) for r in rows])
    nodes = np.unique(teff)
    gmin = np.array([logg[teff == t].min() for t in nodes])
    gmax = np.array([logg[teff == t].max() for t in nodes])
    return nodes, gmin, gmax, meta.min(), meta.max()


def inside_atmosphere_grid(teff, logg, mh):
    """True where (Teff, logg, [M/H]) lies inside the atmosphere grid. Outside it the colors
    module clamps to the grid edge, so the magnitudes there are not trustworthy."""
    nodes, gmin, gmax, mmin, mmax = _kurucz_nodes()
    teff, logg, mh = map(np.asarray, (teff, logg, mh))
    ok = (teff >= nodes[0]) & (teff <= nodes[-1]) & (mh >= mmin - 1e-6) & (mh <= mmax + 1e-6)
    i = np.clip(np.searchsorted(nodes, teff) - 1, 0, len(nodes) - 2)
    lo = np.maximum(gmin[i], gmin[i + 1])  # both bracketing Teff columns must cover this logg
    hi = np.minimum(gmax[i], gmax[i + 1])
    return ok & (logg >= lo - 1e-6) & (logg <= hi + 1e-6)


def zero_point_offsets():
    """AB - Vega offset per filter (mag), measured from the colors module (data/zp_offsets.json)."""
    import json
    return json.loads((C.DATA / "zp_offsets.json").read_text())["offsets"]


def load_track(name):
    """History trimmed to ZAMS onwards, with phase, photospheric [M/H], grid flag and SDSS in AB."""
    h = trim_pre_ms(read(name))
    zp = zero_point_offsets()
    d = {k: np.asarray(h[k], float) for k in h.dtype.names}
    for b in C.SDSS_BANDS:
        d[b] = d[b] + zp[b]  # SDSS is an AB system; MESA runs are Vega
    d["ph"] = phases(h)
    d["mh"] = photospheric_mh(h)
    d["teff"] = 10.0 ** d["log_Teff"]
    d["ok"] = inside_atmosphere_grid(d["teff"], d["log_g"], d["mh"])
    return d
