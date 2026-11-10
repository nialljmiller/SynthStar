"""Measure AB - Vega offsets per filter from the colors module itself.

Two identical short 1 Msun runs, one with mag_system = 'Vega' and one with 'AB'.
The per-band difference is a constant (to ~1e-14), and it is exactly the offset
the module applies. Writes data/zp_offsets.json; SDSS is then reported in AB.
"""

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import config as C
from .grids import calibration
from .runner import prepare, run_dir


def main():
    cal = calibration()
    base = dict(mass=1.0, Z=cal["Z_sun"], Y=cal["Y_sun"], alpha=cal["alpha_sun"], max_age=3e8)
    dirs = {s: prepare(dict(base, name=f"_zeropoints/{s}"), mag_system=s, run_dir=C.RUNS / "_zeropoints" / s)
            for s in ("Vega", "AB")}
    with ThreadPoolExecutor(2) as ex:
        list(ex.map(run_dir, dirs.values()))
    v, a = (np.genfromtxt(dirs[s] / "LOGS" / "history.data", skip_header=5, names=True) for s in ("Vega", "AB"))
    n = min(len(v), len(a))
    off = {}
    for b in C.ALL_BANDS:
        d = a[b][:n] - v[b][:n]
        assert np.ptp(d) < 1e-8, (b, np.ptp(d))
        off[b] = round(float(np.median(d)), 5)
    res = dict(note="AB minus Vega magnitude per filter, measured from the colors module "
                    "(identical 1 Msun runs with mag_system Vega vs AB; constant to 1e-14).", offsets=off)
    (C.DATA / "zp_offsets.json").write_text(json.dumps(res, indent=1) + "\n")
    for b, o in off.items():
        print(f"  {b:7s} AB - Vega = {o:+.4f}")
