"""Solar calibration: tune (alpha_MLT, Y_sun) so a 1 Msun model at 4.57 Gyr has L = Lsun and
R = Rsun, at fixed surface (Z/X)_sun = 0.0230 (GS98, the colors z_over_x_ref).
Writes data/solar_calibration.json, which anchors the alpha_MLT axis and the dY/dZ law."""

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import config as C
from .composition import ZX_SUN, solar_zy
from .runner import prepare, run_dir

AGE_SUN = 4.57e9
TOL = 2e-4  # dex in L and R


def _evaluate(points):
    dirs = []
    for i, (alpha, y) in enumerate(points):
        z, _ = solar_zy(y)
        t = dict(name=f"_calibration/cal_{i}", mass=1.0, Z=z, Y=y, alpha=alpha, max_age=AGE_SUN)
        dirs.append(prepare(t, run_dir=C.RUNS / t["name"]))
    with ThreadPoolExecutor(len(dirs)) as ex:
        list(ex.map(lambda d: run_dir(d, threads=2), dirs))
    out = []
    for d in dirs:
        h = np.genfromtxt(d / "LOGS" / "history.data", skip_header=5, names=True)
        assert abs(h["star_age"][-1] - AGE_SUN) / AGE_SUN < 1e-3, d
        out.append(np.array([h["log_L"][-1], h["log_R"][-1]]))
    return out


def main(alpha=1.9, y=0.2735, da=0.02, dy=0.003):
    for it in range(8):
        f0, fa, fy = _evaluate([(alpha, y), (alpha + da, y), (alpha, y + dy)])
        print(f"iter {it}: alpha={alpha:.4f} Y={y:.4f}  logL={f0[0]:+.5f} logR={f0[1]:+.5f}", flush=True)
        if np.all(np.abs(f0) < TOL):
            break
        J = np.column_stack([(fa - f0) / da, (fy - f0) / dy])
        step = np.linalg.solve(J, -f0)
        alpha, y = alpha + step[0], y + step[1]
    else:
        raise SystemExit("solar calibration did not converge")
    z_sun, _ = solar_zy(y)
    res = dict(alpha_sun=round(alpha, 4), Y_sun=round(y, 5), Z_sun=round(z_sun, 6), ZX_sun=ZX_SUN,
               logL=float(f0[0]), logR=float(f0[1]), age=AGE_SUN)
    (C.DATA / "solar_calibration.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
