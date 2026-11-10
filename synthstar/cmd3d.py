"""Interactive 3D CMD (single self-contained HTML) for a grid: outputs/<grid>_cmd3d.html.

Per track: pre-MS trimmed, phase labels, points outside the atmosphere grid flagged
(hidden by default), and the track thinned to steps of >= MIN_STEP mag in every CMD.
"""

import json
import time

import numpy as np

from . import config as C
from .grids import calibration, load
from .history import load_track, zero_point_offsets
from .runner import status, termination

MIN_STEP = 0.004  # mag


def _thin(cmds, ph):
    keep = [0]
    for i in range(1, len(ph)):
        j = keep[-1]
        step = max(np.hypot(c[i] - c[j], m[i] - m[j]) for c, m in cmds)
        if step >= MIN_STEP or ph[i] != ph[j] or i == len(ph) - 1:
            keep.append(i)
    return np.array(keep)


def _track_payload(t, d):
    k = _thin([(d["Gbp"] - d["Grp"], d["G"]), (d["beta"] - d["rho"], d["rho"]),
               (d["SDSS_g"] - d["SDSS_r"], d["SDSS_r"])], d["ph"])
    r = lambda a, n=3: [round(float(v), n) for v in np.asarray(a)[k]]  # noqa: E731
    term = termination(C.RUNS / t["name"])
    return dict(
        name=t["name"], mass=t["mass"], feh=t["feh"], alpha=t["alpha"], dalpha=t["dalpha"],
        population=t["population"], why=t["why"], kind=t["kind"] if t["kind"] in ("base", "alpha") else
        ("base" if t["dalpha"] == 0 else "alpha"),
        X=round(t["X"], 5), Y=round(t["Y"], 5), Z=round(t["Z"], 6), term=term or "unfinished",
        age=[float(f"{v:.4g}") for v in d["star_age"][k]], teff=[int(round(v)) for v in d["teff"][k]],
        logg=r(d["log_g"]), logL=r(d["log_L"]), logR=r(d["log_R"]), mh=r(d["mh"]),
        ph=[int(v) for v in d["ph"][k]], ok=[int(v) for v in d["ok"][k]],
        mstar=r(d["star_mass"], 4), mhe=r(d["he_core_mass"], 4), xc=r(d["center_h1"], 4), yc=r(d["center_he4"], 4),
        # log L_H / L_He are ~-99 when a burning stage is off; floor at -10 for plotting
        logLH=r(np.maximum(d["log_LH"], -10)), logLHe=r(np.maximum(d["log_LHe"], -10)),
        interp=r(d["Interp_rad"], 4), mbol=r(d["Mag_bol"]),
        G=r(d["G"]), BP=r(d["Gbp"]), RP=r(d["Grp"]), beta=r(d["beta"]), rho=r(d["rho"]),
        u=r(d["SDSS_u"]), g=r(d["SDSS_g"]), r=r(d["SDSS_r"]), i=r(d["SDSS_i"]), z=r(d["SDSS_z"]),
    )


def build(grid, quiet=False):
    tracks, out, missing = load(grid), [], []
    for t in tracks:
        try:
            d = load_track(t["name"])
        except (OSError, ValueError, KeyError):
            missing.append(t["name"])
            continue
        if len(d["star_age"]) < 20:
            missing.append(t["name"])
            continue
        out.append(_track_payload(t, d))
    payload = dict(calibration=calibration(), zp=zero_point_offsets(), tracks=out)
    html = (C.REPO / "templates" / "cmd3d.html").read_text()
    html = html.replace("/*__PLOTLY__*/", (C.REPO / "vendor" / "plotly-gl3d.min.js").read_text())
    html = html.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":")))
    C.OUTPUTS.mkdir(exist_ok=True)
    path = C.OUTPUTS / f"{grid}_cmd3d.html"
    path.write_text(html)
    if not quiet:
        print(f"wrote {path}  ({path.stat().st_size / 1e6:.2f} MB, {len(out)}/{len(tracks)} tracks"
              + (f"; not yet available: {len(missing)}" if missing else "") + ")")
    return path


def live(grid, every=300):
    """Rebuild every `every` seconds while any track of the grid is still running/pending."""
    while True:
        build(grid, quiet=True)
        states = [status(t["name"]) for t in load(grid)]
        print(time.strftime("%H:%M"), {s: states.count(s) for s in set(states)}, flush=True)
        if not any(s in ("running", "pending") for s in states):
            break
        time.sleep(every)
    build(grid)
