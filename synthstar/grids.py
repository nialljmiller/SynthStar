"""Load grid definitions from grids/<name>.py into track dicts."""

import importlib.util
import json

from . import composition
from . import config as C
from .runner import DEFAULT_MAX_AGE, track_name


def calibration():
    return json.loads((C.DATA / "solar_calibration.json").read_text())


class _Maker:
    """Callable passed to grids/*.tracks(): builds a complete track dict."""

    def __init__(self, cal):
        self.cal = cal
        self.alpha_sun = cal["alpha_sun"]

    def name(self, mass, feh, alpha, max_age=DEFAULT_MAX_AGE):
        return track_name(mass, feh, alpha, max_age)

    def __call__(self, mass, feh, alpha, max_age=DEFAULT_MAX_AGE, **meta):
        x, y, z = composition.from_feh(feh, self.cal["Y_sun"], self.cal["Z_sun"])
        alpha = round(alpha, 4)
        name = track_name(mass, feh, alpha, max_age) + (f"_{meta['tag']}" if meta.get("tag") else "")
        return dict(name=name, mass=mass, feh=round(feh, 4), alpha=alpha,
                    dalpha=round(alpha - self.alpha_sun, 2), X=x, Y=y, Z=z, max_age=max_age, **meta)


def available():
    return sorted(p.stem for p in (C.REPO / "grids").glob("*.py"))


def load(name):
    path = C.REPO / "grids" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"grids.{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tracks = mod.tracks(_Maker(calibration()))
    names = [t["name"] for t in tracks]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise SystemExit(f"grid '{name}' has duplicate track names: {sorted(dup)}")
    for t in tracks:
        t["grid"] = name
    return tracks
