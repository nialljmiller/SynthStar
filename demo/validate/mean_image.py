#!/usr/bin/env python3
"""Reference noiseless images from the movie's own code, for demo/validate/mean_image.mjs.

    python demo/validate/mean_image.py   # -> demo/build/validate_mean/<band>_<exposure>.f64

Exposures 0, 150 and 298 (first, middle, last) of f294 in both bands: night_movie's Scene (its seeing grid), the variables' rates from
their SED at each exposure's exact airmass (as the movie; the page interpolates on AIRMASS_GRID), ArgusSim's sky.
"""

import copy
import json
from pathlib import Path
import sys
import warnings

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "demo"))
import night_movie as M  # noqa: E402
import export_demo as E  # noqa: E402
from argus_sim import ABPhot, SpectralSky, SystemThroughput  # noqa: E402
from argus_sim import c as ARGUS_CONFIG  # noqa: E402
from argus_sim.observatory import Observatory, seasonal_seeing_params, seeing_at_airmass  # noqa: E402
from argus_image.argus import BANDS, FLAM  # noqa: E402

EXPOSURES = (0, 150, 298)


def main():
    F = E.FIELD
    M.use_field(F, E.SCENE)
    M.set_block(M.FIELDS[F].get("block_n", 3))
    M.AREA_HALF = max(240.0, M.HALF + M.MARGIN)
    cfg = copy.deepcopy(ARGUS_CONFIG)
    for key, value in (M.FIELDS[F]["site"] or {}).items():
        setattr(cfg.observatory, key, value)
    tp = SystemThroughput(filters={"g": True, "rho": True}, throughput_loss=cfg.telescope.throughput_loss)
    ab = ABPhot(throughput=tp, collecting_area=cfg.telescope.collecting_area)
    stars, pulsators = M.load_scene(), M.load_pulsators()
    src = np.asarray(stars["Source"]).astype(np.int64)
    follow = M.choose_followed(stars, pulsators, 2)
    times, bjd, airmass, alt, az = M.night_schedule(M.FIELDS[F]["date"], None, cfg)
    d = json.loads((REPO / "demo" / "build" / f"{F}.json").read_text())
    static = {b: E_unpack(d["stars"]["rate"][b]) for b in BANDS}  # the export's template rates (same code as main)

    real = {v["source"]: REPO / v["file"] for v in M.load_placed() if v["source"] > 0}
    models = {k: M.RSPStar(real[int(src[k])], ab) for k in follow}
    zenith = seasonal_seeing_params(times[0])[0]
    seeing = seeing_at_airmass(zenith, airmass)
    grid = np.round(np.arange(np.floor(seeing.min() * 20) / 20, seeing.max() + 0.05, 0.05), 2)

    class Args:
        field_angle, exptime = 1.44, 60.0

    scene = M.Scene(stars, Args, cfg, ab, grid)
    scene.set_seeing(np.abs(seeing[:, None] - grid[None, :]).argmin(axis=1))
    observatory = Observatory(cfg)
    out = {}
    for j in EXPOSURES:
        comp = observatory.sky_components_at(float(alt[j]), float(az[j]), times[j])
        for b, band in BANDS.items():
            sky = SpectralSky(ab.tp, ab, band).sky_electrons(*(float(np.ravel(comp[f"{n}_s10"])[0]) for n in
                                                                 ("airglow", "zodiacal", "starlight", "moon", "artif"))) * scene.plate**2
            scene.set_sky({b: np.full(len(times), sky)})
            x = float(np.clip(airmass[j], M.AIRMASS_GRID[0], M.AIRMASS_GRID[-1]))
            i1 = int(np.clip(np.searchsorted(M.AIRMASS_GRID, x), 1, len(M.AIRMASS_GRID) - 1))
            f = (x - M.AIRMASS_GRID[i1 - 1]) / (M.AIRMASS_GRID[i1] - M.AIRMASS_GRID[i1 - 1])
            rates = static[b][:, i1 - 1] * (1 - f) + static[b][:, i1] * f
            for k in follow:
                v = pulsators[int(src[k])]
                sed = models[k].sed(M.phases_at(v, bjd[j:j + 1])[0])
                rates[k] = ab.electrons_from_sed(band, models[k].wave, sed * FLAM, airmass=airmass[j]).value
            A = scene.design_at(b, j)
            out[f"{b}_{j}"] = scene.background(b, j) + A @ (rates * Args.exptime)
            out[f"seeing_{j}"] = seeing[j]
    path = REPO / "demo" / "build" / "validate_mean"
    path.mkdir(parents=True, exist_ok=True)
    for key, value in out.items():
        np.asarray(value, "<f8").tofile(path / f"{key}.f64")
    print(f"{path}: {sorted(out)}")


def E_unpack(p):
    import base64
    import gzip

    return np.frombuffer(gzip.decompress(base64.b64decode(p["data"])), np.float32).reshape(p["shape"]).astype(float)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        main()
