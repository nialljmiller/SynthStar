#!/usr/bin/env python3
"""Export one field's precomputed physics for the interactive Argus page (demo/argus_demo.html).

    python observe.py examples/observations/f294.toml       # writes the scene (outputs/pseudo_real/f294/scene/)
    python demo/export_demo.py                               # -> demo/build/f294.json
    python demo/export_demo.py --scene outputs/<category>/<name>/scene  # another observation's scene

Everything here is the movie's own physics (night_movie.py, ArgusSim), evaluated once:
- every Gaia star: position, catalogue colour, template electron rates per band on AIRMASS_GRID (as the movie);
  the same for a BP-RP grid at G = 15, for stars added on the page;
- the night: per exposure UT, BJD, airmass, seeing and ArgusSim's five sky components per band (e/s/pixel);
- ArgusSim Zemax PSFs per band on a 0.05" seeing grid x 8 x 8 sub-pixel phases (as the movie's Scene);
- variable-star models (the field's own MESA RSP stars and a library, one per type): electron rates per band x
  AIRMASS_GRID and above-atmosphere magnitudes at every model step of one cycle (the page interpolates linearly in
  phase between steps, as RSPStar.sed does), and the 300-1000 nm SED at N_SED_PHASE of the steps.
Library models are re-reddened to this field's E(B-V) and keep their own distance; the page rescales them.
Arrays are gzipped float32, base64 encoded.
"""

import base64
import copy
import gzip
import json
from pathlib import Path
import sys
import warnings

import astropy.units as u
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import night_movie as M  # noqa: E402
from argus_sim import ABPhot, SpectralSky, SystemThroughput  # noqa: E402
from argus_sim import c as ARGUS_CONFIG  # noqa: E402
from argus_sim.observatory import Observatory, seasonal_seeing_params, seeing_at_airmass  # noqa: E402
from argus_image.argus import BANDS, FLAM, system_response  # noqa: E402

SCENE = REPO / "outputs" / "pseudo_real" / "f294" / "scene"  # an observation's scene (observe.py); --scene sets another
FIELD = SCENE.parent.name
N_SED_PHASE = 100  # SED panel: this many of the model's steps per cycle
SEEING_GRID = np.round(np.arange(0.60, 3.001, 0.05), 2)  # arcsec FWHM: the page snaps to it, as the movie does
LIBRARY = {  # one model per type: Gaia variables already modelled (runs/rsp_gaia_G<source>)
    "RRab": "rsp_gaia_G416445908838640384",
    "RRc": "rsp_gaia_G4096776721732855040",
    "HADS": "rsp_gaia_G4444906841983925504",
    "DCEP F": "rsp_gaia_G229602496313775872",
    "BL Her": "rsp_gaia_G4389902192257874560",
    "W Vir": "rsp_gaia_G4521180822614788480",
    "EB": "rsp_gaia_G1001448801332222848",
}


def pack(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    return dict(shape=list(a.shape), data=base64.b64encode(gzip.compress(a.tobytes(), 9)).decode())


def model_tables(model, ab):
    """A variable-star model at its own steps of one cycle: phases, rates (phase, airmass) per band, magnitudes, SEDs."""
    seds = model.flux
    rate = {b: np.array([[ab.electrons_from_sed(band, model.wave, f * FLAM, airmass=x).value for x in M.AIRMASS_GRID]
                         for f in seds]) for b, band in BANDS.items()}
    mag = {b: np.array([float(ab.mag_from_sed(band, model.wave, f * FLAM)) for f in seds]) for b, band in BANDS.items()}
    w = model.wave.to_value(u.nm)
    shown = (w >= 300) & (w <= 1000)
    rows = np.unique(np.searchsorted(model.phase, np.arange(N_SED_PHASE) / N_SED_PHASE).clip(0, len(seds) - 1))
    return dict(kind=str(model.attrs["kind"]), period=float(model.period), phase=pack(model.phase),
                rate={b: pack(r) for b, r in rate.items()}, mag={b: pack(m) for b, m in mag.items()},
                sed_wave_nm=pack(w[shown]), sed_phase=pack(model.phase[rows]), sed=pack(seds[rows][:, shown]))


def main():
    M.use_field(FIELD, SCENE)
    M.set_block(M.FIELDS[FIELD].get("block_n", 3))
    M.AREA_HALF = max(30.0 * 8.0, M.HALF + M.MARGIN)  # the movie's default --field 8
    date = M.FIELDS[FIELD]["date"]
    cfg = copy.deepcopy(ARGUS_CONFIG)
    for key, value in (M.FIELDS[FIELD]["site"] or {}).items():
        setattr(cfg.observatory, key, value)
    tp = SystemThroughput(filters={"g": True, "rho": True}, throughput_loss=cfg.telescope.throughput_loss)
    ab = ABPhot(throughput=tp, collecting_area=cfg.telescope.collecting_area)
    stars, pulsators = M.load_scene(), M.load_pulsators()
    src = np.asarray(stars["Source"]).astype(np.int64)
    G = M.column(stars, "Gmag")
    follow = M.choose_followed(stars, pulsators, 2)
    times, bjd, airmass, alt, az = M.night_schedule(date, None, cfg)

    # every star's colour and template rates (the movie's own)
    static_rate, static_mag, templates, colour = M.constant_star_rates(stars, pulsators, ab)
    gaia_var = np.array([int(s) in pulsators for s in src])

    # templates for stars added on the page: the movie's choice (Templates.index) on a BP-RP grid, rates at G = 15
    added = dict(bp_rp=np.round(np.arange(-0.3, 3.001, 0.05), 2).tolist(), G=15.0, rate={b: [] for b in BANDS},
                 mag={b: [] for b in BANDS})
    for c_ in added["bp_rp"]:
        f_ = templates.sed(c_, added["G"]) * FLAM
        for b, band in BANDS.items():
            added["rate"][b].append([ab.electrons_from_sed(band, templates.wave, f_, airmass=x).value for x in M.AIRMASS_GRID])
            added["mag"][b].append(float(ab.mag_from_sed(band, templates.wave, f_)))
    added["rate"] = {b: pack(v) for b, v in added["rate"].items()}
    added["mag"] = {b: pack(v) for b, v in added["mag"].items()}

    # the field's own variables (as the movie) and the library (re-reddened to this field)
    real = {v["source"]: REPO / v["file"] for v in M.load_placed() if v["source"] > 0}  # the scene's real Gaia variables
    field_ebv = float(json.loads((M.DATA / "field.json").read_text())["ebv"])
    models, native = [], []
    for k in follow:
        model = M.RSPStar(real[int(src[k])], ab)
        t = model_tables(model, ab)
        v = pulsators[int(src[k])]
        t.update(name=f"Gaia DR3 {src[k]}", origin=FIELD, gaia_period=v["period"], epoch_bjd=v["epoch"], gaia_kind=v["kind"])
        native.append(dict(star=int(k), model=len(models)))
        models.append(t)
    for kind, run in LIBRARY.items():
        model = M.RSPStar(REPO / "runs" / run / "pulsation.h5", ab, ebv=field_ebv)
        t = model_tables(model, ab)
        t.update(name=f"{kind} library model", origin=run.removeprefix("rsp_"), library=True)
        models.append(t)

    # seeing, PSFs (the movie's Scene, on SEEING_GRID) and the sky per exposure
    zenith_seeing = float(seasonal_seeing_params(times[0])[0])
    seeing = seeing_at_airmass(zenith_seeing, airmass)

    class Args:
        field_angle, exptime = 1.44, 60.0

    print(f"PSFs: {len(SEEING_GRID)} seeing values x {M.PSF_PHASES ** 2} phases x {len(BANDS)} bands …", flush=True)
    scene = M.Scene(stars, Args, cfg, ab, SEEING_GRID)
    # the movie fixes the stamp centroid c0 from its first seeing value; recompute it as the movie would for this night
    movie_grid = np.round(np.arange(np.floor(seeing.min() * 20) / 20, seeing.max() + 0.05, 0.05), 2)
    p0 = scene.lookup["rho", int(np.argmin(np.abs(SEEING_GRID - movie_grid[0]))), 0, 0]
    yy, xx = np.indices(p0.shape)
    c0 = ((p0 * yy).sum(), (p0 * xx).sum())
    psf = {b: np.array([[scene.lookup[b, si, iy, ix] for iy in range(M.PSF_PHASES) for ix in range(M.PSF_PHASES)]
                        for si in range(len(SEEING_GRID))]) for b in BANDS}
    observatory = Observatory(cfg)
    per_exposure = [observatory.sky_components_at(float(a_), float(z_), t_) for a_, z_, t_ in zip(alt, az, times)]
    comp = {key: np.array([float(np.ravel(c_[key])[0]) for c_ in per_exposure]) for key in per_exposure[0]}
    names = ("airglow", "zodiacal", "starlight", "moon", "artif")
    sky = {}
    for b, band in BANDS.items():
        s = SpectralSky(ab.tp, ab, band)
        sky[b] = {n_: getattr(s, f"e_per_s10_{n_}") * comp[f"{n_}_s10"] * scene.plate**2 for n_ in names}
    ut = (times.jd - np.floor(times[0].jd - 0.5) - 0.5) * 24
    response = {b: dict(base=pack(system_response(ab.tp, band, 0.0)), atmosphere=pack(np.asarray(tp.filt_wav.atmosphere, float)))
                for b, band in BANDS.items()}

    out = dict(
        field=FIELD, name=M.FIELDS[FIELD]["name"], date=date, centre_label=M.block_centre_label(),
        block_n=M.BLOCK_N, half_arcsec=M.HALF, minipix_arcsec=M.MINIPIX_PX * M.SKY_PX,
        plate=scene.plate, n=scene.n, centre=scene.centre, exptime=60.0, fullwell=scene.fullwell,
        readnoise=scene.readnoise, dark=scene.dark, psf_phases=M.PSF_PHASES, c0=list(c0),
        airmass_grid=M.AIRMASS_GRID.tolist(), seeing_grid=SEEING_GRID.tolist(), bands=list(BANDS),
        zenith_seeing=zenith_seeing, field_ebv=field_ebv,
        stars=dict(source=[str(s) for s in src], x=pack(scene.x), y=pack(scene.y), dx=pack(stars["dx"]), dy=pack(stars["dy"]),
                   G=pack(G), bp_rp_gaia=pack(M.column(stars, "BP-RP")), colour=pack(colour),
                   in_block=pack(np.asarray(stars["in_block"], float)), gaia_var=pack(gaia_var.astype(float)),
                   rate={b: pack(r) for b, r in static_rate.items()}, mag={b: pack(m) for b, m in static_mag.items()}),
        native=native, models=models, templates=added,
        night=dict(ut=pack(ut), bjd=(bjd - 2460000.0).tolist(), airmass=pack(airmass), seeing=pack(seeing),
                   moon_illum=pack(comp["illum_frac"]), moon_alt=pack(comp["moon_alt"]), sky_v=pack(comp["v_mag"]),
                   sky={b: {n_: pack(v) for n_, v in c_.items()} for b, c_ in sky.items()}),
        psf={b: pack(p) for b, p in psf.items()}, response=response, response_wave_nm=pack(ab.tp.wav.to_value(u.nm)),
    )
    path = REPO / "demo" / "build" / f"{FIELD}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    print(f"{path}: {path.stat().st_size / 1e6:.1f} MB; {len(stars)} stars, {len(times)} exposures, {len(models)} models")


if __name__ == "__main__":
    import argparse

    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--scene", type=Path, default=SCENE, help="an observation's scene folder (observe.py)")
    SCENE = a.parse_args().scene.resolve()
    FIELD = SCENE.parent.name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        main()
