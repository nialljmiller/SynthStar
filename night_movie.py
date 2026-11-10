#!/usr/bin/env python3
"""The engine: one night of Argus 60 s exposures of a scene, simulated and measured; the movie (CMD, colour exposure,
difference image, light curves, SEDs) and the photometry CSV. observe.py writes the scene and runs this, once per band.

    python night_movie.py --scene outputs/pseudo_real/f294/scene --band rho --stride 10   # by hand
    python night_movie.py --field m3                                                                    # a hand-made field

The night, step by step (main):
  1. the stars: the catalogue (constant stars) and the followed stars (real variables, placed stars, transients)
  2. the night: ArgusSim's exposure schedule at the field (argus_image.night), airmass
  3. constant stars: the MESA SED matched to each star's Gaia BP-RP, scaled to its G -> electrons/s per band (ArgusSim)
  4. followed stars: their SED at every exposure (pulsation / orbit phase, or time since t0), through the observation's
     dust -> electrons/s
  5. conditions per exposure (ArgusSim): seeing (seasonal median x airmass^0.6), ArgusSim's PSF with that seeing
     ([argussim.telescope] psf_model: the Zemax PSF, or the Gaussian spot with no Zemax file read), sky
  6. the difference-image reference: the median of the best-seeing exposures
  7. every exposure: PSFs + sky -> Poisson, full well, read noise -> hdps source detector, matched to the simulated
     stars, per-exposure zero point (argus_image.pipeline)
  8. the movie, frame by frame (the same exposures again, from the same seeds)

Assumptions (field angle, coverage, seeing, transparency, ...): docs/ASSUMPTIONS.md.
"""
import argparse
import copy
import csv
import json
import shutil
import textwrap
import warnings
from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astropy.utils import iers
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.axis as maxis  # noqa: E402
import matplotlib.text as mtext  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.colors import AsinhNorm  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402

# (ZemaxPSF is imported only where psf_model is "zemax" (Scene): the public argus_sim has none)
from argus_sim import ABPhot, SpectralSky, SystemThroughput  # noqa: E402
from argus_sim import c as ARGUS_CONFIG  # noqa: E402
from argus_sim import skymap_shim as sh  # noqa: E402
from argus_sim.observatory import Observatory, seasonal_seeing_params, seeing_at_airmass  # noqa: E402

from synthstar import dust  # noqa: E402
from argus_image.argus import BANDS, BLUE, COLORS, FLAM, NAVY, RED, SYMBOLS, add_logo, apply_style, system_response  # noqa: E402
from synthstar.seds import Templates  # noqa: E402

iers.conf.auto_download = False
iers.conf.auto_max_age = None  # the bundled Earth-orientation table at any age: sub-second UT1, sub-arcsec polar motion
_REPO = Path(__file__).resolve().parent  # data/, runs/, outputs/ are in the repo, wherever this is run from

from argus_image.engine import state  # noqa: E402  the shared mutable field/block state (keystone of the stage split)
from argus_image.engine.state import (MINIPIX_PX, SKY_PX, MARGIN, AIRMASS_GRID, PSF_PHASES, FIELDS,  # noqa: E402
                                     use_field, set_block, block_label)


def __getattr__(name):  # forward the shared mutable state (night_movie.HALF, .TILE, .FIELD, ...) to engine.state, live
    try:
        return getattr(state, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


# ---------------------------------------------------------------- the stars
from argus_image.engine.imaging import (block_offsets, column, block_centre_label, star_catalogue, load_followed, load_placed, engine_of, load_scene, load_pulsators, pulsator_title, choose_followed, night_schedule, survey_schedule, phases_at, catmull_rom_matrix, Scene, constant_star_rates, template_photometry, followed_star_light, conditions, block_radec, binned, raw_frame_path, write_raw_frame, window_footprint)  # noqa: F401  extracted to synthstar/engine/imaging.py






















from synthstar.models import RSPStar, TransientSource  # noqa: E402  (extracted to synthstar/engine/models.py)


# ---------------------------------------------------------------- the night






# ---------------------------------------------------------------- the detector




# ---------------------------------------------------------------- the steps of main












PREVIEW_BIN = 4  # the pipeline movie's raw-frame preview: block means of 4 x 4 sensor pixels
PHOTOMETRY_COLUMNS = ("exposure", "ut_hours", "airmass", "seeing", "star", "true_mag", "saturated", "measured_mag",
                      "measured_err", "band", "tel_id", "ratchet", "field_angle", "time_utc")  # <name>_night_<band>.photometry.csv




from argus_image.engine.downstream import (slim_record, movie_records, alert_table, measure_frame, block_cut, block_cutouts, WindowMosaic, ReferenceMask, alerts_of, alert_marks, draw_alert_marks, set_alert_mark, alert_spans, model_hours, choose_shown, movie_data, _FrameWriter, render, draw, MovieFrame)  # noqa: F401  extracted to synthstar/engine/downstream.py
























# ---------------------------------------------------------------- main
def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", type=Path, help="a scene folder written by observe.py")
    p.add_argument("--field", help="a hand-made field (m3, m15, rr308108, rr416445); with --scene, its name")
    p.add_argument("--catalogues", type=Path, help="also write every measured exposure's detections (parquet) here")
    p.add_argument("--observation", type=Path, help="the resolved observation file (observe.py): hdps chain, window, camera")
    p.add_argument("--hdps-pass", choices=["reference", "science"], default="science",
                   help="reference: the reference nights, hdps coadds only; science: the whole chain against them")
    p.add_argument("--work", type=Path, help="hdps's working folder (raw frames, products, coadds), per band below it")
    p.add_argument("--pipeline-movie", action="store_true", help="also draw the pipeline movie (argus_image.pipeline_movie)")
    p.add_argument("--reference", type=Path, help="hdps's reference coadds (the reference pass's coadds/): SFFT's reference_dir")
    p.add_argument("--no-render", action="store_true", help="measure (and write FITS) but draw no movie")
    p.add_argument("--measure-only", action="store_true",
                   help="measure again from the raw frames and hdps's products a run left in --work (a run that stopped "
                        "after hdps): no rendering, no hdps; then the outputs as usual")
    p.add_argument("--stop-after-render", action="store_true",
                   help="render the raw frame stack to --work and stop before hdps (the image-maker half of a decoupled "
                        "run); the fused run (no flag) renders and runs hdps together as before")
    p.add_argument("--hdps-from-stack", action="store_true",
                   help="run hdps on the raw frame stack already in --work, then measure as usual (the hdps half of a "
                        "decoupled run): no rendering")
    p.add_argument("--render-only", action="store_true",
                   help="draw the movies from a finished run's <output>.movie.joblib (output.movie_data): no "
                        "rendering, no hdps; the scene is rebuilt as in the run")
    p.add_argument("-o", "--output", type=Path,
                   help="default: next to the scene; a hand-made field: outputs/pseudo_real/<field>/<field>_night_<band>.mp4")
    p.add_argument("--date", help="local date the night starts (default: the field's own)")
    p.add_argument("--max-airmass", type=float, default=None, help="a further airmass limit (the array's minimum altitude always applies)")
    p.add_argument("--start", help="UT HH:MM to start (default: when the field is first observable)")
    p.add_argument("--end", help="UT HH:MM to stop (default: when the field is last observable)")
    p.add_argument("--exptime", type=float, default=60.0)
    p.add_argument("--seeing", type=float, default=None, help="zenith seeing FWHM (arcsec); default: ArgusSim's seasonal median")
    p.add_argument("--field-angle", type=float, default=1.44, help="deg from the optical axis; 1.44 = area-weighted median")
    p.add_argument("--stride", type=int, default=1, help="render and measure every Nth exposure")
    p.add_argument("--fps", type=float, default=24)
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--highlight", type=int, default=6, help="hand-made fields: pulsators to follow")
    p.add_argument("--seed", type=int, default=3, help="noise")
    p.add_argument("--area", type=float, default=8.0, help="side (arcmin) of the simulated area (kernel and zero point)")
    p.add_argument("--block", type=int, default=None, help="minipixes across the block shown (default: the field's)")
    p.add_argument("--band", choices=list(BANDS), default="rho", help="band shown in the images, light curves and CMD")
    return p


def main(argv=None):
    import os

    args = parser().parse_args(argv)
    if args.scene is not None:  # relative paths are relative to where the command was run
        args.scene = (Path.cwd() / args.scene) if not args.scene.is_absolute() else args.scene
        args.field = args.field or args.scene.resolve().parent.name
    for key in ("catalogues", "reference", "observation", "work"):
        if getattr(args, key) is not None and not getattr(args, key).is_absolute():
            setattr(args, key, Path.cwd() / getattr(args, key))
    if args.output is None:
        folder = args.scene.parent if args.scene is not None else _REPO / "outputs" / "pseudo_real" / args.field
        args.output = folder / f"{args.field}_night_{args.band}.mp4"
    if not args.output.is_absolute():
        args.output = Path.cwd() / args.output
    os.chdir(_REPO)
    use_field(args.field, args.scene)
    set_block(args.block or FIELDS[args.field].get("block_n", 3))
    if args.observation is not None:  # the stars kept: out to a readout window's far corner, wherever it falls
        from argus_image import runconfig

        resolved = runconfig.resolve(runconfig.load(args.observation))
        window = resolved["field"]["window"]
        side = (window or ARGUS_CONFIG.telescope.n_pixels_x) * MINIPIX_PX * SKY_PX if window else \
            ARGUS_CONFIG.telescope.n_pixels_x * ARGUS_CONFIG.telescope.plate_scale
        state.AREA_HALF = max(side * np.sqrt(2), state.HALF + MARGIN)
    else:
        state.AREA_HALF = max(30.0 * args.area, state.HALF + MARGIN)
    args.date = args.date or FIELDS[args.field]["date"]
    B = args.band  # the band shown in the images, light curves and CMD
    cfg = copy.deepcopy(ARGUS_CONFIG)
    for key, value in (FIELDS[args.field]["site"] or {}).items():  # the instrument stays Argus; only the site moves
        setattr(cfg.observatory, key, value)
    if args.observation is not None:  # the observation's PSF ([argussim.telescope] psf_model, spot_size) for the Scene
        # and the sensor's PSFs (argus_image.sensor.PSFs); the other [argussim] values are not applied to ArgusSim here
        telescope = resolved["argussim"]["telescope"]
        cfg.telescope.psf_model, cfg.telescope.spot_size = telescope["psf_model"], telescope["spot_size"]
    print(f"PSF: ArgusSim's {cfg.telescope.psf_model} model"
          + (f" (GaussianPSF: optical spot {cfg.telescope.spot_size} um rms with the Moffat seeing; no Zemax file read)"
             if cfg.telescope.psf_model == "gaussian" else ""), flush=True)
    ab = ABPhot(throughput=SystemThroughput(filters={"g": True, "rho": True}, throughput_loss=cfg.telescope.throughput_loss),
                collecting_area=cfg.telescope.collecting_area)

    # 1. the stars
    stars = load_scene()
    pulsators = load_pulsators()
    source = np.asarray(stars["Source"]).astype(np.int64)  # Gaia id, or -1, -2, ... for placed stars
    in_block = np.asarray(stars["in_block"])
    G = column(stars, "Gmag")
    gaia_var = np.array([int(s) in pulsators for s in source])  # Gaia-classified: constant unless followed
    follow = choose_followed(stars, pulsators, args.highlight)  # row indices; k below is always a row index
    placed_label = {v["source"]: v["label"] for v in load_placed()}
    labels = {k: placed_label[int(source[k])] if source[k] < 0 else f"V{i + 1}" for i, k in enumerate(follow)}
    n_placed = int(np.sum(source < 0))
    if args.catalogues is not None:  # what each simulated star is, for the truth columns (pipeline.lightcurves)
        args.catalogues.mkdir(parents=True, exist_ok=True)
        (args.catalogues / "truth_stars.json").write_text(json.dumps(
            {"label": {str(k): v for k, v in labels.items()}, "source": [int(s) for s in source]}))
    print(f"{len(stars) - n_placed}{'synthetic stars' if FIELDS[args.field].get('synthetic') else 'Gaia sources'} "
          f"({int(in_block.sum()) - n_placed} in the block)" + (f"; {n_placed} transients" if n_placed else "")
          + f"; following {len(follow) - n_placed} pulsators with MESA RSP models; the other "
          f"{int((gaia_var & in_block).sum()) - len(follow)} Gaia-classified variables in the block are held constant")

    # 2. the night
    if args.exptime > cfg.survey.base_cadence_s:
        raise SystemExit(f"--exptime {args.exptime} s is longer than ArgusSim's cadence ({cfg.survey.base_cadence_s} s)")
    survey = None  # observing "survey": the exposures ArgusSim's array takes of the field, each in one band
    if (state.DATA / "exposures.ecsv").exists():
        times, bjd, airmass, alt, az, survey = survey_schedule(B, cfg)
        if args.observation is not None and not args.render_only:  # here, before the photometry: see 6. below
            from argus_image.survey import check_layout

            check_layout(survey, resolved["argussim"]["packing_strategy"]["layout_file"])
        print(f"survey ({B}): {len(times)} exposures through {len(np.unique(survey['tel_id']))} telescopes in "
              f"{len(np.unique(survey['ratchet']))} ratchets, UT {times[0].iso[11:16]}-{times[-1].iso[11:16]}, "
              f"airmass {airmass.min():.2f}-{airmass.max():.2f}")
    else:
        times, bjd, airmass, alt, az = night_schedule(args.date, args.max_airmass, cfg, args.start, args.end)
        print(f"night of {args.date}: {len(times)} exposures, UT {times[0].iso[11:16]}-{times[-1].iso[11:16]}, "
              f"airmass {airmass.min():.2f}-{airmass.max():.2f}")
    bands = [B] if survey is not None else list(BANDS)  # the bands each exposure is taken in
    ut = (times.jd - np.floor(times[0].jd - 0.5) - 0.5) * 24  # UT hours since 0h on the first date

    # 3. constant stars
    workers = (runconfig.resolve(runconfig.load(args.observation))["stars"]["photometry_workers"]
               if args.observation is not None else 1)
    static_rate, static_mag, templates, _ = constant_star_rates(stars, pulsators, ab, workers)
    G = column(stars, "Gmag")  # as seen where observed (a synthetic field: through its own sightline)

    # 4. followed stars: real variables and placed stars from their model files, with the observation's dust
    from synthstar import rsp

    model_file = {v["source"]: _REPO / "runs" / f"rsp_{rsp.PREFIX}{name}" / "pulsation.h5"
                  for name, v in load_followed().items()}
    model_file.update({v["source"]: _REPO / v["file"] for v in load_placed() if v["timing"] == "periodic"})
    model_ebv = {v["source"]: v["ebv"] for v in load_placed() if v["timing"] == "periodic"}
    model_distance = {v["source"]: v.get("distance_pc") for v in load_placed() if v["timing"] == "periodic"}
    models, follow_sed, follow_rate, follow_mag, follow_cycle = {}, {}, {}, {}, {}
    transient = {k for k in follow if "transient" in pulsators[int(source[k])]}
    for k in follow:
        v = pulsators[int(source[k])]
        models[k], follow_sed[k], follow_rate[k], follow_mag[k], follow_cycle[k] = followed_star_light(
            k, v, args, ab, bjd, airmass, model_file.get(int(source[k])), model_ebv.get(int(source[k])),
            model_distance.get(int(source[k])))
        cyc = follow_cycle[k]["rho"]
        if k in transient:
            tr = v["transient"]
            print(f"  {labels[k]}: {tr['kind']} {tr.get('engine', tr.get('generator', 'sncosmo'))} {tr['model']}, "
                  f"z {tr['z']:g}, t0 BJD {tr['t0_bjd']:.4f} (night starts {bjd[0] - tr['t0_bjd']:+.3f} d after t0); "
                  + (f"rho {np.nanmin(cyc):.2f}-{np.nanmax(cyc):.2f} over the night" if np.isfinite(cyc).any()
                     else "no light during the night"))
        else:
            print(f"  {labels[k]}: {v['kind']} {engine_of(v)}, period {models[k].period:.5f} d "
                  f"({'given' if 'placed' in v else 'Gaia'} {v['period']:.5f}); rho {cyc.min():.2f}-{cyc.max():.2f} "
                  f"(amplitude {np.ptp(cyc):.2f}); Gaia mean G {G[k]:.2f}")
    # the catalogue / reference brightness of a periodic star: its cycle-mean SED; a transient is not in the reference
    ref_rate = {}
    for k in follow:
        if k in transient:
            ref_rate[k] = {b: np.zeros(len(airmass)) for b in BANDS}
            continue
        mean_sed = np.mean([models[k].sed(ph) for ph in np.linspace(0, 1, 200, endpoint=False)], axis=0) * FLAM
        ref_rate[k] = {b: np.array([ab.electrons_from_sed(band, models[k].wave, mean_sed, airmass=x).value for x in airmass])
                       for b, band in BANDS.items()}
    # the reference a candidate is subtracted against: its telescope's coadd of the reference exposures that telescope
    # kept ([night] reference_ratchets: its best-seeing ratchets). A periodic star is in that coadd as it was at those
    # exposures, not at its cycle mean; SFFT matches the coadd's photometric scale to the science frame, so the star's
    # flux there is its mean SED over those exposures at the science exposure's airmass (electrons/s on AIRMASS_GRID,
    # {star: {telescope: rates}}). Before 2026-10-30 the truth used the cycle mean: off by up to ~20% per telescope.
    ref_by_tel = {}
    ref_scene = state.DATA.parent / "reference" / "scene" / "exposures.ecsv"
    if survey is not None and ref_scene.exists() and state.DATA.parent.name != "reference":
        from astropy.table import Table
        from astropy.time import Time

        from argus_image import night

        ref = Table.read(ref_scene)
        ref = ref[ref["band"] == B]
        ref_tel = np.asarray(ref["tel_id"], int)
        ref_bjd = night.bjd_tdb(Time(list(ref["time"]), scale="utc"), state.CENTRE.ra.deg, state.CENTRE.dec.deg, cfg=cfg)
        for k in follow:
            if k in transient:
                continue
            v = pulsators[int(source[k])]
            ref_by_tel[k] = {}
            for tel in np.unique(ref_tel):
                mean_sed = np.mean([models[k].sed(ph) for ph in phases_at(v, ref_bjd[ref_tel == tel])], axis=0) * FLAM
                ref_by_tel[k][int(tel)] = np.array([ab.electrons_from_sed(BANDS[B], models[k].wave, mean_sed, airmass=x).value
                                                    for x in AIRMASS_GRID])
        print(f"truth: each periodic star's reference flux from its telescope's reference exposures ({len(ref)} in {B}, "
              f"{len(np.unique(ref_tel))} telescopes)", flush=True)

    def rates_at(b, j, reference=False):
        """Electrons/s of every star in exposure j (reference: as the difference reference sees it, no transients)."""
        x = float(np.clip(airmass[j], AIRMASS_GRID[0], AIRMASS_GRID[-1]))
        i1 = int(np.clip(np.searchsorted(AIRMASS_GRID, x), 1, len(AIRMASS_GRID) - 1))
        f = (x - AIRMASS_GRID[i1 - 1]) / (AIRMASS_GRID[i1] - AIRMASS_GRID[i1 - 1])
        out = static_rate[b][:, i1 - 1] * (1 - f) + static_rate[b][:, i1] * f  # linear in airmass
        for k in follow:
            out[k] = 0.0 if (reference and k in transient) else follow_rate[k][b][j]
        return out

    def true_mag(b, k, j):
        return float(follow_mag[k][b][j]) if k in follow_mag else static_mag[b][k]

    # 5. conditions per exposure
    seeing, scene = conditions(times, airmass, alt, az, args, stars, cfg, ab, survey)

    # 6. the camera: every exposure on its telescope's sensor (ArgusSim's geometry, argus_image.sensor), in the
    # observation's readout window, written as a raw camera frame (argus_image.camera); 7. hdps's worker processes
    # them (argus_image.hdps_worker). The reference pass makes hdps's coadds (transients left out); the science pass
    # runs the whole chain against them, then the measurements from hdps's products (truth labels) and the movies.
    if survey is None:
        raise SystemExit("the engine observes as ArgusSim's survey does: observe.py writes scene/exposures.ecsv")
    from argus_image import hdps_worker, runconfig, sensor
    from argus_image.survey import array_layout

    obs = runconfig.resolve(runconfig.load(args.observation))
    # ArgusSim's ring layout ([argussim.packing_strategy] layout_file): each exposure's ota is a row of the layout
    # observe.py found the exposures with (exposures.ecsv meta), so the two must be the same (checked in 2. above, with
    # survey.check_layout). --render-only draws from the movie data alone (the frames' positions saved with it) and
    # places nothing with the layout, so it is not checked there: finished runs of either layout draw again.
    layout = array_layout(obs["argussim"]["packing_strategy"]["layout_file"])
    # the field movie's images: the whole readout window (output.movie_window_px; hdps's images of every tile it
    # touches, WindowMosaic) when the movie or its data is made, else the block (block_cutouts: the tile it is in)
    whole = obs["output"]["movie_window_px"] > 0 and (not args.no_render or obs["output"]["movie_data"])
    for t in obs["hdps"]["tasks"]:  # hdps's tile images: those the movie shows (None: every tile hdps differenced)
        if t["type"] == "TileImageOutputTask":
            t["config"]["tile_ids"] = None if whole else [int(state.TILE)]
    reference_pass = args.hdps_pass == "reference"
    window_px = int(round(obs["field"]["window"] * MINIPIX_PX * SKY_PX / ARGUS_CONFIG.telescope.plate_scale))
    star_ra, star_dec = block_radec(np.asarray(stars["dx"]), np.asarray(stars["dy"]))
    centre_ra, centre_dec = block_radec(np.zeros(1), np.zeros(1))
    projector = sensor.Projector(np.r_[centre_ra, star_ra], np.r_[centre_dec, star_dec], times, layout)
    east, north = obs["field"]["window_offset"]  # where the readout window's centre lies from the field (arcmin)
    window_ra, window_dec = block_radec(np.array([-60.0 * east]), np.array([60.0 * north]))  # the block's +x is west
    window_projector = sensor.Projector(window_ra, window_dec, times, layout)
    psfs = sensor.PSFs(cfg, wing=obs["image"])  # the observation's psf_model, with its [image] PSF options
    if psfs.gaussian and obs["image"]["psf_halo"]:
        print("PSF: [image] psf_halo_taper not used (it rounds Zemax's square optics array; the Gaussian has none)",
              flush=True)
    work = args.work / B
    expected_flux = static_rate[B][:, 0] * args.exptime
    for k in follow:
        expected_flux[k] = 0.0 if k in transient else np.mean(ref_rate[k][B]) * args.exptime
    frames = range(0, len(times), args.stride)
    seeds = np.random.SeedSequence(args.seed).spawn(len(times))
    calibration = None  # the simulation's WCS and zero point in each raw frame, when the chain takes them from it
    if any(t["type"] == "TrueCalibrationTask" for t in obs["hdps"]["tasks"]):
        from argus_image import true_calibration

        detection = next(t["config"] for t in obs["hdps"]["tasks"] if t["type"] == "SourceDetectionTask")
        calibration = dict(zp_rate=true_calibration.ab_zero_point(ab, BANDS[B], AIRMASS_GRID), detection=detection,
                           zp=[], share=[], residual=[])
    movie_file = args.output.with_suffix(".movie.joblib")

    def movies(frames, measured, alerted, cutouts, records, window=None):
        """8. the movies: the field movie (the panels as before, its images hdps's tile and difference image of the
        whole readout window (window: WindowMosaic.summary), or around the block) and the pipeline movie (each raw
        frame's preview, where it falls on the sensor, hdps's detections and WCS)."""
        if not args.no_render:
            o = obs["output"]
            # the field movies asked for ([output] movie_views): the window view (<name>_night_<band>.mp4) and the
            # block's own tile (_block.mp4); without whole-window images, the block alone under the first name
            views = list(dict.fromkeys(o["movie_views"])) if window is not None else ["block"]
            coadds = Path(args.reference) if args.reference else args.scene.parent / "reference" / "hdps" / B / "coadds"
            mask = (ReferenceMask(coadds, o["movie_mask_reference_e"], o["movie_mask_radius_px"], o["movie_mask_growth_px"])
                    if o["movie_mask_reference_e"] > 0 and coadds.is_dir() else None)
            cameras = [f"ARGUSSIM-{int(r['tel_id']):04d}" for r in records]
            cuts = [mask.block(c, k) for c, k in zip(cameras, cutouts)] if mask is not None else cutouts
            for view in views:
                win = window if view == "window" else None
                if win is not None and mask is not None:
                    mosaic = WindowMosaic(window["footprints"], o["movie_window_px"], o["movie_window_zoom"])
                    win = dict(window, mosaics=[mask.window(c, m, mosaic, window["tiles"])
                                                for c, m in zip(cameras, window["mosaics"])])
                draw_data = movie_data(args, stars, scene, follow, transient, models, follow_sed, follow_rate,
                                       follow_cycle, ref_rate, static_mag, rates_at, true_mag, measured, frames, times,
                                       bjd, ut, airmass, seeing, G, in_block, ab, cuts, alerted=alerted, window=win)
                draw_data.update(alerted=alerted, frame_j=list(frames), labels=labels, pulsators=pulsators,
                                 source=source, gaia_var=gaia_var, templates=templates, survey=survey, cutouts=cuts,
                                 masked=mask is not None)
                one = copy.copy(args)
                if view == "block" and len(views) > 1:
                    one.output = args.output.with_name(args.output.stem + "_block.mp4")
                render(one, draw_data, frames)
                print(f"field movie ({view}) -> {one.output.name}", flush=True)
        if args.pipeline_movie:
            from argus_image import pipeline_movie

            pipeline_movie.render(records, args.output.with_name(args.output.stem + "_pipeline.mp4"), state.FIELD, B,
                                  args.fps, args.dpi, ut, airmass, seeing)

    if args.render_only:  # a finished run's movie data: the movies only
        import joblib

        if not movie_file.exists():
            raise SystemExit(f"{movie_file}: no movie data (run with output.movie_data = true)")
        saved = joblib.load(movie_file)
        movies(saved["frames"], saved["measured"], saved["alerted"], saved["cutouts"], movie_records(saved["records"]),
               saved.get("window"))  # (movie data from before the whole-window images: the block)
        print(f"saved {args.output} (from {movie_file.name})")
        return

    print(f"{'reference pass' if reference_pass else 'observing'}: {len(frames)} exposures ({B}) as raw frames "
          f"({window_px} px window) -> {work / 'raw'}" + (" (already there: measuring them again)"
                                                           if args.measure_only or args.hdps_from_stack
                                                           else ""), flush=True)
    if not (args.measure_only or args.hdps_from_stack):  # a fresh stack; hdps ingests every raw/*.fits, so start clean
        shutil.rmtree(work / "raw", ignore_errors=True)
    records = []
    for n, j in enumerate(frames):  # 6. the camera: each exposure rendered and written as a raw frame
        row = survey[j]
        x, y, angle = projector.pixels(j, row["theta_deg"], row["ota"])
        wx, wy, _ = window_projector.pixels(j, row["theta_deg"], row["ota"])
        win = sensor.window(wx[0], wy[0], window_px)
        if args.measure_only or args.hdps_from_stack:  # the frame already on disk: read it, don't render
            from astropy.io import fits

            path = raw_frame_path(work / "raw", row, args)[0]
            image = fits.getdata(path).astype(np.float32)
        else:
            electrons = rates_at(B, j, reference=reference_pass) * args.exptime
            image = sensor.render(x[1:], y[1:], electrons, angle[1:], B, seeing[j], scene.background(B, j), win, psfs,
                                  np.random.default_rng(seeds[j]), ARGUS_CONFIG.telescope.readnoise,
                                  ARGUS_CONFIG.telescope.fullwell)
            cards = None
            if calibration is not None:
                w, residual, residual_rms = projector.wcs(j, row["theta_deg"], row["ota"], win[:2])
                zp_total = 2.5 * np.log10(np.interp(airmass[j], AIRMASS_GRID, calibration["zp_rate"]) * args.exptime)
                ai = int(np.abs(psfs.angles(B) - angle[0]).argmin())  # the field's tabulated field angle
                share = true_calibration.aperture_fraction(psfs, B, seeing[j], ai, calibration["detection"])
                cards = true_calibration.cards(w, residual, residual_rms, zp_total, share)
                for k, v in (("zp", zp_total), ("share", share), ("residual", residual)):
                    calibration[k].append(v)
            path = write_raw_frame(work / "raw", image, win, row, args, layout, cards)
        sx, sy = x[1:] - win[0], y[1:] - win[1]  # every star in the window's pixels (truth)
        bright = (G < 17) & (sx > 20) & (sx < win[2] - 20) & (sy > 20) & (sy < win[3] - 20)  # for the WCS check
        records.append(dict(path=path, window=win, field_xy=(x[0], y[0]), j=j, sx=sx, sy=sy,
                            preview=binned(image, PREVIEW_BIN), preview_bin=PREVIEW_BIN,
                            truth=(sx[bright], sy[bright], star_ra[bright], star_dec[bright]), tel_id=int(row["tel_id"]),
                            ratchet=int(row["ratchet"]), field_angle=float(row["field_angle"]),
                            footprint=window_footprint(win, row, layout) if whole else None))
        if (n + 1) % max(1, len(frames) // 10) == 0:
            print(f"  rendered {n + 1}/{len(frames)}", flush=True)
    if calibration is not None and calibration["zp"]:
        print(f"calibration from the simulation (TrueCalibrationTask): AB zero point {min(calibration['zp']):.3f}-"
              f"{max(calibration['zp']):.3f} mag (total e-), hdps's aperture holds {min(calibration['share']):.3f}-"
              f"{max(calibration['share']):.3f} of a star's e-; true WCS fit within "
              f"{max(calibration['residual']) * 1e3:.1f} mpx", flush=True)
    mosaic = None  # the field movie's whole-window images (WindowMosaic), made from hdps's tile images below
    if whole and not reference_pass:
        zoom = obs["output"]["movie_window_zoom"]
        mosaic = WindowMosaic([r["footprint"] for r in records], obs["output"]["movie_window_px"], zoom)
        if zoom > 1:  # hdps's tile images: only the tiles the zoomed view reaches
            for t in obs["hdps"]["tasks"]:
                if t["type"] == "TileImageOutputTask":
                    t["config"]["tile_ids"] = mosaic.tiles()

    if args.stop_after_render:  # the image-maker half of a decoupled run: raw stack on disk, stop before hdps
        print(f"rendered {len(frames)} frames -> {work / 'raw'}; stopping before hdps (--stop-after-render)", flush=True)
        return

    # 7. hdps's worker over the raw frames (argus_image.hdps_worker). The reference pass makes hdps's coadds
    # (transients left out); the science pass runs the whole chain against them.
    if reference_pass:
        result = hdps_worker.run(obs, work / "raw", work / "coadds", stop_after="CoaddTapTask",
                                 worker_id=f"{state.FIELD}_{B}_reference")
        shutil.rmtree(work / "coadds" / "solved", ignore_errors=True)
        print(f"hdps worker: {result}; reference coadds -> {work / 'coadds'}", flush=True)
        if not obs["output"]["raw"]:
            shutil.rmtree(work / "raw", ignore_errors=True)
        return
    if not args.measure_only:
        result = hdps_worker.run(obs, work / "raw", work, reference_dir=args.reference, worker_id=f"{state.FIELD}_{B}")
        print(f"hdps worker: {result}", flush=True)
    tile_bytes = sum(p.stat().st_size for p in work.glob("epoch_*_tile_*.fits"))
    print(f"hdps's tile images: {tile_bytes / 1e9:.2f} GB ({tile_bytes / 1e6 / max(1, len(records)):.0f} MB per frame)"
          + ("" if obs["output"]["raw"] else ", deleted at the end (output.raw = false)"), flush=True)

    # the measurements, from hdps's products: the solved frame's catalogue and zero point, the tile images, alerts
    rows, measured, alerted, cutouts, alert_rows = [], [], [], [], []
    mosaics = []
    n_alerts = n_skipped = 0
    for record in records:
        j, stem = record["j"], record["path"].stem
        row = survey[j]
        catalogue, zp, star_of, wcs = measure_frame(work, stem, B, obs, record["sx"], record["sy"], expected_flux)
        record.update(catalogue=catalogue, zp=zp, wcs=wcs, failed=None if catalogue is not None else "not solved by hdps")
        cutouts.append(block_cutouts(work, stem))
        mosaics.append(mosaic.frame(work, stem) if mosaic is not None else None)
        n_skipped += catalogue is None
        frame_alerts = alerts_of(work, stem, follow, star_ra, star_dec)
        table = alert_table(work, stem)
        if table is not None and len(table):  # every alert, labelled with what the simulation put there
            from hdps.alerts.features import FEATURE_SPEC_VERSION

            from argus_image.pipeline import truth

            for at, (key, value) in enumerate(dict(observation=state.FIELD, band=B, exposure=int(j), time_utc=str(row["time"]),
                                                   exptime=args.exptime, airmass=float(airmass[j]),
                                                   seeing=float(seeing[j]), tel_id=int(row["tel_id"]),
                                                   ratchet=int(row["ratchet"]),
                                                   field_angle=float(row["field_angle"]),
                                                   feature_spec=FEATURE_SPEC_VERSION).items()):
                table.insert(1 + at, key, value)
            for key, values in truth.label_alerts(table["ra_deg"], table["dec_deg"], star_ra, star_dec, expected_flux,
                                                  set(follow), transient, labels, obs["truth"]["alert_radius"]).items():
                table[key] = values
            table["truth_mag"] = [true_mag(B, int(k), j) if k >= 0 else np.nan for k in table["truth_star"]]
            # the change the difference image should show at the star (electrons): its flux now minus its flux in
            # the reference (a transient: none; a followed variable: as in its telescope's reference coadd, ref_by_tel,
            # else its cycle mean; any other star: unchanged). What decides real or bogus for a real/bogus training
            # set (tools/rb_training_set.py).
            now = rates_at(B, j) * args.exptime
            k = table["truth_star"].to_numpy()
            flux_now = np.where(k >= 0, now[np.maximum(k, 0)], np.nan)
            flux_ref = np.where(k >= 0, flux_now, np.nan)
            tel = int(row["tel_id"])
            for i, kk in enumerate(k):
                if kk in transient:
                    flux_ref[i] = 0.0
                elif kk in follow:
                    grid = ref_by_tel.get(kk, {}).get(tel)
                    flux_ref[i] = (float(np.interp(airmass[j], AIRMASS_GRID, grid)) if grid is not None
                                   else float(np.mean(ref_rate[kk][B]))) * args.exptime
            table["truth_flux_now"], table["truth_flux_reference"] = flux_now, flux_ref
            table["truth_diff_flux"] = flux_now - flux_ref
            alert_rows.append(table)
        alerted.append([(a["key"], a["dx"], a["dy"]) for a in frame_alerts])
        n_alerts += len(frame_alerts)
        if args.catalogues is not None and catalogue is not None:
            args.catalogues.mkdir(parents=True, exist_ok=True)
            star = np.full(len(catalogue), -1)
            star[star_of[star_of >= 0]] = np.nonzero(star_of >= 0)[0]
            catalogue["star"] = star
            meta = dict(field=state.FIELD, band=B, exposure=int(j), date_obs=times[j].isot, bjd_tdb=float(bjd[j]),
                        airmass=float(airmass[j]), seeing=float(seeing[j]), exptime=args.exptime,
                        tel_id=int(row["tel_id"]), ratchet=int(row["ratchet"]), fieldang=float(row["field_angle"]),
                        **{f"zp_{key}": value for key, value in (zp or {}).items()})
            catalogue.meta = meta
            catalogue.write(args.catalogues / f"{state.FIELD}_{B}_{j:04d}.parquet", overwrite=True)
        this_frame = {}
        for k in follow:
            d = int(star_of[k]) if star_of is not None else -1
            ok = d >= 0 and np.isfinite(catalogue["mag"][d]) and catalogue["snr"][d] >= 5  # S/N >= 5
            mag, err = (float(catalogue["mag"][d]), float(catalogue["mag_err"][d])) if ok else (np.nan, np.nan)
            this_frame[k] = (mag, err)
            rows.append(dict(exposure=j, ut_hours=ut[j], airmass=airmass[j], seeing=seeing[j], star=labels[k],
                             true_mag=true_mag(B, k, j), saturated=bool(d >= 0 and catalogue["saturated"][d]),
                             measured_mag=mag, measured_err=err, band=B, tel_id=int(row["tel_id"]),
                             ratchet=int(row["ratchet"]), field_angle=float(row["field_angle"]), time_utc=row["time"]))
        measured.append(this_frame)
    print(f"hdps: {n_alerts} alerts from {len(frames)} frames ({n_skipped} frames hdps could not process)", flush=True)
    window_images = None
    if mosaic is not None:
        side, zoom = obs["field"]["window"], obs["output"]["movie_window_zoom"]
        window_images = mosaic.summary(mosaics, [r["footprint"] for r in records],
                                       (f"a {side} × {side}-minipix readout window" if side else "the whole sensor")
                                       + (f" (×{zoom:g} about the block)" if zoom > 1 else ""))
        print(f"field movie: {window_images['label']}, hdps's tiles {window_images['tiles']} in {mosaic.bin} x "
              f"{mosaic.bin} tile pixels per image pixel ({mosaic.shape[1]} x {mosaic.shape[0]}); "
              f"{sum(m is not None for m in mosaics)}/{len(mosaics)} frames with an image", flush=True)
    if alert_rows:  # <name>_night_<band>.alerts.csv: every hdps alert row, its features and its truth
        import pandas as pd

        alerts = pd.concat(alert_rows, ignore_index=True)
        alerts.to_csv(args.output.with_suffix(".alerts.csv"), index=False)
        accepted = alerts[alerts["accepted"]]
        print("alert truth (accepted): " + ", ".join(f"{c} {n}" for c, n in accepted["truth_class"].value_counts().items()),
              flush=True)
    with open(args.output.with_suffix(".photometry.csv"), "w", newline="") as handle:  # (no rows: no star followed)
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else PHOTOMETRY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    if obs["output"]["movie_data"]:  # what the movies need, so they can be drawn later (--render-only)
        import joblib

        joblib.dump(dict(frames=list(frames), measured=measured, alerted=alerted, cutouts=cutouts,
                         records=[slim_record(r) for r in records], window=window_images), movie_file, compress=3)
        print(f"movie data -> {movie_file.name} ({movie_file.stat().st_size / 1e6:.0f} MB)", flush=True)
    movies(frames, measured, alerted, cutouts, movie_records([slim_record(r) for r in records]), window_images)
    if not obs["output"]["raw"]:  # the raw frames and hdps's bulky per-frame products (read above)
        shutil.rmtree(work / "raw", ignore_errors=True)
        shutil.rmtree(work / "solved", ignore_errors=True)
        for tile in work.glob("epoch_*_tile_*.fits"):
            tile.unlink()
    print(f"saved {args.output.with_suffix('.photometry.csv') if args.no_render else args.output}")


# ---------------------------------------------------------------- the movie
























if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        main()
