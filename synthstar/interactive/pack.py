"""Field packs: one finished observation's night in one band, as the field movie draws it, saved for the interactive
app (data/interactive/packs/<pack>_<band>.joblib).

The engine's own setup runs once (night_movie.py --render-only on the run's movie data) and its draw data is kept:
the dict the movie's drawing (night_movie.MovieFrame) reads, with what cannot be saved replaced by tables (true magnitudes of the stars it
shows, at every frame) and the module's settings (the block, the tile) alongside. Added for the app:

- candidates: every hdps alert candidate of each frame (the run kept them all: AlertVettingTask's
  emit_all_candidates with no model), its position, its movie key (the followed star within 2", as the movie's
  alerts) and the 18 features hdps computed, so the app scores them with any real/bogus model and cut;
- zp: each frame's zero point on the cut-out (hdps's source detector on hdps's tile, matched to hdps's own calibrated
  catalogue magnitudes of the same stars): an added star measured with it is on hdps's scale for that frame.

    python -m synthstar.interactive.pack outputs/pseudo_real/interactive_m3 --pack m3
"""
import argparse
import json
import os
import pickle
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PACKS = REPO / "data" / "interactive" / "packs"
CACHE = PACKS / "cache"  # per pack (and its file's time stamp): an uncompressed copy to memory-map, derived values
BANDS = ("beta", "rho")
VERSION = 1


def engine_argv(out, band):
    """night_movie.py's arguments for a finished observation, as observe.run_engine passes them, drawing from the
    run's movie data."""
    from synthstar import runconfig

    obs = runconfig.resolve(runconfig.load(out / "observation.toml"))
    o, name = obs["output"], obs["field"]["name"]
    n_placed = len(json.loads((out / "scene" / "sources.json").read_text()))
    return ["--scene", str(out / "scene"), "--field", name, "--band", band, "-o", str(out / f"{name}_night_{band}.mp4"),
            "--observation", str(out / "observation.toml"), "--work", str(out / "hdps"), "--stride", str(o["stride"]),
            "--fps", str(o["fps"]), "--dpi", str(o["dpi"]), "--seed", str(o["noise_seed"]),
            "--exptime", str(obs["argussim"]["schedule"]["exptime_s"]), "--highlight", str(n_placed), "--render-only"]


def capture(out, band):
    """(night_movie module, args, draw data, frames) of a finished observation: the engine's setup, then its render
    call intercepted."""
    import night_movie as nm

    got = {}
    real_render = nm.render
    nm.render = lambda args, d, frames: got.update(args=args, d=d, frames=list(frames))
    try:
        nm.main(engine_argv(out, band))
    finally:
        nm.render = real_render
    if not got:
        raise SystemExit(f"{out}: the engine drew nothing (no movie data for {band}?)")
    return nm, got["args"], got["d"], got["frames"]


def _picklable(v):
    try:
        pickle.dumps(v)
        return True
    except Exception:
        return False


def module_settings(nm):
    """night_movie's module-level settings (the field, block, tile, ...) as main left them: draw() reads them."""
    return {k: v for k, v in vars(nm).items() if k.isupper() and not callable(v) and _picklable(v)}


def candidates(nm, out, args, d, frames):
    """Per frame: dict(key [movie keys], dx, dy [arcsec], features (n x 18, the bundle's order), truth [class]) of
    every hdps alert candidate in the run's alerts.csv."""
    import pandas as pd

    from hdps.alerts.features import FEATURE_NAMES

    path = args.output.with_suffix(".alerts.csv")
    empty = dict(key=[], dx=np.zeros(0), dy=np.zeros(0), features=np.zeros((0, len(FEATURE_NAMES)), np.float32),
                 truth=np.array([], dtype=object))
    if not path.exists():
        return [empty for _ in frames]
    table = pd.read_csv(path)
    follow = np.asarray(d["follow"], int)
    star_ra, star_dec = nm.block_radec(np.asarray(d["stars"]["dx"], float), np.asarray(d["stars"]["dy"], float))
    out_frames = []
    for j in frames:
        t = table[table["exposure"] == j]
        if not len(t):
            out_frames.append(empty)
            continue
        ra, dec = t["ra_deg"].to_numpy(float), t["dec_deg"].to_numpy(float)
        dx, dy = nm.block_offsets(ra, dec)
        keys = []
        for i in range(len(t)):  # as night_movie.alerts_of: the followed star within 2", else the position
            if len(follow):
                sep = np.hypot((star_ra[follow] - ra[i]) * np.cos(np.radians(star_dec[follow])),
                               star_dec[follow] - dec[i]) * 3600
                near = np.nonzero(sep <= 2.0)[0]
            else:
                near = []
            keys.append(int(follow[near[np.argmin(sep[near])]]) if len(near)
                        else ("pos", round(float(dx[i]) / 3), round(float(dy[i]) / 3)))
        feats = np.column_stack([t[f"feat_{n}"].to_numpy(float) for n in FEATURE_NAMES]).astype(np.float32)
        out_frames.append(dict(key=keys, dx=np.asarray(dx, float), dy=np.asarray(dy, float), features=feats,
                               truth=t["truth_class"].to_numpy(object) if "truth_class" in t else
                               np.full(len(t), "", object)))
    return out_frames


def cutout_zero_points(out, args, d, frames, mag_max=17.0, match_arcsec=1.5):
    """Each frame's zero point on its science cut-out: median of hdps's calibrated catalogue magnitude + 2.5 log10
    (the cut-out flux hdps's detector measures) over the unsaturated stars brighter than mag_max found in both. A frame
    with fewer than 5 such stars (a sparse block: M45 beta, 34 of 224 frames) takes hdps's zero point of the frame +
    the cut-out's offset from it, the median over the frames that have both (the tile's flux scale: M45 beta with
    TrueCalibrationTask 0.279 mag, rms 0.008)."""
    from astropy.table import Table

    from synthstar.interactive import measure

    sdx, sdy = np.asarray(d["stars"]["dx"], float), np.asarray(d["stars"]["dy"], float)
    folder = out / "catalogues"
    name = args.field
    zps, frame_zp = [], np.full(len(frames), np.nan)
    for n, j in enumerate(frames):
        c = d["cutouts"][n]
        path = folder / f"{name}_{args.band}_{j:04d}.parquet"
        if c is None or not path.exists():
            zps.append(np.nan)
            continue
        cat = Table.read(path)
        frame_zp[n] = float(cat.meta.get("zp_zp", np.nan))
        ok = (np.asarray(cat["star"]) >= 0) & ~np.asarray(cat["saturated"], bool) & (np.asarray(cat["mag"]) < mag_max)
        cat = cat[ok]
        found = measure.detect(c["science"])
        if len(cat) == 0 or len(found) == 0:
            zps.append(np.nan)
            continue
        fx, fy = measure.to_offsets(c["extent"], np.shape(c["science"]), np.asarray(found["xcentroid"]),
                                    np.asarray(found["ycentroid"]))
        k = np.asarray(cat["star"], int)
        diffs = []
        for x, y, m in zip(sdx[k], sdy[k], np.asarray(cat["mag"], float)):
            sep = np.hypot(fx - x, fy - y)
            i = int(np.argmin(sep))
            if sep[i] <= match_arcsec and found["flux"][i] > 0 and not found["is_saturated"][i]:
                diffs.append(m + 2.5 * np.log10(float(found["flux"][i])))
        zps.append(float(np.median(diffs)) if len(diffs) >= 5 else np.nan)
    zps = np.array(zps)
    both = np.isfinite(zps) & np.isfinite(frame_zp)
    if both.sum() >= 10:
        fill = ~np.isfinite(zps) & np.isfinite(frame_zp)
        zps[fill] = frame_zp[fill] + np.median(zps[both] - frame_zp[both])
    return zps


def sfft_settings(out):
    """The run's SFFTSubtractionTask settings that shape what can alert (its bright-star detection mask)."""
    from synthstar import runconfig

    obs = runconfig.resolve(runconfig.load(Path(out) / "observation.toml"))
    task = next((t for t in obs["hdps"]["tasks"] if t["type"] == "SFFTSubtractionTask"), None)
    return dict(task["config"]) if task is not None else {}


def chain(out):
    """The run's hdps Task chain (its types, in order): whether hdps solved each frame or took the simulation's
    calibration (TrueCalibrationTask)."""
    from synthstar import runconfig

    return [t["type"] for t in runconfig.resolve(runconfig.load(Path(out) / "observation.toml"))["hdps"]["tasks"]]


def image_settings(out):
    """The run's [image] settings (its PSF wing): the edited star is drawn as the run drew its stars."""
    from synthstar import runconfig

    return dict(runconfig.resolve(runconfig.load(Path(out) / "observation.toml"))["image"])


def build(out, band, pack_name):
    """Capture one observation's band and write data/interactive/packs/<pack_name>_<band>.joblib."""
    import joblib

    out = Path(out).resolve()
    nm, args, d, frames = capture(out, band)
    shown = list(d.get("shown", d["follow"]))
    source = {k: int(d["source"][k]) for k in shown}
    pack = dict(
        version=VERSION, observation=out.name, pack=pack_name, band=band,
        args=dict(band=args.band, exptime=float(args.exptime), fps=int(args.fps), date=str(args.date),
                  field=str(args.field), dpi=int(args.dpi)),
        settings=module_settings(nm),
        frames=list(frames), cutouts=d["cutouts"],
        times=d["times"], bjd=np.asarray(d["bjd"]), ut=np.asarray(d["ut"]), airmass=np.asarray(d["airmass"]),
        seeing=np.asarray(d["seeing"]),
        survey={c: np.asarray(d["survey"][c]) for c in ("tel_id", "field_angle")},
        stars={c: np.asarray(d["stars"][c], float) for c in ("dx", "dy")}, G=np.asarray(d["G"], float),
        in_block=np.asarray(d["in_block"], bool),
        static_mag={b: np.asarray(d["static_mag"][b], float) for b in BANDS},
        n_follow=len(d["follow"]), shown=shown, source=source,
        pulsators={source[k]: d["pulsators"][source[k]] for k in shown},
        labels={k: d["labels"][k] for k in shown},
        follow_cycle={k: d["follow_cycle"][k] for k in shown},
        model_curve={k: d["model_curve"][k] for k in shown if k in d["model_curve"]},
        follow_sed={k: np.asarray(d["follow_sed"][k], np.float32) for k in shown},
        wave={k: d["models"][k].wave for k in shown},
        measured={k: np.array([f[k] for f in d["measured"]], float) for k in shown},
        true={b: {k: np.array([d["true_mag"](b, k, j) for j in frames], float) for k in shown} for b in BANDS},
        rel={k: d["rel"][k] for k in shown},
        rgb=d["rgb"], dlim=float(d["dlim"]), centre=d["centre"],
        candidates=candidates(nm, out, args, d, frames),
        zp=cutout_zero_points(out, args, d, frames),
        sfft=sfft_settings(out),
        image=image_settings(out),
        chain=chain(out),
    )
    n_all = len(frames)
    pack = drop_failed(pack)
    frames = pack["frames"]
    if len(frames) < n_all:
        print(f"{n_all - len(frames)} of {n_all} frames left out: hdps failed on them (no cut-outs)", flush=True)
    PACKS.mkdir(parents=True, exist_ok=True)
    path = PACKS / f"{pack_name}_{band}.joblib"
    joblib.dump(pack, path, compress=3)
    n_cand = sum(len(c["key"]) for c in pack["candidates"])
    print(f"{path.relative_to(REPO)}: {len(frames)} frames, {len(shown)} variables shown of {len(d['follow'])} followed, "
          f"{n_cand} alert candidates, cut-out zero points {np.nanmedian(pack['zp']):.3f} "
          f"(rms {np.nanstd(pack['zp']):.3f}, {int(np.isfinite(pack['zp']).sum())} frames)", flush=True)
    return path


def drop_failed(pack):
    """The pack without the frames hdps failed on (no cut-outs: no difference image to show or measure the star on);
    every per-frame entry kept in step."""
    keep = [n for n, c in enumerate(pack["cutouts"]) if c is not None]
    if len(keep) == len(pack["frames"]):
        return pack
    for k in ("frames", "cutouts", "candidates"):
        pack[k] = [pack[k][n] for n in keep]
    pack["zp"] = np.asarray(pack["zp"])[keep]
    pack["measured"] = {k: np.asarray(v)[keep] for k, v in pack["measured"].items()}
    pack["true"] = {b: {k: np.asarray(v)[keep] for k, v in t.items()} for b, t in pack["true"].items()}
    return pack


def _replace_into(path, write):
    """write(part) a file beside path, then move it into place whole (several processes may make it at once)."""
    part = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.part")
    write(part)
    os.replace(part, path)


def load(pack_name, band):
    """A pack, with night_movie's module settings restored as they were when it was made (draw() reads them). Read
    memory-mapped from an uncompressed copy (made on its first load, in pack.CACHE): instant, and the cut-outs are
    shared between the processes that read them rather than copied into each. pack["stamp"] names this version of
    the pack (its file's modification time), for values derived from it (cached)."""
    import joblib

    import night_movie as nm

    path = PACKS / f"{pack_name}_{band}.joblib"
    stamp = f"{pack_name}_{band}.{path.stat().st_mtime_ns}"
    flat = CACHE / f"{stamp}.joblib"
    if not flat.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        _replace_into(flat, lambda part: joblib.dump(joblib.load(path), part))
        for old in CACHE.glob(f"{pack_name}_{band}.*"):  # what was derived from an earlier version of the pack
            if not old.name.startswith(stamp) and not old.name.endswith(".part"):
                old.unlink(missing_ok=True)
    pack = joblib.load(flat, mmap_mode="r")
    pack["stamp"] = stamp
    if pack.get("version") != VERSION:
        raise SystemExit(f"{pack_name}_{band}: pack version {pack.get('version')}, this code reads {VERSION}: rebuild it")
    for k, v in pack["settings"].items():
        setattr(nm, k, v)
    pack["models"] = {k: SimpleNamespace(wave=w) for k, w in pack["wave"].items()}
    return pack


def cached(pack, what, make):
    """A value derived from a pack alone (make()), kept in pack.CACHE for this version of the pack."""
    import joblib

    path = CACHE / f"{pack['stamp']}.{what}.joblib"
    if path.exists():
        return joblib.load(path)
    value = make()
    save_cached(path, value)
    return value


def save_cached(path, value):
    """Save a value in pack.CACHE (whole, even with several processes saving it at once)."""
    import joblib

    CACHE.mkdir(parents=True, exist_ok=True)
    _replace_into(path, lambda part: joblib.dump(value, part))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("observation_dir", type=Path, help="a finished observation with movie data (output.movie_data)")
    p.add_argument("--pack", required=True, help="the pack's name (the app's field dropdown key), e.g. m3")
    p.add_argument("--bands", nargs="+", default=list(BANDS))
    args = p.parse_args(argv)
    sys.path.insert(0, str(REPO))
    for band in args.bands:
        build(args.observation_dir, band, args.pack)


if __name__ == "__main__":
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        main()
