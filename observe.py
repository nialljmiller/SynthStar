#!/usr/bin/env python3
"""Observe built stars with Argus: one night of 60 s exposures of a pointing, measured, as a movie, a photometry CSV,
FITS exposures and/or the interactive page.

    python observe.py examples/observations/f294.toml
    python observe.py examples/observations/synthetic_transients.toml --set output.stride=10 --date 2026-07-19

Stars must be built first (make_stars.py); an observation only names them. The observation file
(examples/observations/template.toml lists every key):

    [field]    where Argus points and how big a block is shown
    [night]    which night ("auto": the night the placed real stars change most), UT window, site; observing
               "ideal" (every exposure in both bands, the season's median seeing, one PSF field angle) or "survey"
               (the exposures ArgusSim's array actually takes: telescope, band, field angle, seeing per ratchet)
    [stars]    the background: a built star set, at its real positions or scattered at random (a synthetic field)
    [[place]]  each star: where (x, y in percent of the image; real Gaia stars sit where they are) and when (t0 for
               transients, epoch for periodic stars; real stars keep their own ephemeris)
    [pipeline] which implementation runs each measurement stage (mesa_argus/pipeline/)
    [measure]  the stages' settings
    [output]   movie, bands, FITS, page; noise seed, stride, PSF and kernel options

Everything about observing is added here: the pointing, the night (ArgusSim's schedule), the Milky Way dust (the 3D map
at each star's distance along its sightline, mesa_argus/dust.py), positions and timing. night_movie.py then simulates and
measures the exposures from its scene/.

Outputs in outputs/pseudo_real/<name>/ (real Gaia stars where they are) or outputs/fully_synthetic/<name>/
(layout random): <name>_night_<band>.mp4 and .photometry.csv, fits/, observation.toml (the resolved file:
running it makes the same data), summary.md, scene/.
"""
import argparse
import json
import shutil
import os
import subprocess
import sys
from datetime import date as Date
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
STARS = REPO / "stars"

from mesa_argus import dust, runconfig  # noqa: E402


def read_observation():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("observation", type=Path)
    p.add_argument("--ra", type=float)
    p.add_argument("--dec", type=float)
    p.add_argument("--name")
    p.add_argument("--date", help="YYYY-MM-DD or auto")
    p.add_argument("--bands", nargs="+")
    p.add_argument("--place", action="append", default=[], help="a [[place]] as a TOML inline table (replaces the file's)")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="any key, e.g. output.stride=10")
    p.add_argument("--scene-only", action="store_true", help="write the scene and stop")
    p.add_argument("--decoupled", action="store_true",
                   help="render every band's raw stack first, then run hdps on the stacks (two passes sharing one work "
                        "dir), instead of the fused render+hdps: the image-maker/hdps split of a run")
    a = p.parse_args()
    flags = (("ra", "field.ra"), ("dec", "field.dec"), ("name", "field.name"), ("date", "night.date"),
             ("bands", "output.bands"))
    overrides = {dotted: getattr(a, flag) for flag, dotted in flags if getattr(a, flag) is not None}
    for item in a.set:
        dotted, _, v = item.partition("=")
        overrides[dotted.strip()] = runconfig.parse_value(v.strip())
    if a.place:
        overrides["place"] = [runconfig.parse_value(t if t.strip().startswith("{") else "{" + t + "}") for t in a.place]
    obs = runconfig.resolve(runconfig.merge(runconfig.load(a.observation), overrides))
    obs["field"]["name"] = obs["field"]["name"] or a.observation.stem
    return obs, a.scene_only, a.decoupled


def built_star(name):
    folder = STARS / name
    if not (folder / "meta.json").exists():
        raise SystemExit(f"no built star {name!r}: build it first (python make_stars.py <a star file naming it>)")
    return folder, json.loads((folder / "meta.json").read_text())


def choose_night(f, n, window, real_stars, site, max_airmass):
    """The night asked for, or the nearest one the field is up >= 3 h (with a UT window: at all), or "auto"."""
    from mesa_argus import night

    if n["date"] == "auto":  # the night the placed real stars change most
        picks = [dict(kind=r["kind"], period=r["period"], epoch_bjd=r["epoch_bjd"], amp_G=r["amp_G"], veb=r.get("veb"))
                 for r in real_stars]
        date, score = night.best_night(f["ra"], f["dec"], Date.today().isoformat(), site, *window, max_airmass, picks)
        print(f"night chosen for the placed real stars (date auto): {date} (visibility score {score:.2f} of {len(picks)})")
        return date
    asked = n["date"] or Date.today().isoformat()
    shortest = 5 if any(window) else 180  # minutes
    date, minutes = night.nearest_night(f["ra"], f["dec"], asked, site, *window, max_airmass, shortest)
    if date != asked:
        print(f"not up on the night of {asked}: using the nearest night it is, {date}")
    return date


def write_background(scene_dir, s, star_set, ra, dec, rng):
    """The background stars into the scene: the set's catalogue at the stars' real positions, or at random positions
    within 0.12 deg of the pointing (synthetic). Each star gets its distance and two E(B-V): in front of it where Gaia
    measured it (ebv_own), and where it is observed (ebv); the engine takes the first out and puts the second in.
    Returns a line for the summary."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.table import Table

    from mesa_argus.gaia import RADIUS_DEG, col, distances

    catalogue = STARS / s["background"] / "catalogue"
    gaia_stars = Table.read(catalogue / "gaia_dr3_core.ecsv")
    n = len(gaia_stars)
    distance, how = distances(catalogue, gaia_stars["Source"])
    real_ra, real_dec = col(gaia_stars, "RA_ICRS"), col(gaia_stars, "DE_ICRS")
    ebv_own, which = dust.ebv(real_ra, real_dec, distance)
    counts = (f"distances: {int(np.sum(how == 'Bailer-Jones'))} Bailer-Jones, {int(np.sum(how == 'GSP-Phot'))} GSP-Phot, "
              f"{int(np.sum(how == 'none'))} none (the whole column); dust from {'Bayestar19' if (which == 'bayestar19').all() else 'Bayestar19 and SFD'}")
    if s["layout"] == "real":
        offset = SkyCoord(ra * u.deg, dec * u.deg).separation(SkyCoord(*star_set["centre"], unit="deg")).deg
        if offset > 0.02:  # real positions only make sense at the set's own pointing
            raise SystemExit(f"background {s['background']} is centred {offset:.3f} deg from the pointing")
        for f in catalogue.glob("*.ecsv"):
            shutil.copy(f, scene_dir / f.name)
        gaia_stars["distance_pc"], gaia_stars["ebv_own"], gaia_stars["ebv"] = distance, ebv_own, ebv_own
        gaia_stars.write(scene_dir / "gaia_dr3_core.ecsv", overwrite=True)
        return (f"Background: `{s['background']}` ({star_set['params']}), at the stars' real positions; each constant at "
                f"its Gaia G and colour (colour-matched MESA SED through the dust in front of it: {counts}).")
    radius, angle = RADIUS_DEG * np.sqrt(rng.random(n)), 2 * np.pi * rng.random(n)  # uniform over the circle
    pos = SkyCoord(ra * u.deg, dec * u.deg).spherical_offsets_by(radius * np.cos(angle) * u.deg,
                                                                 radius * np.sin(angle) * u.deg)
    synthetic = Table({"Source": np.arange(1, n + 1), "RA_ICRS": pos.ra.deg, "DE_ICRS": pos.dec.deg})
    for column in ("Gmag", "BPmag", "RPmag", "BP-RP"):
        synthetic[column] = gaia_stars[column]
    synthetic["distance_pc"], synthetic["ebv_own"] = distance, ebv_own
    synthetic["ebv"] = dust.ebv(pos.ra.deg, pos.dec.deg, distance)[0]
    synthetic.meta.update(dict(synthetic=True, parent=f"star set {s['background']}", seed=s["seed"],
                               note="Gaia G, BP, RP of every source (as Gaia measured them, through ebv_own); positions "
                                    "uniform at random around the pointing, seen through ebv (the engine re-reddens)"))
    synthetic.write(scene_dir / "synthetic_stars.ecsv", overwrite=True)
    return (f"Background: `{s['background']}` ({star_set['params']}): its {n} stars at positions drawn uniformly at "
            f"random within {RADIUS_DEG} deg of the pointing (seed {s['seed']}), at their own distances, each seen through "
            f"the dust in front of it there ({counts}); every star constant.")


def catalogue_entry(sed_file, ebv, distance):
    """Intensity-mean Gaia G, BP-RP and G amplitude of a periodic model at `distance` (pc) through its dust: what a
    catalogue would list for it (the movie calibrates against these, as for a Gaia star)."""
    import h5py

    from mesa_argus.binary import vega_mag

    with h5py.File(sed_file) as h:
        wave, flux_10pc, t = h["wavelength"][:], h["flux"][:], h["time_days"][:]
    seen = dust.redden(wave, flux_10pc * (10.0 / distance) ** 2, ebv)
    weight = np.gradient(t)  # the samples need not be evenly spaced in time
    mags = {band: np.array([vega_mag(wave, f, band) for f in seen]) for band in ("G", "Gbp", "Grp")}
    mean = {band: -2.5 * np.log10(np.sum(weight * 10 ** (-0.4 * m)) / np.sum(weight)) for band, m in mags.items()}
    return float(mean["G"]), float(mean["Gbp"] - mean["Grp"]), float(np.ptp(mags["G"]))


def placed_distance(p, folder, meta):
    """A placed star's distance (pc): a periodic model's is [[place]] distance, else its file's (an intrinsic model,
    stored at 10 pc, needs one); a transient's is its model's if it has one (a dwarf nova); none for a supernova or a
    microlensing source (they sit behind all the dust)."""
    import h5py

    if meta["timing"] != "periodic":
        if p["distance"] is not None:
            raise SystemExit(f"{p['star']}: a transient's distance is its model's; [[place]] distance is for periodic stars")
        return meta.get("distance_pc", meta.get("d_pc"))
    if p["distance"] is not None:
        return float(p["distance"])
    with h5py.File(folder / "sed.h5") as h:
        dm = float(h.attrs["distance_modulus"])
    if dm == 0:
        raise SystemExit(f"{p['star']} is an intrinsic model (10 pc): give its distance (pc) in [[place]]")
    return 10 ** (dm / 5 + 1)


def inject(obs, night_start_bjd, night_end_bjd, half_block):
    """[inject]: library stars scattered over the readout window, as [[place]] entries (their positions in percent of
    the block, beyond 0..100 where outside it): a random template from each family in turn; a position in the disc
    margin_arcsec inside the window's half-width (inside every telescope's window at any rotation) and
    separation_arcsec from the others; a transient's t0 so the nights fall inside
    its model (its peak within the nights where the model allows), a periodic star's phase at random and its distance
    for a mean G drawn from mag_min..mag_max. Returns [(placement, folder, meta)]."""
    import h5py

    import night_movie

    c = obs["inject"]
    if not c["number"]:
        return []
    rng = np.random.default_rng(c["seed"])
    pools = {f: sorted(d.name for d in STARS.glob(f"{f}_*") if (d / "meta.json").exists() and (d / "sed.h5").exists())
             for f in c["families"]}
    pools = {f: names for f, names in pools.items() if names}
    if not pools:
        raise SystemExit(f"[inject]: no built library stars for {c['families']}")
    half_window = obs["field"]["window"] * night_movie.MINIPIX_PX * night_movie.SKY_PX / 2 - c["margin_arcsec"]
    families = list(pools)
    chosen, positions = [], []
    while len(chosen) < c["number"]:
        family = families[len(chosen) % len(families)]
        name = pools[family][rng.integers(len(pools[family]))]
        folder, meta = built_star(name)
        for _ in range(100):  # a position clear of the others
            r, angle = half_window * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)  # uniform in the disc that
            dx, dy = r * np.cos(angle), r * np.sin(angle)  # every telescope's window holds, whatever its rotation
            if all(np.hypot(dx - a, dy - b) >= c["separation_arcsec"] for a, b in positions):
                break
        else:
            raise SystemExit(f"[inject]: {c['number']} stars do not fit {c['separation_arcsec']}\" apart")
        p = dict(star=name, x=float(50 + dx / (2 * half_block) * 100), y=float(50 + dy / (2 * half_block) * 100), t0=None,
                 epoch=None, distance=None)
        with h5py.File(folder / "sed.h5") as h:
            first, last = float(h["time_days"][0]), float(h["time_days"][-1])
            wave = h["wavelength"]
            covers = float(wave[0]) <= 3000.0 and float(wave[-1]) >= 10000.0  # Angstrom: ArgusSim's beta and rho curves
        if not covers:  # a template ending inside a band (some sncosmo/SALT sources stop at ~9200 A): not this one
            continue
        if meta["timing"] == "transient":
            lo, hi = night_end_bjd - last, night_start_bjd - first  # the nights inside the model
            if lo > hi:
                continue  # a model shorter than the nights: not this one
            peak = float(meta.get("peak_days") or 0.0)
            want_lo, want_hi = max(lo, night_start_bjd - 3.0 - peak), min(hi, night_end_bjd + 1.0 - peak)
            p["t0"] = float(rng.uniform(want_lo, want_hi) if want_lo < want_hi else rng.uniform(lo, hi))
        else:
            p["epoch"] = float(night_start_bjd + rng.uniform(0, meta["period"]))
            absolute = catalogue_entry(folder / "sed.h5", 0.0, 10.0)[0]  # mean G at 10 pc, no dust
            p["distance"] = float(10.0 * 10 ** ((rng.uniform(c["mag_min"], c["mag_max"]) - absolute) / 5))
        chosen.append((p, folder, meta))
        positions.append((dx, dy))
    return chosen


def place_stars(placements, scene_dir, night_info, half_width, rng):
    """One row per placed star for the engine (sources.json): its SED file, position, timing and dust (the 3D map at
    its distance on the pointing's sightline; a real star, at its own position and distance); and the
    ephemerides of real HADS / binaries, which the Gaia SOS tables do not hold (extra_pulsators.json)."""
    import h5py
    from astropy.table import Table

    from mesa_argus import night

    start, night_start_bjd, night_end_bjd, ra, dec, site = night_info
    catalogue = scene_dir / "gaia_dr3_core.ecsv"
    in_catalogue = set(np.asarray(Table.read(catalogue)["Source"]).astype(np.int64).tolist()) if catalogue.exists() else set()
    rows, extra, count = [], {}, {"transient": 0, "periodic": 0}
    for i, (p, folder, meta) in enumerate(placements, 1):
        row = dict(star=p["star"], generator=meta["generator"], engine=meta["engine"], timing=meta["timing"],
                   kind=meta["kind"], model=meta.get("model", meta["generator"]), params=meta["params"],
                   reference=meta.get("reference", ""), file=str((folder / "sed.h5").relative_to(REPO)))

        if meta.get("real"):  # a real star: its own position, ephemeris and sightline dust; already in the catalogue
            real = meta["real"]
            if real["source"] not in in_catalogue:
                raise SystemExit(f"{p['star']} is Gaia DR3 {real['source']}: observe it in its own field (layout real)")
            rows.append(dict(row, label="", source=real["source"], real=True, ebv=real["sightline_ebv"],
                             period=real["period"], epoch_bjd=real["epoch_bjd"]))
            if real["kind"] in ("HADS", "EB"):
                extra[str(real["source"])] = dict(period=real["period"], epoch_bjd=real["epoch_bjd"], kind=real["kind"],
                                                  amp=real["amp_G"], G=real["G"])
            continue

        timing = meta["timing"]
        when = night.resolve_time(p["t0"] if timing == "transient" else p["epoch"], start, ra, dec, site)
        x, y = (rng.uniform(0, 100) if v == -1 else float(v) for v in (p["x"], p["y"]))  # -1: random
        count[timing] += 1
        label = ("T" if timing == "transient" else "P") + str(count[timing])
        dx, dy = (x / 100 - 0.5) * 2 * half_width, (y / 100 - 0.5) * 2 * half_width  # arcsec from the block centre
        distance = placed_distance(p, folder, meta)
        ebv = float(dust.ebv(ra, dec, distance)[0][0])
        row.update(label=label, source=-i, x_percent=x, y_percent=y, dx=float(dx), dy=float(dy), ebv=ebv,
                   distance_pc=float(distance) if distance else None)
        if timing == "transient":
            with h5py.File(folder / "sed.h5") as h:
                first, last = float(h["time_days"][0]), float(h["time_days"][-1])
            if night_end_bjd - when > last or (first < 0 and night_start_bjd - when < first):  # no extrapolation
                raise SystemExit(f"{p['star']}: the night runs outside its model ({first:.2f} to {last:.2f} d from t0)")
            row.update(t0_bjd=when, z=float(meta.get("z", np.nan)), peak_days=float(meta.get("peak_days", np.nan)))
        else:
            G, bp_rp, amplitude = catalogue_entry(folder / "sed.h5", ebv, distance)
            row.update(period=meta["period"], epoch_bjd=when, G=G, bp_rp=bp_rp, amp_G=amplitude)
        rows.append(row)
    return rows, extra


def run_engine(obs, out, n_placed, extra=(), reference=None, decoupled=False):
    """The engine (night_movie.py) for each band at once (each is single-threaded): the band's exposures as raw frames
    through hdps, with the band's hdps coadds as references (reference: the folder holding <band>/coadds).
    decoupled: render every band's raw stack first (--stop-after-render), then run hdps on those stacks
    (--hdps-from-stack) as a separate pass sharing the same --work, instead of the fused render+hdps in one call."""
    name, o = obs["field"]["name"], obs["output"]
    options = ["--observation", str(out / "observation.toml"), "--work", str(out / "hdps"), "--stride", str(o["stride"]),
               "--fps", str(o["fps"]), "--dpi", str(o["dpi"]), "--seed", str(o["noise_seed"]),
               "--exptime", str(obs["argussim"]["schedule"]["exptime_s"]), "--highlight", str(n_placed)]
    if not o["movie"]:
        options += ["--no-render"]
    if o["pipeline_movie"] and "--hdps-pass" not in extra:
        options += ["--pipeline-movie"]
    if o["catalogues"]:
        options += ["--catalogues", str(out / "catalogues")]
    options += list(extra)

    def launch(phase_extra, suffix):  # start night_movie for every band at once, wait, raise on any failure
        running = []
        for band in o["bands"]:
            mp4 = out / f"{name}_night_{band}.mp4"
            print(f"observing {band} -> {mp4.relative_to(REPO)} (log {mp4.with_suffix(suffix).relative_to(REPO)})", flush=True)
            log = open(mp4.with_suffix(suffix), "w")
            band_options = ["--reference", str(reference / band / "coadds")] if reference is not None else []
            command = [sys.executable, "night_movie.py", "--scene", str(out / "scene"), "--field", name, "--band", band,
                       "-o", str(mp4), *options, *phase_extra, *band_options]
            # NUMBA_BOUNDSCHECK: an out-of-range index in a numba kernel (hdps's CPU mirrors) raises IndexError, which
            # hdps's worker counts as one failed frame, instead of a segmentation fault that kills the band's whole run
            # (two ρ runs died that way on 2026-10-23, in a numba kernel during difference detection). Its own
            # NUMBA_CACHE_DIR: kernels cached without the check would otherwise be reused without it.
            env = dict(os.environ, NUMBA_BOUNDSCHECK="1", NUMBA_CACHE_DIR=os.path.expanduser("~/.cache/numba_boundscheck"))
            running.append((band, log, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)))
        failed = []
        for band, log, process in running:
            if process.wait() != 0:
                failed.append(band)
            log.close()
        if failed:
            raise SystemExit(f"the engine failed for {', '.join(failed)}: see the log(s) in {out.relative_to(REPO)}/")

    if decoupled:  # the image-maker half (every band's raw stack), then the hdps half, two passes sharing one --work
        launch(["--stop-after-render"], ".render.log")
        launch(["--hdps-from-stack"], ".log")
    else:
        launch([], ".render.log" if "--render-only" in extra else ".log")
    if o["catalogues"] and o["lightcurves"] and "--hdps-pass" not in extra and "--render-only" not in extra:
        from mesa_argus.pipeline import lightcurves

        catalogue = lightcurves.positional(out / "catalogues", out, dict(obs["lightcurves"]))
        print(f"catalogue.csv: {len(catalogue)} objects; their light curves in lightcurves/", flush=True)


def main():
    import os

    obs, scene_only, decoupled = read_observation()
    os.chdir(REPO)
    observe(obs, runconfig.output_folder(obs["field"]["name"], obs["stars"]["layout"]), scene_only, decoupled=decoupled)


def make_reference(obs, out, placements, science=None):
    """hdps's references, by its coadd bootstrap (docker/demo/config.yaml): the [night] reference_nights before the
    first night are observed (the same field and stars, the transients left out), and each band's frames go through
    hdps up to CoaddTapTask, whose coadds are the references. science: the science night's exposures, for [night]
    reference_geometry_px. Returns the folder holding <band>/coadds."""
    import copy
    from datetime import date as Date, timedelta

    reference = copy.deepcopy(obs)
    n = obs["night"]["reference_nights"]
    reference["night"].update(date=(Date.fromisoformat(obs["night"]["date"]) - timedelta(days=n)).isoformat(), nights=n,
                              only_nights=[])  # all of its own nights, whichever of the science nights run
    reference["place"] = [place for place, folder, meta in placements if meta["timing"] != "transient"]
    reference["inject"]["number"] = 0  # the injected periodic stars are in [[place]] above; no transients
    # every exposure, whatever the science night's stride: a reference is a deep stack (hdps's coadds of whole ratchets);
    # a strided one stacks 1-3 frames, leaving residuals unlike any the real/bogus model was trained on
    reference["output"].update(movie=False, pipeline_movie=False, catalogues=False, lightcurves=False, stride=1)
    ref_out = out / "reference"
    best = obs["night"]["reference_ratchets"]
    near = obs["night"]["reference_geometry_px"]
    if obs["night"]["reference_oracle"]:  # TEST ONLY: tonight's own exposures, another noise draw
        reference["night"].update(date=obs["night"]["date"], nights=1)
        reference["output"]["noise_seed"] = obs["output"]["noise_seed"] + 1
        best, near, science = 0, 0, None
        from argus_sim import c as cfg

        shift = obs["night"]["reference_oracle_shift_px"] * cfg.telescope.plate_scale / 3600 / np.cos(np.radians(obs["field"]["dec"]))
        tweak = (obs["night"]["reference_oracle_seeing"], shift)  # seeing factor, tracking angle offset (deg)
        print("reference: TEST ONLY (reference_oracle): the science night's own exposures, noise seed "
              f"{reference['output']['noise_seed']}, transients left out; seeing x {tweak[0]}, tracking angle "
              f"{tweak[1] * 3600:+.3f}\" ({obs['night']['reference_oracle_shift_px']} px)", flush=True)
    print(f"reference: {n} night(s) from {reference['night']['date']} through hdps's coadd"
          + (f", each telescope's ratchets within {near} px of tonight's sensor position" if near else "")
          + (f", each telescope's {best} best-seeing ratchet(s)" if best else "") + f" -> {ref_out.relative_to(REPO)}/",
          flush=True)
    before = min(science["time"]) if science is not None and len(science) else None
    observe(reference, ref_out, engine_options=["--hdps-pass", "reference"], best_ratchets=best,
            match=(science, near) if near else None, before=before,
            tweak=tweak if obs["night"]["reference_oracle"] else None)
    return ref_out / "hdps"


def only_nights(coverage, n, obs):
    """[night] only_nights: the exposures of those nights (night k: local mean noon of the first date + k - 1 to the
    next)."""
    from argus_sim import c as cfg
    from astropy.time import Time

    noon = Time(f"{n['date']}T12:00:00", scale="utc").jd - cfg.observatory.longitude / 360.0  # local mean noon, UTC
    night_of = np.floor(Time(list(coverage["time"]), scale="utc").jd - noon).astype(int) + 1
    keep = np.isin(night_of, n["only_nights"])
    if not keep.any():
        raise SystemExit(f"[night] only_nights {n['only_nights']}: no exposures on those nights")
    print(f"[night] only_nights {n['only_nights']}: {int(keep.sum())} of {len(coverage)} exposures (nights "
          f"{sorted(set(night_of.tolist()))} in all)", flush=True)
    return coverage[keep]


def ratchet_views(coverage, ra, dec):
    """Each (telescope, band, ratchet) of the exposures at its first exposure: the field centre's sensor x, y (px) and
    its seeing there (the ratchet's zenith seeing x airmass^0.6, ArgusSim's seeing_at_airmass)."""
    from argus_sim import c as cfg
    from argus_sim.observatory import seeing_at_airmass
    from astropy.time import Time

    from mesa_argus import night, survey
    from mesa_argus.sensor import Projector

    times = Time(list(coverage["time"]), scale="utc")
    airmass, _, _ = night.sky_position(times, ra, dec, night.location(cfg))
    seeing = seeing_at_airmass(np.asarray(coverage["seeing_zenith"], float), airmass)
    projector = Projector(np.array([ra]), np.array([dec]), times, survey.array_layout(coverage.meta["layout"]))
    views = {}
    for j in range(len(coverage)):
        key = (int(coverage["tel_id"][j]), str(coverage["band"][j]), int(coverage["ratchet"][j]))
        if key not in views:
            x, y, _ = projector.pixels(j, coverage["theta_deg"][j], coverage["ota"][j])
            views[key] = (float(np.ravel(x)[0]), float(np.ravel(y)[0]), float(seeing[j]))
    return views, seeing


def best_seeing_ratchets(coverage, ra, dec, best, match=None):
    """The exposures of each telescope's `best` ratchets with the best seeing at the field (per band; 0: all), from
    ratchet_views. match: (the science night's exposures, px): only the telescopes observing the field tonight, each
    from its ratchets within px of where tonight's ratchet put the field centre on its sensor, preferring those at
    least as sharp as tonight's (none within px: the nearest, with a warning)."""
    views, seeing = ratchet_views(coverage, ra, dec)
    tonight = ratchet_views(match[0], ra, dec)[0] if match else {}
    chosen = set()
    far = []
    for tel, band in sorted({k[:2] for k in views}):
        mine = {k: v for k, v in views.items() if k[:2] == (tel, band)}
        if match:
            here = [v for k, v in tonight.items() if k[:2] == (tel, band)]
            if not here:
                continue  # not observing the field tonight in this band: no reference needed
            x, y, sharp = here[0]  # tonight's first ratchet of this telescope
            dist = {k: np.hypot(v[0] - x, v[1] - y) for k, v in mine.items()}
            near = [k for k in mine if dist[k] <= match[1]]
            if not near:
                near = [min(mine, key=dist.get)]
                far.append(f"{tel} {band} {dist[near[0]]:.0f} px")
            sharper = [k for k in near if mine[k][2] <= sharp]
            mine = {k: mine[k] for k in (sharper or near)}
        chosen.update(sorted(mine, key=lambda k: mine[k][2])[:best] if best else mine)
    keep = np.array([(int(t), str(b), int(r)) in chosen for t, b, r in
                     zip(coverage["tel_id"], coverage["band"], coverage["ratchet"])])
    print(f"reference: " + (f"each telescope's ratchets within {match[1]} px of tonight's view, " if match else "")
          + (f"its {best} best-seeing ratchet(s) per band" if best else "all of them") + f": {keep.sum()} of "
          f"{len(coverage)} exposures, seeing {seeing[keep].min():.2f}-{seeing[keep].max():.2f}\" (all: "
          f"{seeing.min():.2f}-{seeing.max():.2f}\")", flush=True)
    if far:
        print(f"reference: WARNING no ratchet within {match[1]} px of tonight's view for {', '.join(far)}: the nearest",
              flush=True)
    return coverage[keep]


def observe(obs, out, scene_only=False, engine_options=None, best_ratchets=0, match=None, before=None, tweak=None, decoupled=False):
    """Write the observation's scene into `out`, then run the engine on it (engine_options: extra engine arguments;
    None for a full observation, which makes the reference first if alerts are asked for; best_ratchets: observe only
    each telescope's best-seeing ratchets, best_seeing_ratchets, 0 for all; match: (science exposures, px) for
    best_seeing_ratchets; before: a reference's exposures end before this UTC time, the science night's first: a
    reference window that choose_night moved later, to a night the field is up, must not reach the science night;
    tweak: TEST ONLY (seeing factor, tracking-angle offset in deg) applied to the exposures, reference_oracle_*)."""
    from argus_sim import skymap_shim
    from astropy.time import Time

    import night_movie
    from mesa_argus import night

    f, n, s = obs["field"], obs["night"], obs["stars"]
    name, ra, dec = f["name"], f["ra"], f["dec"]
    window = (None, None)  # the whole dark night (ArgusSim's schedule)
    scene_dir = out / "scene"
    shutil.rmtree(scene_dir, ignore_errors=True)
    scene_dir.mkdir(parents=True)

    # the stars
    star_set = built_star(s["background"])[1]
    from mesa_argus.gaia import RADIUS_DEG

    if star_set.get("generator") == "Gaia field" and star_set.get("radius_deg", 0.12) < RADIUS_DEG:  # see gaia.RADIUS_DEG
        raise SystemExit(f"{s['background']} holds Gaia stars within {star_set.get('radius_deg', 0.12)} deg only; the "
                         f"readout window needs {RADIUS_DEG} (its outer part would be empty and hdps's WCS solve fails): "
                         f"rebuild it (python make_stars.py <its star file> --rebuild {s['background']})")
    placements = []
    for p in obs["place"]:
        if s["missing"] == "skip" and not (STARS / p["star"] / "meta.json").exists():
            print(f"{p['star']}: not built; skipped ([stars] missing = \"skip\")")
            continue
        placements.append((p, *built_star(p["star"])))

    # the night
    site = night.SITES["argus"]
    max_airmass = None
    n["date"] = choose_night(f, n, window, [m["real"] for _, _, m in placements if m.get("real")], site, max_airmass)
    from astropy.table import unique
    from mesa_argus import survey  # the exposures ArgusSim's array takes of the pointing (mesa_argus/survey.py)

    coverage = survey.exposures(ra, dec, n["date"], n["nights"], n["seed"],
                                layout=obs["argussim"]["packing_strategy"]["layout_file"])
    if len(coverage) == 0:
        raise SystemExit(f"the array does not cover RA {ra}, Dec {dec} on the night of {n['date']}")
    if tweak is not None:  # TEST ONLY: [night] reference_oracle_seeing, reference_oracle_shift_px
        coverage["seeing_zenith"] = np.asarray(coverage["seeing_zenith"], float) * tweak[0]
        coverage["theta_deg"] = np.asarray(coverage["theta_deg"], float) + tweak[1]
    if before is not None:
        late = Time(list(coverage["time"]), scale="utc") >= Time(before, scale="utc")
        if late.any():
            print(f"reference: {int(late.sum())} exposures at or after the science night's first ({before}) left out",
                  flush=True)
            coverage = coverage[~late]
    if best_ratchets or match:
        coverage = best_seeing_ratchets(coverage, ra, dec, best_ratchets, match)
    exposures = Time(list(unique(coverage, keys="time")["time"]), scale="utc").jd
    start = Time(exposures[0], format="jd", scale="utc")
    night_start_bjd = float(night.bjd_tdb(start, ra, dec, site))
    night_end_bjd = float(night.bjd_tdb(Time(exposures[-1], format="jd", scale="utc"), ra, dec, site)) + 10 / 1440
    half_block = f["block"] / 2 * night_movie.MINIPIX_PX * night_movie.SKY_PX  # as set_block below makes HALF
    injected = inject(obs, night_start_bjd, night_end_bjd, half_block)
    placements += injected
    if injected:
        print(f"[inject]: {len(injected)} library stars over the readout window", flush=True)
    if n["only_nights"] and engine_options is None:  # [night] only_nights: the scene above is all the nights'
        coverage = only_nights(coverage, n, obs)

    # the field: HDPS tile and block, dust, background, placed stars
    tile_id, minipix = skymap_shim.radec_to_composite_id(np.array([ra]), np.array([dec]))
    tile, block = int(tile_id[0]), (int(minipix[0]) % 64, int(minipix[0]) // 64)
    night_movie.set_block(f["block"])
    ebv = float(dust.ebv(ra, dec)[0][0])  # the whole column at the pointing (for the summary)
    rng = np.random.default_rng(s["seed"])  # background positions first, then any random placements
    background_note = write_background(scene_dir, s, star_set, ra, dec, rng)
    night_info = (start, night_start_bjd, night_end_bjd, ra, dec, site)
    rows, extra = place_stars(placements, scene_dir, night_info, night_movie.HALF, rng)
    (scene_dir / "sources.json").write_text(json.dumps(rows, indent=1, default=float))
    (scene_dir / "extra_pulsators.json").write_text(json.dumps(extra, indent=1))
    field_json = dict(name="field" if s["layout"] == "real" else f"synthetic, seed {s['seed']}", centre=[ra, dec],
                      tile=tile, block=list(block), block_n=f["block"], date=n["date"], site=site,
                      synthetic=s["layout"] == "random", ebv=ebv)
    (scene_dir / "field.json").write_text(json.dumps(field_json, indent=1))
    coverage.write(scene_dir / "exposures.ecsv", overwrite=True)  # the engine observes exactly these
    dust.maps.cache_clear()  # the 3D map (~6 GB) is not needed while the engine runs

    # what was done, for reading and for making it again
    resolved = out / "observation.toml"
    resolved.write_text(runconfig.dumps(obs, f"observe.py: {name}, resolved (python observe.py "
                                             f"{resolved.relative_to(REPO)} makes it again)"))
    observing = (f"observing: ArgusSim's survey ({len(coverage)} exposures at {len(exposures)} times (overlapping telescopes "
                 f"take some together) through {len(set(coverage['tel_id']))} telescopes: "
                 f"{int(np.sum(coverage['band'] == 'beta'))} beta, {int(np.sum(coverage['band'] == 'rho'))} rho; "
                 f"{n['nights']} night(s), seed {n['seed']}; scene/exposures.ecsv); each a raw frame of a "
                 f"{f['window']}-minipix readout window through hdps")
    summary = [f"# {name}", "", observing, "", f"RA {ra:.5f}, Dec {dec:+.5f}; first night {n['date']}; Milky Way E(B-V) "
               f"{ebv:.3f} (the whole column at the pointing; each "
               "star sees the 3D map's dust in front of it, mesa_argus/dust.py)", "",
               background_note, "", "## Stars placed", ""]
    for r in rows:
        if r.get("real"):
            where = f"at its real position, own ephemeris and E(B-V) {r['ebv']:.3f} in front of it"
        else:
            when = f"t0 BJD {r['t0_bjd']:.5f}" if r["timing"] == "transient" else f"epoch BJD {r['epoch_bjd']:.5f}"
            far = f"{r['distance_pc']:.0f} pc" if r.get("distance_pc") else "behind all the dust"
            where = f"at {r['x_percent']:.1f}%, {r['y_percent']:.1f}% of the image; {when}; E(B-V) {r['ebv']:.3f} ({far})"
        summary.append(f"- **{r['label'] or 'V'}** `{r['star']}`: {r['kind']} ({r['generator']}; {r['engine']}), {where}")
    summary += ["", "What each star is: stars/<name>/summary.md. Instrument, PSF, sky, noise, photometry: docs/ASSUMPTIONS.md."]
    (out / "summary.md").write_text("\n".join(summary) + "\n")
    if scene_only:
        print(f"scene: {scene_dir.relative_to(REPO)}/")
        return
    reference = None
    if engine_options is None:
        engine_options = []
        if obs["output"]["alerts"]:
            reference = make_reference(obs, out, placements, coverage)
        if n["only_nights"]:  # the frames' noise apart from other nights' runs (the reference keeps noise_seed)
            obs = dict(obs, output=dict(obs["output"], noise_seed=obs["output"]["noise_seed"] + min(n["only_nights"])))
            print(f"[night] only_nights: frame noise seed {obs['output']['noise_seed']}", flush=True)
    shutil.rmtree(out / "hdps", ignore_errors=True)
    run_engine(obs, out, len(rows), engine_options, reference, decoupled=decoupled)
    print(f"done: {out.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
