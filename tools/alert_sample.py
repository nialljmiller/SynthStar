"""An alert sample for an alert broker: hdps's packets exactly as hdps wrote them, and beside them what the
simulation put at each one.

For each observation folder (outputs/pseudo_real/<name>) and band:
    <out>/<name>/<band>/packets/<epochId>.avro   hdps's packets for that exposure (an Avro object container file, hdps's
                                                 alert schema embedded: hdps_alerts/schemas): copied byte for byte, or,
                                                 with --cut, only those whose realBogusScore is at or above the cut,
                                                 each record unchanged
    <out>/<name>/<band>/truth.parquet            one row per packet, keyed by alertId (and the packet's epochId, ra,
                                                 dec): what is there (truth_class: transient / variable / constant /
                                                 none), which simulated source (truth_label, its kind and model), its
                                                 true magnitude and its true flux change since the reference, and the
                                                 real/bogus label as tools/rb_training_set.py defines it
    <out>/manifest.json                          per night and band: date, Moon illumination, the hdps settings that
                                                 decide which alerts exist (bright-star mask, real/bogus model and
                                                 cut), the schema version, packet counts by truth class
    <out>/viewer.html                            the alert viewer (tools/alert_viewer.html) as a page of its own: opened
                                                 in a browser it reads this folder (cutouts, light curves, sky, truth)

The packets are the stream Argus would send (argus_image.alert_files: hdps's AlertVettingTask builds every packet,
the file stands in for Kafka); nothing in them is changed or added. The truth comes from the engine's
<name>_night_<band>.alerts.csv (every hdps candidate with its features and its truth, alertId from the packets).

A night run in two halves, one band each (observation files <night>_beta and <night>_rho with [output] bands = one
band: one engine per run fits a smaller machine), is the same night: no seed derives from the name or the bands
(the stars, [inject] and [night] seeds are explicit; the engine's noise is seeded per exposure from [output]
noise_seed), so both halves hold the same scene, as long as both run on one machine (the [inject] templates are drawn
from that machine's stars/ library). Such a half is filed under its night, <out>/<night>/<band>/, and the halves of a
night are checked to hold the same scene (scene/sources.json and scene/exposures.ecsv): a night whose halves differ
stops the run.

The cut: hdps sends a packet only for a candidate whose score is at or above AlertVettingTask's score_cut
(emit_all_candidates adds below-cut rows to the alert catalogue, never packets), and no packet field depends on the
cut. So a night run with score_cut = 0 (a packet for every candidate, the real/bogus model loaded) and selected here
with --cut C holds exactly the packets a run at score_cut = C sends (checked on hdps's v4.0 AlertVettingTask: every
field equal but alertId and the build timestamp, which differ between any two runs), and the cut can change without
running the night again. --cut takes a number or an operating point of --bundle (e.g. fa10).

    python tools/alert_sample.py outputs/pseudo_real/rb_f299_911m08_104_20261023 -o data/alert_sample
    python tools/alert_sample.py <folders> --cut fa10 --bundle data/rb/<bundle> -o data/alert_sample
"""
import argparse
import json
import shutil
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# What each truth row carries from the simulated source (scene/sources.json, by label), when it is a placed star.
SOURCE_KEYS = ("kind", "generator", "engine", "model", "params", "z", "t0_bjd", "peak_days", "period", "ebv",
               "distance_pc", "star")


def moon_illumination(time_utc):
    """Fraction of the Moon's disc lit at this time (astropy: Sun-Moon elongation)."""
    from astropy.coordinates import get_body, get_sun
    from astropy.time import Time

    t = Time(time_utc)
    return float((1.0 - np.cos(get_sun(t).separation(get_body("moon", t)).rad)) / 2.0)


def window_deg2(folder, band, obs):
    """The readout window's area (deg2): its side in sensor pixels from the engine's log, times the plate scale."""
    import re

    from argus_image.runconfig import plate_scale

    log = folder / f"{folder.name}_night_{band}.log"
    m = re.search(r"\((\d+) px window\)", log.read_text(errors="replace")) if log.exists() else None
    return (int(m[1]) * plate_scale(obs) / 3600.0) ** 2 if m else None


def night_settings(folder, band):
    """The resolved observation's settings that decide which alerts exist."""
    from datetime import date, timedelta

    obs = tomllib.loads((folder / "observation.toml").read_text())
    tasks = {t["type"]: t.get("config", {}) for t in obs["hdps"]["tasks"]}
    sfft, vet = tasks.get("SFFTSubtractionTask", {}), tasks.get("AlertVettingTask", {})
    area = window_deg2(folder, band, obs)
    inject = len(sources_by_label(folder))  # the library stars placed over the window (scene/sources.json)
    first = min(obs["night"].get("only_nights") or [1])  # [night] only_nights: night k of [night] date's observation
    image = obs.get("image", {})
    return dict(field=obs["stars"].get("background", ""),
                date=(date.fromisoformat(obs["night"]["date"]) + timedelta(days=first - 1)).isoformat(),
                true_calibration="TrueCalibrationTask" in tasks, psf_box=image.get("psf_box", 28),
                psf_halo=bool(image.get("psf_halo")),
                tpv_terms=None if "TrueCalibrationTask" in tasks  # (no TPV fit runs)
                else tasks.get("TPVWCSSolverTask", {}).get("tpv_terms", 17),
                mask_bright_stars=sfft.get("mask_bright_stars"),
                bright_star_threshold=sfft.get("bright_star_threshold"), model_bundle=vet.get("model_bundle") or "",
                score_cut=vet.get("score_cut"), emit_all_candidates=vet.get("emit_all_candidates"),
                stride=obs["output"].get("stride"), injected=inject, window_deg2=round(area, 4) if area else None,
                injected_per_deg2=round(inject / area, 1) if area else None)


def sources_by_label(folder):
    path = folder / "scene" / "sources.json"
    if not path.exists():
        return {}
    return {s["label"]: s for s in json.loads(path.read_text()) if s.get("label")}


def night_of(folder, band):
    """The night a run belongs to: a one-band half (<night>_<band>, [output] bands = [band]) files under <night>."""
    obs = tomllib.loads((folder / "observation.toml").read_text())
    bands = obs.get("output", {}).get("bands", ["beta", "rho"])
    suffix = f"_{band}"
    return folder.name[:-len(suffix)] if bands == [band] and folder.name.endswith(suffix) else folder.name


def scene_digest(folder):
    """A digest of the run's scene (scene/sources.json and scene/exposures.ecsv): equal for the halves of one night."""
    import hashlib

    h = hashlib.sha256()
    for part in ("sources.json", "exposures.ecsv"):
        path = folder / "scene" / part
        h.update(path.read_bytes() if path.exists() else b"")
    return h.hexdigest()[:16]


def exposures_run(folder, band):
    """The exposures the engine ran through hdps for this band: its log's "hdps: N alerts from M frames" (M), else the
    scene's exposures in the band (scene/exposures.ecsv) at the run's stride."""
    import re

    log = folder / f"{folder.name}_night_{band}.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    m = re.search(r"^hdps: \d+ alerts from (\d+) frames", text, re.M)
    if m:
        return int(m[1])
    path = folder / "scene" / "exposures.ecsv"
    if not path.exists():
        return None
    from astropy.table import Table

    stride = tomllib.loads((folder / "observation.toml").read_text())["output"].get("stride", 1)
    return int(len(range(0, int((Table.read(path)["band"] == band).sum()), stride)))


def resolve_cut(cut, bundle):
    """A cut as a number: a literal, or an operating point from the bundle's manifest.json (hdps's cut_for)."""
    if cut is None:
        return None
    try:
        return float(cut)
    except ValueError:
        if bundle is None:
            raise SystemExit(f"--cut {cut}: an operating point needs --bundle")
        points = json.loads((Path(bundle) / "manifest.json").read_text()).get("operating_points", {})
        if cut not in points:
            raise SystemExit(f"--cut {cut}: {bundle} offers {sorted(points)}")
        return float(points[cut])


def one_band(folder, band, out, min_sigma, cut=None, validate=False, write=True):
    """Copy a night's packets for one band (with a cut: those at or above it) and write their truth; returns the
    manifest entry (None: no packets)."""
    from fastavro import writer

    from argus_image import alert_files
    from rb_training_set import label

    run = folder.name
    name = night_of(folder, band)  # the night (a one-band half files under it)
    packet_dir = folder / "hdps" / band / "alerts"
    files = sorted(packet_dir.glob("*.avro"))
    table_path = folder / f"{run}_night_{band}.alerts.csv"
    if not files or not table_path.exists():
        return None
    table = pd.read_csv(table_path, low_memory=False, dtype={"alertId": str})
    table["alertId"] = table["alertId"].fillna("")
    table["label"] = label(table, min_sigma)
    by_id = table[table["alertId"] != ""].drop_duplicates("alertId").set_index("alertId")
    # a packet without its alertId in the table (schema v4: hdps_worker writes <epochId>.avro and the engine's
    # alert_table looks for <frame>.avro, so the column is empty) is the row at its candidate's ra, dec: the packet's
    # are the row's, written from the same values
    at = {(r, d): k for k, (r, d) in enumerate(zip(table["ra_deg"], table["dec_deg"]))}
    sources = sources_by_label(folder)
    dest = out / name / band / "packets"
    if write:  # (write=False: the manifest entry only; packets and truth.parquet as they are)
        dest.mkdir(parents=True, exist_ok=True)
    rows, versions, n_read = [], set(), 0
    cutouts = {"science": 0, "difference": 0, "template": 0}
    for path in files:
        packets = alert_files.read(path)
        n_read += len(packets)
        if cut is None:
            if write:
                shutil.copy2(path, dest / path.name)
        else:
            packets = [k for k in packets if (k["candidate"]["realBogusScore"] or 0.0) >= cut]
            if not packets:
                continue
            with open(path, "rb") as f:  # the file's own embedded schema, the records as they were
                from fastavro import reader

                schema = reader(f).writer_schema
            if write:
                with open(dest / path.name, "wb") as f:
                    writer(f, schema, packets, codec="deflate")
        for packet in packets:
            versions.add(packet["schemaVersion"])
            if validate:  # hdps's own packet model (the cutouts' 4096-byte rule included); raises on a bad packet
                from hdps.alerts.models import Alert

                Alert(**packet)
            for kind in cutouts:  # v4.0: 4096 bytes when present (v3.0 packets have no such fields)
                value = packet.get(f"cutout{kind.capitalize()}")
                cutouts[kind] += value is not None and len(value) == 4096
            c = packet["candidate"]
            alert_id = str(packet["alertId"])
            row = dict(alertId=alert_id, epochId=c["epochId"], ra=c["ra"], dec=c["dec"],
                       packet_file=f"packets/{path.name}")
            t = (by_id.loc[alert_id] if alert_id in by_id.index
                 else table.iloc[at[(c["ra"], c["dec"])]] if (c["ra"], c["dec"]) in at else None)
            if t is not None:
                row.update(exposure=int(t["exposure"]), time_utc=t["time_utc"], truth_class=t["truth_class"],
                           truth_label=t["truth_label"] if isinstance(t["truth_label"], str) else "",
                           truth_sep_arcsec=t["truth_sep_arcsec"], truth_mag=t["truth_mag"],
                           truth_flux_now=t["truth_flux_now"], truth_flux_reference=t["truth_flux_reference"],
                           truth_diff_flux=t["truth_diff_flux"], label=int(t["label"]))
                source = sources.get(row["truth_label"], {})
                row["injected"] = bool(source)  # one of the library stars injected over the window
                row.update({f"source_{k}": source.get(k) for k in SOURCE_KEYS})
            rows.append(row)
    truth = pd.DataFrame(rows)
    missing = int(truth["truth_class"].isna().sum()) if "truth_class" in truth else len(truth)
    if write:
        # Variable-star truth goes in its own file: its truth_diff_flux and real/bogus label depend on the reference
        # coadd of the camera that took the alert (truth_flux_reference), which is right only from the per-telescope
        # reference fix (night_movie.py, 2026-10-30); samples run before that have unreliable variable labels.
        is_var = truth["truth_class"] == "variable" if "truth_class" in truth else pd.Series(False, index=truth.index)
        truth[~is_var].to_parquet(out / name / band / "truth.parquet", index=False)
        truth[is_var].to_parquet(out / name / band / "truth_variables.parquet", index=False)
    settings = night_settings(folder, band)
    mid = table["time_utc"].iloc[len(table) // 2] if len(table) else None
    models = sorted({str(m) for m in table.get("model_version", pd.Series(dtype=str)).dropna().unique()})
    with_candidates = int(table["exposure"].nunique()) if len(table) else 0  # (every candidate is a row)
    with_packets = int(truth["exposure"].nunique()) if "exposure" in truth else 0
    return dict(observation=name, run=run, scene=scene_digest(folder), band=band, **settings, sample_cut=cut,
                rb_model_versions=models, exposures_run=exposures_run(folder, band),
                exposures_with_candidates=with_candidates, exposures_with_packets=with_packets,
                moon_illumination=round(moon_illumination(mid), 3) if mid else None,
                schema_versions=sorted(versions), packets=len(truth), packets_before_cut=n_read,
                cutouts_4096_bytes=cutouts,
                packet_files=len(files),
                packets_without_truth=missing,
                by_truth_class={k: int(v) for k, v in truth["truth_class"].value_counts().items()}
                if "truth_class" in truth else {},
                by_label={str(k): int(v) for k, v in truth["label"].value_counts().items()} if "label" in truth else {})


CUTOUTS = """\
## Cutouts

Every packet carries cutoutScience, cutoutDifference and cutoutTemplate, as ADR-0032 sec. 4 (ADR-0030 until
2026-10-28) and TN-0004 ("Cutouts") specify: 32 x 32 little-endian IEEE 754 float32 in electrons, 4096 bytes each,
row-major in the TAN segment's [y, x]
order (byte 4 * (32 * row + col) holds segment pixel (x0 + col, y0 + row), (x0, y0) the stamp's lowest-index corner;
no rotation, CDELT1 negative: +y north, -x east at the segment's tangent point). Read one with
`numpy.frombuffer(packet["cutoutDifference"], "<f4").reshape(32, 32)`.

- Centered on every alert: the exposures are 60 s, and the BOOM/Argus meeting chose centered cutouts at 15-minute
  and longer cadence and, Plan A, at 1-minute cadence. TN-0004 leaves the centered placement TBD; here the
  candidate's rounded pixel (round(y), round(x)) is at [row 16, col 16] (x0 = round(x) - 16, y0 = round(y) - 16), the
  convention of hdps's own alert stamps extended to an even size. Pixels beyond the segment's edge are NaN.
- Science: the science tile before subtraction (sky included); difference: SFFT's difference plane; template: the
  reference coadd on the tile that SFFT subtracts against (hdps's retained presub_science and presub_reference: the
  planes of hdps's own alert stamps). The template ships for v4 until the broker hosts Argus reference images
  (optional per the meeting).
- Pixel values are as hdps's tiles hold them: electrons per sensor-pixel area (hdps's reprojection keeps surface
  brightness, 0.879" tile pixels against 1.007" sensor pixels), so a star summed over a cutout is about 1.31 times its
  electrons. hdps PR #488 (open) makes tile pixels flux-conserving (each pixel x ~0.755), which would change this.
- The filling is our implementation of ADR-0032 sec. 4 in hdps's AlertVettingTask (a local commit on top of hdps PR
  #474, which defers cutout extraction): hdps's own code, the planes hdps already cuts its alert stamps from, until
  hdps's producer PR lands.
"""

def limitations(nights):
    """The README's simulation limitations, for the nights' PSF and calibration settings (checked 2026-10-30 against the
    Argus documents, hdps and the published specifications)."""
    if all(n.get("psf_box", 28) > 28 and n.get("psf_halo") for n in nights):
        boxes = "/".join(str(b) for b in sorted({n["psf_box"] for n in nights}))
        psf = (f"- Bright stars' PSFs: ArgusSim's Zemax PSF (optics, delivered blur, Moffat seeing) drawn in a {boxes} px "
               "box, with the delivered PSF's static halo as measured on sky (H. Corbett: an hdps tile stack, "
               "unsaturated Gaia stars of G 8.5-12.5, fitted over 8-60 tile px) added after the seeing, so the wings "
               "continue past the Zemax grid; the halo stops at the box's edge, 60 sensor px out, where it is ~1e-9 "
               "of the star per pixel.")
    else:
        psf = ("- Bright stars' PSFs stop at ArgusSim's Zemax stamp, 28 x 28 sensor pixels: around the brightest stars "
               "the wings end in a visible square, at about 50 e- per pixel, which can show in the cutouts of alerts "
               "near bright stars.")
    if all(n.get("true_calibration") for n in nights):
        wcs = ("- ArgusSim applies no lens distortion: the true WCS of every exposure is a plain TAN projection, and "
               "TrueCalibrationTask hands it to hdps in place of the solve (above). The PSF zero point is the "
               "total-light one; hdps's own solve calibrates against aperture fluxes.")
    else:
        wcs = ("- ArgusSim applies no lens distortion: the true WCS of every exposure is a plain TAN projection. hdps "
               "fits its TPV distortion (TPVWCSSolverTask, 17 terms by default) from an identity guess, because the "
               "simulated cameras have no calibration file; that fit is what fails on the exposures without tiles "
               "(above).")
    return "\n".join(["## Simulation limitations", "", psf, wcs,
        "- PSF photometry reads about 5.5% low: hdps's PSF flux of an unsaturated source is 0.952 (beta) and 0.938 "
        "(rho) of its true flux, in the science image as in the difference, at every S/N and seeing; hdps's Moffat PSF "
        "does not hold the halo and wings above. The flux errors are statistical only, so at S/N above ~50 this offset "
        "is many sigma.",
        "- Saturation: ArgusSim's full well is 21,657 e- (65,535 ADU at an older gain of 0.33 e-/ADU) and cores of "
        "saturated stars are flat there. The production IMX461 cameras read 0.264-0.269 e-/ADU with a ~502 ADU bias, "
        "so the 16-bit rail is ~17.3 ke- and hdps's saturation level ~16 ke-: real stars saturate ~0.3 mag fainter than "
        "here. About 13% of the real transient alerts come from injected sources whose cores saturate (true flux above "
        "~1.5e5 e-, brighter than mag ~12.8); they carry flag bit 0, and their PSF fluxes are 15-85% of the truth.",
        "- Periodic stars' truth: truth_flux_reference is the star's cycle-mean flux, but each camera's reference holds "
        "the star as it was during that camera's reference exposures, so truth_diff_flux (and the label) of a periodic "
        "star can be off by up to ~20% of its flux (detached binaries out of eclipse are the worst: no change in the "
        "difference image, a large one in the truth). The engine takes the reference flux from each camera's own "
        "reference exposures from 2026-10-30.",
        "- References: each camera's single best-seeing ratchet over the 5 nights before; the team's design stacks the "
        "best few per tile across cameras (deeper and sharper), so the difference images here are noisier than "
        "planned.",
        "- Bands: beta is ArgusSim's b (362-527 nm); rho is 560-825 nm (the PI's bands, 2026-09-05), wider than the "
        "r of the public specifications (560-700 nm), with the two bands split 1:1 across telescopes.",
        "- Sky: ArgusSim's dark sky here is 1.2-1.5 e-/s per pixel in rho (V 21.35-21.59 mag/arcsec2); the Argus "
        "documents measure 21-25 e-/s on moonlit frames and report no dark-sky measurement.",
        "- Saturated cores are flat at the full well (ArgusSim's camera model); read noise 1.2 e- (the public "
        "specifications give 1.4 e-).",
        ""])


KNOWN_GAPS = """\
## Known gaps

The schema (checked 2026-11-02): these packets follow hdps PR #474 (schema v4.0) as its documents define it at head
35bd60dc (TN-0004, alert-packet.md; the ADR is now 0032). The envelope carries a `crossMatches` array (empty here: no
reference-catalog crossmatch runs in the simulation), and the candidate carries a copy of the envelope `alertId`; the
per-candidate `refcatName`, `refcatVersion` and `nearestSrc*` fields are gone (they moved into `crossMatches`). The PR
is open with changes requested and its own .avsc files still lag its documents, so the schema may change before it
merges. procVer 6c0309d.

In these packets:
- schemaVersion is "4.0" (the BOOM/Argus discussion asks for major.minor.fix);
- times are MJD(TT) and Argus epoch (TT seconds since 2025-01-01 00:00 TT, ADR-0012), as the schema defines; the
  discussion asks for JD on BOOM's time scale, not yet in the schema;
- timestampMjdTt / timestampArgusEpoch are when hdps built the packet, i.e. when the simulation was processed (weeks
  after the simulated night), not the observation time plus a realistic latency;
- epochId is the frame key of ADR-0024: the telescope code (here ArgusSim's telescope number + 1) above the exposure
  midpoint in TT on a 1/256 s grid; cameraIds/otaIds/filterIds/focuserIds/subarrayIds are simulated asset ids
  (100000/200000/300000/400000 + telescope code, 500000 + subarray), one per packet;
- filter is 1 (beta) or 2 (rho); alertCadence 60; integrationTime 60 s; coadd false; coaddSpacing null;
- flags: bit 0 (1) saturated, from hdps's detector-quality mask; bit 3 (8) near the segment's edge; no other bit is set;
- eventAssociation.eventId is the HEALPix NESTED pixel at order 16 (provisional upstream); ssoAssociation is always
  null (no solar-system association runs) and crossMatches is always empty (no reference-catalog crossmatch);
- the candidate's alertId equals the envelope alertId (schema v4.0);
- not produced (null or 0): starGalaxy*, artifact*, placeholderPhotometricCalibration (schema placeholders),
  numReferenceSources, withdrawal times, raDecCovariance (0);
- candidates without a sky position or a finite difference flux are not sent;
- on Kafka each message would be one Avro record (topic and framing are for a distribution ADR still to come); here
  each exposure's packets are one Avro object container file (deflate codec) with the schema embedded.
"""


def gaps_sentence(nights):
    """Why exposures run can lack candidates (README): hdps's TPV fit, or none of it with TrueCalibrationTask."""
    if all(n.get("true_calibration") for n in nights):
        return ("These nights ran with TrueCalibrationTask in place of hdps's astrometric and photometric solve "
                "(ReferenceRetrievalTask, LinearWCSSolverTask, SourceMatchingTask, TPVWCSSolverTask): each frame "
                "carries the simulation's own WCS and zero point, as the solve would set them. The solve, started "
                "from an identity guess because the simulated cameras have no camera calibration, fails on some "
                "telescopes (residuals of 100-200\" against about 0.2\" on the others); segment reprojection then "
                "finds a non-finite WCS grid on every tile and drops them all. With the true calibration no exposure "
                "loses its tiles that way.")
    return ("In the others hdps dropped every tile of the exposure: its TPV distortion fit (TPVWCSSolverTask), "
            "started from an identity guess because the simulated cameras have no camera calibration, fails on some "
            "telescopes (residuals of 100-200\" against about 0.2\" on the others), and segment reprojection then "
            "finds a non-finite WCS grid on every tile and skips them all, so light curves have gaps there. "
            + tpv_sentence(nights))


def tpv_sentence(nights):
    """What the nights' TPV setting means for the gaps (README)."""
    terms = sorted({n.get("tpv_terms", 17) for n in nights} - {None})
    if terms == [17]:
        return ("These nights ran with hdps's default of 17 TPV terms. With tpv_terms = 7 (a TPVWCSSolverTask setting, "
                "no code change) the same exposures are live (tested on f359 2026-09-10 beta: 31 of 31 exposures with "
                "candidates against 14 of 31, the live ones unharmed).")
    return (f"These nights ran with TPVWCSSolverTask tpv_terms = {' / '.join(map(str, terms))} (an hdps setting, "
            "no code change). With hdps's default of 17 terms the same nights (the v1 sample) had 803 of 1,804 "
            "exposures without a tile, 22% to 100% live per night and band; the table gives this set's.")


def write_readme(out, manifest):
    """README.md for the sample, from its manifest."""
    nights = manifest["nights"]
    cuts = sorted({f"{n['score_cut']} (= {n['score_cut_value']:.4g})" if n.get("score_cut_value") is not None
                   else str(n["score_cut"]) for n in nights})
    models = sorted({m for n in nights for m in n["rb_model_versions"]})
    bundles = sorted({n["model_bundle"] for n in nights})
    versions = sorted({v for n in nights for v in n["schema_versions"]})
    total = sum(n["packets"] for n in nights)
    real = sum(n["by_label"].get("1", 0) for n in nights)
    lines = [f"# Simulated Argus alert sample{(' ' + manifest['name']) if manifest.get('name') else ''}", "",
             "Alerts from simulated Argus Array observations of real Gaia fields (MESA_ARGUS: ArgusSim images, hdps, "
             "the Argus pipeline, run as production runs it), for training and testing an alert broker.", "",
             f"- packets: {total} in {sum(n['packet_files'] for n in nights)} files, hdps alert schema "
             f"{', '.join(versions)} (argus.alert; the .avsc is embedded in every file)",
             f"- real/bogus model: {', '.join(models)} ({', '.join(bundles)}), loaded in hdps's AlertVettingTask",
             f"- cut: score_cut {', '.join(cuts)}: only candidates at or above it are sent, as production sends "
             "them (\"fa10\": the score at which 10 false alerts per square degree per exposure pass, measured on "
             "simulated nights the model did not train on)",
             f"- of the packets, {real} are real by the truth rule below, {total - real} bogus or below it", "",
             "## Nights", "",
             "| observation | field | night | Moon lit | band | exposures run / with candidates / with packets | "
             "window deg2 | injected (per deg2) | packets | real | by truth class |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in nights:
        lines.append(f"| {n['observation']} | {n['field']} | {n['date']} | {n['moon_illumination']} | {n['band']} | "
                     f"{n.get('exposures_run')} / {n.get('exposures_with_candidates')} / "
                     f"{n.get('exposures_with_packets')} | {n['window_deg2']} | "
                     f"{n['injected']} ({n['injected_per_deg2']}) | "
                     f"{n['packets']} | {n['by_label'].get('1', 0)} | "
                     + ", ".join(f"{k} {v}" for k, v in n["by_truth_class"].items()) + " |")
    run = sum(n.get("exposures_run") or 0 for n in nights)
    live = sum(n.get("exposures_with_candidates") or 0 for n in nights)
    lines += ["", "## Exposures and gaps", "",
              "Exposures run: those ArgusSim took of the window and the engine passed to hdps (one band, every "
              "exposure of the night at the run's stride). With candidates: those for which hdps produced at least one "
              "difference-image candidate. With packets: those that sent at least one packet. "
              f"Over the sample, {live} of {run} exposures run produced candidates "
              f"({100 * live / run:.0f}%). " if run else "",
              gaps_sentence(nights),
              "Rates in this sample count exposures that produced candidates: the fa10 "
              "cut's 10 false alerts per square degree per exposure was measured over the calibration nights' "
              "exposures with candidates, and the packets per exposure of a night divide by its exposures with "
              "candidates, not by its exposures run.", "",
              "## Injection", "",
              "Each night's readout window holds the field's real Gaia stars (constant unless injected) plus the "
              "injected library stars in the table: supernovae, dwarf novae, microlensing events, RR Lyrae, eclipsing "
              "binaries, modelled light curves placed at random over the window. That density is far above what "
              "Argus would see in such a window; it gives the sample many real alerts. The bogus alerts (subtraction "
              "residuals of constant stars, noise, artefacts) come at the rate the cut allows.", "",
              "## Files", "",
              "- `<observation>/<band>/packets/<epochId>.avro`: one exposure's packets, exactly as hdps built them "
              "(read with any Avro reader, e.g. `fastavro.reader`).",
              "- `<observation>/<band>/truth.parquet`: one row per packet, joined on `alertId` (also `epochId`, "
              "`ra`, `dec`): `injected` (true: one of the injected library stars), `truth_class` (transient: an "
              "injected source; constant: a real star that did "
              "not change; none: nothing simulated there), `truth_label` and `source_*` (the injected source: kind, "
              "model, redshift, t0, ...), `truth_mag`, `truth_diff_flux` (its true flux change since the reference, "
              "electrons), and `label`: 1 real (an injected source whose true change is at least 3 times the "
              "candidate's PSF flux error), 0 bogus (a constant star or nothing), -1 an injected source changing less "
              "than that. The variable-star rows are NOT here; they are in truth_variables.parquet.",
              "- `<observation>/<band>/truth_variables.parquet`: the same columns, for the variable-star alerts only. "
              "Kept separate because their `truth_diff_flux` and `label` are unreliable in this sample (see the "
              "simulation limitations): the truth reference flux was the star's cycle mean, not its flux in each "
              "camera's reference coadd.",
              "- `manifest.json`: everything in the table above, per night and band (also cutout counts).",
              "- `viewer.html`: an alert viewer. Open it in a browser (Chrome, Edge, Firefox or Safari) and choose this "
              "folder, or drop the folder on it: it reads the packets and truth.parquet in the browser (nothing is "
              "uploaded; reading truth.parquet loads one library from cdn.jsdelivr.net) and shows each alert's cutouts, "
              "packet fields and truth, its source's light curve over the nights, and where it is on the sky.", "",
              CUTOUTS, limitations(nights), KNOWN_GAPS]
    (out / "README.md").write_text("\n".join(lines))


VIEWER_HEAD = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
               '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
               '<style>:root{color-scheme:light;padding:env(safe-area-inset-top,0px) 0 env(safe-area-inset-bottom,0px)}'
               'body{margin:0;font:14px system-ui,-apple-system,"Segoe UI",sans-serif}img{max-width:100%}'
               '[hidden]{display:none!important}</style></head><body>\n')


def write_viewer(out):
    """viewer.html: the alert viewer page (tools/alert_viewer.html, written as the body of a claude.ai artifact) in a
    document of its own, beside the sample it reads."""
    (out / "viewer.html").write_text(VIEWER_HEAD + (REPO / "tools" / "alert_viewer.html").read_text() + "\n</body></html>\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("observations", nargs="+", type=Path, help="observation folders (outputs/pseudo_real/<name>)")
    p.add_argument("-o", "--out", type=Path, default=REPO / "data" / "alert_sample")
    p.add_argument("--bands", nargs="+", default=["beta", "rho"])
    p.add_argument("--min-sigma", type=float, default=3.0, help="the real/bogus label's rule (tools/rb_training_set.py)")
    p.add_argument("--cut", help="keep packets with realBogusScore >= this: a number, or an operating point of --bundle")
    p.add_argument("--bundle", type=Path, help="the real/bogus bundle the nights were scored with (for a --cut label)")
    p.add_argument("--name", help="the sample's name, for the README")
    p.add_argument("--readme-only", action="store_true", help="write manifest.json and README.md again, from the same "
                                                               "inputs, leaving packets and truth.parquet as they are")
    p.add_argument("--validate", action="store_true", help="check every packet with hdps.alerts.models (the schema's "
                                                            "hdps on PYTHONPATH)")
    args = p.parse_args(argv)
    cut = resolve_cut(args.cut, args.bundle)
    sys.path.insert(0, str(REPO / "tools"))
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"nights": []}
    for folder in args.observations:
        for band in args.bands:
            obs = tomllib.loads((folder / "observation.toml").read_text())
            if band not in obs.get("output", {}).get("bands", ["beta", "rho"]):
                continue  # (the other half of a one-band run)
            entry = one_band(folder.resolve(), band, args.out, args.min_sigma, cut, args.validate,
                             write=not args.readme_only)
            if entry is None:
                print(f"{folder.name} {band}: no packets or no alerts.csv; skipped")
                continue
            night = entry["observation"]
            manifest["nights"] = [n for n in manifest["nights"] if (n["observation"], n["band"]) != (night, band)]
            manifest["nights"].append(entry)
            print(f"{night} {band}{' (from ' + folder.name + ')' if night != folder.name else ''}: "
                  f"{entry['packets']} packets in {entry['packet_files']} files, schema "
                  f"{','.join(entry['schema_versions'])}, {entry['packets_without_truth']} without truth; "
                  f"{entry['by_truth_class']}", flush=True)
    scenes = {}  # the halves of a night hold one scene
    for n in manifest["nights"]:
        scenes.setdefault(n["observation"], set()).add(n.get("scene"))
    differ = sorted(night for night, digests in scenes.items() if len(digests - {None}) > 1)
    if differ:
        raise SystemExit(f"the bands of {', '.join(differ)} come from different scenes (runs on different machines or "
                         "with different seeds): not one night")
    if args.name:
        manifest["name"] = args.name
    for n in manifest["nights"]:  # the cut each night ran at, as a number (a label resolved in its bundle)
        bundle = Path(n["model_bundle"]) if n["model_bundle"] else None
        if bundle is not None and not bundle.is_absolute():
            bundle = REPO / bundle
        try:
            n["score_cut_value"] = resolve_cut(str(n["score_cut"]), bundle) if n["model_bundle"] else None
        except SystemExit:
            n["score_cut_value"] = None
    manifest_path.write_text(json.dumps(manifest, indent=1, default=str))
    write_readme(args.out, manifest)
    write_viewer(args.out)


if __name__ == "__main__":
    main()
