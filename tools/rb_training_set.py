"""A real/bogus training set from simulated observations, for a model hdps's AlertVettingTask loads
(hdps.alerts.classifier.RFBundleClassifier: a joblib random forest over hdps.alerts.features.FEATURE_NAMES).

Every observation's <name>_night_<band>.alerts.csv (night_movie.py) holds every candidate hdps's AlertVettingTask
found, its feat_* features exactly as hdps computed them, and what the simulation put there. Each row is labelled:

    real (1)    a placed transient or followed variable whose true flux change since the reference
                (truth_diff_flux, electrons) is at least --min-sigma times the candidate's PSF flux error in
                electrons (flux_psf_err_e): the difference image really holds that source's change
    bogus (0)   a constant star (a subtraction residual) or nothing simulated there (noise, artefact)
    dropped     a transient or variable whose change is below that (a candidate there is noise on a real source)

Each row also carries its night's provenance, from the run's resolved observation.toml: field (the field set),
night_date (the night used, which for a run asked with --date can differ from the date in its name),
mask_bright_stars (hdps's bright-star detection mask, on in runs before 2026-10-25), and the Moon: moon_phase (new,
first quarter, full, last quarter: the Sun-Moon ecliptic longitude difference, within 45 deg of 0, 90, 180, 270, at the
night's middle frame), moon_illumination (that night) and moon_alt (deg, at each candidate's exposure, from the site).

Writes <out>/training_set.parquet (label, the features under hdps's names in FEATURE_NAMES order with non-finite
values as -999.0 as the bundle serves them, provenance, and where each candidate's stamps are) and
<out>/training_set.json (feature names, feature spec, the rule, counts, sources). A training set made under one hdps
feature spec cannot train a model for another (the bundle refuses to load).

    python tools/rb_training_set.py outputs/pseudo_real/f294_hdps_full outputs/pseudo_real/sn_f065 -o data/rb
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MISSING = -999.0  # RFBundleClassifier maps non-finite feature values to this, at training and serving alike


def label(table, min_sigma):
    """1 real, 0 bogus, -1 dropped (see the module docstring). The error is in electrons, as the truth is: flux_psf
    (and flux_psf / snr_psf) is in hdps's segment units, 1.31 x electrons on our tiles (surface brightness), which
    set the threshold 1.31 x too high before 2026-10-28."""
    if "flux_psf_err_e" not in table:
        raise SystemExit("an alerts.csv without flux_psf_err_e (made before 2026-10-28): its PSF flux error is not in "
                         "electrons, so it cannot be set against truth_diff_flux")
    flux_err = np.abs(table["flux_psf_err_e"].to_numpy(float))
    change = np.abs(table["truth_diff_flux"].to_numpy(float))
    source = table["truth_class"].isin(["transient", "variable"]).to_numpy()
    out = np.where(table["truth_class"].isin(["constant", "none"]).to_numpy(), 0, -1)
    out[source & (change >= min_sigma * flux_err)] = 1
    return out


PHASES = ["new", "first quarter", "full", "last quarter"]


def night_provenance(folder):
    """field, night_date, mask_bright_stars and the site of an observation, from its resolved observation.toml."""
    import tomllib

    obs = tomllib.loads((folder / "observation.toml").read_text())
    sfft = [t["config"] for t in obs.get("hdps", {}).get("tasks", []) if t.get("type") == "SFFTSubtractionTask"]
    site = obs.get("argussim", {}).get("observatory", {})
    return dict(field=obs["stars"]["background"], night_date=obs["night"]["date"],
                mask_bright_stars=bool(sfft[0].get("mask_bright_stars", False)) if sfft else None,
                site=(site.get("latitude"), site.get("longitude"), site.get("altitude")))


def moon(times_utc, site):
    """(phase name, illumination) at the middle of these times, and the Moon's altitude (deg) at each."""
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, GeocentricTrueEcliptic, get_body, get_sun
    from astropy.time import Time

    unique, index = np.unique(np.asarray(times_utc, dtype=str), return_inverse=True)
    t = Time(list(unique), scale="utc")
    lat, lon, height = site
    where = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=(height or 0.0) * u.m)
    body = get_body("moon", t, where)
    alt = body.transform_to(AltAz(obstime=t, location=where)).alt.deg
    mid = t[len(t) // 2]
    m = get_body("moon", mid).transform_to(GeocentricTrueEcliptic(equinox=mid))
    sun = get_sun(mid).transform_to(GeocentricTrueEcliptic(equinox=mid))
    angle = float((m.lon - sun.lon).wrap_at(360 * u.deg).deg)
    illumination = float((1 - np.cos(get_sun(mid).separation(get_body("moon", mid)).rad)) / 2)
    return PHASES[int(((angle + 45) % 360) // 90)], illumination, alt[index]


def main(argv=None):
    from hdps.alerts.features import FEATURE_NAMES, FEATURE_SPEC_VERSION

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("observations", nargs="+", type=Path, help="observation output folders (outputs/<category>/<name>)")
    p.add_argument("-o", "--out", type=Path, required=True, help="folder for training_set.parquet and .json")
    p.add_argument("--min-sigma", type=float, default=3.0,
                   help="a transient/variable candidate is real if its true change is at least this many flux errors")
    args = p.parse_args(argv)

    parts, sources = [], []
    for folder in args.observations:
        night = night_provenance(folder)
        for csv in sorted(folder.glob("*_night_*.alerts.csv")):
            t = pd.read_csv(csv)
            t.insert(0, "source_file", str(csv))
            t.insert(1, "observation_dir", str(folder))
            phase, illumination, alt = moon(t["time_utc"], night["site"]) if len(t) else (None, np.nan, [])
            t = t.assign(field=night["field"], night_date=night["night_date"], mask_bright_stars=night["mask_bright_stars"],
                         moon_phase=phase, moon_illumination=illumination, moon_alt=alt)
            parts.append(t)
            sources.append(dict(file=str(csv), rows=len(t), field=night["field"], night_date=night["night_date"],
                                mask_bright_stars=night["mask_bright_stars"], moon_phase=phase,
                                moon_illumination=round(illumination, 3)))
    if not parts:
        sys.exit("no *_night_*.alerts.csv in " + ", ".join(map(str, args.observations)))
    table = pd.concat(parts, ignore_index=True)
    specs = sorted(set(table["feature_spec"].astype(str)))
    if specs != [FEATURE_SPEC_VERSION]:
        sys.exit(f"feature spec {specs} in the inputs; this hdps serves {FEATURE_SPEC_VERSION}: re-run them")

    table.insert(0, "label", label(table, args.min_sigma))
    kept = table[table["label"] >= 0].copy()
    for name in FEATURE_NAMES:  # hdps's names, in its order, non-finite as served
        values = kept.pop(f"feat_{name}").to_numpy(float)
        kept[name] = np.where(np.isfinite(values), values, MISSING)
    keep = ["label", *FEATURE_NAMES, "truth_class", "truth_label", "truth_star", "truth_mag", "truth_diff_flux",
            "flux_psf", "flux_psf_e", "flux_psf_err_e", "ra_deg", "dec_deg", "x", "y", "tanseg_id", "camera_id", "ratchet_num", "observation", "band",
            "exposure", "time_utc", "exptime", "airmass", "seeing", "tel_id", "ratchet", "field_angle", "alertId",
            "observation_dir", "catalog_file", "catalog_row", "stamps_file", "source_file", "field", "night_date",
            "mask_bright_stars", "moon_phase", "moon_illumination", "moon_alt"]
    kept = kept[[c for c in keep if c in kept.columns]]
    args.out.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(args.out / "training_set.parquet", index=False)

    counts = dict(real=int((kept["label"] == 1).sum()), bogus=int((kept["label"] == 0).sum()),
                  dropped=int((table["label"] < 0).sum()),
                  by_class={c: int(n) for c, n in table["truth_class"].value_counts().items()},
                  by_observation_band={f"{o} {b}": int(n) for (o, b), n in kept.groupby(["observation", "band"]).size().items()},
                  by_moon_phase_mask={f"{ph}, mask {'on' if m else 'off'}": int(n)
                                      for (ph, m), n in kept.groupby(["moon_phase", "mask_bright_stars"]).size().items()})
    manifest = dict(feature_names=list(FEATURE_NAMES), feature_spec=FEATURE_SPEC_VERSION, missing_value=MISSING,
                    label_rule=f"real: transient/variable with |truth_diff_flux| >= {args.min_sigma} x flux_psf_err_e; "
                               "bogus: constant star or nothing simulated; else dropped",
                    min_sigma=args.min_sigma, counts=counts, sources=sources,
                    stamps="hdps/<band>/<stamps_file> under observation_dir: npz with row_index (the catalog_row), "
                           "ref, sci, diff (31 x 31)")
    (args.out / "training_set.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(kept)} rows ({counts['real']} real, {counts['bogus']} bogus; {counts['dropped']} dropped) "
          f"-> {args.out / 'training_set.parquet'}")


if __name__ == "__main__":
    main()
