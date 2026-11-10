"""Light curves: every detection of every frame, linked into objects; a catalogue of the objects and one light curve
file per object.

positional: the frames in time order (both bands together); each frame's detections matched one-to-one to the objects
so far, nearest first, within link_arcsec; a detection with no object within that starts a new one. Then objects
closer than link_arcsec are merged and every detection is given the nearest object (_merge). An object's position is
the mean of its detections'. Every detection is kept: an object seen once is a noise peak or a short
transient, and its n_det says so.

Written into the observation's folder:
  catalogue.csv            one row per object: id, ra, dec, per band: n_det, median mag, its scatter (1.4826 MAD) and
                           median error; first and last detection (MJD); the nearest ATLAS-Refcat2 star within
                           link_arcsec (its position at the epoch, g, r, i, separation); the light curve file; truth_* (the simulated star it is,
                           for scoring only: no stage reads it)
  lightcurves/NNNNNN.csv   one row per detection, in time order: when (UTC, MJD, BJD_TDB), band, exposure, telescope,
                           airmass, seeing, the frame's zero point; everything the extraction and calibration measured
                           (position, flux, error, S/N, shape, flags, mag, mag_err, ...); truth_* as above
"""
import re
from pathlib import Path

import numpy as np

SETTINGS = dict(link_arcsec=1.5, refcat_max_mag=19.0)
_REFCAT = {}


def reference_stars(ra, dec, radius, max_mag, epoch_year):
    """ATLAS-Refcat2 around (ra, dec) from hdps's loader (hdps.astrometry.providers.ATLASRefcat2): its table with
    positions moved to the epoch (proper motions in mas/yr from 2015.5)."""
    key = (round(ra, 4), round(dec, 4), radius, max_mag)
    if key not in _REFCAT:
        from hdps.astrometry.providers import ATLASRefcat2

        _REFCAT[key] = ATLASRefcat2().get_reference_stars(ra, dec, radius, min_g=0.0, max_g=max_mag).table
    t = _REFCAT[key]
    dt = epoch_year - 2015.5
    ra_now = np.asarray(t["ra"]) + np.nan_to_num(np.asarray(t["pmra"], float)) * dt / 3.6e6 / np.cos(np.radians(t["dec"]))
    dec_now = np.asarray(t["dec"]) + np.nan_to_num(np.asarray(t["pmdec"], float)) * dt / 3.6e6
    return t, ra_now, dec_now
FRAME_COLUMNS = ("exposure", "band", "date_obs", "bjd_tdb", "exptime", "airmass", "seeing", "tel_id", "ratchet",
                 "fieldang", "zp_zp", "zp_rms", "zp_n", "zp_tier")


def _frames(catalogue_dir):
    """Every frame's detections (astropy Table, its frame metadata in .meta), in time order."""
    from astropy.table import Table

    files = [f for f in Path(catalogue_dir).glob("*.parquet") if re.search(r"_\d{4}\.parquet$", f.name)]  # not the
    # difference-image tables (*_candidates, *_transients)
    tables = [Table.read(f) for f in files]
    return sorted(tables, key=lambda t: t.meta["date_obs"])


def _truth(catalogue_dir):
    """The engine's truth_stars.json: (label of star index, Gaia source of star index); placed stars' source < 0."""
    import json

    path = Path(catalogue_dir) / "truth_stars.json"
    if not path.exists():
        return None
    t = json.loads(path.read_text())
    labels = {int(k): v for k, v in t["label"].items()}
    source = np.asarray(t["source"], np.int64)
    return (lambda k: labels.get(int(k), "star" if k >= 0 else "")), (lambda k: int(source[k]) if k >= 0 else 0)


def _merge(detections, obj_ra, obj_dec, obj_n, radius):
    """Second pass. Objects closer than the link radius are one source: the detector merges detections within its
    de_dup_radius (6 px, 5.3"), so two real sources this close are never both detected; the first pass can still
    split one when a detection falls between it and a duplicate. Such objects are merged (friends of friends; position
    weighted by detections), and every detection is given the nearest merged object. Returns each detection's
    object, numbered 0, 1, ... in order of first detection."""
    from scipy.spatial import cKDTree

    cos = np.cos(np.radians(np.mean(obj_dec)))
    parent = np.arange(len(obj_ra))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in cKDTree(np.c_[obj_ra * cos, obj_dec]).query_pairs(radius):
        parent[root(a)] = root(b)
    _, group = np.unique([root(i) for i in range(len(obj_ra))], return_inverse=True)
    weight = np.bincount(group, obj_n)
    merged = np.c_[np.bincount(group, obj_ra * obj_n) / weight * cos, np.bincount(group, obj_dec * obj_n) / weight]
    det = np.c_[np.asarray(detections["ra"], float) * cos, np.asarray(detections["dec"], float)]
    nearest = cKDTree(merged).query(det)[1]
    _, first = np.unique(nearest, return_index=True)  # renumber by first detection
    order = np.argsort(np.argsort(first))
    return order[np.searchsorted(np.unique(nearest), nearest)]


def positional(catalogue_dir, out_dir, cfg):
    """Link the detections in catalogue_dir (the per-frame catalogues) into objects; write catalogue.csv and
    lightcurves/ into out_dir. Returns the catalogue."""
    from astropy.coordinates import SkyCoord
    from astropy.table import Table, vstack
    from astropy.time import Time
    import astropy.units as u

    c = {**SETTINGS, **cfg}
    truth = _truth(catalogue_dir)
    radius = c["link_arcsec"] / 3600.0
    obj_ra, obj_dec, obj_n = [], [], []  # running mean positions
    rows = []
    for t in _frames(catalogue_dir):
        ra, dec = np.asarray(t["ra"], float), np.asarray(t["dec"], float)
        object_of = np.full(len(t), -1)
        if obj_ra:
            cos = np.cos(np.radians(np.mean(dec)))
            o_ra, o_dec = np.asarray(obj_ra), np.asarray(obj_dec)
            pairs = []
            for i in range(len(t)):  # candidate pairs within the radius, then one-to-one, nearest first
                d = np.hypot((o_ra - ra[i]) * cos, o_dec - dec[i])
                for k in np.nonzero(d <= radius)[0]:
                    pairs.append((d[k], i, k))
            taken = set()
            for _, i, k in sorted(pairs):
                if object_of[i] < 0 and k not in taken:
                    object_of[i] = k
                    taken.add(k)
        for i in range(len(t)):
            k = object_of[i]
            if k < 0:
                k = object_of[i] = len(obj_ra)
                obj_ra.append(ra[i]), obj_dec.append(dec[i]), obj_n.append(1)
            else:
                obj_n[k] += 1
                obj_ra[k] += (ra[i] - obj_ra[k]) / obj_n[k]
                obj_dec[k] += (dec[i] - obj_dec[k]) / obj_n[k]
        frame = Table(t, copy=True)
        frame.meta = {}
        frame["object"] = object_of
        for key in FRAME_COLUMNS:
            if key in t.meta:
                frame[key.replace("zp_zp", "zp")] = t.meta[key]
        rows.append(frame)
    detections = vstack(rows, metadata_conflicts="silent")
    detections["object"] = _merge(detections, np.asarray(obj_ra), np.asarray(obj_dec), np.asarray(obj_n, float), radius)
    detections["mjd"] = Time(list(detections["date_obs"]), format="isot", scale="utc").mjd
    if "star" in detections.colnames:
        detections.rename_column("star", "truth_star")
        if truth is not None:
            detections["truth_label"] = [truth[0](k) for k in detections["truth_star"]]
            detections["truth_source"] = [truth[1](k) for k in detections["truth_star"]]

    lc_dir = Path(out_dir) / "lightcurves"
    lc_dir.mkdir(parents=True, exist_ok=True)
    for old in lc_dir.glob("*.csv"):
        old.unlink()
    lead = ["object", "date_obs", "mjd", "bjd_tdb", "band", "exposure", "tel_id", "airmass", "seeing", "exptime", "zp",
            "zp_rms", "mag", "mag_err", "flux", "flux_err", "snr", "ra", "dec", "xcentroid", "ycentroid", "saturated"]
    lead = [k for k in lead if k in detections.colnames]
    order = lead + [k for k in detections.colnames if k not in lead]
    detections = detections[order]
    detections.sort(["object", "mjd"])
    bounds = np.r_[0, np.nonzero(np.diff(detections["object"]))[0] + 1, len(detections)]

    catalogue = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        lc = detections[a:b]
        k = int(lc["object"][0])
        name = f"{k:06d}"
        lc.write(lc_dir / f"{name}.csv", format="ascii.csv", overwrite=True)
        row = dict(id=k, ra=float(np.mean(lc["ra"])), dec=float(np.mean(lc["dec"])), n_det=len(lc), mjd_first=float(lc["mjd"].min()),
                   mjd_last=float(lc["mjd"].max()))
        for band in ("beta", "rho"):
            m = lc[lc["band"] == band]
            mag = np.asarray(m["mag"], float)
            ok = np.isfinite(mag)
            row[f"n_det_{band}"] = len(m)
            row[f"mag_{band}"] = float(np.median(mag[ok])) if ok.any() else np.nan
            row[f"mag_{band}_scatter"] = float(1.4826 * np.median(np.abs(mag[ok] - np.median(mag[ok])))) if ok.sum() > 1 else np.nan
            row[f"mag_{band}_err"] = float(np.median(np.asarray(m["mag_err"], float)[ok])) if ok.any() else np.nan
        row["lightcurve"] = f"lightcurves/{name}.csv"
        if "truth_star" in lc.colnames:
            stars, counts = np.unique(np.asarray(lc["truth_star"]), return_counts=True)
            row["truth_star"] = int(stars[np.argmax(counts)])
            if truth is not None:
                row["truth_label"], row["truth_source"] = truth[0](row["truth_star"]), truth[1](row["truth_star"])
        catalogue.append(row)
    catalogue = Table(rows=catalogue)

    # the nearest Refcat2 star (proper motion to the first detection's epoch)
    centre = SkyCoord(float(np.mean(catalogue["ra"])) * u.deg, float(np.mean(catalogue["dec"])) * u.deg)
    radius = float(centre.separation(SkyCoord(catalogue["ra"] * u.deg, catalogue["dec"] * u.deg)).deg.max()) + 0.01
    ref, ref_ra, ref_dec = reference_stars(centre.ra.deg, centre.dec.deg, radius, c["refcat_max_mag"],
                                           Time(float(np.min(catalogue["mjd_first"])), format="mjd").jyear)
    j, sep, _ = SkyCoord(catalogue["ra"] * u.deg, catalogue["dec"] * u.deg).match_to_catalog_sky(
        SkyCoord(ref_ra * u.deg, ref_dec * u.deg))
    near = sep.arcsec <= c["link_arcsec"]
    catalogue["refcat2_ra"] = np.where(near, ref_ra[j], np.nan)  # Refcat2 as read here has no star ID (rix: a row
    catalogue["refcat2_dec"] = np.where(near, ref_dec[j], np.nan)  # index of the query): its position identifies it
    for band in ("g", "r", "i"):
        catalogue[f"refcat2_{band}"] = np.where(near, np.asarray(ref[band], float)[j], np.nan)
    catalogue["refcat2_sep_arcsec"] = np.where(near, sep.arcsec, np.nan)
    last = [k for k in ("truth_star", "truth_label", "truth_source") if k in catalogue.colnames]
    catalogue = catalogue[[k for k in catalogue.colnames if k not in last] + last]
    catalogue.write(Path(out_dir) / "catalogue.csv", format="ascii.csv", overwrite=True)
    return catalogue
