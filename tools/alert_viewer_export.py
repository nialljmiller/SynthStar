"""The data of the alert viewer (tools/alert_viewer.html): an alert sample (tools/alert_sample.py's output) as compact
binary files a static web page can load.

    <out>/index.json            the sample, its night-bands, the column layout of every file, the injected sources
    <out>/<night-band>.meta.txt every packet of the night-band: one column after another (index.json gives each
                                column's type and byte offset), from the packet (time, position, photometry, score,
                                ids) and its truth row (truth.parquet: class, label, source, true magnitude and flux)
    <out>/<night-band>.cutNN.png every packet's three cutouts (science, difference, template), PER_ATLAS packets to
                                an 8-bit grey PNG, PER_ROW to a row, each a 96 x 32 px block (the planes side by
                                side, cutout row 0 at the top of the block); the image's last pixel row holds 0..255
                                so the page can check (and undo) anything the browser does to the grey levels. The
                                atlases go to the artifact's asset store (1 GiB; a version's own files hold 256 MB,
                                the week's cutouts ~650 MB as PNG)

The .meta.txt files are gzip-compressed and base64-encoded (artifacts serve text, not raw bytes; the page decodes them
with the browser's DecompressionStream).

A cutout plane is stored as 8 bits on an asinh scale about its own median, so the page can show any stretch and read
back pixel values: q = round(254 (a - a_min) / (a_max - a_min)), a = asinh((v - m) / s), m the median and s 1.4826 x
the median absolute deviation (at least 1e-3 of the range); 255 is NaN (beyond the segment's edge). Back:
v = m + s sinh(a_min + q (a_max - a_min) / 254). Per plane m, s, a_min, a_max are the meta column qp (12 float32 per
packet: science, difference, template).

Times are the truth rows' time_utc, in days since the first night's date (00:00 UTC). Positions are offsets from the
field centre in arcsec: east dx = (ra - ra0) cos(dec0) 3600 (ra - ra0 wrapped to +-180: f359's window crosses
RA 0), north dy = (dec - dec0) 3600; the page inverts them exactly.

    python tools/alert_viewer_export.py outputs/viewer_stage -o outputs/alert_viewer_data
"""
import argparse
import base64
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CLASSES = ("transient", "variable", "constant", "none")
SOURCE_KEYS = ("kind", "generator", "engine", "model", "params", "z", "t0_bjd", "peak_days", "period", "ebv",
               "distance_pc", "star")
# name, type, where from (packet candidate field, truth column, or computed)
COLUMNS = [("epoch", "f64"),
           ("t", "f32"), ("dx", "f32"), ("dy", "f32"), ("mag", "f32"), ("magErr", "f32"), ("flux", "f32"),
           ("fluxErr", "f32"), ("snr", "f32"), ("score", "f32"), ("magSci", "f32"), ("limMag", "f32"), ("zp", "f32"),
           ("fwhm", "f32"), ("x", "f32"), ("y", "f32"), ("tMag", "f32"), ("tDiff", "f32"), ("tNow", "f32"),
           ("tRef", "f32"), ("tSep", "f32"),
           ("qp", "f32x12"), ("cut", "i32"), ("exposure", "i32"), ("tanseg", "i32"), ("camera", "i32"),
           ("src", "i16"),
           ("cls", "u8"), ("label", "i8"), ("uuid", "u8x16")]
DTYPES = {"f64": "<f8", "f32": "<f4", "i32": "<i4", "i16": "<i2", "u8": "u1", "i8": "i1", "u8x16": "u1", "f32x12": "<f4"}
PER = {"u8x16": 16, "f32x12": 12}  # values per packet of a vector column
PER_ATLAS, PER_ROW = 2048, 32  # packets per atlas image, packets per atlas row
CANDIDATE = {"mag": "psfMagnitudeDif", "magErr": "psfMagnitudeDifErr", "flux": "psfFluxDif",
             "fluxErr": "psfFluxDifErr", "snr": "psfSnr", "score": "realBogusScore", "magSci": "psfMagnitudeSci",
             "limMag": "psfLimMag", "zp": "psfZeropoint", "fwhm": "fwhm", "x": "x", "y": "y", "tanseg": "tansegId"}
TRUTH = {"tMag": "truth_mag", "tDiff": "truth_diff_flux", "tNow": "truth_flux_now", "tRef": "truth_flux_reference",
         "tSep": "truth_sep_arcsec", "exposure": "exposure"}
PLANES = ("cutoutScience", "cutoutDifference", "cutoutTemplate")


def field_centre(field):
    """The field centre from its name (f359_539p61_214 -> 359.539, +61.214)."""
    m = re.match(r"f(\d+)_(\d+)([pm])(\d+)_(\d+)", field)
    if not m:
        raise SystemExit(f"{field}: no centre in the field's name (give --centre)")
    return float(f"{m[1]}.{m[2]}"), (1 if m[3] == "p" else -1) * float(f"{m[4]}.{m[5]}")


def quantize(raw):
    """One cutout plane (4096 bytes, float32) -> (m, s, a_min, a_max), 1024 codes (module docstring)."""
    v = np.frombuffer(raw, "<f4").astype(np.float64)
    ok = np.isfinite(v)
    q = np.full(v.size, 255, np.uint8)
    if not ok.any():
        return (0.0, 1.0, 0.0, 0.0), q
    m = float(np.median(v[ok]))
    s = max(1.4826 * float(np.median(np.abs(v[ok] - m))), 1e-3 * float(np.ptp(v[ok])), 1e-6)
    a = np.arcsinh((v[ok] - m) / s)
    lo, hi = float(a.min()), float(a.max())
    q[ok] = np.round(254 * (a - lo) / (hi - lo)) if hi > lo else 0
    return (m, s, lo, hi), q


def write_atlases(blocks, out, name):
    """Cutout blocks (n, 32, 96) -> PNG atlases (module docstring); returns their index.json entries."""
    from PIL import Image

    files = []
    for a in range(0, len(blocks), PER_ATLAS):
        part = blocks[a:a + PER_ATLAS]
        rows = -(-len(part) // PER_ROW)
        pad = np.full((rows * PER_ROW, 32, 96), 255, np.uint8)
        pad[:len(part)] = part
        img = np.full((rows * 32 + 1, PER_ROW * 96), 255, np.uint8)
        img[:-1] = pad.reshape(rows, PER_ROW, 32, 96).transpose(0, 2, 1, 3).reshape(rows * 32, PER_ROW * 96)
        img[-1, :256] = np.arange(256)
        f = f"{name}.cut{a // PER_ATLAS:02d}.png"
        Image.fromarray(img, "L").save(out / f, "PNG", optimize=True)
        files.append(dict(file=f, n=len(part), width=img.shape[1], height=img.shape[0]))
    return files


def night_band(sample, entry, ra0, dec0, t0, sources, out):
    """Write one night-band's two files; returns its index.json entry."""
    from argus_image import alert_files

    folder = sample / entry["observation"] / entry["band"]
    truth = pd.read_parquet(folder / "truth.parquet").set_index("alertId")
    packets = {}
    for path in sorted((folder / "packets").glob("*.avro")):
        for p in alert_files.read(path):
            packets[str(p["alertId"])] = p
    ids = [a for a in truth.index if a in packets]  # (every packet has its truth row: alert_sample.py's join)
    n = len(ids)
    col = {name: np.zeros(PER.get(kind, 1) * n, DTYPES[kind]) for name, kind in COLUMNS}
    tr = truth.loc[ids]
    for name, key in CANDIDATE.items():
        col[name][:] = [np.nan if packets[a]["candidate"][key] is None else packets[a]["candidate"][key] for a in ids] \
            if col[name].dtype.kind == "f" else [packets[a]["candidate"][key] or 0 for a in ids]
    col["epoch"][:] = [float(packets[a]["candidate"]["epochId"]) for a in ids]
    col["camera"][:] = [(packets[a]["candidate"]["cameraIds"] or [0])[0] for a in ids]
    for name, key in TRUTH.items():
        col[name][:] = tr[key].to_numpy()
    col["t"][:] = (pd.to_datetime(tr["time_utc"], utc=True) - t0).dt.total_seconds().to_numpy() / 86400.0
    ra, dec = tr["ra"].to_numpy(), tr["dec"].to_numpy()
    col["dx"][:] = ((ra - ra0 + 180.0) % 360.0 - 180.0) * np.cos(np.radians(dec0)) * 3600.0
    col["dy"][:] = (dec - dec0) * 3600.0
    col["cls"][:] = [CLASSES.index(c) if c in CLASSES else 255 for c in tr["truth_class"].fillna("")]
    col["label"][:] = tr["label"].to_numpy()
    col["src"][:] = [sources.get(lab, -1) for lab in tr["truth_label"].fillna("")]
    col["uuid"][:] = np.frombuffer(b"".join(bytes.fromhex(a.replace("-", "")) for a in ids), np.uint8)

    has = np.array([all(len(packets[a].get(p) or b"") == 4096 for p in PLANES) for a in ids], bool)
    col["cut"][:] = -1
    col["cut"][has] = np.arange(int(has.sum()))
    blocks = np.full((int(has.sum()), 32, 96), 255, np.uint8)
    for k, row in enumerate(np.flatnonzero(has)):
        for j, plane in enumerate(PLANES):
            col["qp"][12 * row + 4 * j:12 * row + 4 * j + 4], q = quantize(packets[ids[row]][plane])
            blocks[k, :, 32 * j:32 * j + 32] = q.reshape(32, 32)

    name = f"{entry['observation'].rsplit('_', 1)[-1]}_{entry['band']}"  # n<k>_<band>
    offsets, blob = {}, bytearray()
    for c, kind in sorted(COLUMNS, key=lambda c: -np.dtype(DTYPES[c[1]]).itemsize):  # (typed arrays need alignment)
        blob += b"\0" * (-len(blob) % 8)
        offsets[c] = len(blob)
        blob += col[c].tobytes()
    (out / f"{name}.meta.txt").write_bytes(base64.b64encode(gzip.compress(bytes(blob), 9, mtime=0)))
    atlases = write_atlases(blocks, out, name)
    night = re.search(r"_n(\d+)$", entry["observation"])
    labels = col["label"]
    size = sum((out / a["file"]).stat().st_size for a in atlases)
    print(f"{name}: {n} packets, cutouts for {len(blocks)} in {len(atlases)} atlases ({size / 1e6:.1f} MB), meta "
          f"{(out / f'{name}.meta.txt').stat().st_size / 1e6:.1f} MB", flush=True)
    return dict(id=name, observation=entry["observation"], band=entry["band"], night=int(night[1]) if night else 0,
                date=entry["date"], moon=entry.get("moon_illumination"), n=n, ncut=len(blocks), offsets=offsets,
                meta=f"{name}.meta.txt", atlases=atlases, exposures=entry.get("exposures_run"),
                window_deg2=entry.get("window_deg2"), real=int((labels == 1).sum()), bogus=int((labels == 0).sum()),
                below=int((labels == -1).sum()))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sample", type=Path, help="an alert sample (tools/alert_sample.py -o)")
    p.add_argument("-o", "--out", type=Path, required=True)
    p.add_argument("--centre", type=float, nargs=2, metavar=("RA", "DEC"), help="default: from the field's name")
    p.add_argument("--only", nargs="+", metavar="NIGHT-BAND", help="only these night-bands (e.g. n1_beta)")
    args = p.parse_args(argv)
    manifest = json.loads((args.sample / "manifest.json").read_text())
    entries = sorted(manifest["nights"], key=lambda e: (e["date"], e["observation"], e["band"] != "beta"))
    if args.only:
        entries = [e for e in entries if f"{e['observation'].rsplit('_', 1)[-1]}_{e['band']}" in args.only]
    ra0, dec0 = args.centre or field_centre(entries[0]["field"])
    t0 = pd.Timestamp(min(e["date"] for e in entries), tz="UTC")
    labels = set()  # the injected sources (one scene over the week): their order is the page's source index
    info = {}
    for e in entries:
        truth = pd.read_parquet(args.sample / e["observation"] / e["band"] / "truth.parquet")
        for _, r in truth[truth["truth_label"].fillna("") != ""].drop_duplicates("truth_label").iterrows():
            labels.add(r["truth_label"])
            info[r["truth_label"]] = {k: (None if pd.isna(r[f"source_{k}"]) else r[f"source_{k}"])
                                      for k in SOURCE_KEYS if f"source_{k}" in r}
    order = sorted(labels, key=lambda s: (s[0], int(re.sub(r"\D", "", s) or 0)))
    sources = {lab: k for k, lab in enumerate(order)}
    args.out.mkdir(parents=True, exist_ok=True)
    bands = [night_band(args.sample, e, ra0, dec0, t0, sources, args.out) for e in entries]
    first = entries[0]
    index = dict(name=manifest.get("name", ""), field=first["field"], centre=[ra0, dec0],
                 t0_utc=t0.isoformat(), model=sorted({m for e in entries for m in e["rb_model_versions"]}),
                 bundle=first["model_bundle"], score_cut=first["score_cut"], score_cut_value=first.get("score_cut_value"),
                 schema=sorted({v for e in entries for v in e["schema_versions"]}),
                 atlas=dict(per_atlas=PER_ATLAS, per_row=PER_ROW, block=[96, 32]),
                 classes=list(CLASSES), columns=[dict(name=c, type=k) for c, k in COLUMNS],
                 cutout_planes=["science", "difference", "template"], nightbands=bands,
                 sources=[dict(label=lab, **info[lab]) for lab in order])
    (args.out / "index.json").write_text(json.dumps(index, default=str))
    print(f"index: {len(bands)} night-bands, {sum(b['n'] for b in bands)} packets, {len(order)} sources", flush=True)


if __name__ == "__main__":
    main()
