"""Bring a packaged alert sample (tools/alert_sample.py output) up to the alert schema v4.0 of hdps PR #474 as the
docs define it (2026-11-02), and split the variable-star truth into a sidecar.

Two changes, both in place on <sample>/<night>/<band>/:

  packets/*.avro   each packet re-emitted under the updated schema (hdps alert_schema()): the candidate gets a copy of
                   the envelope alertId; the per-candidate refcatName/refcatVersion/nearestSrc* fields are removed
                   (they moved to the envelope's crossMatches array); the envelope gets crossMatches = [] (no
                   reference-catalog crossmatch runs in the simulation). ssoAssociation stays null. Every other field
                   and all three cutouts are byte-for-byte unchanged; only the record shape changes.

  truth.parquet    keeps transient / constant / none rows only
  truth_variables.parquet   the variable-star rows, moved out: their truth (truth_diff_flux and the real/bogus label)
                   is unreliable here, because truth_flux_reference was the star's cycle mean rather than its flux in
                   each camera's reference coadd (fixed in the engine 2026-10-30; this already-run sample predates it).

Idempotent: a packet already carrying candidate.alertId, or a truth.parquet already split, is left as it is.

    python tools/alert_repack_v4.py outputs/alert_week_v1
"""
import argparse
import io
import sys
from pathlib import Path

import pandas as pd
from fastavro import reader, writer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DROP = ("refcatName", "refcatVersion", "nearestSrcId", "nearestSrcSep", "nearestSrcMag")


def repack_packets(folder, schema):
    """Rewrite every packets/*.avro under the new schema; returns (files, packets) changed, (0, 0) if already v4."""
    paths = sorted(folder.glob("packets/*.avro"))
    if not paths:
        return 0, 0
    # decide from the first file whether this sample is already repacked
    first = list(reader(open(paths[0], "rb")))
    if first and "alertId" in first[0]["candidate"]:
        return 0, 0
    n = 0
    for path in paths:
        packets = list(reader(open(path, "rb")))
        for p in packets:
            p["candidate"] = {"alertId": p["alertId"], **{k: v for k, v in p["candidate"].items() if k not in DROP}}
            p["crossMatches"] = []
        buf = io.BytesIO()
        writer(buf, schema, packets, codec="deflate")
        path.write_bytes(buf.getvalue())
        n += len(packets)
    return len(paths), n


def split_truth(folder):
    """Move the variable rows of truth.parquet into truth_variables.parquet; returns (kept, variable), (-1, -1) if
    already split."""
    main = folder / "truth.parquet"
    side = folder / "truth_variables.parquet"
    if not main.exists() or side.exists():
        return -1, -1
    t = pd.read_parquet(main)
    var = t["truth_class"] == "variable"
    t[var].to_parquet(side, index=False)
    t[~var].to_parquet(main, index=False)
    return int((~var).sum()), int(var.sum())


def main(argv=None):
    from synthstar import alert_files

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sample", type=Path, help="a packaged alert sample (tools/alert_sample.py -o)")
    args = p.parse_args(argv)
    schema = alert_files.alert_schema()
    nb = sorted(d for d in args.sample.glob("*/*") if (d / "packets").is_dir() or (d / "truth.parquet").exists())
    tot_f = tot_p = tot_v = 0
    for folder in nb:
        f, n = repack_packets(folder, schema)
        kept, var = split_truth(folder)
        tot_f += f; tot_p += n; tot_v += max(var, 0)
        rel = folder.relative_to(args.sample)
        print(f"{rel}: packets {'already v4' if f == 0 else f'{f} files / {n} repacked'}; "
              f"truth {'already split' if kept < 0 else f'{kept} kept, {var} variables moved out'}", flush=True)
    print(f"done: {tot_f} packet files repacked, {tot_p} packets; {tot_v} variable truth rows moved to "
          f"truth_variables.parquet", flush=True)


if __name__ == "__main__":
    main()
