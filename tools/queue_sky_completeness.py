#!/usr/bin/env python3
"""Validate and enqueue pilot coverage for faint stars, LPV structure and PHOEBE binaries.

Default is a dry run. The MESA LPV entries are evolutionary tracks with restart
photos; they are not yet pulsation SED cycles because the cool AGB atmosphere is
outside the current Kurucz grid.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import tempfile
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from synthstar.grids import load
from synthstar.sources import check
from synthstar.tomlfile import table
from make_stars import read_star_files


def roche(q):
    q23 = q ** (2.0 / 3.0)
    return 0.49 * q23 / (0.6 * q23 + math.log1p(q ** (1.0 / 3.0)))


def semimajor(period, m1, m2):
    G, MSUN, RSUN = 6.674e-8, 1.989e33, 6.957e10
    return (G * (m1 + m2) * MSUN * (period * 86400) ** 2 / (4 * math.pi ** 2)) ** (1 / 3) / RSUN


def binaries():
    detached = [
        dict(name="libcomp_eb_cool_edge", period=0.65, incl=89.0, teff1=3770, teff2=3560,
             r1=0.55, r2=0.45, m1=0.55, m2=0.45, ecc=0.0, per0=90.0),
        dict(name="libcomp_eb_low_q", period=3.0, incl=87.0, teff1=8500, teff2=3660,
             r1=1.9, r2=0.50, m1=2.0, m2=0.50, ecc=0.0, per0=90.0),
        dict(name="libcomp_eb_a_f", period=2.4, incl=84.0, teff1=9000, teff2=7500,
             r1=2.0, r2=1.6, m1=2.0, m2=1.5, ecc=0.0, per0=90.0),
        dict(name="libcomp_eb_bstar", period=6.0, incl=82.0, teff1=22000, teff2=12000,
             r1=4.5, r2=2.2, m1=8.0, m2=4.0, ecc=0.1, per0=70.0),
        dict(name="libcomp_eb_near_roche", period=0.55, incl=89.0, teff1=5800, teff2=5200,
             r1=1.10, r2=0.90, m1=1.0, m2=0.8, ecc=0.0, per0=90.0),
        dict(name="libcomp_eb_subgiant", period=4.0, incl=88.0, teff1=5000, teff2=5800,
             r1=3.0, r2=1.0, m1=1.5, m2=1.0, ecc=0.0, per0=90.0),
        dict(name="libcomp_eb_giant_pair", period=22.0, incl=86.0, teff1=3900, teff2=4800,
             r1=18.0, r2=3.5, m1=1.5, m2=1.3, ecc=0.0, per0=90.0),
        dict(name="libcomp_eb_eccentric", period=8.0, incl=83.0, teff1=8500, teff2=6800,
             r1=1.9, r2=1.3, m1=2.0, m2=1.4, ecc=0.6, per0=90.0),
    ]
    for d in detached:
        a_peri = semimajor(d["period"], d["m1"], d["m2"]) * (1 - d["ecc"])
        assert d["r1"] < roche(d["m1"] / d["m2"]) * a_peri, d["name"]
        assert d["r2"] < roche(d["m2"] / d["m1"]) * a_peri, d["name"]
        d.update(generator="detached EB", distance=10.0, mh=0.0)

    contact = [
        dict(name="libcomp_wuma_short_lowq", period=0.24, q=0.15, incl=88.0,
             teff1=4800, teff2=4320, fillout=0.05),
        dict(name="libcomp_wuma_mid", period=0.38, q=0.40, incl=82.0,
             teff1=5600, teff2=5260, fillout=0.25),
        dict(name="libcomp_wuma_hot", period=0.72, q=0.75, incl=75.0,
             teff1=7000, teff2=6860, fillout=0.50),
        dict(name="libcomp_wuma_long_twin", period=0.95, q=1.0, incl=89.0,
             teff1=7500, teff2=7500, fillout=0.80),
    ]
    for c in contact:
        c.update(generator="W UMa", distance=10.0, mh=0.0)
    return detached + contact


def groups():
    faint = [{"name": "libcomp_" + t["name"], "generator": "MESA evolution track",
              "grid": "faint_stars", "track": t["name"]} for t in load("faint_stars")]
    lpv = [{"name": "libcomp_" + t["name"], "generator": "MESA evolution track",
            "grid": "lpv_tracks", "track": t["name"]} for t in load("lpv_tracks")]
    return {
        "sky_20261021_10_faint_star_tracks": faint,
        "sky_20261021_20_lpv_checkpoint_tracks": lpv,
        "sky_20261021_30_phoebe_binaries": binaries(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enqueue", action="store_true")
    args = parser.parse_args()
    batches = groups()
    all_names = [e["name"] for entries in batches.values() for e in entries]
    if len(all_names) != len(set(all_names)):
        raise RuntimeError("duplicate names in this coverage batch")
    for entries in batches.values():
        for entry in entries:
            gen, keys = check(entry)
            if gen.engine == "MESA evolution":
                track = {t["name"]: t for t in load(keys["grid"])}[keys["track"]]
                from synthstar.runner import render_inlist
                render_inlist(track)

    occupied = set()
    for path in (REPO / "queue").glob("*.toml"):
        occupied.update(e[2]["name"] for e in read_star_files([path]))
    for name in all_names:
        if name in occupied or (REPO / "stars" / name).exists():
            raise RuntimeError(f"refusing to overwrite existing model {name}")

    print(json.dumps({"models_by_batch": {name: len(items) for name, items in batches.items()},
                      "generators": dict(Counter(e["generator"] for es in batches.values() for e in es)),
                      "lpv_note": "AGB structure pilots, not pulsation light curves; cool AGB atmospheres remain unsupported",
                      "checkpointing": "LPV photos every 500 models; up to four total attempts resume from latest photo"}, indent=2))
    staged = []
    with tempfile.TemporaryDirectory(prefix="sky-completeness-") as tmp:
        for batch, entries in sorted(batches.items()):
            text = "# Sky completeness pilot. Coverage samples, not occurrence weights.\n\n"
            text += "\n".join(line for entry in entries for line in table("[[star]]", entry)) + "\n"
            path = Path(tmp) / f"{batch}.toml"
            path.write_text(text)
            parsed = read_star_files([path])
            if len(parsed) != len(entries):
                raise RuntimeError(f"parsed {len(parsed)} entries from {batch}; expected {len(entries)}")
            target = REPO / "queue" / path.name
            if target.exists() and target.read_text() != text:
                raise RuntimeError(f"refusing to replace different queue file {target}")
            staged.append((target, text))
    for target, _ in staged:
        print(f"queue/{target.name}")
    if args.enqueue:
        (REPO / "queue").mkdir(exist_ok=True)
        for target, text in staged:
            if target.exists():
                continue
            tmp = target.with_suffix(".staging")
            tmp.write_text(text)
            os.replace(tmp, target)
        print("Published queue files.")
    else:
        print("Dry run only; pass --enqueue to publish.")


if __name__ == "__main__":
    main()
