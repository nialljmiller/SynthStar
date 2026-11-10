#!/usr/bin/env python3
"""Add ordinary-star SED coverage and catalogue-selected Cepheids without changing RR jobs.

Default is a dry run; --enqueue validates then atomically publishes new queue files.
Gaia counts are observed catalogue occupancy, NOT completeness-corrected sky weights.
"""
import argparse
from collections import defaultdict, Counter
from pathlib import Path
import json
import sys
import tempfile
import os

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from synthstar.grids import load
from synthstar.tomlfile import table
from make_stars import read_star_files


def evolution_batches():
    import importlib.util
    spec = importlib.util.spec_from_file_location("sky_spec", REPO / "grids/sky_library.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tracks = load("sky_library")
    groups = defaultdict(list)
    reused = []
    for t in tracks:
        if t["tag"] == "sed":
            if not (REPO / "runs" / t["name"] / "seds.h5").exists():
                raise RuntimeError(f"expected existing SED track missing: {t['name']}")
            reused.append(t["name"])
            continue
        if t["tag"] == "sky_hb_pilot":
            group = "sky_20261019_30_hb_pilot"
        elif (t["mass"], t["feh"]) in mod.PILOT:
            group = "sky_20261019_00_evolution_pilot"
        else:
            group = "bulk"
        groups[group].append(dict(name="lib_evol_" + t["name"], generator="MESA evolution track",
                                  grid="sky_library", track=t["name"]))
    bulk = groups.pop("bulk")
    for i in range(0, len(bulk), 6):
        groups[f"sky_20261019_20_evolution_{i//6:03d}"] = bulk[i:i+6]
    return dict(groups), reused


def cepheids():
    from astropy.table import Table
    cache = REPO / "data/library/gaia_cepheids_population.ecsv"
    t = Table.read(cache)
    def col(name):
        return np.asarray(np.ma.filled(t[name], np.nan), float)
    pf, po, mh, dec = (col(n) for n in ("PF", "P1O", "[M/H]", "DE_ICRS"))
    cells = defaultdict(list)
    for i, row in enumerate(t):
        # Loose ever-visible declination at McDonald. No claim of cadence/detection completeness.
        if dec[i] < -59:
            continue
        cl, sub, mode = (str(row[n]).strip() for n in ("Class", "SubClass", "ModeClass"))
        mode = {"FUNDAMENTAL": "F", "FIRST_OVERTONE": "1O"}.get(mode, mode)
        if cl == "DCEP" and mode in ("F", "1O"):
            kind = "DCEP " + mode
        elif cl == "T2CEP" and sub in ("BL_HER", "W_VIR"):
            kind = sub
        else:
            continue
        period = po[i] if mode == "1O" else pf[i]
        if not np.isfinite(period) or period <= 0:
            continue
        metal = round(mh[i] / 0.5) if np.isfinite(mh[i]) else "unknown"
        key = (kind, int(np.floor(np.log10(period)/0.1)), metal)
        cells[key].append((int(row["Source"]), period, float(row["Gmagavg"])))
    if not cells:
        raise RuntimeError("no supported Cepheids: check catalogue class column values")
    ordered = sorted(cells, key=lambda k: (-len(cells[k]), str(k)))
    # Retain some coverage of less common supported modes; don't call this class prevalence.
    chosen = []
    for kind in ("DCEP F", "DCEP 1O", "BL_HER", "W_VIR"):
        chosen.extend([k for k in ordered if k[0] == kind][:4])
    chosen += [k for k in ordered if k not in chosen][:max(0, 36-len(chosen))]
    chosen.sort(key=lambda k: (-len(cells[k]), str(k)))
    entries, occupancy = [], []
    for key in chosen:
        members = sorted(cells[key], key=lambda x: (not np.isfinite(x[2]), x[2], x[0]))
        # A relatively bright real member reduces parameter/distance failures.
        source, period, g = members[0]
        entries.append(dict(name=f"lib_gaia_cep_{source}", generator="Gaia variable", source=source))
        occupancy.append(dict(kind=key[0], log_period_bin=key[1], mh_bin=key[2],
                              catalogue_count=len(members), representative_source=source,
                              period_days=period, G=g))
    batches = {f"sky_20261019_10_cepheids_{i//6:03d}": entries[i:i+6]
               for i in range(0, len(entries), 6)}
    return batches, occupancy


def hads():
    from astropy.table import Table
    t = Table.read(REPO / "data/library/asassn_hads_population.ecsv")
    cells = defaultdict(list)
    for r in t:
        if np.ma.is_masked(r["GaiaDR3"]) or np.ma.is_masked(r["Per"]):
            continue
        source, period = int(r["GaiaDR3"]), float(r["Per"])
        if source <= 0 or float(r["DEJ2000"]) < -59 or not 0.03 <= period <= 0.3:
            continue
        v = float(r["Vmag"]) if not np.ma.is_masked(r["Vmag"]) else 99.0
        cells[int(np.floor(np.log10(period)/0.05))].append((source, period, v))
    ordered = sorted(cells, key=lambda k: (-len(cells[k]), k))
    chosen = ordered[:10]
    for k in (min(cells), max(cells)):
        if k not in chosen:
            chosen.append(k)
    chosen += [k for k in ordered if k not in chosen][:12-len(chosen)]
    entries, occupancy = [], []
    for key in chosen:
        source, period, v = min(cells[key], key=lambda x: (x[2], x[0]))
        entries.append(dict(name=f"lib_gaia_hads_{source}", generator="Gaia variable", source=source))
        occupancy.append(dict(log_period_bin=key, catalogue_count=len(cells[key]),
                              representative_source=source, period_days=period, V=v))
    return {f"sky_20261019_15_hads_{i//6:03d}": entries[i:i+6]
            for i in range(0, len(entries), 6)}, occupancy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--enqueue", action="store_true")
    a = p.parse_args()
    groups, reused = evolution_batches()
    cep, occupancy = cepheids()
    groups.update(cep)
    hads_groups, hads_occupancy = hads()
    groups.update(hads_groups)
    all_names = [e["name"] for es in groups.values() for e in es]
    if len(all_names) != len(set(all_names)):
        raise RuntimeError("duplicate new library names")
    queued = set()
    for q in (REPO / "queue").glob("*.toml"):
        if q.stem not in groups:
            queued.update(e[2]["name"] for e in read_star_files([q]))
    if queued.intersection(all_names):
        raise RuntimeError("models already queued: " + str(queued.intersection(all_names)))
    manifest = dict(created="2026-10-19", existing_evolution_tracks=reused,
                    new_models=Counter(e["generator"] for es in groups.values() for e in es),
                    batches={n: es for n, es in sorted(groups.items())}, cepheid_occupancy=occupancy,
                    hads_occupancy=hads_occupancy,
                    caveat="Coverage library, not a probability sample. Gaia observed counts are not "
                    "completeness corrected; apply age/SFH/IMF, spatial distribution, dust and Argus selection.")
    staged = []
    with tempfile.TemporaryDirectory(prefix="sky-library-") as tmp:
        for name, entries in sorted(groups.items()):
            text = "# General Argus library expansion, 2026-10-19. Coverage, NOT abundance weights.\n\n"
            text += "\n".join(line for e in entries for line in table("[[star]]", e))
            path = Path(tmp) / (name + ".toml")
            path.write_text(text)
            parsed = read_star_files([path])
            assert len(parsed) == len(entries)
            for _, gen, definition in parsed:
                if gen.name == "MESA evolution track":
                    from synthstar.runner import render_inlist
                    ts = {t["name"]: t for t in load(definition["grid"])}
                    render_inlist(ts[definition["track"]])
            target = REPO / "queue" / path.name
            if target.exists() and target.read_text() != text:
                raise RuntimeError(f"refusing to replace different existing queue file {target}")
            staged.append((target, text))
            print(f"{name}: {len(entries)} validated models")
        print(json.dumps({"models": manifest["new_models"], "reused": len(reused),
                          "batches": len(staged), "cepheid_classes": Counter(o["kind"] for o in occupancy)}, indent=2))
        if a.enqueue:
            manifest_path = REPO / "data/library/sky_library_20261019.json"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            for target, text in staged:
                if target.exists():
                    continue
                temp = target.with_suffix(".staging")
                temp.write_text(text)
                os.replace(temp, target)
            print("Published queue files. Existing RR files and running processes unchanged.")
        else:
            print("Dry run only; use --enqueue to publish.")


if __name__ == "__main__":
    main()
