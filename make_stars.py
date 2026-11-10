#!/usr/bin/env python3
"""Simulate stars: each [[star]] / [[set]] of a star file is built into its own folder, stars/<name>/.

    python make_stars.py examples/stars/variables.toml                  # build every star in the file
    python make_stars.py examples/stars/f294_field.toml examples/stars/transients.toml   # several files
    python make_stars.py examples/stars/variables.toml --only wuma_035
    python make_stars.py examples/stars/variables.toml --rebuild wuma_035  # build again (e.g. after changing its definition)
    python make_stars.py --gaia-list 294.7844 -11.6235 --write my_field.toml --write-observation my_obs.toml
    python make_stars.py --gaia-list 294.78 -11.62 --gaia-list 273.04 -20.71 --write s.toml --write-batch b.toml
    python make_stars.py --list                                          # every built star

A star knows nothing about observing: no dust, no position in an image, no night, no instrument (observe.py adds
those). The generators and their keys: python -m synthstar.sources. A star folder holds

    star.toml    the definition it was built from (every key)
    meta.json    what was built: kind, engine, model parameters; real stars also position, ephemeris, sightline E(B-V)
    sed.h5       the model: SEDs against time (transients, before dust) or one cycle at 10 pc (periodic)
    catalogue/   (sets) the catalogue files
    summary.md   the same, to read

Stars already built from the same definition are skipped; a changed definition under an existing name is refused
unless the name is given to --rebuild.
"""
import argparse
import json
import shutil
import sys
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
STARS = REPO / "stars"

from synthstar import config as C  # noqa: E402
from synthstar import sources  # noqa: E402
from synthstar.tomlfile import table, value  # noqa: E402


def read_star_files(paths):
    """[(table name, generator, definition)] of the star files. A name may repeat across files with the same definition
    (e.g. a field's set in both its own file and its variables' file)."""
    entries = {}
    for path in paths:
        doc = tomllib.loads(Path(path).read_text())
        for kind in ("set", "star"):
            for entry in doc.get(kind, []):
                gen, keys = sources.check(entry)
                definition = dict(name=entry["name"], generator=gen.name, **keys)
                if entry["name"] in entries and entries[entry["name"]][2] != definition:
                    raise SystemExit(f"{entry['name']} is defined differently in two star files")
                entries[entry["name"]] = (kind, gen, definition)
    return list(entries.values())


def is_built(kind, definition):
    star_toml = STARS / definition["name"] / "star.toml"
    written = {k: v for k, v in definition.items() if v is not None}  # None keys are not written to star.toml
    return star_toml.exists() and tomllib.loads(star_toml.read_text())[kind] == written


def build_star(kind, gen, definition, keys):
    """Run the generator into stars/<name>/ and write star.toml, meta.json and summary.md next to its output."""
    name = definition["name"]
    folder = STARS / name
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    print(f"  {name}: building ({gen.engine}) …", flush=True)
    meta = gen.make(folder, keys, name)
    if meta is None:
        print(f"  {name}: NOT built (no acceptable model; see runs/)", flush=True)
        shutil.rmtree(folder)
        return False
    meta.pop("generator", None)  # the transient makers' own name for their engine
    meta = dict(name=name, generator=gen.name, engine=meta.pop("engine", gen.engine), timing=gen.timing,
                built=time.strftime("%Y-%m-%d %H:%M:%S"), **meta)
    as_json = json.dumps(meta, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    (folder / "meta.json").write_text(as_json)
    (folder / "star.toml").write_text(f"# made by make_stars.py; stars/{name}/\n" + "\n".join(table(f"[{kind}]", definition)))
    summary = [f"# {name}", "", f"**{meta['kind']}**: generator `{gen.name}` ({meta['engine']}; {meta.get('reference', '')})",
               "", f"- {meta['params']}", f"- timing: {gen.timing}" + (f", period {meta['period']:g} d" if meta.get("period") else "")]
    if meta.get("real"):
        r = meta["real"]
        summary.append(f"- a real star: Gaia DR3 {r['source']} at RA {r['ra']:.5f}, Dec {r['dec']:+.5f}; epoch BJD "
                       f"{r['epoch_bjd']:.5f}; sightline E(B-V) {r['sightline_ebv']}")
    summary += [f"- {k}: {v}" for k, v in (meta.get("notes") or {}).items()]
    summary += ["", "Definition: star.toml. Dust, position, timing and instrument are the observation's (observe.py)."]
    (folder / "summary.md").write_text("\n".join(summary) + "\n")
    print(f"  {name}: built -> stars/{name}/ ({meta['kind']}; {meta['params']})", flush=True)
    return True


def build(paths, only, rebuild):
    todo = []
    for kind, gen, definition in read_star_files(paths):
        name = definition["name"]
        if only and name not in only:
            continue
        if is_built(kind, definition) and name not in rebuild:
            print(f"  {name}: built already")
            continue
        if (STARS / name / "star.toml").exists() and name not in rebuild:
            raise SystemExit(f"stars/{name} was built from another definition: rename it, or --rebuild {name}")
        todo.append((kind, gen, definition, dict(definition)))

    failed = []
    for kind, gen, definition, keys in todo:  # network queries and shared state: one star at a time
        if gen.prepare:
            try:
                gen.prepare(keys, definition["name"])
            except SystemExit as reason:  # this star cannot be modelled; the others go on
                print(f"  {reason}")
                failed.append(definition["name"])
    todo = [t for t in todo if t[2]["name"] not in failed]

    def one(item):
        try:  # one failure must not stop a long batch
            if not build_star(*item):
                failed.append(item[2]["name"])
        except Exception as err:
            print(f"  {item[2]['name']}: FAILED: {err!r}", flush=True)
            failed.append(item[2]["name"])

    one_at_a_time = [t for t in todo if not (t[1].parallel and t[3].get("parallel", True))]
    side_by_side = [t for t in todo if t not in one_at_a_time]  # MESA RSP: config.toml [run] workers at once
    for item in one_at_a_time:
        one(item)
    with ThreadPoolExecutor(int(C.config()["run"]["workers"])) as pool:
        list(pool.map(one, side_by_side))
    if failed:
        raise SystemExit(f"not built: {failed}")



def list_built():
    for meta_file in sorted(STARS.glob("*/meta.json")):
        m = json.loads(meta_file.read_text())
        print(f"  {m['name']:32s} {m['timing']:9s} {m['kind']:16s} {m['generator']} ({m['engine']})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stars", nargs="*", help="star file(s)")
    p.add_argument("--only", nargs="+", default=[], help="build only these names")
    p.add_argument("--rebuild", nargs="+", default=[], help="rebuild these names")
    p.add_argument("--list", action="store_true", help="list the built stars")
    a = p.parse_args()
    if a.list:
        list_built()
    else:
        build(a.stars, a.only, a.rebuild)
