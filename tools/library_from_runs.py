"""Every finished Gaia-variable model on this machine (runs/rsp_gaia_G<source>/pulsation.h5, MESA RSP or PHOEBE) as a
star file, so make_stars.py can wrap them into stars/ (it reuses the runs: nothing is recomputed), with each
variable's field as a [[set]] and one observation file per field (its variables at their real positions, on the
night they change most).

    python tools/library_from_runs.py                       # -> examples/stars/gaia_library.toml, outputs/library_obs/
    python make_stars.py examples/stars/gaia_library.toml   # wrap them (Gaia queries for each star's parameters)
    python observe.py outputs/library_obs/<field>.toml

A variable's field is the cached Gaia pointing (data/gaia/<field>/) that holds it; failed models (FAILED) are left out.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def pointing(name):
    """RA, Dec of a cache name f<ra>p|m<dec> (3 decimals, as gaia.field_name)."""
    ra, rest = name[1:].split("p" if "p" in name[1:] else "m", 1)
    sign = 1 if "p" in name[1:] else -1
    return float(ra.replace("_", ".")), sign * float(rest.replace("_", "."))


def main():
    from astropy.table import Table

    from argus_image.runconfig import _value

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--write", type=Path, default=REPO / "examples" / "stars" / "gaia_library.toml")
    p.add_argument("--observations", type=Path, default=REPO / "outputs" / "library_obs")
    a = p.parse_args()
    sources = sorted(int(d.name.removeprefix("rsp_gaia_G")) for d in (REPO / "runs").glob("rsp_gaia_G*")
                     if (d / "pulsation.h5").exists() and not (d / "FAILED").exists())
    caches = {c.parent.name: set(np.asarray(Table.read(c)["Source"]).astype(np.int64).tolist())
              for c in sorted((REPO / "data" / "gaia").glob("*/gaia_dr3_core.ecsv"))}
    fields, orphans = {}, []
    for s in sources:
        hit = [f for f, members in caches.items() if s in members]
        (fields.setdefault(hit[0], []).append(s) if hit else orphans.append(s))
    text = [f"# Every finished Gaia-variable model ({len(sources) - len(orphans)}) as a star, with its field. Written by "
            "tools/library_from_runs.py from runs/; make_stars.py reuses the runs (nothing is recomputed).", ""]
    a.observations.mkdir(parents=True, exist_ok=True)
    import tomllib

    for f, members in sorted(fields.items()):
        built = REPO / "stars" / f"{f}_field" / "star.toml"  # an existing set keeps its exact pointing
        ra, dec = ((lambda t: (t["ra"], t["dec"]))(tomllib.loads(built.read_text())["set"]) if built.exists()
                   else pointing(f))
        text += ["[[set]]", f'name = "{f}_field"', 'generator = "Gaia field"', f"ra = {ra}", f"dec = {dec}", ""]
        text += sum((["[[star]]", f'name = "gaia_{s}"', 'generator = "Gaia variable"', f"source = {s}", ""]
                     for s in members), [])
        (a.observations / f"{f}.toml").write_text("\n".join(
            [f"# The Gaia field {f}: its modelled variables at their real positions, on the night they change most.",
             "[field]", f"ra = {ra}", f"dec = {dec}", "block = 9", "", "[night]", 'date = "auto"', "", "[stars]",
             f"background = {_value(f + '_field')}", 'layout = "real"', 'missing = "skip"', "",
             "[output]", 'bands = ["rho"]', ""] + sum((["[[place]]", f'star = "gaia_{s}"', ""] for s in members), [])))
    a.write.write_text("\n".join(text))
    print(f"{len(sources)} models -> {sum(len(m) for m in fields.values())} stars in {len(fields)} fields: {a.write}; "
          f"observations in {a.observations}" + (f"; no cached field for {orphans}" if orphans else ""))


if __name__ == "__main__":
    main()
