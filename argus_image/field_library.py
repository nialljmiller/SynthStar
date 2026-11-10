"""argus_image.field_library -- the Gaia field inventory: which catalogued variables are in an Argus FIELD OF VIEW
(decided here, from ArgusSim's tile geometry and the detector scene), and a star file + observation files for them.
The "which variables can be modelled and how" decision is synthstar's (synthstar.gaia.classify_variables), called
from here. This is the architecture's inversion: argus_image decides the FOV and hands the source list to synthstar;
synthstar never looks at a field.

    python -m argus_image.field_library --gaia-list RA DEC [--block N] [--write stars.toml] [--write-observation obs.toml]
"""
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from synthstar import gaia
from synthstar.gaia import classify_variables
from synthstar.sources.gaia import CACHE
from synthstar.tomlfile import table, value


def field_sources(data, ra, dec, block_n=3):
    """The engine's star list for an Argus pointing -- which stars fall in the block_n x block_n block (in_block) --
    via ArgusSim's tile geometry and the detector scene (argus_image.engine). The field-of-view decision."""
    from argus_sim import skymap_shim as sh

    from argus_image.engine import imaging, state

    tile_id, minipix = sh.radec_to_composite_id(np.array([ra]), np.array([dec]))
    tile, block = int(tile_id[0]), (int(minipix[0]) % 64, int(minipix[0]) // 64)
    state.set_block(block_n)
    state.FIELDS["_probe"] = dict(name="", centre=(ra, dec), tile=tile, block=block, date="", site=None)
    state.FIELD, state.DATA, state.TILE, state.BLOCK = "_probe", data, tile, block
    state.CENTRE = SkyCoord(ra * u.deg, dec * u.deg)
    return imaging.load_scene(), tile, block


def inventory(data, ra, dec, block_n=3):
    """Gaia-classified variables in the block: the Argus FOV (field_sources) joined to synthstar's classification
    (classify_variables). Returns (rows, stars, tile, block), as the old synthstar.gaia.inventory did."""
    stars, tile, block = field_sources(data, ra, dec, block_n)
    return classify_variables(stars, data), stars, tile, block


def gaia_list(pointings, block, write, write_observation, write_batch):
    """The catalogued variables around each pointing, and a star file for them: the pointing's Gaia field set and every
    variable that can be modelled. Optionally an observation file per field, or one observation batch over the fields
    (each field's variables at their real positions, on the night they change most)."""
    from synthstar import gaia
    from synthstar.sources.gaia import CACHE

    star_file = ["# Gaia fields: each one's stars as a set, and each catalogued variable that can be modelled",
                 "# (Gaia variables: MESA RSP ~0.5-2 h each, PHOEBE minutes).", ""]
    fields = []  # (ra, dec, field name, the variables' star names)
    for ra, dec in pointings:
        field = gaia.field_name(ra, dec)
        gaia.fetch(ra, dec, CACHE / field)
        rows, stars, _, _ = inventory(CACHE / field, ra, dec, block)
        print(f"\nRA {ra}, Dec {dec}:")
        gaia.show(rows, block)
        star_file += table("[[set]]", dict(name=f"{field}_field", generator="Gaia field", ra=ra, dec=dec))
        variables = []
        for r in rows:
            if r["model"]:
                star_file += [f"# {r['kind']}, G {r['G']:.2f}, P {r['period']:.5f} d ({r['model']})"]
                star_file += table("[[star]]", dict(name=f"gaia_{r['source']}", generator="Gaia variable", source=r["source"]))
                variables.append(f"gaia_{r['source']}")
        fields.append((ra, dec, field, variables))
    if write:
        Path(write).write_text("\n".join(star_file))
        print(f"wrote {write}")
    else:
        print("\n" + "\n".join(star_file))

    def observation(ra, dec, field, variables, header):
        lines = header + [""] + table("[field]", dict(ra=ra, dec=dec, block=block)) + table("[night]", dict(date="auto"))
        lines += table("[stars]", dict(background=f"{field}_field", layout="real", missing="skip"))
        for v in variables:
            lines += table("[[place]]", dict(star=v))
        return "\n".join(lines)

    if write_observation:
        for ra, dec, field, variables in fields:
            path = Path(write_observation)
            if len(fields) > 1:
                path = path.with_name(f"{path.stem}_{field}{path.suffix}")
            path.write_text(observation(ra, dec, field, variables,
                                        [f"# The Gaia field at RA {ra}, Dec {dec}, on the night its variables change most."]))
            print(f"wrote {path}")
    if write_batch:  # tools/stage_batch.py: the base observation, and the fields zipped with their variables
        batch = Path(write_batch)
        base = batch.with_name(batch.stem + "_base.toml")
        base.write_text(observation(*fields[0][:3], [], ["# The observation each field of the batch gets (the batch sets "
                                                          "the pointing, background, name and stars)."]))
        lines = [f"# Observe each Gaia field of {write or 'the star file'} on the night its variables change most.",
                 f"#   python tools/stage_batch.py {batch}", "", f'base = "{base.name}"', 'combine = "zip"', "", "[vary]",
                 '"field.ra,field.dec,stars.background,field.name" = [']
        lines += [f"  [{ra}, {dec}, {value(field + '_field')}, {value(field)}]," for ra, dec, field, _ in fields] + ["]"]
        lines += ["place = ["] + [f"  [{', '.join('{star = ' + value(v) + '}' for v in vs)}]," for *_, vs in fields] + ["]", ""]
        batch.write_text("\n".join(lines))
        print(f"wrote {batch} and {base}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gaia-list", nargs=2, type=float, action="append", metavar=("RA", "DEC"), required=True,
                   help="a Gaia field's variables (repeatable)")
    p.add_argument("--block", type=int, default=9, help="minipixes across the block searched")
    p.add_argument("--write", help="write the star file here")
    p.add_argument("--write-observation", help="also write an observation file per field here")
    p.add_argument("--write-batch", help="also write an observation batch over the fields here")
    a = p.parse_args()
    gaia_list(a.gaia_list, a.block, a.write, a.write_observation, a.write_batch)
