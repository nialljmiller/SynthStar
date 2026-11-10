"""Assemble the combined Gaia + Argus + SDSS filter instrument for the colors module.

Gaia and Argus curves come from your MESA colors data (paths in config.toml);
the SDSS ugriz responses (SVO SLOAN/SDSS, Doi et al. 2010) ship with this repo.
"""

import shutil

from . import config as C

GAIA = ("G.dat", "Gbp.dat", "Grp.dat")
ARGUS = ("beta.dat", "rho.dat")
SDSS = tuple(f"SDSS_{b}.dat" for b in "ugriz")


def build():
    from pathlib import Path

    src = {
        Path(C.colors_path("gaia_filters")): GAIA,
        Path(C.colors_path("argus_filters")): ARGUS,
        C.DATA / "filters" / "SDSS": SDSS,
    }
    if C.INSTRUMENT.exists():
        shutil.rmtree(C.INSTRUMENT)
    C.INSTRUMENT.mkdir(parents=True)
    names = []
    for d, files in src.items():
        for f in files:
            shutil.copy(d / f, C.INSTRUMENT / f)
            names.append(f)
    (C.INSTRUMENT / C.INSTRUMENT.name).write_text("\n".join(names) + "\n")
    return C.INSTRUMENT, names
