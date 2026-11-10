"""Repository paths and machine configuration (config.toml, analysis.toml)."""

import os
import tomllib
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"
OUTPUTS = REPO / "outputs"
DATA = REPO / "data"
INLISTS = REPO / "inlists"
INSTRUMENT = REPO / "filters" / "ArgusGaiaSDSS" / "ArgusGaiaSDSS"
STAR_BIN = REPO / "mesa_work" / "build" / "bin" / "star"

# Filter files in the combined instrument -> history column names.
GAIA_BANDS = ("G", "Gbp", "Grp")
ARGUS_BANDS = ("beta", "rho")
SDSS_BANDS = ("SDSS_u", "SDSS_g", "SDSS_r", "SDSS_i", "SDSS_z")
ALL_BANDS = GAIA_BANDS + ARGUS_BANDS + SDSS_BANDS


@cache
def config():
    return tomllib.loads((REPO / "config.toml").read_text())


@cache
def analysis():
    return tomllib.loads((REPO / "analysis.toml").read_text())


def mesa_dir():
    return Path(config()["paths"]["mesa_dir"] or os.environ["MESA_DIR"])


def colors_data():
    d = config()["paths"]["colors_data"]
    return Path(d) if d else mesa_dir() / "data" / "colors_data"


def colors_path(key):
    """Absolute path for a [colors] entry in config.toml."""
    p = Path(config()["colors"][key])
    out = p if p.is_absolute() else colors_data() / p
    # MESA treats directory paths with a trailing slash; keep it for stellar_atm
    return str(out) + ("/" if config()["colors"][key].endswith("/") else "")
