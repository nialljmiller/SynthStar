"""Build the galaxy libraries: the spectra (BC03) and shapes (COSMOS) libraries are make_stars generators
(synthstar/sources/galaxies.py); this is the same build as

    python make_stars.py examples/stars/galaxy_library.toml

The earlier single-file grid (data/galaxy_templates/bc03_grid.h5: 90 dust-free, line-free solar-metallicity
spectra with Argus magnitudes) is superseded and no longer read.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, str(REPO / "make_stars.py"),
                              str(REPO / "examples" / "stars" / "galaxy_library.toml"), *sys.argv[1:]]))
