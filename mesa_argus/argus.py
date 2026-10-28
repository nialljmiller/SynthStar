"""The Argus bands as the movies and plots use them: ArgusSim filter names, symbols, colours, units, plot style.

The style is the Argus Array branding (branding/): its font (Helvetica Neue), its colours (branding/argus_colors.txt:
text navy, the logo's red and blue dots; the bands take the dots, beta blue and rho red) and its logo."""
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

BRANDING = Path(__file__).resolve().parent.parent / "branding"
BRAND = dict(line.strip().split(":", 1) for line in (BRANDING / "argus_colors.txt").read_text().splitlines() if ":" in line)
NAVY, RED, BLUE = BRAND["text"], BRAND["red_dots"], BRAND["blue_dots"]

FLAM = u.erg / (u.s * u.cm**2 * u.AA)
BANDS = {"beta": "g", "rho": "rho"}
COLORS = {"beta": BLUE, "rho": RED, "colour": "#7853a6"}
SYMBOLS = {"beta": r"$\beta$", "rho": r"$\rho$"}
MAG_FACTOR = 2.5 / np.log(10.0)


def system_response(tp, band, airmass):
    """Filter x optics x QE x atmosphere^airmass, as in ABPhot.electrons_from_sed (for plotting)."""
    response = np.asarray(tp.filt_wav[band] * tp.throughput_wav * tp.qe_wav, dtype=float)
    return response * np.asarray(tp.filt_wav.atmosphere, dtype=float) ** airmass


def apply_style():
    """The Argus branding for every plot: Helvetica Neue (branding/HelveticaNeue.ttc), navy text and axes."""
    from matplotlib import font_manager

    font = BRANDING / "HelveticaNeue.ttc"
    if font.exists():  # Apple's font, not redistributable: copies made for others leave it out (DejaVu Sans then)
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#8a8fa6",
            "axes.labelcolor": NAVY,
            "axes.titlecolor": NAVY,
            "text.color": NAVY,
            "xtick.color": NAVY,
            "ytick.color": NAVY,
            "font.family": ["Helvetica Neue", "DejaVu Sans"],  # DejaVu for the glyphs Helvetica Neue lacks
            "mathtext.fontset": "dejavusans",
            "savefig.facecolor": "white",
        }
    )


def add_logo(fig, rect):
    """The Argus Array logo (branding/argus_array_logo.jpeg) in figure coordinates rect = (left, bottom, width, height);
    its aspect ratio is kept, anchored at the rect's left."""
    logo = plt.imread(BRANDING / "argus_array_logo.jpeg")
    ax = fig.add_axes(rect, anchor="W")
    ax.imshow(logo, interpolation="lanczos")
    ax.set_axis_off()
    return ax
