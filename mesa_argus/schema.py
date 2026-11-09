"""Every parameter of an observation: the one file that defines a run (examples/observations/template.toml is written
from this module: python -m mesa_argus.schema > examples/observations/template.toml).

Each parameter has a default, a unit, its allowed values (a range, a set of choices, or a text form) and the
justification for them, with the source. runconfig.py checks every value against these before any work starts.

Sources of the defaults:
- [argussim.*]: ArgusSim, read from it (argus_sim.config, argus_sim.coverage_exact.Schedule) when the file is
  resolved, so a run records exactly what it used. ArgusSim defines everything about taking the image. One exception:
  [argussim.packing_strategy] layout_file defaults to A170r_1200, not ArgusSim's thinB (its note says why).
- [hdps] and [[hdps.tasks]]: hdps's production pipeline, docker/demo/config.yaml (the Task chain and its settings),
  and each Task's constructor defaults (hdps.orchestrate.tasks). hdps processes the raw frames.
- [camera]: the toy camera (mesa_argus/camera.py): what ArgusSim does not define (electrons to ADU).
- the rest: this repository (the scene, the outputs), with the reason for each.

TOML has no null: "none" stands for an unset optional value.
"""
from dataclasses import dataclass, field
from typing import Any, Callable

NONE = "none"
REQUIRED = "required"
POSITIVE = "positive"  # allowed: > 0


@dataclass
class P:
    """One parameter. allowed: (lo, hi) for numbers (None: open), a set of choices, a callable(value, resolved) that
    returns an error text or None, or None (any value of the default's type). why: the justification and source."""
    key: str
    default: Any
    unit: str
    allowed: Any
    why: str
    optional: bool = False  # "none" allowed
    allowed_text: str = ""  # how a callable `allowed` reads in the template


def argus(path):
    """The live ArgusSim default at dotted path (e.g. "telescope.readnoise")."""
    from argus_sim import c

    obj = c
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def schedule(name):
    """ArgusSim's ratchet schedule default (coverage_exact.Schedule)."""
    from argus_sim.coverage_exact import Schedule

    return getattr(Schedule(), name)


def layouts():
    """ArgusSim's registered ring layouts (argus_sim.ring_layout.LAYOUTS)."""
    from argus_sim.ring_layout import LAYOUTS

    return set(LAYOUTS)


def sensor_px():
    from argus_sim import c

    return c.telescope.n_pixels_x, c.telescope.n_pixels_y


def minipix_arcsec():
    from argus_sim import skymap_shim as sh

    return sh.MINIPIX_PIXEL_SIZE * sh.PIXEL_SCALE_DEG * 3600


# ---------------------------------------------------------------- this repository: the field, nights, scene, outputs
FIELD = [
    P("ra", REQUIRED, "deg", (0.0, 360.0), "ICRS right ascension of the field (the block's centre)."),
    P("dec", REQUIRED, "deg", (-90.0, 90.0), "ICRS declination; whether Argus can see it at all is ArgusSim's coverage "
      "(observe.py stops when no exposure covers it)."),
    P("name", "", "", None, "Output folder name; \"\": the observation file's name."),
    P("block", 9, "minipix", (1, 64), "Side of the block shown in the movie and followed in the summary, in hdps "
      "minipixes (32 tile px, 28.1\"). At most 64: one hdps tile (2048 px) is 64 x 64 minipixes."),
    P("window", 56, "minipix", lambda v, r: _window_check(v),
      "Side of the simulated readout window, in minipixes, centred on the field's position on the sensor (shifted to "
      "stay on it); 0: the whole sensor (ArgusSim telescope.n_pixels_x x n_pixels_y). A real camera can read out a "
      "window (XORIGIN, YORIGIN, XSIZE, YSIZE in its header). At least 28 (782 px): hdps's source detector needs 3 of "
      "its 256 px background regions across. Default 56 (1564 px, 26.3'): of the windows tried, the one hdps handles "
      "best end to end (16 exposures of f294, 2026-10-21: WCS solved 13/16, every touched tile reprojected 11/16; 28: "
      "14/16 and 0/16, 40: 14/16 and 7/16, 80: 11/16 and 0/16; production processes whole frames). The field's Gaia "
      "stars must reach the window's far corner: radius ~0.5 deg for 56.",
      allowed_text="0, or 28 .. the sensor's short side (313 for ArgusSim's default sensor)"),
    P("window_offset", [0.0, 0.0], "arcmin", lambda v, r: _window_offset_check(v, r),
      "Where the readout window's centre lies from the field (the block's centre), [east, north]; the block stays on "
      "the field, only the window moves. [0, 0]: the window centred on the field. A real Argus frame is centred on the "
      "telescope's pointing, not on a target, so a target generally sits off-centre. hdps's linear WCS solver starts "
      "from the detections in a circle about the frame centre; with a crowded target there (M3's core) it accepts "
      "false alignments with the pixel scale 5-15x wrong and every tile is dropped (docs/ARGUS_INTEGRATION.md, "
      "2026-10-24), so such a target is observed off-centre. Limits: the block stays inside the window, and the "
      "window's far corner within the field's Gaia stars (0.52 deg); 0 for the whole sensor (window = 0).",
      allowed_text="[east, north]: the block inside the window, the far corner within 0.52 deg"),
]


def _window_offset_check(v, r):
    if not (isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                                        for x in v)):
        return "[east, north] in arcmin"
    section = (r or {}).get("field", {})
    window, block = section.get("window", 56), section.get("block", 9)
    if not window:
        return None if v == [0, 0] else "[0, 0] with the whole sensor (window = 0)"
    room = (window - block) / 2 * minipix_arcsec() / 60  # arcmin the block can move from the window's centre
    if max(abs(v[0]), abs(v[1])) > room:
        return f"each within {room:.1f}' (the {block}-minipix block inside the {window}-minipix window)"
    import math

    corner = math.hypot(abs(v[0]) + window / 2 * minipix_arcsec() / 60, abs(v[1]) + window / 2 * minipix_arcsec() / 60)
    from mesa_argus.gaia import RADIUS_DEG

    return None if corner <= RADIUS_DEG * 60 else f"the window's far corner {corner:.1f}' beyond the field's Gaia stars ({RADIUS_DEG * 60:.0f}')"


def _only_nights_check(v, r):
    nights = (r or {}).get("night", {}).get("nights", 7)
    if not isinstance(v, list) or not all(isinstance(k, int) and not isinstance(k, bool) and 1 <= k <= nights for k in v):
        return f"[] or night numbers 1..{nights}"
    return None if len(set(v)) == len(v) else "each night once"


def _window_check(v):
    if v == 0:
        return None
    nx, ny = sensor_px()
    from argus_sim import c

    most = int(ny * c.telescope.plate_scale / minipix_arcsec())
    return None if 28 <= v <= most else f"0 (whole sensor) or 28..{most}"


NIGHT = [
    P("date", "auto", "", lambda v, r: None if v == "auto" or _is_date(v) else "YYYY-MM-DD or \"auto\"",
      "Local date of the first night (the night starts on that evening); \"auto\": the night the placed real stars "
      "change most.", allowed_text="YYYY-MM-DD | \"auto\""),
    P("nights", 7, "nights", (1, 366), "Consecutive nights observed. 7 is the standard observation (a week); up to a "
      "year."),
    P("only_nights", [], "", lambda v, r: _only_nights_check(v, r), "Observe only these of the nights (1: the first): "
      "the scene (transient times over all the nights), the reference and every exposure are made as for all of them "
      "and only these nights' exposures are observed, so runs of different nights add up to one observation of all of "
      "them (a week split across machines). Such a run draws its frames' noise from output.noise_seed + its first "
      "night here (the engine seeds a frame by its order in the run, so runs of different nights would otherwise repeat "
      "each other's noise); its reference keeps output.noise_seed. []: every night.",
      allowed_text="[] or a list of night numbers 1..nights"),
    P("seed", 0, "", (0, 2**31 - 1), "Seed of ArgusSim's per-ratchet draws (seasonal zenith seeing); the same seed "
      "makes the same nights."),
    P("reference_nights", 1, "nights", (1, 200), "Nights observed before the first night to make hdps's references "
      "(its coadd bootstrap: pass 1 coadds these nights' frames with CoaddTapTask, pass 2 subtracts against those "
      "coadds; docker/demo/config.yaml header). Same stars, transients left out. 1: the night before. Up to 200 for "
      "reference_geometry_px: a telescope's view of a field repeats only every few weeks (f294 2026-09-10: every "
      "telescope within 122 px of tonight's sensor position on 07-31, within ~257 px on 08-23)."),
    P("reference_ratchets", 0, "ratchets", (0, 100), "The reference nights' ratchets each telescope keeps for its "
      "references: its N with the best seeing at the field (the ratchet's zenith seeing x airmass^0.6, ArgusSim's), per "
      "band; the other exposures are not observed. SFFT matches the reference to the frame by convolving it, so a frame "
      "sharper than its reference cannot be matched and its stars stay in the difference (M67 beta 2026-10-27: a "
      "frame's bright-star residual against a 1-night reference of all ratchets went from +17% of the star at 0.85\" to "
      "-10% at 2.1\", correlation -0.96 with the frame's seeing; ~680 hdps candidates per frame). Seeing is drawn per "
      "ratchet, so a cut within one night drops whole cameras (they then take another camera's coadd): use with "
      "reference_nights > 1, as Argus's references will be. 0: every ratchet (as before 2026-10-27)."),
    P("reference_geometry_px", 0, "px", (0, 5000), "Each telescope's reference from its own ratchets that saw the field "
      "where tonight's does on its sensor: within this many pixels of the field centre's sensor position in that "
      "telescope's ratchet tonight (the reference_ratchets best-seeing of them, those at least as sharp as tonight's "
      "first; all of them with reference_ratchets = 0); telescopes not observing the field tonight are not observed. "
      "A telescope's view changes from night to night (f294: 0.3-1.9 deg off-axis), and with it the PSF's width and "
      "shape (ArgusSim's optics: 2.6 px near the axis to 3.2 px at 1.9 deg, tile pixels); a reference from another "
      "position left each star's core +5 to +12% (frames sharper than it) or -4 to -10% (broader) of its peak in the "
      "difference (f294 beta 2026-09-10, iter3z), red or blue by telescope. 300 px is ~0.08 deg. None within it: the "
      "telescope's nearest ratchet, with a warning. 0: off (reference_ratchets alone)."),
    P("reference_oracle", False, "", {True, False}, "TEST ONLY: the reference from the science night's own exposures "
      "(another noise draw: output.noise_seed + 1), the transients left out, instead of reference_nights before it. The "
      "best any reference could be: what it leaves at constant stars is the subtraction's own floor (hdps's kernel fit, "
      "the reprojection of an undersampled PSF), which no choice of reference removes. Never for data."),
    P("reference_oracle_seeing", 1.0, "", (0.5, 2.0), "TEST ONLY, with reference_oracle: the reference's seeing x this "
      "(its ratchets' zenith seeing scaled): whether a reference sharper (< 1) or broader than the frames, all else "
      "the same, leaves the stars' cores in the difference."),
    P("reference_oracle_shift_px", 0.0, "px", (0.0, 2.0), "TEST ONLY, with reference_oracle: the reference's tracking "
      "angle moved so the stars land this many sensor pixels along the sky's east-west from tonight's: whether the "
      "stars' sub-pixel positions (an undersampled PSF, ~2.6 tile pixels FWHM) alone, all else the same, leave their "
      "cores in the difference."),
]


def _is_date(v):
    from datetime import date

    try:
        date.fromisoformat(str(v))
        return True
    except ValueError:
        return False


STARS = [
    P("background", REQUIRED, "", None, "The star set (stars/<name>/) every field star comes from: Gaia stars of the "
      "field built by make_stars.py."),
    P("layout", "real", "", {"real", "random"}, "real: each star at its Gaia position; random: scattered uniformly "
      "over the window (a synthetic field)."),
    P("seed", 1, "", (0, 2**31 - 1), "Seed of the random layout and random placements."),
    P("missing", "stop", "", {"stop", "skip"}, "A placed star not built yet: stop the run, or skip it."),
    P("photometry_workers", 1, "", (1, 64), "Processes for ArgusSim's photometry of the constant stars (each MESA "
      "template through both bands at its stars' E(B-V) and every tabulated airmass), forked from the engine (Linux; "
      "macOS forks unsafely with Accelerate: keep 1 there). The results do not depend on it. 1: in the engine's own "
      "process; f065's 35k stars took 67 min of each pass (reference and night) on newton that way."),
]

PLACE = [
    P("star", REQUIRED, "", None, "A built star (stars/<name>/)."),
    P("x", 50.0, "%", lambda v, r: None if v == -1 or -1000 <= v <= 1000 else "-1000..1000, or -1 (random)",
      "Position across the block, percent from its left edge (outside 0..100: beyond the block, in the readout "
      "window, as [inject] places them); -1: random in the block. Ignored for a real Gaia star (its own position).",
      allowed_text="-1000 .. 1000 | -1"),
    P("y", 50.0, "%", lambda v, r: None if v == -1 or -1000 <= v <= 1000 else "-1000..1000, or -1 (random)",
      "Position up the block, percent from its bottom edge (as x); -1: random in the block.",
      allowed_text="-1000 .. 1000 | -1"),
    P("t0", NONE, "", None, "Transients: the model's t0 (ISO UTC, \"start+D\" days after the first exposure, or BJD).",
      optional=True),
    P("epoch", NONE, "", None, "Periodic stars: a primary eclipse / maximum (as t0).", optional=True),
    P("distance", NONE, "pc", (0.0, 1e7), "Distance for the dust in front of it (the 3D map); none: behind all the "
      "dust in its sightline, or the star's own.", optional=True),
]

CAMERA = [
    P("gain", None, "e-/ADU", (0.01, 100.0), "Toy: ArgusSim defines the detector in electrons only. Default: "
      "ArgusSim's full well (telescope.fullwell) at hdps's raw saturation level (saturation_adu), so a full pixel is "
      "where hdps flags saturation; 0.373 for 21657 e- (hdps's prototype QHY600M: 0.33)."),
    P("offset", 0, "ADU", (0, 65535), "Toy: bias level (none in ArgusSim)."),
    P("saturation_adu", 58000.0, "ADU", (1.0, 65535.0), "hdps.calibration.ImageReducer's saturation_value: raw "
      "pixels at or above it are flagged saturated."),
]

def _radii(v, lo, hi):
    """A pair [inner, outer] of radii (px) with lo <= inner < outer <= hi."""
    ok = (isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)
          and lo <= v[0] < v[1] <= hi)
    return None if ok else f"[inner, outer] with {lo} <= inner < outer <= {hi}"


def _join_check(v, r):
    err = _radii(v, 2, 13)
    if err:
        return err
    fit = (r or {}).get("image", {}).get("psf_wing_fit", [6, 11])
    return None if v[0] >= fit[0] else f"the blend starts at or beyond the fitted radii (psf_wing_fit {fit})"


def _box_check(v, r):
    if not (isinstance(v, int) and not isinstance(v, bool) and 28 <= v <= 301):
        return "an integer, 28 <= psf_box <= 301"
    return None if v == 28 or not (r or {}).get("image", {}).get("psf_wing") else "psf_box above 28 with psf_wing"


def _halo_check(v, r):
    if v not in (True, False):
        return "true or false"
    return None if not v or (r or {}).get("image", {}).get("psf_box", 28) > 28 else "psf_halo needs psf_box above 28"


def _halo_params(v, r):
    ok = (isinstance(v, (list, tuple)) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v)
          and 0 <= v[0] < 1e-2 and 1 <= v[1] <= 200 and 1 < v[2] <= 20)
    return None if ok else "[A, r0, beta], 0 <= A < 1e-2, 1 <= r0 <= 200, 1 < beta <= 20"


IMAGE = [  # this repository's renderer (mesa_argus/sensor.py: ArgusSim has no image renderer)
    P("psf_wing", False, "", {True, False}, "false: ArgusSim's PSF stamp as it is; it ends at ArgusSim's Zemax grid "
      "(28 x 28 px), so a bright star sits in a visible square (its light stops at ~2e-5 of the star per pixel). "
      "true: the stamp continued by a power law fitted to its own outer profile (mesa_argus/sensor.py PSFs), drawn "
      "out to where it falls below psf_wing_floor. Default false: runs as before 2026-10-26 (the v1 alert sample)."),
    P("psf_wing_fit", [6, 11], "px", lambda v, r: _radii(v, 2, 12), "Radii (from the stamp's centroid) whose mean "
      "profile is fitted with A r^-alpha: beyond the seeing core (3 FWHM at 2\" seeing), inside the stamp (its edge "
      "is 13.5-14 px from the centroid; 12 keeps the last annulus whole). Over 6-11 px the slope is -2.7 to -3.6 "
      "(1-2\" seeing, both bands, every field angle), near diffraction's -3.",
      allowed_text="[inner, outer], 2 <= inner < outer <= 12"),
    P("psf_wing_join", [10, 13], "px", _join_check, "The star is drawn as the stamp inside the inner radius, the "
      "fitted law beyond the outer, a linear blend between (no seam where the stamp ends). The outer radius at most "
      "13: the stamp reaches 13.5 px from its centroid in every direction; the inner at or beyond the fit's inner "
      "radius, where the law describes the stamp.", allowed_text="[inner, outer], 2 <= inner < outer <= 13, inner >= "
      "psf_wing_fit's inner"),
    P("psf_wing_radius", 400, "px", (14, 2000), "The PSF is normalised to unit flux within this radius and drawn at "
      "most this far out. At 400 px the r^-3 wing of a beta ~3 star (a Pleiad, ~1e10 e- in 60 s) is ~1e1 e- per "
      "pixel, below the sky's photon noise (~30 e-); the light beyond (2 pi A / R for r^-3) is ~0.1% of a star's. "
      "Larger costs time only for the brightest stars. At least 14: beyond the stamp."),
    P("psf_wing_floor", 0.5, "e-/px", POSITIVE, "A star's wing is drawn out to where it falls to this: well below "
      "the sky's photon noise per pixel (> 15 e- in 60 s even at new Moon) and the read noise, so leaving out the "
      "rest changes nothing measurable while faint stars cost nothing extra. Positive."),
    P("psf_box", 28, "px", _box_check, "The PSF stamp's side. 28: ArgusSim's stamp (survey.stamp_size) as it is, "
      "cut from its Zemax array of 29 px after the seeing, so a bright star's light stops at a square edge 14 px out "
      "(~2e-5 of the star per pixel there: ~200 e- for a 1e7 e- star). Larger: ArgusSim's same PSF (Zemax optics, "
      "delivered-EE50 blur, Moffat seeing) made with the Zemax array padded first, so the seeing blurs that edge "
      "instead of cutting it, and room for psf_halo. The optics data itself still ends 14.5 px out (the light falls "
      "~1000x between 15 and 18 px): without psf_halo the square only softens. 121 holds the halo to 60 px (~70 tile "
      "px), where it is ~1e-9 of the star per pixel. Default 28: runs as before 2026-10-26. Not with psf_wing (that "
      "continues the 28 px stamp).", allowed_text="an integer, 28 <= psf_box <= 301; above 28 only with psf_wing = "
      "false"),
    P("psf_box_floor", 0.5, "e-/px", POSITIVE, "With psf_box above 28, each star is drawn in the square outside which "
      "its PSF is below this: as psf_wing_floor, below the sky's photon noise and the read noise, so faint stars keep "
      "a 29 px square and cost nothing extra (the light they leave out is below the floor in every pixel and outside "
      "hdps's aperture). Positive."),
    P("psf_halo", False, "", _halo_check, "true: the delivered PSF's static halo, as measured on sky, added to "
      "ArgusSim's PSF after the seeing (needs psf_box above 28). Measured by H. Corbett (synthref halo_measure.py, "
      "2026-09: hdps tile 363381, a 15 x 60 s stack, unsaturated non-variable Gaia XP stars of G 8.5-12.5, every other "
      "star removed, fitted over 8-60 tile px): beyond ~10 tile px the real PSF lies well above the Zemax model "
      "(~3e-5 of the star per pixel at 15 tile px, ~1e-5 at 20, ~2e-6 at 30), and the halo holds 3.6% (beta) / 4.0% "
      "(rho) of the core's light within 60 tile px. This continues the PSF past the Zemax grid, so the square goes. "
      "Default false: runs as before 2026-10-26.", allowed_text="true or false; true only with psf_box above 28"),
    P("psf_halo_beta", [4.9e-5, 34.1, 6.0], "[per tile px, tile px, ]", _halo_params, "The beta halo's A, r0, beta "
      "in A [1 + (r/r0)^2]^-beta, per hdps tile pixel (0.879\") as a share of the star's core light, r from the star "
      "in tile pixels: Corbett's fit (synthref figs/halo_profile.png, beta, 21 stars).",
      allowed_text="[A, r0, beta], 0 <= A < 1e-2, 1 <= r0 <= 200, 1 < beta <= 20"),
    P("psf_halo_rho", [8.0e-5, 28.3, 6.0], "[per tile px, tile px, ]", _halo_params, "As psf_halo_beta, for rho: "
      "Corbett's fit (synthref figs/halo_profile.png, rho, 11 stars).",
      allowed_text="[A, r0, beta], 0 <= A < 1e-2, 1 <= r0 <= 200, 1 < beta <= 20"),
    P("psf_halo_taper", [10, 14], "px", lambda v, r: _radii(v, 2, 14), "With psf_halo, ArgusSim's Zemax optics are "
      "tapered (cos^2) from 1 at the inner radius to 0 at the outer, a circle inside their 29 px square, before the "
      "seeing; the halo carries the light beyond. Without it the square's corners (14.5-20.5 px out) keep optics light "
      "the sides lack, and a bright star shows a square inside its halo. Corbett's model that matched the measured "
      "PSF is the same: a +-12 px optical stamp plus the halo. The taper removes ~0.1% of the light (renormalised).",
      allowed_text="[inner, outer], 2 <= inner < outer <= 14"),
    P("psf_seeing_step", 0.0, "arcsec", (0.0, 0.1), "Each exposure's PSF is made for its seeing rounded to this, so "
      "exposures within a step share one set of stamps. ArgusSim's seeing (a ratchet's zenith seeing x airmass^0.6) "
      "differs in every exposure, and with psf_box above 28 a new seeing costs ~0.3 s per tabulated field angle (11 "
      "across a 56-minipix window): 3.7 s of a 50,000-star frame's 4 s on the Mac. 0.02 changes a ~1\" FWHM by at "
      "most 1%. 0: the exact seeing (runs as before 2026-10-27). At most 0.1 (5% at 1\").",
      allowed_text="0.0 .. 0.1"),
]

LIGHTCURVES = [
    P("link_arcsec", 1.5, "arcsec", (0.1, 5.3), "A detection joins the nearest object within this. Above ~1 px "
      "(1.0\") so centroid noise links; below hdps's de-duplication radius (6 px, 6.0\"), inside which two "
      "detections in one frame cannot both exist."),
]

TRUTH = [
    P("match_radius", 2.0, "px", (0.5, 6.0), "Scoring only: a detection or candidate is a simulated star within "
      "this (tile px). Below hdps's de-dup radius (6 px)."),
    P("alert_radius", 2.0, "arcsec", (0.5, 6.0), "Scoring only: an alert is the simulated star within this of its "
      "position (hdps's tile WCS against the true position; the median WCS error is ~0.1\"). About two pixels; "
      "below the 6 px de-dup radius so a neighbour is not taken."),
]

FAMILIES = ["libnm_sncosmo", "libnm_salt_shape", "libnm_microlensing", "libnm_dwarf_nova", "libnm_contact",
            "libnm_detached_ms", "libnm_detached_evolved", "lib_rrab", "lib_rrc"]
INJECT = [  # library stars scattered over the readout window: labelled real sources for a real/bogus training set
    P("number", 0, "", (0, 5000), "Library stars injected at random (0: none). For training data: many per window, "
      "so each exposure holds many labelled real sources; not a sky abundance."),
    P("families", FAMILIES, "", lambda v, r: None if isinstance(v, list) and v else "a non-empty list of star-name "
      "prefixes", "Library families drawn from, equally per family (stars/<prefix>_*: supernovae, microlensing, "
      "dwarf novae, eclipsing and contact binaries, RR Lyrae).", allowed_text="list of stars/ name prefixes"),
    P("mag_min", 13.0, "mag", (5.0, 25.0), "Periodic stars: mean Gaia G drawn uniformly from mag_min..mag_max (the "
      "distance set to give it, then the 3D dust at that distance). 13: below the 60 s saturation (~11-12)."),
    P("mag_max", 19.5, "mag", (5.0, 25.0), "Periodic stars: faintest mean G; ~19.5 is about a single frame's limit, "
      "so the faint end is mostly undetected (labelled, dropped from training). Transients keep their model's own "
      "brightness (redshift, baseline, distance)."),
    P("margin_arcsec", 60.0, "arcsec", (0.0, 600.0), "Kept from the readout window's edge (hdps's detector and SFFT "
      "edges)."),
    P("separation_arcsec", 20.0, "arcsec", (0.0, 300.0), "Least separation between injected stars: ~20 px, beyond "
      "hdps's de-dup radius and the alert stamp, so each candidate is one injected star."),
    P("seed", 1, "", (0, None), "Random draws (templates, positions, times)."),
]

OUTPUT = [
    P("bands", ["beta", "rho"], "", lambda v, r: None if v and set(v) <= {"beta", "rho"} else "beta and/or rho",
      "Argus bands kept (ArgusSim's filter_strategy has b, named g there, and rho).",
      allowed_text="a non-empty list of \"beta\", \"rho\""),
    P("movie", True, "", {True, False}, "Render the movie (the third output)."),
    P("catalogues", True, "", {True, False}, "Keep hdps's per-frame source catalogues."),
    P("alerts", True, "", {True, False}, "Run difference imaging and keep hdps's alert products."),
    P("lightcurves", True, "", {True, False}, "Link detections into catalogue.csv and lightcurves/ (needs "
      "catalogues)."),
    P("pipeline_movie", True, "", {True, False}, "Also render the pipeline movie (each raw frame on its sensor, hdps's "
      "detections and WCS); drawn from the same run, about 15-25% more time."),
    P("raw", False, "", {True, False}, "Keep the raw camera frames (raw/); they are always written for hdps, and "
      "deleted after unless kept."),
    P("movie_data", True, "", {True, False}, "Save what both movies need (<name>_night_<band>.movie.joblib: each "
      "frame's block cut-outs, a 4 x 4-binned raw-frame preview, detections, WCS, measurements, alerts; ~1 MB per "
      "frame), so they can be drawn after the run (tools/render_movies.py) with movie = false during it."),
    P("movie_window_px", 800, "pixels", lambda v, r: None if isinstance(v, int) and not isinstance(v, bool) and
      (v == 0 or 100 <= v <= 4000) else "0, or an integer 100-4000", "Field movie: its Exposure and Difference panels "
      "show the whole readout window, hdps's images of every tile it touches (TileImageOutputTask tile_ids: all, when "
      "the movie or its data is made) on the block's tile grid, each pixel the mean of a power-of-two square of tile "
      "pixels, the smallest that brings the frames' windows (their union) to at most this many pixels across. 800: "
      "f294's 56-minipix windows (placed on 8 telescopes' sensors) span ~3100 tile pixels -> 4 x 4, 730 x 785 pixels, "
      "~2 MB per frame more in the movie data; the panel is ~300 screen pixels at dpi 100 (drawn antialiased). Disk "
      "while the run lasts: 2 x 16 MB per tile per frame (f294: 4 tiles, 128 MB per frame against 32 MB for the "
      "block's tile alone; deleted at the end unless raw). 0: the block alone, from the tile it is in.",
      allowed_text="0, or 100-4000"),
    P("movie_window_zoom", 1.0, "", (1.0, 10.0), "Field movie, with movie_window_px: its panels show the central "
      "1/zoom of the whole-window view, about the block, and the images cover only that, binned as movie_window_px "
      "says (so finer: f294 at 3, ~956\" across, 2 x 2 tile pixels, ~545 pixels; at 1, 4 x 4); hdps writes only the "
      "tiles that view reaches. 1: the whole window (all of every frame's window, the view centred on them). Up to 10: "
      "f294's view is then ~290\", about the 9 x 9 block itself (closer: movie_window_px = 0, the block's own view)."),
    P("movie_views", ["window", "block"], "", lambda v, r: None if isinstance(v, list) and v and set(v) <= {"window", "block"}
      else "a non-empty list of \"window\", \"block\"", "Field movies drawn from one run (each from its movie data, so "
      "tools/render_movies.py draws them all again): window, the whole readout window or its zoom about the block "
      "(movie_window_px, movie_window_zoom; <name>_night_<band>.mp4); block, the 9 x 9 block's own tile at hdps's "
      "pixels (<name>_night_<band>_block.mp4; the window view instead when movie_window_px = 0). The pipeline movie is "
      "pipeline_movie.", allowed_text="list of \"window\", \"block\""),
    P("movie_mask_reference_e", 8000.0, "electrons", (0.0, None), "Field movies' Difference panel: pixels about the "
      "stars brighter than this in the frame's own reference coadd (electrons per tile pixel above its background; the "
      "coadd hdps subtracted, by its rule: the frame's camera, the deepest) are drawn grey. Saturated and bright stars "
      "keep a few % of their light in hdps's difference whatever the reference (f294 beta iter5, a reference of the "
      "night's own frames: the ~130 brightest constant stars +4% net, the rest 0.1-0.3%), and draw the eye; the "
      "transients are not in the reference and stay. Display only: hdps, its alerts and the movie data are unchanged. "
      "8000: about a third of the 60 s full well (21657 e-). 0: off."),
    P("movie_mask_radius_px", 4, "pixels", (0, 50), "The grey disc about each masked pixel (tile pixels, 0.879\") at "
      "movie_mask_reference_e; ~1.5 x the PSF's FWHM (2.6-3.2 tile pixels)."),
    P("movie_mask_growth_px", 2, "pixels", (0, 20), "The disc's growth per magnitude a pixel is brighter than "
      "movie_mask_reference_e in the reference (a star's residual reaches further the brighter it is; hdps's "
      "bright-star mask grows the same way, SFFTSubtractionTask mask_dilation_growth)."),
    P("stride", 1, "", (1, 1000), "Movie: every Nth frame."),
    P("fps", 24, "frames/s", (1, 120), "Movie frame rate."),
    P("dpi", 100, "", (50, 300), "Movie resolution."),
    P("noise_seed", 3, "", (0, 2**31 - 1), "Seed of the photon and read noise."),
]

# ---------------------------------------------------------------- ArgusSim: everything about taking the image
# Ranges: ArgusSim validates none of these, so each range is what the quantity can physically be; an open bound (no
# upper limit) where nothing limits it.
_A = "ArgusSim argus_sim/config.py"
ARGUSSIM = {
    "observatory": [
        P("latitude", argus("observatory.latitude"), "deg", (-90.0, 90.0), f"Site latitude, Mt Fowlkes ({_A}). A "
          "latitude."),
        P("longitude", argus("observatory.longitude"), "deg", (-180.0, 180.0), f"Site longitude, east positive "
          f"({_A}). A longitude."),
        P("altitude", argus("observatory.altitude"), "m", (-500.0, 9000.0), f"Site height ({_A}). Between the "
          "lowest and highest land on Earth."),
        P("min_alt", argus("observatory.min_alt"), "deg", (0.0, 90.0), f"Lowest altitude the array points at "
          f"({_A}). Above the horizon."),
        P("seasonal_seeing", argus("observatory.seasonal_seeing"), "", {True, False}, f"Zenith seeing per ratchet "
          f"from the site's DIMM seasons (Barker et al. 2003; {_A}, PI 2026-09-19)."),
        P("seeing_mean", argus("observatory.seeing_mean"), "arcsec", POSITIVE, f"Lognormal seeing, used only "
          f"when seasonal_seeing is false ({_A}). Positive."),
        P("seeing_std", argus("observatory.seeing_std"), "arcsec", (0.0, None), f"As seeing_mean ({_A}). Not "
          "negative."),
        P("transparency", argus("observatory.transparency"), "", (0.0, 1.0), f"Grey transparency, 1 = photometric "
          f"({_A}). A transmitted fraction."),
        P("artificial_light_cd_m2", argus("observatory.artificial_light_cd_m2"), "ucd/m2", (0.0, None),
          f"Artificial sky glow ({_A}). Not negative."),
        P("solar_activity", argus("observatory.solar_activity"), "", (0.8, 2.0), f"Airglow scaling when "
          f"survey.solar_activity_prior is \"fixed\"; ArgusSim's stated range, 0.8 (solar minimum) to 2.0 (maximum) "
          f"({_A})."),
    ],
    "telescope": [
        P("aperture_diameter", argus("telescope.aperture_diameter"), "mm", POSITIVE, f"({_A}). Positive."),
        P("central_obstruction_diameter", argus("telescope.central_obstruction_diameter"), "mm", (0.0, None),
          f"({_A}). Not negative; smaller than the aperture."),
        P("focal_length", argus("telescope.focal_length"), "mm", POSITIVE, f"({_A}); with pixel_size, the plate "
          "scale. Positive."),
        P("pixel_size", argus("telescope.pixel_size"), "um", POSITIVE, f"({_A}). Positive."),
        P("n_pixels_x", argus("telescope.n_pixels_x"), "px", (768, None), f"Sensor long axis ({_A}). At least 768: "
          "hdps's source detector needs 3 of its 256 px background regions across."),
        P("n_pixels_y", argus("telescope.n_pixels_y"), "px", (768, None), f"Sensor short axis ({_A}). As "
          "n_pixels_x."),
        P("readnoise", argus("telescope.readnoise"), "e-", (0.0, None), f"({_A}). A noise: not negative."),
        P("dark_current", argus("telescope.dark_current"), "e-/s/px", (0.0, None), f"({_A}). A rate: not "
          "negative."),
        P("fullwell", argus("telescope.fullwell"), "e-", (1, 65535 * 100), f"({_A}). Positive; with [camera] gain "
          "it must fit the 16-bit raw frame (fullwell / gain <= 65535), checked with [camera]."),
        P("throughput_loss", argus("telescope.throughput_loss"), "", (0.0, 1.0), f"Multiplicative throughput "
          f"degradation ({_A}). A fraction."),
        P("psf_model", argus("telescope.psf_model"), "", {"zemax", "gaussian"}, f"Delivered optics (zemax, the "
          f"default since 2026-09-19) or a Gaussian spot; ArgusSim's two models ({_A})."),
        P("spot_size", argus("telescope.spot_size"), "um", POSITIVE, f"RMS spot radius of the Gaussian PSF "
          f"({_A}). Positive."),
        P("ee50_max_px", argus("telescope.ee50_max_px"), "px", POSITIVE, f"Cap on the delivered EE50 diameter "
          f"({_A}); none: uncapped. Positive.", optional=True),
        P("randomize_pixel_phase", argus("telescope.randomize_pixel_phase"), "", {True, False}, f"({_A})"),
    ],
    "packing_strategy": [
        P("layout_file", "A170r_1200", "", layouts(), "The array's ring layout, by its name in ArgusSim "
          "(argus_sim.ring_layout.LAYOUTS): every telescope's pointing (alt/az), tel_id, subarray and band, which "
          "coverage_exact.covering observes with. Default A170r_1200, not ArgusSim's own default (thinB, 1190 "
          "telescopes): the canonical layout (argus-hdps/array_arrangement 9206cc3, PI 2026-09-23; 1200 telescopes on "
          "8 subarrays, its band table 600 beta / 600 rho), adopted here 2026-10-28; runs before then used thinB. "
          "observe.py records it in scene/exposures.ecsv, and the engine stops when the two differ (each exposure's "
          "telescope is a row of this layout). The *-strips layouts assign bands by ArgusSim's strip rule instead of "
          "the table; legacy1202 is the pre-thinning layout."),
    ],
    "survey": [
        P("min_sun_alt", argus("survey.min_sun_alt"), "deg", (-90.0, 0.0), f"Observing only with the Sun below this "
          f"({_A}). Below the horizon."),
        P("min_moon_sep", argus("survey.min_moon_sep"), "deg", (0.0, 180.0), f"({_A}). An angle on the sky."),
        P("solar_activity_prior", argus("survey.solar_activity_prior"), "", {"epoch", "fixed"}, f"Per-night solar "
          f"activity: ArgusSim's 2028-2033 prior, or observatory.solar_activity ({_A})."),
        P("apply_weather", argus("survey.apply_weather"), "", {True, False}, f"Nights gated by the site's clear "
          f"fractions ({_A}, PI 2026-09-19)."),
    ],
    "schedule": [
        P("exptime_s", schedule("exptime_s"), "s", POSITIVE, "Exposure time (ArgusSim coverage_exact.Schedule). "
          "Positive; at most track_min, checked with it."),
        P("track_min", schedule("track_min"), "min", POSITIVE, "Tracking per ratchet (Schedule). Positive."),
        P("reset_min", schedule("reset_min"), "min", (0.0, None), "Reset per ratchet, no exposures (Schedule). Not "
          "negative."),
        P("theta_start_deg", schedule("theta_start_deg"), "deg", (-180.0, 180.0), "Tracking angle at a ratchet's "
          "start (Schedule). An angle."),
        P("sun_alt_max_deg", schedule("sun_alt_max_deg"), "deg", (-90.0, 0.0), "Exposures with the Sun below this "
          "(Schedule). Below the horizon."),
    ],
}

# ---------------------------------------------------------------- hdps: the pipeline
_H = "hdps docker/demo/config.yaml"
HDPS = [
    P("refcat", "~/.rico_cache/atlas_refcat2", "", None, f"ATLAS-Refcat2 for ReferenceRetrievalTask ({_H}: "
      "RICO_CACHE_DIR/atlas_refcat2)."),
]
HDPS_COADD = [  # the config's top-level coadd section (the worker's accumulator)
    P("strategy", "inverse_variance", "", {"inverse_variance", "mean"}, f"Frame weighting in the coadd ({_H})."),
    P("clip_kappa", 3.0, "sigma", POSITIVE, f"Outlier clip on the sky-subtracted residual ({_H}). Positive."),
    P("clip_min_frames", 3, "frames", (1, None), f"Fewest frames before clipping ({_H}). At least 1."),
    P("accumulation_dtype", "float32", "", {"float32", "float64"}, f"({_H}: float32, sub-1e-6 relative difference "
      "from float64)."),
]


def _task(type_, values):
    return dict(type=type_, config=values)


# The production chain (docker/demo/config.yaml), every constructor parameter written out. Changes from it, each a
# toy for something the simulation does not have: ImageReductionTask (the toy camera's gain and serial number, no
# masters: provider "file", no paths); ReferenceRetrievalTask pixscale (ArgusSim's plate scale, filled at resolve);
# AlertVettingTask enabled with no model bundle (real/bogus not run: every score 1). Two hdps output Tasks added for
# the outputs: LightweightFITSOutputTask (the solved frame's catalogues, WCS and zero point; as the AEGIS
# astrometric-reduction config, after TPVWCSSolverTask) and TileImageOutputTask (each differenced tile's science and
# difference planes, for the movies).
TASKS = [
    _task("DataCleanupTask", dict(target_dtype="float32", nan_replacement=0.0)),
    _task("ImageReductionTask", dict(camera_specs={}, bias_path=NONE, dark_path=NONE, flat_path=NONE,
                                     overscan_trimming=False, bias_subtract=False, gain=NONE, gpu_reduce=True,
                                     provider="file")),
    _task("SourceDetectionTask", dict(detect_sigma=3.0, bg_region_size=256, max_sources=10000000, flux_radius=4.0,
                                      matched_filter_width=4.0, centroid_n_iter=3, cr_detect_contrast=0.25,
                                      de_dup_sources=True, de_dup_radius=6.0, saturation_threshold=0.9999)),
    _task("SourceFilterTask", dict(min_x=0, min_y=0, max_x=NONE, max_y=NONE, min_snr=NONE)),
    _task("ReferenceRetrievalTask", dict(catalog_provider="ATLAS", radius_deg=2.5, min_mag=0.0, max_mag=19.1,
                                         pixscale=NONE, cacheable=True, cache_precision=0.1, cache_max_age_days=7.0)),
    _task("LinearWCSSolverTask", dict(n_stars=150)),
    _task("SourceMatchingTask", dict(n_sigma_clip=5.0, region_shape=[4, 6])),
    _task("TPVWCSSolverTask", dict(tpv_terms=17, niter=3, n_sigma_clip=5.0, use_annealing=True, use_chebyshev=True)),
    _task("LightweightFITSOutputTask", dict(output_dir="./solved")),
    _task("ReprojectionTask", dict(segment_shape=[2048, 2048], grid_resolution=33)),
    _task("CoaddTapTask", dict(background_mode="regional")),
    _task("SFFTSubtractionTask", dict(reference_mode="coadd", kernel_half_width=6, kernel_poly_order=0, background_order=2,
                                      mask_saturated_in_fit=True, mask_no_coverage_in_fit=True, mask_bright_stars=False,
                                      bright_star_threshold=3000.0, mask_dilation_growth=4, variance_conv="direct",
                                      rejection_iters=1, rejection_sigma=3.0, rejection_grow=2, retain_planes=True)),
    _task("DiffDetectionTask", dict(detect_sigma=5.0, bg_region_size=256, max_sources=100000,
                                    matched_filter_width=4.0)),
    _task("TransientOutputTask", dict(fmt="parquet")),
    _task("AlertVettingTask", dict(enabled=True, model_bundle="", detect_sigma=3.0, score_cut=0.5,
                                   emit_all_candidates=False, emit_stamps=True, stamp_size=31, emit_kafka=False,
                                   fmt="parquet")),
    _task("TileImageOutputTask", dict(enabled=True, output_dir="", science_output_dir="", difference_output_dir="",
                                      save_science=True, save_difference=True, save_difference_variance=False,
                                      compress=False, tile_ids=NONE, filename_prefix="epoch")),
]

# Toy Tasks a chain may use in place of hdps's own, never in the default chain (each documented in its module):
# TrueCalibrationTask, the simulation's WCS and zero point in place of ReferenceRetrievalTask, LinearWCSSolverTask,
# SourceMatchingTask and TPVWCSSolverTask (mesa_argus/true_calibration.py; user 2026-10-26: the solve is not needed
# and often fails, ~36% of exposures without a tile in the v1 alert sample). It takes no settings.
TOY_TASKS = [
    _task("TrueCalibrationTask", dict()),
]

# Allowed values and why, per (Task, setting). hdps itself validates only the enumerated modes (variance_conv, fmt,
# background_mode, ...: "hdps checks"); every other range is what the setting means, open where nothing limits it.
# Source of each value: the production config line (_H), else "hdps default" (the Task's constructor).
_SIG = "A threshold in sigma: positive."
_PX = "A size in pixels: positive."
_N = "A count: at least 1."
TASK_RULES = {
    ("DataCleanupTask", "target_dtype"): ({"float32", "float64"}, f"{_H}. A float type the later Tasks take."),
    ("DataCleanupTask", "nan_replacement"): (None, f"{_H}. Any value."),
    ("ImageReductionTask", "camera_specs"): (None, "Toy camera: filled at resolve ({serial_number}); no overscan "
                                             "geometry, so nothing is trimmed."),
    ("ImageReductionTask", "bias_path"): (None, "Toy camera: no masters (none)."),
    ("ImageReductionTask", "dark_path"): (None, "Toy camera: no masters (none)."),
    ("ImageReductionTask", "flat_path"): (None, "Toy camera: no masters (none)."),
    ("ImageReductionTask", "overscan_trimming"): ({True, False}, f"{_H} has true; the toy camera has no overscan."),
    ("ImageReductionTask", "bias_subtract"): ({True, False}, f"{_H}"),
    ("ImageReductionTask", "gain"): (None, "none: [camera] gain (the pipeline applies it once, here)."),
    ("ImageReductionTask", "gpu_reduce"): ({True, False}, "hdps default; falls back to the CPU without CuPy."),
    ("ImageReductionTask", "provider"): ({"file", "opsdb"}, f"{_H} uses opsdb (the ops database); file: no "
                                         "database, masters from paths. hdps checks."),
    ("SourceDetectionTask", "detect_sigma"): (POSITIVE, f"{_H}. {_SIG}"),
    ("SourceDetectionTask", "bg_region_size"): ((1, None), f"{_H}. {_PX} The window must be 3 regions across "
                                                "(checked with [field] window)."),
    ("SourceDetectionTask", "max_sources"): ((1, None), f"hdps default. {_N}"),
    ("SourceDetectionTask", "flux_radius"): (POSITIVE, f"hdps default (aperture radius). {_PX}"),
    ("SourceDetectionTask", "matched_filter_width"): (POSITIVE, f"{_H} (FWHM). {_PX}"),
    ("SourceDetectionTask", "centroid_n_iter"): ((1, None), f"hdps default. {_N}"),
    ("SourceDetectionTask", "cr_detect_contrast"): ((0.0, 1.0), "hdps default. A fraction of the peak."),
    ("SourceDetectionTask", "de_dup_sources"): ({True, False}, "hdps default."),
    ("SourceDetectionTask", "de_dup_radius"): (POSITIVE, f"hdps default. {_PX}"),
    ("SourceDetectionTask", "saturation_threshold"): ((0.0, 1.0), "hdps default. A fraction of 65534."),
    ("SourceFilterTask", "min_x"): ((0, None), "hdps default. A pixel index: not negative."),
    ("SourceFilterTask", "min_y"): ((0, None), "hdps default. A pixel index: not negative."),
    ("SourceFilterTask", "max_x"): (None, "hdps default (none: no limit)."),
    ("SourceFilterTask", "max_y"): (None, "hdps default (none: no limit)."),
    ("SourceFilterTask", "min_snr"): (None, "hdps default (none: no cut)."),
    ("ReferenceRetrievalTask", "catalog_provider"): ({"ATLAS"}, f"{_H}. The one provider with a local catalogue "
                                                     "here (Refcat2)."),
    ("ReferenceRetrievalTask", "radius_deg"): ((0.0, 180.0), f"{_H}. An angle on the sky; to cover a whole frame, "
                                               "at least the sensor's half-diagonal (2.0 deg for ArgusSim's)."),
    ("ReferenceRetrievalTask", "min_mag"): (None, "hdps default."),
    ("ReferenceRetrievalTask", "max_mag"): (None, f"{_H}; fainter than min_mag."),
    ("ReferenceRetrievalTask", "pixscale"): (None, "none: ArgusSim's plate scale (telescope pixel_size / "
                                             "focal_length); hdps's config has its prototype camera's 1.42."),
    ("ReferenceRetrievalTask", "cacheable"): ({True, False}, "hdps default."),
    ("ReferenceRetrievalTask", "cache_precision"): (POSITIVE, "hdps default (deg). Positive."),
    ("ReferenceRetrievalTask", "cache_max_age_days"): ((0.0, None), "hdps default. Not negative."),
    ("LinearWCSSolverTask", "n_stars"): ((3, None), f"{_H}. At least 3 stars fix a linear (6-parameter) solution."),
    ("SourceMatchingTask", "n_sigma_clip"): (POSITIVE, f"{_H}. {_SIG}"),
    ("SourceMatchingTask", "region_shape"): (None, f"{_H}: [rows, columns] of matching regions."),
    ("TPVWCSSolverTask", "tpv_terms"): (None, f"{_H}. A TPV term count (hdps's solver decides which it takes)."),
    ("TPVWCSSolverTask", "niter"): ((1, None), f"{_H}. {_N}"),
    ("TPVWCSSolverTask", "n_sigma_clip"): (POSITIVE, f"hdps default. {_SIG}"),
    ("TPVWCSSolverTask", "use_annealing"): ({True, False}, f"{_H}"),
    ("TPVWCSSolverTask", "use_chebyshev"): ({True, False}, f"{_H}"),
    ("ReprojectionTask", "segment_shape"): (None, "hdps default: the 2048 x 2048 px tile of hdps's sky "
                                            "segmentation; fixed by that scheme."),
    ("ReprojectionTask", "grid_resolution"): ((2, None), "hdps default. A grid of at least 2 points per axis."),
    ("CoaddTapTask", "background_mode"): ({"scalar", "regional"}, f"{_H}. hdps checks."),
    ("SFFTSubtractionTask", "reference_mode"): ({"coadd", "first_frame"}, f"{_H}: coadd, the references are earlier "
                                                "coadds (its reference_dir: the run's reference pass, set by "
                                                "mesa_argus.hdps_worker)."),
    ("SFFTSubtractionTask", "kernel_half_width"): ((1, None), "hdps default. hdps checks it is below the tile "
                                                   "overlap minus 2 px."),
    ("SFFTSubtractionTask", "kernel_poly_order"): ((0, None), f"{_H}. A polynomial order: not negative."),
    ("SFFTSubtractionTask", "background_order"): ((0, None), f"{_H} (2 beats 1). A polynomial order: not negative."),
    ("SFFTSubtractionTask", "mask_saturated_in_fit"): ({True, False}, "hdps default."),
    ("SFFTSubtractionTask", "mask_no_coverage_in_fit"): ({True, False}, "true (hdps_full_cpu; hdps main has no such "
                                                         "setting and fits across them): pixels with no reference or "
                                                         "no science data kept out of the kernel fit. Fitting across "
                                                         "them gives the kernel a flat pedestal (a kernel-sized square "
                                                         "around every star) and a background gradient."),
    ("SFFTSubtractionTask", "mask_bright_stars"): ({True, False}, f"{_H} and hdps default: false. True keeps "
                                                   "bright-star residual footprints out of detection (inf Var(D); "
                                                   "the kernel fit is unchanged), but hdps thresholds the science "
                                                   "tile with its sky still in, so nothing brighter than about beta "
                                                   "15 can alert and a bright-sky tile is masked whole "
                                                   "(docs/ARGUS_INTEGRATION.md). Off, as Argus runs: the alerts are "
                                                   "the ones Argus would send (true until 2026-10-25)."),
    ("SFFTSubtractionTask", "bright_star_threshold"): (POSITIVE, "electrons per pixel in the reference (sky-"
                                                       "subtracted coadd) or the science tile (which still holds its "
                                                       "sky, ~1000-1300 e- in rho here): above the science sky, or "
                                                       "the whole science tile is masked. hdps default 5000."),
    ("SFFTSubtractionTask", "mask_dilation_growth"): ((0, None), "hdps default: px added to the mask radius per "
                                                      "magnitude above the threshold."),
    ("SFFTSubtractionTask", "variance_conv"): ({"direct", "fft"}, "hdps default. hdps checks."),
    ("SFFTSubtractionTask", "rejection_iters"): ((0, None), f"{_H}. A count: not negative."),
    ("SFFTSubtractionTask", "rejection_sigma"): (POSITIVE, f"{_H}. {_SIG}"),
    ("SFFTSubtractionTask", "rejection_grow"): ((0, None), f"{_H}. Pixels: not negative."),
    ("SFFTSubtractionTask", "retain_planes"): ({True, False}, "true (hdps default false): AlertVettingTask's ref/sci "
                                               "stamps need the pre-subtraction planes (its emit_stamps note)."),
    ("DiffDetectionTask", "detect_sigma"): (POSITIVE, f"{_H}. {_SIG}"),
    ("DiffDetectionTask", "bg_region_size"): ((1, None), f"{_H}. {_PX}"),
    ("DiffDetectionTask", "max_sources"): ((1, None), f"hdps default. {_N}"),
    ("DiffDetectionTask", "matched_filter_width"): (POSITIVE, f"hdps default (FWHM). {_PX}"),
    ("TransientOutputTask", "fmt"): ({"parquet", "csv"}, f"{_H}. hdps checks."),
    ("AlertVettingTask", "enabled"): ({True, False}, f"{_H} has false (no classifier deployed); true here: the "
                                      "alerts are an output."),
    ("AlertVettingTask", "model_bundle"): (None, "\"\": no model, every candidate scores 1 (decision 2026-10-20); "
                                           "else a bundle directory (tools/train_rb.py, or hdps's own), which hdps's "
                                           "RFBundleClassifier loads; score_cut may then name an operating point "
                                           "(\"fa10\")."),
    ("AlertVettingTask", "detect_sigma"): (POSITIVE, f"{_H}. {_SIG}"),
    ("AlertVettingTask", "score_cut"): (lambda v, r: None if (isinstance(v, str) and v) or (not isinstance(v, bool) and
                                         isinstance(v, (int, float)) and 0 <= v <= 1) else "0..1, or an operating "
                                         "point of the bundle (\"fa10\")",
                                         "A real/bogus score is a probability: a number 0..1, or with a model bundle "
                                         "one of its operating points (\"fa10\": 10 false alerts per deg2 per "
                                         "exposure, as hdps's production config). With every score 1, any number "
                                         "accepts every candidate."),
    ("AlertVettingTask", "emit_all_candidates"): ({True, False}, "hdps default."),
    ("AlertVettingTask", "emit_stamps"): ({True, False}, "true (hdps default false): ref/sci/diff stamps with each "
                                          "alert."),
    ("AlertVettingTask", "stamp_size"): (lambda v, r: None if isinstance(v, int) and v > 0 and v % 2 else
                                         "a positive odd integer", "hdps default. Odd, so the candidate sits on the "
                                         "centre pixel."),
    ("AlertVettingTask", "emit_kafka"): ({False}, f"{_H}: no message broker here."),
    ("AlertVettingTask", "fmt"): ({"parquet", "csv"}, "hdps default."),
    ("LightweightFITSOutputTask", "output_dir"): (None, "hdps default; unused in the worker (its product goes to "
                                                  "the data-products sink, fits_image)."),
    ("TileImageOutputTask", "enabled"): ({True, False}, "hdps default false; true here: the field movie's images "
                                         "are hdps's tiles."),
    ("TileImageOutputTask", "output_dir"): (None, "hdps default \"\": the data-products sink."),
    ("TileImageOutputTask", "science_output_dir"): (None, "hdps default (output_dir)."),
    ("TileImageOutputTask", "difference_output_dir"): (None, "hdps default (output_dir)."),
    ("TileImageOutputTask", "save_science"): ({True, False}, "hdps default."),
    ("TileImageOutputTask", "save_difference"): ({True, False}, "hdps default."),
    ("TileImageOutputTask", "save_difference_variance"): ({True, False}, "hdps default."),
    ("TileImageOutputTask", "compress"): ({True, False}, "false (hdps default true): exact float32 with NaN kept; "
                                          "RICE quantizes and maps NaN to 0."),
    ("TileImageOutputTask", "tile_ids"): (None, "hdps default (none: every tile)."),
    ("TileImageOutputTask", "filename_prefix"): (None, "hdps default."),
}

SECTIONS = {"field": FIELD, "night": NIGHT, "stars": STARS, "inject": INJECT, "camera": CAMERA, "image": IMAGE,
            "lightcurves": LIGHTCURVES,
            "truth": TRUTH, "output": OUTPUT}


def check(where, p, v, resolved=None):
    """None if v is allowed for p, else the error text."""
    if p.optional and v == NONE:
        return None
    a = p.allowed
    if a is None:
        return None
    if a == POSITIVE:
        ok = not isinstance(v, bool) and isinstance(v, (int, float)) and v > 0
        return None if ok else f"{where} = {v!r}: > 0"
    if callable(a):
        err = a(v, resolved)
        return f"{where} = {v!r}: {err}" if err else None
    if isinstance(a, set):
        return None if v in a else f"{where} = {v!r}: one of {sorted(a, key=str)}"
    lo, hi = a
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return f"{where} = {v!r}: a number, {_range(p)}"
    return None if (lo is None or v >= lo) and (hi is None or v <= hi) else f"{where} = {v!r}: {_range(p)}"


def _value(v):
    from mesa_argus.tomlfile import value

    return value(v)


def _range(p):
    a = p.allowed
    if a is None:
        return "any"
    if a == POSITIVE:
        return "> 0"
    if callable(a):
        return p.allowed_text
    if isinstance(a, set):
        return " | ".join(_value(x) for x in sorted(a, key=str))
    lo, hi = a
    if hi is None:
        return f">= {lo} (no upper limit)"
    return f"{lo} .. {hi}"


def template():
    """The documented observation file: every parameter, its default, unit, allowed values and why."""
    import textwrap

    def notes(text):
        return [f"# {line}" for line in textwrap.wrap(text, 116)]

    def block(title, params, header=None):
        out = [f"[{title}]"] + (notes(header) if header else [])
        for p in params:
            default = "\"required\"" if p.default == REQUIRED else _value(NONE if p.default is None else p.default)
            note = (f"[{p.unit}] " if p.unit else "") + f"allowed: {_range(p)}" + (" or \"none\"" if p.optional else "")
            out += notes(f"{note}. {p.why}") + [f"{p.key} = {default}"]
        return out + [""]

    lines = notes("An observation: the one file that defines a run. Written by python -m mesa_argus.schema; every "
                  "parameter with its default, unit, allowed values and the justification.") + [""]
    lines += block("field", FIELD) + block("night", NIGHT) + block("stars", STARS)
    for sub, ps in ARGUSSIM.items():
        lines += block(f"argussim.{sub}", ps, "ArgusSim: how the image is taken. Defaults are read from ArgusSim.")
    lines += block("camera", CAMERA, "The toy camera (mesa_argus/camera.py): electrons to ADU, which ArgusSim does "
                   "not define. none: the default described.")
    lines += block("image", IMAGE, "This repository's renderer (mesa_argus/sensor.py; ArgusSim has no image "
                   "renderer): the PSF's outer wing.")
    lines += block("hdps", HDPS, "hdps: the pipeline the raw frames go through.") + block("hdps.coadd", HDPS_COADD)
    lines += notes("[[hdps.tasks]]: hdps's Task chain, in order, run by hdps.orchestrate's PipelineExecutor; each "
                   "[hdps.tasks.config] holds the Task's constructor parameters.") + [""]
    for t in TOY_TASKS:
        lines += notes(f"Toy Task {t['type']}: allowed in the chain in place of hdps's own (mesa_argus/schema.py "
                       "TOY_TASKS), not in the default chain.")
    lines.append("")
    for t in TASKS:
        lines += ["[[hdps.tasks]]", f"type = \"{t['type']}\"", "[hdps.tasks.config]"]
        for k, v in t["config"].items():
            allowed, why = TASK_RULES[(t["type"], k)]
            text = "a positive odd integer" if callable(allowed) else _range(P(k, v, "", allowed, why))
            lines += notes(f"allowed: {text}. {why}") + [f"{k} = {_value(v)}"]
        lines.append("")
    lines += block("lightcurves", LIGHTCURVES) + block("truth", TRUTH) + block("output", OUTPUT)
    lines += notes("[[place]]: any number of built stars placed in the field (each key below).")
    for p in PLACE:
        lines += notes(f"{p.key}: [{p.unit}] allowed {_range(p)}. {p.why}")
    lines += ["# [[place]]", "# star = \"dwarf_nova_400pc\"", "# x = 60", "# y = 40", "# t0 = \"start+0.05\"", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "examples" / "observations" / "template.toml"
    out.write_text(template())
    print(f"wrote {out}")
