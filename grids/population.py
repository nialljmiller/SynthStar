"""Population grid: realistic (initial mass, [M/H], alpha_MLT) combinations for display.

NOT a full cube. Every track is a star that plausibly exists in the populations
Argus sees (Milky Way thin/thick disk, halo, globular clusters, LMC), and each
one is there for a reason, given in its `why`.

  * [M/H] levels are tied to real populations. Low-[M/H] levels only carry
    masses old enough (<= ~0.85 Msun) to still exist there today; massive stars
    only appear where star formation is ongoing (thin disk, LMC).
  * alpha_MLT is solar-calibrated for every track, plus +/-0.3 variants.
    0.3 roughly brackets the spread that 3D convection simulations
    (Trampedach+2014, Magic+2015) and APOGEE/Kepler giants (Tayar+2017) give
    around the solar value. Variants are only run where an outer convective
    envelope sets Teff (cool dwarfs, giants, blue loops).
  * The atmospheres are scaled-solar Kurucz, so [M/H] is total metallicity.
    For alpha-enhanced halo stars, [M/H] ~= [Fe/H] + 0.3 (Salaris+1993).
"""

DA = 0.3  # alpha_MLT variant offset

# (population label, [M/H], [(mass, why), ...])
POPULATIONS = [
    ("Metal-rich disk / NGC 6791", +0.3, [
        (0.8, "metal-rich K dwarf: line-blanketing shift of the lower MS"),
        (1.0, "metal-rich solar analogue / old open cluster turnoff (NGC 6791, ~8 Gyr)"),
        (1.5, "young metal-rich F star: tests colour-[M/H] degeneracy at the turnoff"),
    ]),
    ("Thin disk (solar)", 0.0, [
        (0.6, "late-K/early-M dwarf: lowest mass that stays on the Kurucz grid (Teff >= 3500 K)"),
        (0.8, "K dwarf: lower main sequence, unevolved in a Hubble time"),
        (1.0, "the Sun: MS -> RGB -> He flash -> red clump"),
        (1.5, "F star: convective-core turnoff hook, Kraft break"),
        (2.0, "A star: near the degenerate/non-degenerate He-ignition boundary (secondary clump)"),
        (3.0, "late-B star: intermediate-mass giant branch"),
        (5.0, "mid-B star: blue loop through the Cepheid instability strip"),
        (10.0, "early-B star: upper MS and red-supergiant branch"),
    ]),
    ("Thick disk / LMC", -0.5, [
        (0.7, "thick-disk K dwarf"),
        (0.9, "thick-disk turnoff (~10 Gyr)"),
        (1.2, "LMC intermediate-age population"),
        (3.0, "LMC young population: intermediate-mass giant"),
        (5.0, "LMC Cepheid progenitor: blue loop at sub-solar [M/H]"),
        (10.0, "LMC massive star: upper MS / supergiant"),
    ]),
    ("Metal-weak thick disk / 47 Tuc-M4", -1.0, [
        (0.7, "metal-poor dwarf (subdwarf sequence)"),
        (0.85, "old turnoff -> RGB -> HB (~11-12 Gyr)"),
    ]),
    ("Inner halo / M3-M13", -1.5, [
        (0.7, "halo subdwarf"),
        (0.8, "GC turnoff -> RGB -> HB"),
    ]),
    ("Outer halo / M92-M15", -2.0, [
        (0.7, "very metal-poor subdwarf"),
        (0.8, "oldest GC turnoff -> RGB -> HB"),
    ]),
]

# alpha_MLT variants: (mass, [M/H], why); each run at alpha_sun - DA and alpha_sun + DA
ALPHA_VARIANTS = [
    (0.6, 0.0, "M-dwarf radius inflation: reduced alpha is a proposed fix (e.g. Feiden & Chaboyer 2013)"),
    (1.0, 0.0, "solar analogue: alpha sets RGB Teff, the classic calibration lever"),
    (1.0, +0.3, "metal-rich giants: APOGEE/Kepler alpha-[M/H] trend (Tayar+2017)"),
    (0.85, -1.0, "metal-poor RGB: low-[M/H] end of the Tayar+2017 alpha-[M/H] trend"),
    (0.8, -2.0, "GC RGB colour: historical alpha calibration from globular clusters"),
    (5.0, 0.0, "blue-loop extent and Cepheid colours depend on envelope convection"),
]


def tracks(make):
    """`make(mass, feh, alpha, **meta)` builds a track dict (see synthstar.grids)."""
    a_sun = make.alpha_sun
    pop_of = {feh: label for label, feh, _ in POPULATIONS}
    out = []
    for label, feh, members in POPULATIONS:
        for mass, why in members:
            out.append(make(mass, feh, a_sun, population=label, why=why, kind="base"))
    for mass, feh, why in ALPHA_VARIANTS:
        for s in (-1, +1):
            out.append(make(mass, feh, a_sun + s * DA, population=pop_of[feh], why=why, kind="alpha"))
    return out
