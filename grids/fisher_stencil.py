"""Fisher stencils: small symmetric steps around anchor stars, for numerical derivatives.

Each anchor (M, [M/H], alpha_sun) gets 6 neighbours, +/- one step in each of
[M/H], ln M and alpha_MLT. At matched evolutionary points (EEPs) the central
differences give d(magnitude)/d(parameter) for every band, which feeds the
Fisher-information analysis (synthstar.fisher). Every run here exists to
measure one derivative.

Steps are small enough to stay in the linear regime but large enough that
numerical noise in the tracks does not dominate (a few times larger than the
step-to-step scatter in Teff/L).

max_age is raised to 25 Gyr for the whole stencil. Here age is a nuisance
parameter, not a realism cut: the -dM neighbour of an old anchor must reach
the same EEPs as the anchor, even if it gets there after 13.5 Gyr.
"""

D_FEH = 0.1     # dex
D_LNM = 0.05    # ~5% in mass
D_ALPHA = 0.1
MAX_AGE = 25e9

# (mass, [M/H], label, why)
ANCHORS = [  # fastest first, so the analysis can start on complete anchors early
    (3.0, 0.0, "late-B star", "intermediate-mass, non-degenerate He ignition"),
    (1.0, 0.0, "Sun", "solar-metallicity dwarf -> RGB -> red clump"),
    (1.5, 0.0, "F star", "convective-core turnoff, subgiant, RGB, clump"),
    (0.85, -1.0, "47 Tuc/M4-like", "metal-poor turnoff, RGB, HB"),
    (0.8, -2.0, "M92-like", "very metal-poor turnoff, RGB, HB"),
]


def tracks(make):
    import math

    a_sun = make.alpha_sun
    out = []
    for mass, feh, label, why in ANCHORS:
        anchor = make.name(mass, feh, a_sun, MAX_AGE)
        meta = dict(population=f"Stencil: {label} ({mass} M☉, [M/H] {feh:+.1f})", anchor=anchor, max_age=MAX_AGE)
        out.append(make(mass, feh, a_sun, why=f"anchor: {why}", kind="anchor", param=None, sign=0, **meta))
        for s in (-1, +1):
            out.append(make(mass, feh + s * D_FEH, a_sun, why="d/d[M/H]", kind="stencil", param="feh", sign=s, **meta))
            m = round(mass * math.exp(s * D_LNM), 3)
            out.append(make(m, feh, a_sun, why="d/d ln M", kind="stencil", param="lnM", sign=s, **meta))
            out.append(make(mass, feh, a_sun + s * D_ALPHA, why="d/d alpha_MLT", kind="stencil", param="alpha", sign=s, **meta))
    return out
