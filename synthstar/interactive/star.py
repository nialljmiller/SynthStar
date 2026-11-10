"""The one editable star: a library template (templates.py) with the sliders' settings, evaluated at every frame.

Sliders, by type (the display band's magnitudes; the other band follows the template's own colour):

    periodic (RR Lyrae, binaries)   magnitude: mean; amplitude: peak to peak; period: days (the nearest precomputed
                                    template, its phase stretched to the exact period)
    dwarf nova                      magnitude: quiescence; amplitude: outburst; timescale: the template's time axis
                                    stretched (x1 = as modelled); onset a third of the way through the night
    supernova (Ia)                  magnitude: peak; amplitude: none (greyed); timescale: as the dwarf nova; peak at
                                    the middle of the night (a SN hardly changes in one night)
    microlensing                    magnitude: the source unmagnified; amplitude: brightening at peak (sets u0);
                                    timescale: Einstein time tE (days); peak at the middle of the night

The band integrals are linear in the SED, so every slider is exact arithmetic on the template's tables: the star's
light is a(F0 + s (F - F0)) with F the template's flux, F0 its mean (periodic) or quiescent level, s the amplitude
scale and a the brightness scale, the same in every band and in the SED.
"""
from dataclasses import dataclass

import numpy as np

from synthstar.interactive import templates as T

BANDS = ("beta", "rho")


@dataclass
class Settings:
    kind: str = "RR Lyrae (RRab)"
    band: str = "beta"  # the band the magnitude and amplitude sliders are in (the band shown)
    mag: float = 16.0
    amplitude: float = 0.8  # mag
    period: float = 0.55  # days (periodic types)
    timescale: float = 1.0  # stretch (dwarf nova, supernova) or tE in days (microlensing)
    dx: float = 30.0  # arcsec from the block centre (east is -x on the movie's image, as the field's stars)
    dy: float = -20.0


def _flux(mag):
    return 10.0 ** (-0.4 * np.asarray(mag, float))


def _mag(flux):
    with np.errstate(divide="ignore"):
        return -2.5 * np.log10(np.asarray(flux, float))


def _scale_for_amplitude(F, F0, amplitude, peak=False):
    """s with ptp(mag(F0 + s (F - F0))) = amplitude (peak=True: mag(F0) - min mag), F0 > 0; bisection, kept positive."""
    F, F0 = np.asarray(F, float), float(F0)
    lo_flux = F.min()
    s_max = 50.0 if lo_flux >= F0 else min(50.0, 0.999 * F0 / (F0 - lo_flux))

    def ptp(s):
        m = _mag(F0 + s * (F - F0))
        return (_mag(F0) - m.min()) if peak else (m.max() - m.min())

    if amplitude <= 0 or ptp(s_max) <= 0:
        return 0.0
    if ptp(s_max) < amplitude:
        return s_max
    a, b = 0.0, s_max
    for _ in range(60):
        mid = 0.5 * (a + b)
        a, b = (mid, b) if ptp(mid) < amplitude else (a, mid)
    return 0.5 * (a + b)


def phase_weights(x):
    """Each phase step's share of the cycle (x in [0, 1), increasing, not evenly spaced): half the gaps either side."""
    x = np.asarray(x, float)
    gaps = np.diff(np.r_[x, x[0] + 1.0])
    return 0.5 * (gaps + np.roll(gaps, 1))


def on_even_phases(x, values, n=1000):
    """A cycle tabulated at phases x, on n even phases from 0 (as the movie draws a cycle)."""
    even = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.interp(even, np.r_[x, x[0] + 1.0], np.r_[values, values[0]])


def paczynski(u):
    return (u * u + 2.0) / (u * np.sqrt(u * u + 4.0))


class EditableStar:
    """evaluate(settings, bjd, airmass) -> the star at every frame (see evaluate)."""

    def __init__(self, ebv=0.0):
        self.ebv = float(ebv)

    def template(self, s):
        name = T.nearest(s.kind, s.period) if T.TYPES[s.kind]["timing"] == "periodic" else T.choose(s.kind)
        return T.light(name, self.ebv)

    def evaluate(self, s, bjd, airmass, exptime=60.0, span=None):
        """dict(electrons {band: per frame, this exposure, through the atmosphere at its airmass}, mag {band: per
        frame, above the atmosphere}, sed (per frame, f_lambda at Argus, the template's wavelengths), wave,
        track {band: the CMD cycle (periodic) or the night's track}, reference {band: per frame, the electrons the
        reference coadd holds of it}, label, template, timing, period). span: the night's first and last exposure (BJD;
        default bjd's ends): a periodic star's phase 0 is at its start, a transient's onset or peak placed in it."""
        L = self.template(s)
        timing = T.TYPES[s.kind]["timing"]
        bjd, airmass = np.asarray(bjd, float), np.asarray(airmass, float)
        start, end = span if span is not None else (bjd[0], bjd[-1])
        mid = 0.5 * (start + end)
        # where each frame falls on the template (rows of its tables), and the brightening on top (microlensing)
        boost = np.ones(len(bjd))
        if timing == "periodic":
            x = ((bjd - start) / s.period) % 1.0  # phase 0 (maximum light) at the night's start
            idx = np.interp(x, np.r_[L["x"], 1.0], np.r_[np.arange(len(L["x"])), len(L["x"])]) % len(L["x"])
        elif "dwarf" in s.kind:  # the outburst starts a third of the way through the night
            m = np.nan_to_num(L["mag"][s.band], nan=np.inf)
            rise = float(L["x"][max(int(np.argmax(m < m[0] - 0.1)) - 1, 0)])  # the last step still quiescent
            days = rise + (bjd - (start + (end - start) / 3.0)) / s.timescale
            idx = np.interp(days, L["x"], np.arange(len(L["x"])), left=-1, right=len(L["x"]) - 1)
        elif timing == "transient":  # the peak at the middle of the night
            m = np.nan_to_num(L["mag"][s.band], nan=np.inf)
            days = float(L["x"][int(np.argmin(m))]) + (bjd - mid) / s.timescale
            idx = np.interp(days, L["x"], np.arange(len(L["x"])), left=-1, right=len(L["x"]) - 1)
        else:  # microlensing: the template's faintest step as the unmagnified source
            idx = np.full(len(bjd), int(np.argmax(L["mag"]["rho"])), float)
        rows = np.clip(idx, 0, len(L["x"]) - 1)
        lit = idx >= 0  # a transient before its model starts: no light

        def at(table):  # a template table (rows x ...) at each frame: linear between rows
            i0 = np.floor(rows).astype(int)
            i1 = np.minimum(i0 + 1, len(table) - 1) if timing != "periodic" else (i0 + 1) % len(table)
            w = (rows - i0)[(...,) + (None,) * (np.ndim(table) - 1)]
            return (1 - w) * table[i0] + w * table[i1]

        # the template's flux scale in the display band: F0 (mean, quiescent, or source) and the amplitude scale s
        F = {b: np.nan_to_num(_flux(L["mag"][b]), nan=0.0) for b in BANDS}  # no light (or a model's tiny negative
        # flux before it starts, NaN magnitude) as zero
        w = phase_weights(L["x"]) if timing == "periodic" else None
        if timing == "periodic":  # the mean over the cycle (the phase steps are not evenly spaced)
            F0 = {b: float(np.sum(w * F[b])) for b in BANDS}
            amp = _scale_for_amplitude(F[s.band], F0[s.band], s.amplitude)
            a = _flux(s.mag) / F0[s.band]
        elif "dwarf" in s.kind:  # quiescence: the level before the outburst
            F0 = {b: float(F[b][0]) for b in BANDS}
            amp = _scale_for_amplitude(F[s.band], F0[s.band], s.amplitude, peak=True)
            a = _flux(s.mag) / F0[s.band]
        elif timing == "transient":  # supernova: magnitude at peak, no amplitude
            F0 = {b: 0.0 for b in BANDS}
            amp = 1.0
            a = _flux(s.mag) / float(np.max(F[s.band]))
        else:  # microlensing: u0 from the brightening at peak, tE in days
            F0 = {b: float(F[b][int(rows[0])]) for b in BANDS}
            u0 = 1.0
            target = 10 ** (0.4 * max(s.amplitude, 1e-3))
            lo_u, hi_u = 1e-4, 10.0
            for _ in range(80):
                u0 = np.sqrt(lo_u * hi_u)
                lo_u, hi_u = (u0, hi_u) if paczynski(u0) > target else (lo_u, u0)
            boost = paczynski(np.sqrt(u0 ** 2 + ((bjd - mid) / s.timescale) ** 2))
            amp = 1.0
            a = _flux(s.mag) / F0[s.band]

        def scaled(table_rows, level):  # a(F0 + s (F - F0)) on any linear quantity; boost (lensing) on top
            v = level + amp * (table_rows - level)
            return a * v * (boost if np.ndim(v) == 1 else boost[:, None])

        out = dict(label=s.kind, template=L["name"], wave=L["wave"])
        out["mag"] = {b: np.where(lit, _mag(scaled(at(F[b]), F0[b])), np.inf) for b in BANDS}
        out["electrons"], out["reference"] = {}, {}
        for b in BANDS:
            per_frame = np.array([np.interp(x, T.AIRMASS, row) for x, row in zip(airmass, at(L["electrons"][b]))])
            if timing == "periodic":  # the cycle's mean, at each frame's airmass
                level_row = np.sum(w[:, None] * L["electrons"][b], axis=0)
            elif "dwarf" in s.kind:  # quiescence: the level before the outburst
                level_row = L["electrons"][b][0]
            else:
                level_row = None
            e0 = (np.array([np.interp(x, T.AIRMASS, level_row) for x in airmass]) if level_row is not None
                  else (per_frame if timing == "lens" else np.zeros(len(airmass))))
            out["electrons"][b] = np.where(lit, np.maximum(scaled(per_frame, e0), 0.0), 0.0) * exptime
            # what the reference coadd (another night) holds of it, in this frame's electrons: a variable at its mean, a
            # dwarf nova in quiescence, a lensed source unmagnified, a supernova nothing (the engine leaves every
            # transient-type source out of its reference; physically the persistent light is there)
            out["reference"][b] = a * e0 * exptime
        sed_rows = at(L["seds"])
        if timing == "periodic":
            sed0 = np.sum(w[:, None] * L["seds"], axis=0)
        elif "dwarf" in s.kind:
            sed0 = L["seds"][0]
        elif timing == "transient":
            sed0 = np.zeros(L["seds"].shape[1])
        else:
            sed0 = sed_rows
        v = sed0 + amp * (sed_rows - sed0)
        out["sed"] = np.where(lit[:, None], np.maximum(a * v * boost[:, None], 0.0), 0.0)
        # the CMD: the cycle (periodic), else the night's track
        if timing == "periodic":
            out["track"] = {b: _mag(a * (F0[b] + amp * (on_even_phases(L["x"], F[b]) - F0[b]))) for b in BANDS}
        else:
            out["track"] = {b: np.where(lit, out["mag"][b], np.nan) for b in BANDS}
        out["timing"], out["period"] = timing, (s.period if timing == "periodic" else None)
        return out
