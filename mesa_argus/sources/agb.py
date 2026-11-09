"""Long-period AGB variables (Mira / semiregular) from a saved MESA TP-AGB structure, from parameters alone.

Why not MESA RSP (as the Cepheids / RR Lyrae use): RSP cannot reach the 200+ d fundamental mode of a Mira here. A
Mira is the fundamental radial mode of a hugely extended, very low gravity AGB envelope (large amplitude, shocks);
RSP's static-envelope build and turbulent-convection closure are calibrated for the classical instability strip. This
pipeline already records the failure one step up in period: the 15.3 d Type II Cepheid "could not be modelled ... its
envelope either cools below the opacity tables (logT < 2.7) or diverges within ~10 cycles" (mesa_argus/rsp.py). A
200-400 d Mira envelope is far more extended and cooler, so RSP is not viable for it here.

The viable path, instead (the user's evolve -> save -> branch idea):

 1. SAVE (expensive, computed once, reused): an intermediate-mass star is evolved with the ordinary MESA_ARGUS
    physics up the AGB (grids/lpv_tracks.py: 1.5-3 Msun to log L/Lsun 3.8). Its seds.h5 is the saved structure -- the
    full run of (L, Teff, log g, R, mass, [M/H]) with a colors-module SED at every history step. We do NOT recompute
    the pre-AGB evolution per star.

 2. BRANCH (cheap, this generator, seconds per star): read the AGB-tip structure from that seds.h5, take the
    fundamental-mode period from a published period-mass-radius relation (Ostlie & Cox 1986; Wood 1990) -- no GYRE and
    no hydrodynamics needed -- and drive a phenomenological, one-cycle time-dependent SED: a bolometric light curve
    with an anti-correlated effective-temperature excursion (brightest = hottest, as observed), each phase's SED taken
    from the MESA colors Kurucz surface-flux cube (mesa_argus/binary.surface_sed) and renormalised so its integrated
    flux equals that phase's luminosity. Output is sed.h5 in the RSP pulsators' format, so observe.py / night_movie
    treat it like any other periodic variable.

HONEST FIDELITY LIMIT (reported, not hidden). Genuine Miras are cool (~2500-3500 K) and the pipeline's Kurucz 2003
atmosphere grid stops at 3500 K. Below that edge surface_sed clamps to the 3500 K spectrum, so for a cool AGB tip the
cycle is a correct bolometric (grey) and period behaviour but the optical SED shape is frozen: the TiO/VO-band-driven
optical amplitude and colour swing of a real Mira are NOT reproduced (the M2.0 track's AGB tip is 3211 K, entirely
below the grid). A faithful optical Mira needs a cool-star atmosphere grid (MARCS / PHOENIX / BT-Settl); that is the
documented follow-up, the same gap grids/lpv_tracks.py and grids/faint_stars.py already note.
"""
import json

import numpy as np

from .. import config as C
from . import REQUIRED, Generator

TEFF_SUN = 5772.0
LSUN = 3.828e33
PC_CM = 3.0857e18


def fundamental_period(mass, radius):
    """Mira fundamental-mode period (days) from the period-mass-radius relation (Ostlie & Cox 1986; Wood 1990),
    log P = -1.92 - 0.73 log(M/Msun) + 1.86 log(R/Rsun). Masses and radii in solar units."""
    return float(10.0 ** (-1.92 - 0.73 * np.log10(mass) + 1.86 * np.log10(radius)))


def agb_tip(track):
    """AGB-tip structure (the most luminous model) of a saved AGB structure, as a dict: L0 (Lsun), teff0 (K), mass
    (Msun), mh ([M/H]), in_grid (bool), and the track's X, Z."""
    import h5py

    path = C.RUNS / track / "seds.h5"
    if not path.exists():
        raise SystemExit(f"AGB generator: no saved AGB structure at {path} (run grids/lpv_tracks.py first, or give the "
                         f"track of an lpv_checkpoint run whose seds.h5 exists)")
    with h5py.File(path) as f:
        logL = np.asarray(f["model/log_L"])
        i = int(np.argmax(logL))
        return dict(L0=float(10.0 ** logL[i]), teff0=float(f["model/Teff"][i]), mass=float(f["model/star_mass"][i]),
                    mh=float(f["model/mh"][i]), in_grid=bool(f["model/inside_atm_grid"][i]),
                    X=float(f.attrs.get("X", 0.70)), Z=float(f.attrs.get("Z", 0.0)))


def cycle(L0, teff0, mass, mh, period, amp_bol, teff_swing, rise_fraction, n_phase):
    """One pulsation cycle: phase 0 = maximum light. Returns (wavelength, flux[n_phase+1, n_wave] f_lambda at 10 pc,
    time_days, extras). Bolometric light curve of peak-to-peak amp_bol mag; Teff excursion teff_swing K, in phase with
    brightness; radius from L = 4 pi R^2 sigma Teff^4; each phase's SED is the Kurucz surface flux renormalised to that
    phase's luminosity, so the bolometric light curve is exact even where Teff is below the 3500 K grid edge."""
    from .. import binary as B

    # a warped phase so the rise is faster than the decline (Miras: rise_fraction ~ 0.3-0.4 of the cycle). Phase 0 is
    # maximum light; going forward in phase the star first declines (the first 1-rf of the cycle: ang 0->pi, max->min)
    # then rises back to maximum (the last rf: ang pi->2pi). rise_fraction = 0.5 is a symmetric sinusoid.
    theta = np.linspace(0.0, 1.0, n_phase + 1)
    rf = float(np.clip(rise_fraction, 0.05, 0.95))
    fall = 1.0 - rf
    ang = np.where(theta < fall, np.pi * theta / fall, np.pi + np.pi * (theta - fall) / rf)  # 0..2pi, min at theta=fall
    shape = -np.cos(ang)  # -1 at maximum (theta 0), +1 at minimum (theta 1-rf)
    logL = np.log10(L0) + 0.4 * (amp_bol / 2.0) * (-shape)      # brightest at theta 0
    teff = teff0 + (teff_swing / 2.0) * (-shape)                # hottest at theta 0
    L = 10.0 ** logL
    radius = np.sqrt(L) * (TEFF_SUN / teff) ** 2               # Rsun
    logg = np.log10(B.G_CGS * mass * B.MSUN / (radius * B.RSUN) ** 2)

    wave = None
    flux = []
    d10 = 10.0 * PC_CM
    for Lp, tp, gp in zip(L, teff, logg):
        w, surf = B.surface_sed(tp, gp, mh)   # surface F_lambda (clamps below 3500 K); shape only
        fbol_surf = np.trapezoid(surf, w)
        target = Lp * LSUN / (4.0 * np.pi * d10 ** 2)           # bolometric f at 10 pc
        flux.append(surf * (target / fbol_surf))
        wave = w
    flux = np.asarray(flux)
    time_days = theta * period
    extras = dict(teff=teff, logL=logL, logg=logg, radius=radius)
    return wave, flux, time_days, extras


AMP_FLOOR = 3500.0   # K: the Kurucz grid's cool edge
LOGG_FLOOR = 0.0     # the grid's lowest log g


def make(folder, k, name):
    import h5py

    from .. import binary as B

    tip = agb_tip(k["track"])
    L0, teff0, mass = tip["L0"], tip["teff0"], tip["mass"]
    mh = tip["mh"] if k["mh"] is None else float(k["mh"])
    radius0 = np.sqrt(L0) * (TEFF_SUN / teff0) ** 2
    period = float(k["period"]) if k["period"] is not None else fundamental_period(mass, radius0)
    dm = float(5 * np.log10(k["distance"]) - 5)

    wave, flux, time_days, ex = cycle(L0, teff0, mass, mh, period, k["amp_bol"], k["teff_swing"],
                                      k["rise_fraction"], int(k["n_phase"]))

    # how much of the cycle is actually resolved by the atmosphere grid (Teff and log g on-grid), vs clamped
    on_grid = (ex["teff"] >= AMP_FLOOR) & (ex["logg"] >= LOGG_FLOOR)
    frac_on = float(on_grid.mean())
    g_amp = float(np.ptp([B.vega_mag(wave, f, "G") for f in flux]))
    # a Mira is a large-amplitude LPV; the class is set by amplitude, not period
    kind = "Mira" if k["amp_bol"] >= 0.8 else "semiregular AGB variable"

    with h5py.File(folder / "sed.h5", "w") as f:
        f.attrs.update(label=name, kind=kind, source=0, gaia_period=period, nonlinear_period=period,
                       gaia_epoch_bjd=float("nan"), epoch_bjd=float("nan"), distance_modulus=dm, ebv=0.0,
                       mass=mass, L=L0, teff_static=teff0, radius=radius0, X=tip["X"], Z=tip["Z"], mh=mh,
                       amp_bol=k["amp_bol"], teff_swing=k["teff_swing"], rise_fraction=k["rise_fraction"],
                       atmosphere_floor_K=AMP_FLOOR, cycle_fraction_on_grid=frac_on,
                       flux_units="erg s^-1 cm^-2 A^-1 (f_lambda at 10 pc)")
        f.create_dataset("wavelength", data=wave)
        f.create_dataset("flux", data=flux, compression="gzip", compression_opts=4)
        f.create_dataset("time_days", data=time_days)
        f.create_dataset("log_Teff", data=np.log10(ex["teff"]))
        f.create_dataset("log_L", data=ex["logL"])
        f.create_dataset("log_R", data=np.log10(ex["radius"]))
        f.create_dataset("log_g", data=ex["logg"])

    if frac_on == 0:
        fidelity = (f"optical SED shape frozen: the whole cycle (Teff {ex['teff'].min():.0f}-{ex['teff'].max():.0f} K, "
                    f"log g down to {ex['logg'].min():.2f}) is below the {AMP_FLOOR:.0f} K / log g 0 Kurucz edge, so the "
                    "variability is bolometric (grey) only -- the TiO-band optical amplitude is NOT captured "
                    "(needs a cool-star atmosphere grid)")
    elif frac_on < 1:
        fidelity = (f"partial: only {frac_on:.0%} of the cycle is inside the atmosphere grid; the cool half is clamped "
                    f"at {AMP_FLOOR:.0f} K, so the optical colour swing has a clamp kink and its amplitude is a lower "
                    "limit (needs a cool-star atmosphere grid for a faithful Mira)")
    else:
        fidelity = f"the whole cycle is inside the atmosphere grid (Teff {ex['teff'].min():.0f}-{ex['teff'].max():.0f} K)"
    return dict(kind=kind, reference="MESA AGB structure (grids/lpv_tracks.py) + Ostlie & Cox 1986 / Wood 1990 P-M-R",
                period=period,
                params=f"P {period:.0f} d (fundamental P-M-R); AGB tip L {L0:.0f} Lsun, Teff {teff0:.0f} K, "
                       f"R {radius0:.0f} Rsun, M {mass:.2f} Msun, [M/H] {mh:+.2f}; bolometric amplitude "
                       f"{k['amp_bol']:.2f} mag, Teff swing {k['teff_swing']:.0f} K; delivered G amplitude {g_amp:.2f} mag; "
                       f"at {k['distance']:g} pc",
                notes=dict(trunk=k["track"], fidelity=fidelity, cycle_fraction_on_grid=round(frac_on, 3),
                           use=f"placed at {k['distance']:g} pc (distance_modulus {dm:.2f}); a different distance can be "
                               "given in [[place]]"))


GENERATORS = [
    Generator("AGB variable", "periodic", "MESA AGB structure + phenomenological pulsation",
              "long-period AGB variable (Mira / semiregular): a saved MESA TP-AGB structure driven over one "
              "fundamental-mode cycle; the 200+ d light/SED variability the engine reads for a variable star",
              dict(track=(REQUIRED, "lpv_checkpoint track name whose runs/<track>/seds.h5 gives the AGB structure"),
                   period=(None, "period (days); default: the fundamental-mode P-M-R value of the AGB tip"),
                   distance=(REQUIRED, "distance (pc) the star is placed at; sets its distance modulus (a different "
                                       "distance can still be given in [[place]])"),
                   amp_bol=(1.0, "peak-to-peak bolometric amplitude (mag); Mira bolometric amplitudes ~0.9-1.4 mag "
                                 "(e.g. Whitelock et al. 2000), far below the optical amplitude, which is TiO-driven"),
                   teff_swing=(500.0, "peak-to-peak Teff excursion (K), in phase with brightness; a few hundred K over "
                                      "a Mira cycle (hottest at maximum light)"),
                   rise_fraction=(0.35, "fraction of the cycle spent rising to maximum; Miras rise faster than they "
                                        "decline (0.5 = symmetric sinusoid)"),
                   n_phase=(80, "SED samples over one cycle; the engine interpolates between them"),
                   mh=(None, "[M/H] for the atmosphere lookup; default: the trunk's AGB-tip photospheric [M/H]")),
              make, "Ostlie & Cox 1986; Wood 1990 (P-M-R); MESA colors Kurucz 2003 atmospheres"),
]
