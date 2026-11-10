"""Eclipsing binaries from PHOEBE 2 (synthstar/binary.py), from parameters alone (no Gaia, no dust).

Two stars with MESA-atmosphere SEDs (Kurucz 2003 surface flux at each Teff and log g); PHOEBE gives each one's visible
flux through the orbit (Roche geometry, blackbody intensities, default limb darkening, no irradiation). One orbit of
system SEDs at 10 pc, phase 0 = primary eclipse, seen at `distance`. Masses from the main-sequence Teff-mass relation
(Pecaut & Mamajek 2013) unless given; orbit size from Kepler's third law.
"""
import numpy as np

from . import REQUIRED, Generator


def write_binary(folder, orbit, contact, k, name, kind):
    from .. import binary as B

    wave, sed_10pc, r1, r2, m1, m2 = B.component_seds(orbit, contact, k["mh"])
    phases, seds, t_max = B.orbit_seds(orbit, contact, wave, sed_10pc)
    dm = float(5 * np.log10(k["distance"]) - 5)
    B.write_orbit(folder / "sed.h5", wave, seds, phases * orbit["period"], dict(
        label=name, kind="EB", source_kind=kind, nonlinear_period=orbit["period"], distance_modulus=dm, ebv=0.0,
        mass=m1, mass2=m2, teff1=orbit["teff1"], teff2=orbit["teff2"], R1=r1, R2=r2, incl=orbit["incl"], q=orbit["q"],
        sma=orbit["sma"], ecc=orbit.get("ecc", 0.0), contact=bool(contact), route="PHOEBE from given parameters"))
    return dict(kind=kind, reference="PHOEBE 2 (Prsa et al. 2016); Pecaut & Mamajek 2013", period=orbit["period"],
                params=f"P {orbit['period']:g} d, i {orbit['incl']:g} deg, q {orbit['q']:.3f}, Teff {orbit['teff1']:.0f} / "
                       f"{orbit['teff2']:.0f} K, R {r1:.3f} / {r2:.3f} Rsun, M {m1:.3f} / {m2:.3f} Msun, "
                       f"a {orbit['sma']:.3f} Rsun; at {k['distance']:g} pc")


def wuma(folder, k, name):
    from .. import binary as B

    m1 = B.ms_mass(k["teff1"])
    sma = B.sma_rsun(k["period"], m1 * (1 + k["q"]))
    orbit = dict(period=k["period"], incl=k["incl"], q=k["q"], sma=sma, teff1=k["teff1"], teff2=k["teff2"],
                 requiv1=B.contact_requiv(k["period"], k["q"], sma, k["fillout"]))
    return write_binary(folder, orbit, True, k, name, "W UMa")


def detached(folder, k, name):
    from .. import binary as B

    m1 = k["m1"] or B.ms_mass(k["teff1"])
    m2 = k["m2"] or B.ms_mass(k["teff2"])
    orbit = dict(period=k["period"], incl=k["incl"], q=m2 / m1, sma=B.sma_rsun(k["period"], m1 + m2), teff1=k["teff1"],
                 teff2=k["teff2"], requiv1=k["r1"], requiv2=k["r2"], m1=m1, m2=m2, ecc=k["ecc"], per0=k["per0"])
    return write_binary(folder, orbit, False, k, name, "detached EB")


GENERATORS = [
    Generator("W UMa", "periodic", "PHOEBE", "contact binary (W UMa): two stars sharing a common envelope",
              dict(period=(REQUIRED, "orbital period (days)"),
                   q=(REQUIRED, "mass ratio M2/M1"),
                   incl=(REQUIRED, "inclination (deg)"),
                   teff1=(REQUIRED, "primary Teff (K); its mass from the main-sequence relation"),
                   teff2=(REQUIRED, "secondary Teff (K)"),
                   fillout=(REQUIRED, "PHOEBE fillout factor: 0 inner, 1 outer critical surface"),
                   distance=(REQUIRED, "distance (pc)"),
                   mh=(0.0, "[M/H] of the atmospheres")),
              wuma, "Prsa et al. 2016", parallel=False),
    Generator("detached EB", "periodic", "PHOEBE", "detached eclipsing binary, circular or eccentric",
              dict(period=(REQUIRED, "orbital period (days)"),
                   incl=(REQUIRED, "inclination (deg)"),
                   teff1=(REQUIRED, "primary Teff (K)"),
                   teff2=(REQUIRED, "secondary Teff (K)"),
                   r1=(REQUIRED, "primary equivalent radius (Rsun)"),
                   r2=(REQUIRED, "secondary equivalent radius (Rsun)"),
                   m1=(0.0, "primary mass (Msun); 0 = main-sequence relation"),
                   m2=(0.0, "secondary mass (Msun); 0 = main-sequence relation"),
                   ecc=(0.0, "eccentricity"),
                   per0=(90.0, "argument of periastron (deg)"),
                   distance=(REQUIRED, "distance (pc)"),
                   mh=(0.0, "[M/H] of the atmospheres")),
              detached, "Prsa et al. 2016", parallel=False),
]
