"""Transients: SEDs against time (days since t0), at the observer, before Milky Way dust.

    python -m synthstar.transients --list                                  # the sncosmo models
    python -m synthstar.transients salt3 out.h5 --z 0.03 --peak-absmag -19.3

supernova     sncosmo templates (built from real supernova spectra), at redshift z, scaled to a peak absolute magnitude
microlensing  a point lens magnifying a MESA template star (Paczynski 1986)
dwarf nova    a steady accretion disc whose accretion rate rises and decays, plus the white dwarf (Shakura & Sunyaev 1973)

t0 and the dust are the observation's (observe.py); here t0 = 0 and there is no dust.
"""
import argparse
from pathlib import Path

import astropy.units as u
import h5py
import numpy as np

from argus_sim import ABPhot, SystemThroughput
from argus_sim import c as ARGUS_CONFIG
from synthstar.seds import Templates

WAVE = np.arange(3000.0, 10500.0 + 1e-6, 5.0)  # A: the SED panel's 300-1000 nm, 5 A steps
BAND_RANGE = (3600.0, 8300.0)  # A: where the Argus beta and rho responses are non-zero

# cgs
G, SIGMA_SB, H, C, K_B = 6.674e-8, 5.6704e-5, 6.62607e-27, 2.99792458e10, 1.380649e-16
MSUN, RSUN, PC, YEAR = 1.98847e33, 6.957e10, 3.0857e18, 3.15576e7


def write(out, wave, time, flux, attrs):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f.create_dataset("wavelength", data=wave)
        f.create_dataset("time_days", data=time)
        f.create_dataset("flux", data=flux, compression="gzip", compression_opts=4)
        f.attrs.update(attrs, t0_bjd=0.0, ebv=0.0)
    return attrs


def rho_mag(wave, flam):
    """Argus rho AB magnitude of f_lambda (erg/s/cm2/A), above the atmosphere."""
    tel = ARGUS_CONFIG.telescope
    ab = ABPhot(throughput=SystemThroughput(filters={"g": True, "rho": True}, throughput_loss=tel.throughput_loss),
                collecting_area=tel.collecting_area)
    return float(ab.mag_from_sed("rho", wave * u.AA, flam * u.erg / (u.s * u.cm**2 * u.AA)))


# ---------------------------------------------------------------- supernova (sncosmo)
def catalogue():
    """(name, version, type, reference) of every sncosmo model."""
    from sncosmo.models import _SOURCES

    return [(m["name"], m["version"], m.get("type", ""), m.get("reference", ("", ""))[1])
            for m in _SOURCES.get_loaders_metadata()]


def make_sncosmo(out, name, z, peak_absmag, peak_band="bessellb", magsys="ab", params=None, host_ebv=0.0, version=None):
    """t0 is the model's phase 0 (B maximum for SALT and Hsiao, the explosion for Nugent)."""
    import sncosmo

    found = [r for r in catalogue() if r[0] == name and (version is None or r[1] == version)]
    if not found:
        raise SystemExit(f"no sncosmo model {name!r} {version or ''}")
    name, version, kind, reference = found[-1]  # the latest version, as sncosmo
    if host_ebv > 0:
        model = sncosmo.Model(sncosmo.get_source(name, version=version), effects=[sncosmo.CCM89Dust()],
                              effect_names=["host"], effect_frames=["rest"])
        model.set(hostebv=host_ebv, hostr_v=3.1)
    else:
        model = sncosmo.Model(sncosmo.get_source(name, version=version))
    model.set(z=z, t0=0.0, **(params or {}))
    model.set_source_peakabsmag(peak_absmag, peak_band, magsys)  # sncosmo's default cosmology (WMAP9)
    lo, hi = max(model.minwave(), WAVE[0]), min(model.maxwave(), WAVE[-1])
    if lo > BAND_RANGE[0] or hi < BAND_RANGE[1]:
        raise SystemExit(f"{name} at z = {z} covers {model.minwave():.0f}-{model.maxwave():.0f} A; the Argus bands need "
                         f"{BAND_RANGE[0]:.0f}-{BAND_RANGE[1]:.0f} A")
    wave = np.arange(np.ceil(lo / 5.0) * 5.0, hi + 1e-6, 5.0)
    time = np.arange(model.mintime(), model.maxtime() + 1e-9, 0.25)
    flux = np.asarray(model.flux(time, wave), float)
    attrs = dict(kind=kind, model=name, model_version=str(version), reference=reference, z=float(z),
                 peak_days=float(model.source.peakphase("bessellb")) * (1 + z),
                 params=", ".join(f"{k} = {model.get(k):.6g}" for k in model.param_names if k not in ("z", "t0")))
    return write(out, wave, time, flux, attrs)


# ---------------------------------------------------------------- microlensing
def magnification(separation, rho=0.0):
    """Point lens at separation u (Einstein radii): A = (u^2 + 2) / (u sqrt(u^2 + 4)).
    A disc source of radius rho: A averaged over the disc, summed in rings about the lens."""
    separation = np.atleast_1d(np.asarray(separation, float))
    if rho <= 0:
        return (separation**2 + 2) / (separation * np.sqrt(separation**2 + 4))
    x, w = np.polynomial.legendre.leggauss(400)
    phi, w_phi = 0.5 * np.pi * (x + 1), 0.5 * np.pi * w
    out = np.empty_like(separation)
    for i, sep in enumerate(separation):
        a = max(rho - sep, 0.0)
        inside = np.pi * a * np.sqrt(a**2 + 4)  # rings wholly inside the disc, exactly
        if sep <= 0:
            out[i] = inside / (np.pi * rho**2)
            continue
        r_lo = abs(sep - rho)  # rings cut by the disc's edge, r_lo to sep + rho
        centre, half = 0.5 * (r_lo + sep + rho), 0.5 * (sep + rho - r_lo)
        r = centre - half * np.cos(phi)
        arc = 2 * np.arccos(np.clip((r**2 + sep**2 - rho**2) / (2 * r * sep), -1, 1))
        crossed = np.sum(w_phi * half * np.sin(phi) * arc * (r**2 + 2) / np.sqrt(r**2 + 4))
        out[i] = (inside + crossed) / (np.pi * rho**2)
    return out


def make_microlensing(out, t_e, u0, base_mag, bp_rp, rho=0.0, fs=1.0):
    """The MESA template star of colour bp_rp at rho = base_mag, magnified: light = (fs A + 1 - fs) x the star.
    fs: the source's share of the baseline light (the rest an unlensed blend)."""
    templates = Templates()
    i = templates.index(bp_rp)
    sed = np.interp(WAVE, templates.wave.value, templates.flux[i])
    sed *= 10 ** (-0.4 * (base_mag - rho_mag(WAVE, sed)))
    # time steps fine near t0, coarse far from it; at least +-10 d, so a week's observation fits wherever t0 is in it
    width = t_e * max(u0, rho, 1e-3)
    s = np.arcsinh(max(10 * t_e, 10.0) / width)
    time = width * np.sinh(np.linspace(-s, s, 2001))
    A = magnification(np.sqrt(u0**2 + (time / t_e) ** 2), rho)
    flux = (fs * A + 1 - fs)[:, None] * sed[None, :]
    peak_mag = rho_mag(WAVE, sed * (fs * A.max() + 1 - fs))
    attrs = dict(kind="microlensing", model="microlensing", reference="Paczynski 1986", z=float("nan"), peak_days=0.0,
                 params=f"t_E {t_e:g} d, u0 {u0:g}, rho {rho:g}, fs {fs:g}; source rho {base_mag:g} (AB, before dust), "
                        f"BP-RP {bp_rp:g} (template {templates.colour[i]:.2f}); peak A {A.max():.1f} (rho {peak_mag:.2f})")
    return write(out, WAVE, time, flux, attrs)


# ---------------------------------------------------------------- dwarf nova
def wd_radius(m_wd):
    """White-dwarf radius (cm), Nauenberg 1972."""
    x = m_wd / 1.44
    return 0.0112 * RSUN * np.sqrt(x ** (-2 / 3) - x ** (2 / 3))


def planck(wave, t):
    """B_lambda (erg/s/cm2/A/sr), wave in A."""
    lam = wave * 1e-8
    with np.errstate(over="ignore"):
        return 2 * H * C**2 / lam**5 / np.expm1(H * C / (lam * K_B * t)) * 1e-8


def disc_flux(wave, mdot, m_wd, r_in, r_out, incl, d_pc):
    """A steady disc as blackbody rings, T^4 = 3 G M Mdot / (8 pi sigma R^3) (1 - sqrt(r_in / R)), seen at incl."""
    edges = np.geomspace(r_in, r_out, 301)
    r, dr = np.sqrt(edges[1:] * edges[:-1]), np.diff(edges)
    t4 = 3 * G * m_wd * MSUN * mdot / (8 * np.pi * SIGMA_SB * r**3) * (1 - np.sqrt(r_in / r))
    t = np.maximum(t4, 0) ** 0.25
    area = 2 * np.pi * r * dr
    return np.cos(np.radians(incl)) / (d_pc * PC) ** 2 * np.sum(planck(wave[None, :], t[:, None]) * area[:, None], axis=0)


def make_dwarf_nova(out, d_pc, mdot_q, mdot_peak, t_rise, plateau, tau_decay, r_out, t_wd, m_wd=0.8, incl=60.0):
    """Mdot (Msun/yr): mdot_q, then from t0 an exponential rise to mdot_peak over t_rise days, plateau days at the peak,
    an exponential decay (e-folding tau_decay) back to mdot_q. Light: the disc (white dwarf to r_out Rsun) plus the
    white dwarf as a blackbody. From 3 days before t0 until Mdot is within 1% of mdot_q."""
    r_in = wd_radius(m_wd)
    end_peak = t_rise + plateau
    end = end_peak + tau_decay * np.log(100 * mdot_peak / mdot_q)
    time = np.unique(np.r_[np.linspace(-3.0, 0, 31), np.linspace(0, end, 1200)])
    with np.errstate(over="ignore"):
        mdot = np.where(time <= 0, mdot_q,
                        np.where(time <= t_rise, mdot_q * (mdot_peak / mdot_q) ** (time / t_rise),
                                 np.where(time <= end_peak, mdot_peak,
                                          np.maximum(mdot_peak * np.exp(-(time - end_peak) / tau_decay), mdot_q))))
    white_dwarf = np.pi * planck(WAVE, t_wd) * (r_in / (d_pc * PC)) ** 2
    flux = np.array([disc_flux(WAVE, m * MSUN / YEAR, m_wd, r_in, r_out * RSUN, incl, d_pc) for m in mdot]) + white_dwarf
    rho_q, rho_p = rho_mag(WAVE, flux[0]), rho_mag(WAVE, flux[int(np.argmax(mdot))])
    attrs = dict(kind="dwarf nova", model="dwarf_nova", reference="Shakura & Sunyaev 1973; Nauenberg 1972",
                 z=float("nan"), peak_days=float(t_rise), distance_pc=float(d_pc),  # observe.py: the dust in front of it
                 params=f"Mdot {mdot_q:g} -> {mdot_peak:g} Msun/yr (rise {t_rise:g} d, plateau {plateau:g} d, decay e-fold "
                        f"{tau_decay:g} d); M_wd {m_wd:g}, T_wd {t_wd:g} K, r_out {r_out:g} Rsun, i {incl:g} deg, "
                        f"d {d_pc:g} pc; rho {rho_q:.2f} quiescent, {rho_p:.2f} peak")
    return write(out, WAVE, time, flux, attrs)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", nargs="?")
    p.add_argument("out", nargs="?", type=Path)
    p.add_argument("--list", action="store_true")
    p.add_argument("--z", type=float)
    p.add_argument("--peak-absmag", type=float)
    p.add_argument("--peak-band", default="bessellb")
    p.add_argument("--magsys", default="ab")
    p.add_argument("--host-ebv", type=float, default=0.0)
    p.add_argument("--param", action="append", default=[], help="model parameter, e.g. x1=0.5 (repeatable)")
    a = p.parse_args()
    if a.list:
        for name, version, kind, ref in sorted(catalogue(), key=lambda r: (r[2], r[0])):
            print(f"{kind:10s} {name:22s} {version:8s} {ref}")
    else:
        params = {k: float(v) for k, v in (s.split("=") for s in a.param)}
        print(make_sncosmo(a.out, a.model, a.z, a.peak_absmag, a.peak_band, a.magsys, params, a.host_ebv))
