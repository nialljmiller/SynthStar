"""Eclipsing binaries for the Argus movies: PHOEBE 2 light curves of two stars with MESA-atmosphere SEDs.

Output: runs/rsp_<field>_<label>/pulsation.h5 in the same format as the RSP pulsators (one cycle of SEDs at 10 pc,
time in days), so the movie treats a binary like any other variable.

Inputs (all Gaia DR3, per system)
- Timing and the observed light curve: vari_eclipsing_binary (I/358/veb): period, reference time, the two-Gaussian
  (+ half-period cosine) model of the G light curve, and the derived eclipse phases/depths/durations.
- Geometry, when Gaia has one: the non-single-star EclipsingBinary / EclipsingSpectro solution (I/357/tboeb, tboes):
  eccentricity, argument of periastron, inclination, Roche fill factors, temperature ratio (mass ratio for
  EclipsingSpectro). Used for detached systems (fill factors < 1).
  Otherwise ("fit" route): circular detached or contact systems are fitted with PHOEBE to Gaia's two-Gaussian model
  (inclination, radii / fill, temperature ratio; mass ratio for contact systems), on a hard budget (FIT_MAX_EVALS
  PHOEBE evaluations or FIT_MAX_SECONDS). Eccentric systems without a Gaia solution are not modelled.
- Absolute scale: the system's Gaia GSP-Phot Teff for the brighter (or, for NSS, Gaia's unfitted) component; the
  other from the temperature ratio; masses from the main-sequence Teff-mass relation of Pecaut & Mamajek (2013;
  version 2022.04.16, data/EEM_dwarf_UBVIJHK_colors_Teff.txt); orbit size from Kepler's third law.
- Brightness: the SEDs are scaled so the system's out-of-eclipse G equals Gaia's (geom_model_reference_level), as
  every constant star's brightness comes from Gaia. The absolute scale above (main-sequence masses at a blended
  GSP-Phot Teff) is good to ~0.5-1 mag; its G at the parallax distance is recorded, and a system is rejected only
  beyond DIST_REJECT_MAG (e.g. an evolved star taken for a dwarf). The light curve must reproduce Gaia's eclipses
  within the fit tolerance.

Light: PHOEBE computes each component's visible flux (mesh: intensity x mu x area x visibility) in Gaia BP and RP,
blackbody atmospheres with PHOEBE's default (manual) limb darkening, no irradiation (as Gaia's NSS model: Roche
geometry, blackbody). The system SED at each time is the sum of the two component SEDs (MESA colors' Kurucz 2003
surface fluxes at each star's Teff, log g, [M/H], scaled by (R/10 pc)^2), each weighted by its visible fraction,
interpolated in wavelength between the BP and RP pivot wavelengths (constant outside them).
"""
from functools import lru_cache
import json
import logging
import time

import h5py
import numpy as np

from . import config as C
from .dust import redden

SIGMA_SB = 5.6704e-5
RSUN, PC, G_CGS, MSUN = 6.957e10, 3.0857e18, 6.674e-8, 1.989e33
FIT_MAX_EVALS = 100
FIT_MAX_SECONDS = 180.0
FIT_TOL_MAG = 0.02        # accepted if the rms residual <= max(this, FIT_TOL_FRAC x primary depth)
FIT_TOL_FRAC = 0.15
DIST_REJECT_MAG = 1.5  # model vs Gaia G at the parallax distance: recorded; rejected only beyond this (gross misfit)
LOW_TRIANGLES, HIGH_TRIANGLES = 500, 3000
N_FINE = 400              # SEDs per orbit
DWARFS = C.DATA / "EEM_dwarf_UBVIJHK_colors_Teff.txt"


# ---------------------------------------------------------------- spectra and photometry (MESA colors data)
@lru_cache
def _cube():
    p = C.colors_data() / "stellar_models" / "Kurucz2003all" / "flux_cube.bin"
    with open(p, "rb") as f:
        nt, ng, nm, nl = (int(x) for x in np.fromfile(f, np.int32, 4))
        teff, logg, meta, wave = (np.fromfile(f, np.float64, n) for n in (nt, ng, nm, nl))
        flux = np.fromfile(f, np.float64, nt * ng * nm * nl).reshape((nt, ng, nm, nl), order="F")
    return teff, logg, meta, wave, flux


def surface_sed(teff, logg, mh):
    """Surface flux F_lambda (erg/s/cm2/A) from MESA colors' Kurucz 2003 cube: bilinear in (Teff, log g) in log flux
    at the nearest [M/H]; empty grid points take the nearest filled log g at that Teff."""
    T, Gg, M, wave, flux = _cube()
    im = int(np.argmin(np.abs(M - mh)))
    i1 = int(np.clip(np.searchsorted(T, teff), 1, len(T) - 1))
    j1 = int(np.clip(np.searchsorted(Gg, logg), 1, len(Gg) - 1))
    out = np.zeros_like(wave)
    for i, wt in ((i1 - 1, (T[i1] - teff) / (T[i1] - T[i1 - 1])), (i1, (teff - T[i1 - 1]) / (T[i1] - T[i1 - 1]))):
        for j, wg in ((j1 - 1, (Gg[j1] - logg) / (Gg[j1] - Gg[j1 - 1])), (j1, (logg - Gg[j1 - 1]) / (Gg[j1] - Gg[j1 - 1]))):
            jj = j
            if flux[i, jj, im].max() <= 0:
                filled = [k for k in range(len(Gg)) if flux[i, k, im].max() > 0]
                jj = min(filled, key=lambda k: abs(Gg[k] - Gg[j]))
            out += np.clip(wt, 0, 1) * np.clip(wg, 0, 1) * np.log(np.clip(flux[i, jj, im], 1e-300, None))
    return wave, np.exp(out)


@lru_cache
def _band(name):
    t = np.genfromtxt(C.colors_data() / "filters" / "GAIA" / "GAIA" / f"{name}.dat", delimiter=",", names=True)
    return t["Wavelength"], t["Transmission"]


@lru_cache
def _vega():
    v = np.genfromtxt(C.colors_data() / "stellar_models" / "vega_flam.csv", delimiter=",", skip_header=1)
    return v[:, 0], v[:, 1]


def vega_mag(wave, flam, band):
    """Photon-counting Vega magnitude in a Gaia band (MESA colors' filter curves and Vega spectrum)."""
    bw, bt = _band(band)
    vw, vf = _vega()
    num = np.trapezoid(np.interp(bw, wave, flam) * bt * bw, bw)
    den = np.trapezoid(np.interp(bw, vw, vf) * bt * bw, bw)
    return -2.5 * np.log10(num / den)


def pivot(band):
    bw, bt = _band(band)
    return float(np.sqrt(np.trapezoid(bt * bw, bw) / np.trapezoid(bt / bw, bw)))


@lru_cache
def _dwarfs():
    teff, mass = [], []
    for line in DWARFS.read_text().splitlines()[23:]:
        if line.startswith("#"):
            break
        f = line.split()
        if f[-2] == "...":  # L, T and Y dwarfs: no mass tabulated
            continue
        teff.append(float(f[1]))
        mass.append(float(f[-2]))
    o = np.argsort(teff)
    return np.array(teff)[o], np.array(mass)[o]


def ms_mass(teff):
    """Main-sequence mass (Msun) at Teff: Pecaut & Mamajek (2013), table version 2022.04.16."""
    t, m = _dwarfs()
    return float(np.interp(teff, t, m))


def sma_rsun(period_d, mtot):
    return (G_CGS * mtot * MSUN * (period_d * 86400) ** 2 / (4 * np.pi ** 2)) ** (1 / 3) / RSUN


# ---------------------------------------------------------------- Gaia's two-Gaussian model
def gaia_model_mag(veb, phase):
    """Gaia's geometric model of the G light curve (mag, relative to its reference level) at phases relative to its
    reference time. The half-period cosine's sign is fixed by physics: its faint extrema fall on the eclipses."""
    m = np.zeros_like(phase, dtype=float)
    for k in (1, 2):
        d, ph, s = (float(veb[f"{q}Gauss{k}"]) for q in ("Depth", "Phase", "sigPhase"))
        if np.isfinite(d) and np.isfinite(s) and s > 0:
            for w in (-1, 0, 1):
                m += d * np.exp(-0.5 * ((phase - ph + w) / s) ** 2)
    a, pc = float(veb["AmpCHP"]), float(veb["PhaseCHP"])
    if np.isfinite(a) and np.isfinite(pc):
        sign = 1.0 if np.cos(4 * np.pi * (float(veb["PhaseE1"]) - pc)) > 0 else -1.0
        m += sign * a * np.cos(4 * np.pi * (phase - pc))
    return m


# ---------------------------------------------------------------- PHOEBE
def _phoebe():
    import phoebe

    logging.disable(logging.WARNING)
    if hasattr(phoebe, "progressbars_off"):
        phoebe.progressbars_off()
    return phoebe


def make_bundle(p, phases, contact, ntriangles, mesh=False):
    """A PHOEBE bundle for parameters p (period d, incl deg, q, sma Rsun, teff1, teff2, requiv1, requiv2 Rsun, ecc,
    per0 deg), with primary eclipse (superior conjunction of the primary) at phase 0."""
    phoebe = _phoebe()
    b = phoebe.default_binary(contact_binary=contact)
    b.set_value("period@binary", p["period"])
    b.set_value("t0_supconj@binary", 0.0)
    b.set_value("q@binary", p["q"])
    b.set_value("sma@binary", p["sma"])
    b.set_value("incl@binary", p["incl"])
    if not contact:
        b.set_value("ecc@binary", p.get("ecc", 0.0))
        b.set_value("per0@binary", p.get("per0", 90.0))
    b.set_value("teff@primary", p["teff1"])
    b.set_value("teff@secondary", p["teff2"])
    b.set_value("requiv@primary", p["requiv1"])
    if not contact:
        b.set_value("requiv@secondary", p["requiv2"])
    times = np.asarray(phases) * p["period"]
    for pb in (("Gaia:BP", "Gaia:RP") if mesh else ("Gaia:G",)):
        b.add_dataset("lc", compute_times=times, passband=pb, dataset="lc" + pb.split(":")[1])
    if mesh:
        b.add_dataset("mesh", compute_times=times, dataset="mesh01",
                      columns=["visibilities", "mus", "areas", "abs_intensities@lcBP", "abs_intensities@lcRP"])
    b.set_value_all("atm", "blackbody")
    b.set_value_all("ld_mode", "manual")
    b.set_value_all("ld_mode_bol", "manual")
    b.set_value_all("ntriangles", ntriangles)
    return b


def lc_g(p, phases, contact):
    """PHOEBE G light curve (mag, relative to its brightest point), low resolution."""
    b = make_bundle(p, phases, contact, LOW_TRIANGLES)
    b.run_compute(irrad_method="none")
    f = b.get_value("fluxes", dataset="lcG", context="model")
    return -2.5 * np.log10(f / f.max())


def component_fluxes(p, phases, contact):
    """Visible flux of each component in BP and RP at the phases: {component: {band: array}}."""
    b = make_bundle(p, phases, contact, HIGH_TRIANGLES, mesh=True)
    b.run_compute(irrad_method="none")
    times = np.asarray(phases) * p["period"]
    out = {}
    for c in ("primary", "secondary"):
        out[c] = {}
        for band in ("BP", "RP"):
            fl = []
            for t in times:
                q = b.filter(context="model", kind="mesh", component=c, time=t)
                i_ = q.get_value(qualifier="abs_intensities", dataset="lc" + band)
                mu, a, v = (q.get_value(qualifier=x, dataset="mesh01") for x in ("mus", "areas", "visibilities"))
                fl.append(np.nansum(i_ * np.clip(mu, 0, None) * a * v))
            out[c][band] = np.array(fl)
    return out


# ---------------------------------------------------------------- the two routes
def _teff_ratio_from_depths(veb):
    """Surface-brightness ratio ~ secondary/primary eclipse flux depths (central eclipses), turned into a
    temperature ratio with blackbodies at the G pivot wavelength. A starting value only."""
    d1, d2 = (1 - 10 ** (-0.4 * float(veb[k])) for k in ("DepthE1", "DepthE2"))
    j = d2 / d1 if np.isfinite(d2) and d1 > 0 else 0.5
    lam, c2, t1 = pivot("G"), 1.4388e8, 6000.0
    s = lambda t: 1 / np.expm1(c2 / (lam * t))  # noqa: E731
    grid = np.linspace(0.3, 2.0, 400)
    return float(grid[np.argmin(np.abs(s(grid * t1) / s(t1) - np.clip(j, 0.01, 3)))])


def fit_phases(veb):
    """Where the fit compares light curves: 15 phases across each Gaia eclipse (+-2.5 sigma), 10 elsewhere;
    relative to the primary eclipse."""
    e1 = float(veb["PhaseE1"])
    out = [np.linspace(-0.5, 0.5, 10, endpoint=False)]
    for k in (1, 2):
        ph, s = float(veb[f"PhaseGauss{k}"]), float(veb[f"sigPhaseGauss{k}"])
        if np.isfinite(ph) and np.isfinite(s):
            out.append(ph - e1 + np.linspace(-2.5 * s, 2.5 * s, 15))
    return np.sort(np.concatenate(out) % 1.0)


def fit_route(veb, teff1, contact, seed=None, orbit=None):
    """PHOEBE fit to Gaia's model, budget-capped. orbit: fixed (ecc, per0) for a detached eccentric orbit (from Gaia's
    NSS solution). Returns (params, rms mag, evaluations, seconds)."""
    from scipy.optimize import minimize

    period = 1 / float(veb["Freq"])
    phases = fit_phases(veb)
    e1 = float(veb["PhaseE1"])
    target = gaia_model_mag(veb, phases + e1)
    target = target - target.min()
    tr0 = _teff_ratio_from_depths(veb)
    state = dict(n=0, t0=time.time(), best=(np.inf, None))

    def params(x):
        incl = float(np.clip(x[0], 40, 90))
        tratio = float(np.clip(x[3], 0.3, 2.0))
        teff2 = teff1 * tratio
        m1 = ms_mass(teff1)
        if contact:
            q = float(np.clip(x[2], 0.07, 1.0))
            sma = sma_rsun(period, m1 * (1 + q))
            p = dict(period=period, incl=incl, q=q, sma=sma, teff1=teff1, teff2=teff2)
            # requiv of the primary between the contact limits, from x[1] in [0, 1]
            phoebe = _phoebe()
            b = phoebe.default_binary(contact_binary=True)
            b.set_value("q@binary", q), b.set_value("sma@binary", sma), b.set_value("period@binary", period)
            rmin, rmax = (b.get_value(f"requiv_{k}@primary@component") for k in ("min", "max"))
            p["requiv1"] = rmin + float(np.clip(x[1], 0.02, 0.98)) * (rmax - rmin)
        else:
            m2 = ms_mass(teff2)
            q = m2 / m1
            sma = sma_rsun(period, m1 + m2)
            rsum, k = float(np.clip(x[1], 0.02, 0.8)), float(np.clip(x[2], 0.2, 1.5))
            r1 = rsum / (1 + k)
            p = dict(period=period, incl=incl, q=q, sma=sma, teff1=teff1, teff2=teff2, requiv1=r1 * sma,
                     requiv2=k * r1 * sma, ecc=orbit[0] if orbit else 0.0, per0=orbit[1] if orbit else 90.0)
        return p

    def objective(x):
        if state["n"] >= FIT_MAX_EVALS or time.time() - state["t0"] > FIT_MAX_SECONDS:
            raise StopIteration
        state["n"] += 1
        try:
            p = params(x)
            m = lc_g(p, phases, contact)
        except Exception:  # PHOEBE rejects unphysical systems (e.g. overflowing Roche lobes): penalise, the simplex moves on
            return 10.0
        r = float(np.sqrt(np.mean((m - target) ** 2)))
        if r < state["best"][0]:
            state["best"] = (r, p)
        return r

    if contact:
        x0 = seed or [75.0, 0.5, 0.5, min(tr0, 1.2)]
        step = [8.0, 0.25, 0.25, 0.08]
    else:
        dur = float(veb["DurE1"]) if np.isfinite(float(veb["DurE1"])) else 0.1
        x0 = seed or [86.0, float(np.clip(np.sin(np.pi * dur), 0.05, 0.6)), 0.8, tr0]
        step = [4.0, 0.05, 0.25, 0.1]
    simplex = np.array([x0] + [np.array(x0) + np.eye(4)[i] * step[i] for i in range(4)])
    try:
        minimize(objective, x0, method="Nelder-Mead", options=dict(initial_simplex=simplex, xatol=1e-3, fatol=1e-4,
                                                                    maxfev=FIT_MAX_EVALS))
    except StopIteration:
        pass
    return state["best"][1], state["best"][0], state["n"], time.time() - state["t0"]


def nss_route(veb, nss, teff_gsp):
    """Geometry from Gaia's EclipsingBinary / EclipsingSpectro solution (detached systems)."""
    period = float(nss["Per"])
    tr = float(nss["Tratio"])
    primary_fitted = int(nss["f_Tratio"]) == 1 if np.isfinite(float(nss.get("f_Tratio", np.nan))) else True
    # Gaia: temperature_ratio = fitted / unfitted; the unfitted temperature takes the GSP-Phot value
    teff1, teff2 = (teff_gsp * tr, teff_gsp) if primary_fitted else (teff_gsp, teff_gsp * tr)
    m1, m2 = ms_mass(teff1), ms_mass(teff2)
    q = float(nss["MassRatio"]) if np.isfinite(float(nss.get("MassRatio", np.nan))) else m2 / m1
    sma = sma_rsun(period, m1 * (1 + q))
    p = dict(period=period, incl=float(nss["inc"]), q=q, sma=sma, teff1=teff1, teff2=teff2,
             ecc=float(nss["ecc"]) if np.isfinite(float(nss.get("ecc", np.nan))) else 0.0,
             per0=float(nss["omega"]) if np.isfinite(float(nss.get("omega", np.nan))) else 90.0)
    # fill factor s = r_pole / r_pole(L1 lobe); requiv ~ s x requiv_max (an approximation, exact for spheres)
    phoebe = _phoebe()
    b = phoebe.default_binary()
    for k, v in (("period@binary", period), ("q@binary", q), ("sma@binary", sma), ("ecc@binary", p["ecc"]),
                 ("per0@binary", p["per0"])):
        b.set_value(k, v)
    p["requiv1"] = float(nss["ffactp"]) * b.get_value("requiv_max@primary@component")
    p["requiv2"] = float(nss["ffacts"]) * b.get_value("requiv_max@secondary@component")
    phases = fit_phases(veb)
    target = gaia_model_mag(veb, phases + float(veb["PhaseE1"]))
    rms = float(np.sqrt(np.mean((lc_g(p, phases, False) - (target - target.min())) ** 2)))
    return p, rms


# ---------------------------------------------------------------- the model
def component_seds(p, contact, mh=0.0):
    """Each component's SED at 10 pc (MESA colors' Kurucz surface flux at its Teff, log g, [M/H], x (R / 10 pc)^2),
    masses from the main-sequence Teff-mass relation (the secondary q x the primary when p has q only for a contact
    system). Returns wavelength, [f1, f2], R1, R2 (Rsun), M1, M2 (Msun)."""
    r1, r2 = p["requiv1"], (p["requiv2"] if "requiv2" in p and not contact else p["requiv1"] * p["q"] ** 0.46)
    if contact:  # PHOEBE's constrained secondary radius
        b = make_bundle(p, [0.0], True, LOW_TRIANGLES)
        r2 = b.get_value("requiv@secondary@component")
    logg = lambda m, r: np.log10(G_CGS * m * MSUN / (r * RSUN) ** 2)  # noqa: E731
    m1 = p.get("m1") or ms_mass(p["teff1"])
    m2 = p.get("m2") or m1 * p["q"]
    w, s1 = surface_sed(p["teff1"], logg(m1, r1), mh)
    _, s2 = surface_sed(p["teff2"], logg(m2, r2), mh)
    return w, [s * (r * RSUN / (10 * PC)) ** 2 for s, r in ((s1, r1), (s2, r2))], r1, r2, m1, m2


def orbit_seds(p, contact, w, f10):
    """One orbit of system SEDs from PHOEBE's visible flux of each component (module docstring). Returns phases
    (0 = primary eclipse), SEDs, and the time of maximum light after the primary eclipse (days)."""
    phases = np.linspace(0, 1, N_FINE + 1)
    comp = component_fluxes(p, phases, contact)
    lam_bp, lam_rp = pivot("Gbp"), pivot("Grp")
    x = np.clip((w - lam_bp) / (lam_rp - lam_bp), 0, 1)
    seds = np.zeros((len(phases), len(w)))
    for c, f in zip(("primary", "secondary"), f10):
        wb = comp[c]["BP"] / np.median(comp[c]["BP"])
        wr = comp[c]["RP"] / np.median(comp[c]["RP"])
        seds += (wb[:, None] * (1 - x)[None, :] + wr[:, None] * x[None, :]) * f[None, :]
    light = comp["primary"]["RP"] + comp["secondary"]["RP"]
    return phases, seds, phases[int(np.argmax(light))] * p["period"]


def write_orbit(path, w, seds, time_days, attrs):
    """One cycle of SEDs (f_lambda at 10 pc) against time, in the RSP pulsators' format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs.update(**attrs, flux_units="erg s^-1 cm^-2 A^-1 (f_lambda at 10 pc)")
        f.create_dataset("wavelength", data=w)
        f.create_dataset("flux", data=seds, compression="gzip", compression_opts=4)
        f.create_dataset("time_days", data=time_days)


def contact_requiv(period, q, sma, fillout):
    """The primary's equivalent radius (Rsun) of a contact system with PHOEBE's fillout factor (0: the inner critical
    surface, 1: the outer)."""
    b = _phoebe().default_binary(contact_binary=True)
    for k, v in (("period@binary", period), ("q@binary", q), ("sma@binary", sma)):
        b.set_value(k, v)
    b.flip_constraint("pot@contact_envelope", solve_for="requiv@primary")
    b.flip_constraint("fillout_factor", solve_for="pot")
    b.set_value("fillout_factor@contact_envelope@component", fillout)
    b.run_delayed_constraints()
    return float(b.get_value("requiv@primary@component"))


def model_binary(target, out_dir, field_dm, ebv, mh=0.0):
    """Model one eclipsing binary; write out_dir/pulsation.h5. target: dict with veb (row as dict), nss (row as dict
    or None), teff_gsp, logg_gsp, G (Gaia mean G). Returns (ok, notes)."""
    veb, nss = target["veb"], target.get("nss")
    notes = {}
    period = 1 / float(veb["Freq"])
    teff_gsp = float(target["teff_gsp"])
    if not np.isfinite(teff_gsp):
        return False, dict(reason="no Gaia GSP-Phot Teff for the system")
    if np.isfinite(target.get("logg_gsp", np.nan)) and target["logg_gsp"] < 3.5:
        return False, dict(reason=f"evolved (GSP-Phot log g {target['logg_gsp']:.2f}): the main-sequence mass relation does not apply")
    ecc_like = abs(((float(veb["PhaseE2"]) - float(veb["PhaseE1"])) % 1.0) - 0.5) > 0.01 if np.isfinite(float(veb["PhaseE2"])) else False
    detached_nss = nss is not None and float(nss["ffactp"]) < 1 and float(nss["ffacts"]) < 1
    if detached_nss:
        p, rms = nss_route(veb, nss, teff_gsp)
        contact = False
        notes["route"] = ("Gaia DR3 NSS " + ("EclipsingSpectro" if np.isfinite(float(nss.get("MassRatio", np.nan))) else "EclipsingBinary") +
                          " solution (ecc, omega, incl, fill factors, temperature ratio" +
                          (", mass ratio" if np.isfinite(float(nss.get("MassRatio", np.nan))) else "") + "); requiv ~ fill factor x Roche-lobe requiv")
        n, secs = 1, 0.0
    else:
        if ecc_like:
            return False, dict(reason="eccentric (secondary eclipse not at phase 0.5) and no Gaia NSS solution")
        dur1 = float(veb["DurE1"])
        contact = (nss is not None and float(nss["ffactp"]) >= 1) or (np.isfinite(dur1) and dur1 >= 0.35)
        seed = [float(nss["inc"]), 0.5, 0.5, 1.0] if contact and nss is not None else None
        p, rms, n, secs = fit_route(veb, teff_gsp, contact, seed)
        notes["route"] = (f"PHOEBE fit by us to Gaia's two-Gaussian light-curve model ({'contact' if contact else 'detached, circular'}; "
                          f"{n} evaluations, {secs:.0f} s; cap {FIT_MAX_EVALS} / {FIT_MAX_SECONDS:.0f} s)")
        if p is None:
            return False, dict(reason="fit failed", **notes)
    depth = float(veb["DepthE1"])
    tol = max(FIT_TOL_MAG, FIT_TOL_FRAC * depth)
    if detached_nss and not rms <= tol:
        # Gaia's geometry as translated here (approximate fill factor -> radius; main-sequence mass ratio when Gaia
        # gives none) misses Gaia's eclipses: refit, starting from it, with its eccentricity and periastron fixed
        rsum0 = (p["requiv1"] + p["requiv2"]) / p["sma"]
        seed = [p["incl"], rsum0, p["requiv2"] / p["requiv1"], p["teff2"] / p["teff1"]]
        nss_rms = rms
        p, rms, n, secs = fit_route(veb, p["teff1"], False, seed, orbit=(p["ecc"], p["per0"]))
        notes["route"] += (f"; as translated it missed Gaia's eclipses (rms {nss_rms:.4f} mag), so refitted by us from it "
                           f"with ecc and omega fixed ({n} evaluations, {secs:.0f} s)")
        if p is None:
            return False, dict(reason="NSS-seeded fit failed", **notes)
    notes["light_curve_check"] = f"rms vs Gaia's eclipse model {rms:.4f} mag (tolerance {tol:.4f})"
    if not rms <= tol:
        return False, dict(reason=f"does not reproduce Gaia's eclipses: rms {rms:.4f} mag > {tol:.4f}", **notes)

    # component SEDs (surface flux x (R / 10 pc)^2) and the distance check at maximum light (spheres)
    w, f10, r1, r2, m1, m2 = component_seds(p, contact, mh)
    obs = redden(w, (f10[0] + f10[1]) * 10 ** (-0.4 * field_dm), ebv)
    g_model = vega_mag(w, obs, "G")
    notes["distance_check"] = (f"model G {g_model:.2f} at the parallax distance vs Gaia {target['G']:.2f} "
                               f"({g_model - target['G']:+.2f} mag); SEDs scaled to Gaia's G")
    if not abs(g_model - target["G"]) <= DIST_REJECT_MAG:
        return False, dict(reason=f"grossly inconsistent with the parallax distance: model G {g_model:.2f} vs Gaia {target['G']:.2f}", **notes)
    scale = 10 ** (-0.4 * (target["G"] - g_model))
    f10 = [f * scale for f in f10]

    # one orbit of SEDs
    phases, seds, t_max = orbit_seds(p, contact, w, f10)
    t_ecl1 = float(veb["TimeRef"]) + 2455197.5 + float(veb["PhaseE1"]) * period  # BJD (TCB) of a primary eclipse
    write_orbit(out_dir / "pulsation.h5", w, seds, phases * period, dict(
        label=out_dir.name, kind="EB", gaia_source=int(target["source"]), gaia_period=period,
        gaia_epoch_bjd=t_ecl1 + t_max, nonlinear_period=period, distance_modulus=field_dm, ebv=ebv,
        mass=m1, mass2=m2, teff1=p["teff1"], teff2=p["teff2"], R1=r1, R2=r2, incl=p["incl"], q=p["q"],
        sma=p["sma"], ecc=p.get("ecc", 0.0), contact=bool(contact), route=notes["route"], brightness_scale=scale))
    notes.update(params={k: (round(v, 5) if isinstance(v, float) else v) for k, v in p.items()}, M1=round(m1, 3),
                 M2=round(m2, 3), R1=round(r1, 3), R2=round(r2, 3), epoch_max_light_bjd=t_ecl1 + t_max,
                 masses="main-sequence Teff-mass relation (Pecaut & Mamajek 2013, table 2022.04.16)" +
                        ("; q from the fit" if contact else ""))
    (out_dir / "binary.json").write_text(json.dumps(notes, indent=1, default=float))
    return True, notes
