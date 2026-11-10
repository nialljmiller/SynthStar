"""The editable star's templates: library stars (stars/) by type, and each one's band light integrated once.

A template's SEDs go through ArgusSim's photometry once, when it is chosen (ABPhot: magnitudes above the atmosphere,
electrons through the atmosphere on an airmass grid). Everything the sliders do after that is exact arithmetic on
those tables: the band integrals are linear in the SED, so a brightness or amplitude change scales them as it scales
the SED, and a period change rescales the phase.

Types and their library sources:

    RRab, RRc               lib_rrab_*, lib_rrc_* (MESA RSP): the model nearest the period asked for
    contact binary          libnm_contact_* (PHOEBE W UMa): nearest in period, then stretched to it
    detached binary         libnm_detached_ms_* (PHOEBE): nearest in period, then stretched to it
    dwarf nova              libnm_dwarf_nova_* (steady-disc outburst): its time axis stretched by the timescale
    supernova (Ia)          libnm_salt_shape_* (SALT, SN Ia): its time axis stretched by the timescale
    microlensing            a point lens (Paczynski) on a library source SED: tE and u0 set directly
"""
import json
import os
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
STARS = REPO / "stars"
AIRMASS = np.array([1.0, 1.25, 1.5, 2.0, 2.5, 3.0])  # the electrons table's airmass grid (extinction is smooth in X)
CACHE = REPO / "data" / "interactive" / "templates"  # each integrated template, so choosing one is instant
PER_TYPE = 12  # templates precomputed per periodic type, spread in period (the period slider snaps to these)
VERSION = 2  # of the cached tables (2: periodic templates on an adaptive phase grid)

TYPES = {
    "RR Lyrae (RRab)": dict(prefix="lib_rrab", timing="periodic", period=(0.40, 0.80, 0.55)),
    "RR Lyrae (RRc)": dict(prefix="lib_rrc", timing="periodic", period=(0.22, 0.42, 0.32)),
    "contact binary (W UMa)": dict(prefix="libnm_contact", timing="periodic", period=(0.22, 1.0, 0.35)),
    "detached eclipsing binary": dict(prefix="libnm_detached_ms", timing="periodic", period=(0.5, 10.0, 2.0)),
    "dwarf nova": dict(prefix="libnm_dwarf_nova", timing="transient", timescale=(0.2, 5.0, 1.0)),
    "supernova (Ia)": dict(prefix="libnm_salt_shape", timing="transient", timescale=(0.5, 2.0, 1.0)),
    "microlensing": dict(prefix="libnm_microlensing", timing="lens", timescale=(0.05, 5.0, 0.5)),
}


@lru_cache(maxsize=None)
def library(prefix):
    """[(name, meta)] of the built library stars with this prefix whose SED covers 300-1000 nm."""
    import h5py

    out = []
    for d in sorted(STARS.glob(f"{prefix}_*")):
        if not (d / "meta.json").exists() or not (d / "sed.h5").exists():
            continue
        with h5py.File(d / "sed.h5") as h:
            w = h["wavelength"]
            if float(w[0]) > 3000.0 or float(w[-1]) < 10000.0:
                continue
        out.append((d.name, json.loads((d / "meta.json").read_text())))
    return out


def precomputed(kind):
    """The templates of a type the app offers: for a periodic type PER_TYPE library stars spread evenly in log period
    over the slider's range (each the library star nearest that period); otherwise the one choose() picks."""
    spec = TYPES[kind]
    if spec["timing"] != "periodic":
        return [choose(kind)]
    stars = library(spec["prefix"])
    lo, hi, _ = spec["period"]
    names = []
    for p in np.exp(np.linspace(np.log(lo), np.log(hi), PER_TYPE)):
        name = min(stars, key=lambda s: abs(np.log(float(s[1]["period"]) / p)))[0]
        if name not in names:
            names.append(name)
    return names


def nearest(kind, period):
    """The precomputed template of a periodic type nearest a period (log)."""
    names = precomputed(kind)
    return min(names, key=lambda n: abs(np.log(float(json.loads((STARS / n / "meta.json").read_text())["period"])
                                                 / period)))


def choose(kind, period=None):
    """The template for a type: periodic types the library star nearest the period; others the first suitable one
    (a fixed choice, so a slider never swaps the star's shape)."""
    spec = TYPES[kind]
    stars = library(spec["prefix"])
    if not stars:
        raise FileNotFoundError(f"no built {spec['prefix']}_* stars in {STARS}")
    if spec["timing"] == "periodic" and period is not None:
        return min(stars, key=lambda s: abs(float(s[1]["period"]) - period))[0]
    if kind == "supernova (Ia)":  # a normal SN Ia: SALT x1, c near zero, low host dust
        def oddness(s):
            p = s[1]["params"]
            x1 = float(p.split("x1 = ")[1].split(",")[0]) if "x1 = " in p else 0.0
            c = float(p.split("c = ")[1].split(",")[0]) if "c = " in p else 0.0
            return abs(x1) + 3 * abs(c)
        return min(stars, key=oddness)[0]
    return stars[0][0]


@lru_cache(maxsize=64)
def light(name, ebv=0.0):
    """A template's light (see _integrate), from the disk cache when there."""
    path = CACHE / f"{name}_ebv{ebv:.3f}_v{VERSION}.npz"
    if path.exists():
        import astropy.units as u

        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return dict(name=name, kind=str(z["kind"]), timing=str(z["timing"]), wave=z["wave"] * u.AA,
                    seds=z["seds"], x=z["x"], mag={b: z[f"mag_{b}"] for b in ("beta", "rho")},
                    electrons={b: z[f"electrons_{b}"] for b in ("beta", "rho")},
                    period=None if float(z["period"]) < 0 else float(z["period"]), meta=meta)
    out = _integrate(name, ebv)
    CACHE.mkdir(parents=True, exist_ok=True)
    # written under this process's own name, then moved into place in one step: several processes (the app's
    # workers, the precompute) may integrate the same template at once, and none may read a half-written file
    part = path.with_name(f"{path.stem}.{os.getpid()}.{threading.get_ident()}.part.npz")
    np.savez_compressed(part, kind=out["kind"], timing=out["timing"], wave=out["wave"].to_value("AA"),
                        seds=out["seds"].astype(np.float32), x=out["x"], period=-1.0 if out["period"] is None else out["period"],
                        meta=json.dumps(out["meta"]), **{f"mag_{b}": out["mag"][b] for b in ("beta", "rho")},
                        **{f"electrons_{b}": out["electrons"][b] for b in ("beta", "rho")})
    os.replace(part, path)
    return out


def _integrate(name, ebv=0.0):
    """A template's light, integrated once: dict(kind, timing, wave [A], seds (n x wave) at 10 pc or as stored,
    x (phase 0-1 for periodic, days for transients), mag {beta, rho} above the atmosphere (AB), electrons
    {beta, rho}: (n x airmass) per second through the atmosphere, period (periodic), meta).

    The steps integrated are where the light changes (_steps): a periodic model is sampled at 2000 phases (cheap) and
    kept where its light changed by 1% since the last kept phase, at most 0.01 apart, so a narrow eclipse is
    resolved; a transient likewise in time."""
    import h5py

    import night_movie as nm  # the engine's own models and photometry

    from synthstar import dust

    folder = STARS / name
    meta = json.loads((folder / "meta.json").read_text())
    ab = photometry()
    if meta["timing"] == "periodic":
        model = nm.RSPStar(folder / "sed.h5", ab, ebv=ebv, distance=10.0)
        fine = np.linspace(0, 1, 2000, endpoint=False)
        grid = fine[_steps(fine, np.array([np.nansum(model.sed(p)) for p in fine])[:, None], change=0.01, gap=0.01)]
        seds = np.array([model.sed(p) for p in grid])
        wave, x, period = model.wave, grid, model.period
    else:
        with h5py.File(folder / "sed.h5") as h:
            wave_a, flux, t = h["wavelength"][:], h["flux"][:], h["time_days"][:]
        flux = dust.redden(wave_a, flux, ebv)
        import astropy.units as u

        wave, x, period = wave_a * u.AA, t, None
        keep = _steps(t, flux)
        seds, x = flux[keep], t[keep]
    mag = {b: np.array([float(ab.mag_from_sed(band, wave, f * nm.FLAM)) if f.any() else np.inf for f in seds])
           for b, band in nm.BANDS.items()}
    electrons = {b: np.array([[ab.electrons_from_sed(band, wave, f * nm.FLAM, airmass=a).value if f.any() else 0.0
                               for a in AIRMASS] for f in seds]) for b, band in nm.BANDS.items()}
    return dict(name=name, kind=meta.get("kind", meta.get("generator")), timing=meta["timing"], wave=wave, seds=seds,
                x=np.asarray(x, float), mag=mag, electrons=electrons, period=period, meta=meta)


def _steps(t, flux, change=0.01, gap=0.25):
    """The steps of a transient worth integrating: each where its light (the SED's sum, a cheap proxy) has changed by
    more than `change` (fractional) since the last kept step, or `gap` days have passed; the first and last always.
    A fast rise (a dwarf nova's ~0.05 d) is kept in full, a slow decline sparsely."""
    total = np.nansum(np.clip(flux, 0.0, None), axis=1)
    keep, last_v, last_t = [0], total[0], t[0]
    for i in range(1, len(t) - 1):
        ref = max(abs(last_v), 1e-300)
        if abs(total[i] - last_v) / ref > change or t[i] - last_t >= gap:
            keep.append(i)
            last_v, last_t = total[i], t[i]
    keep.append(len(t) - 1)
    return np.array(keep)


@lru_cache(maxsize=1)
def photometry():
    """ArgusSim's photometry, as the engine builds it (night_movie.main)."""
    from argus_sim import ABPhot, SystemThroughput
    from argus_sim import c as cfg

    return ABPhot(throughput=SystemThroughput(filters={"g": True, "rho": True},
                                              throughput_loss=cfg.telescope.throughput_loss),
                  collecting_area=cfg.telescope.collecting_area)
