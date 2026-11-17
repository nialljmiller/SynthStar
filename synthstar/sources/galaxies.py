"""Galaxy model libraries: the galaxy counterpart of the stellar population library (sources/evolution.py).

Two libraries, both survey-agnostic (no instrument, no position, no Milky Way dust):

  galaxy spectra   BC03 stellar-population spectra (Bruzual & Charlot 2003, 2016 MILES update, via bagpipes) over
                   star-formation history x metallicity x Calzetti (2000) dust x redshift, with nebular emission
                   (bagpipes' Cloudy grid). Each spectrum carries synthetic AB magnitudes in the catalogue bands a
                   real galaxy is matched by (Pan-STARRS, DECam, BASS/MzLS), computed with SED_Tools filter curves.
  galaxy shapes    the parametric (Sersic and bulge+disk) fits of the HST/ACS COSMOS galaxies (GalSim's COSMOS
                   training sample), with their F814W magnitude and photometric redshift, to give a shape to a
                   catalogue galaxy that has no measured one.

synthstar/galaxy_templates.py loads both. Matching a catalogue galaxy, placing it and drawing it are the image
maker's job.
"""
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

from . import Generator

# --------------------------------------------------------------------------------------------- spectra library
# (label, age Gyr, tau Gyr): exponentially declining star formation, old/passive -> young/bursting. The age is capped
# at the age of the universe at each redshift minus 0.3 Gyr (a galaxy cannot be older than the universe).
SFH = [["passive", 11.0, 0.3], ["red", 8.0, 0.6], ["Sab", 6.0, 1.0], ["Sbc", 4.0, 2.0], ["Scd", 2.5, 4.0],
       ["starforming", 1.0, 8.0], ["starburst", 0.3, 10.0]]
BANDS = ["PAN-STARRS/PS1/g", "PAN-STARRS/PS1/r", "PAN-STARRS/PS1/i", "PAN-STARRS/PS1/z", "PAN-STARRS/PS1/y",
         "CTIO/DECam/g", "CTIO/DECam/r", "CTIO/DECam/i", "CTIO/DECam/z", "CTIO/DECam/Y",
         "BOK/BASS/g", "BOK/BASS/r", "KPNO/MzLS/z"]
LOGM_REF = 10.0  # log10 stellar mass formed (Msun) of every spectrum; a catalogue magnitude rescales it


def filter_curves(bands):
    """{band id: SED_Tools FilterCurve} for 'FACILITY/INSTRUMENT/band' ids, fetched from NJM/SVO when not installed."""
    from sed_tools.api import Filters
    from sed_tools.cube.model import FILTER_DIR_DEFAULT
    from sed_tools.photometry import load_filter_curve

    curves = {}
    for band in bands:
        facility, instrument, name = band.split("/")
        path = Path(FILTER_DIR_DEFAULT) / facility / instrument / f"{name}.dat"
        if not path.exists():
            Filters.fetch(facility, instrument)
        curves[band] = load_filter_curve(str(path), name=band)
    return curves


def ab_mags(wave, flam, curves):
    """AB magnitudes of one f_lambda spectrum (erg/s/cm^2/A on wave, A) through each curve (photon-weighted f_nu)."""
    from sed_tools.photometry import AB_ZERO_FLUX, band_average_flux_nu

    out = []
    for curve in curves.values():
        fnu = band_average_flux_nu(wave, flam, curve)
        out.append(-2.5 * np.log10(fnu / AB_ZERO_FLUX) if fnu > 0 else np.nan)
    return out


def make_spectra(folder, k, name):
    import warnings

    import bagpipes as pipes
    import h5py
    from astropy.cosmology import Planck18

    warnings.filterwarnings("ignore")
    wave = np.arange(k["wave_min"], k["wave_max"] + 0.5 * k["wave_step"], k["wave_step"])
    curves = filter_curves(k["bands"])
    for band, curve in curves.items():  # a band outside the spectrum would be silently truncated
        on = curve.wavelength[curve.transmission > 1e-4 * curve.transmission.max()]
        if on.min() < wave[0] or on.max() > wave[-1]:
            raise SystemExit(f"{name}: band {band} ({on.min():.0f}-{on.max():.0f} A) lies outside the spectrum "
                             f"{wave[0]:.0f}-{wave[-1]:.0f} A; widen wave_min/wave_max")
    zgrid = np.round(np.arange(k["z_min"], k["z_max"] + 0.5 * k["z_step"], k["z_step"]), 4)

    rows, flux, mags = [], [], []
    for label, age, tau in k["sfh"]:
        for metal in k["metallicity"]:
            for av in k["av"]:
                model = None
                for z in zgrid:
                    age_z = min(float(age), float(Planck18.age(z).value) - 0.3)
                    comp = {"redshift": float(z),
                            "exponential": {"age": age_z, "tau": float(tau), "massformed": LOGM_REF,
                                            "metallicity": float(metal)},
                            "dust": {"type": "Calzetti", "Av": float(av)},
                            "nebular": {"logU": float(k["logU"])}}
                    if model is None:
                        model = pipes.model_galaxy(comp, spec_wavs=wave)
                    else:
                        model.update(comp)
                    flam = np.asarray(model.spectrum[:, 1], float)
                    if not np.all(np.isfinite(flam)) or flam.max() <= 0:
                        raise SystemExit(f"{name}: bagpipes gave no spectrum for {label} Z={metal} Av={av} z={z}")
                    flux.append(flam.astype(np.float32))
                    mags.append(ab_mags(wave, flam, curves))
                    rows.append((label, float(age), age_z, float(tau), float(metal), float(av), float(z)))
        print(f"  {name}: {label} done ({len(rows)} spectra)", flush=True)

    mags = np.asarray(mags)
    if not np.all(np.isfinite(mags)):
        raise SystemExit(f"{name}: {int((~np.isfinite(mags)).sum())} non-finite synthetic magnitudes")
    with h5py.File(folder / "seds.h5", "w") as f:
        f.create_dataset("wavelength", data=wave)
        f.create_dataset("flux", data=np.asarray(flux), compression="gzip", compression_opts=4, shuffle=True)
        for key, col in zip(("sfh", "age_nominal", "age", "tau", "metallicity", "av", "z"), zip(*rows)):
            f.create_dataset(f"param/{key}", data=np.array(col, dtype="S16" if key == "sfh" else float))
        for j, band in enumerate(curves):
            f.create_dataset(f"mag/{band}", data=mags[:, j])
        f.attrs.update(logm_ref=LOGM_REF, logU=k["logU"], flux_unit="erg/s/cm^2/A, observer frame",
                       wavelength_unit="Angstrom, observer frame", magnitudes="AB, photon-weighted, SED_Tools curves",
                       provenance="BC03 2016 MILES via bagpipes; Calzetti 2000 dust; bagpipes Cloudy nebular grid; "
                                  "Planck18 cosmology for the age cap")
    n = len(rows)
    return dict(kind="galaxy spectra library", spectra=n, reference="Bruzual & Charlot 2003; Carnall et al. 2018 "
                "(bagpipes); Calzetti et al. 2000", params=f"{len(k['sfh'])} SFH x {len(k['metallicity'])} Z x "
                f"{len(k['av'])} Av x {len(zgrid)} z ({zgrid[0]:g}-{zgrid[-1]:g}); {len(wave)} wavelengths; "
                f"{len(curves)} bands", notes=dict(use="observer-frame spectra at 10^10 Msun formed: match a catalogue "
                "galaxy by its colours (+ photo-z), then scale to its magnitude (synthstar.galaxy_templates)."))


# ---------------------------------------------------------------------------------------------- shapes library
COSMOS_URL = "https://zenodo.org/record/3242143/files/COSMOS_{sample}_training_sample.tar.gz"


def prepare_shapes(k, name):
    """Download GalSim's COSMOS training sample catalogues once (the 4.4 GB tarball is streamed; images not kept)."""
    from .. import config as C

    cache = C.REPO / "data" / "cosmos" / f"COSMOS_{k['sample']}_training_sample"
    if (cache / f"real_galaxy_catalog_{k['sample']}_fits.fits").exists():
        return
    print(f"  {name}: downloading the COSMOS {k['sample']} catalogues (streams ~4.4 GB, keeps the catalogues)", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(COSMOS_URL.format(sample=k["sample"])) as stream, \
            tarfile.open(fileobj=stream, mode="r|gz") as tar:
        for member in tar:
            if member.isfile() and "images" not in Path(member.name).name:
                tar.extract(member, cache.parent, filter="data")


def make_shapes(folder, k, name):
    import galsim
    import h5py

    from .. import config as C

    cache = C.REPO / "data" / "cosmos" / f"COSMOS_{k['sample']}_training_sample"
    # GalSim's own catalogue and selection ("marginal" exclusion: failed fits, bad stamps, low S/N removed), so the
    # library holds exactly the galaxies GalSim itself would draw.
    cat = galsim.COSMOSCatalog(sample=k["sample"], dir=str(cache), use_real=False, exclusion_level="marginal")
    rec = cat.param_cat[cat.orig_index]
    # galsim.COSMOSCatalog._buildParametric: sersicfit = [I, hlr, n, q, boxiness, x0, y0, phi]; bulgefit = the disk
    # (n=1) in 0:8 and the bulge (n=4) in 8:16; record hlr/flux = [sersic, bulge, disk] (hlr in arcsec, circularised).
    ser, bul = rec["sersicfit"], rec["bulgefit"]
    hlr, flux = rec["hlr"], rec["flux"]
    with np.errstate(invalid="ignore", divide="ignore"):
        bulge_frac = flux[:, 1] / (flux[:, 1] + flux[:, 2])
    use_bulge = np.asarray(rec["use_bulgefit"], bool) & (bulge_frac >= 0) & (bulge_frac <= 1)
    columns = dict(mag_f814w=rec["mag_auto"], zphot=rec["zphot"],
                   sersic_n=np.clip(ser[:, 2], 0.3, 6.0), sersic_hlr=hlr[:, 0], sersic_q=ser[:, 3],
                   use_bulgefit=use_bulge, bulge_frac=np.where(use_bulge, bulge_frac, np.nan),
                   bulge_hlr=hlr[:, 1], bulge_q=bul[:, 11], disk_hlr=hlr[:, 2], disk_q=bul[:, 3],
                   ident=np.asarray(cat.real_cat.ident[cat.orig_index]) if cat.real_cat is not None
                   else np.asarray(cat.orig_index))
    good = np.isfinite(columns["zphot"]) & (columns["zphot"] > 0) & (columns["sersic_hlr"] > 0)
    good &= np.where(use_bulge, (columns["bulge_hlr"] > 0) & (columns["disk_hlr"] > 0), np.asarray(rec["viable_sersic"], bool))
    with h5py.File(folder / "shapes.h5", "w") as f:
        for key, col in columns.items():
            f.create_dataset(key, data=np.asarray(col)[good])
        f.attrs.update(source=f"GalSim COSMOS {k['sample']} training sample (HST/ACS F814W, Mandelbaum et al. 2012), "
                              "GalSim 'marginal' selection", hlr_unit="arcsec, circularised",
                       url=COSMOS_URL.format(sample=k["sample"]), galsim=galsim.__version__)
    n = int(good.sum())
    m, z = np.asarray(columns["mag_f814w"])[good], np.asarray(columns["zphot"])[good]
    return dict(kind="galaxy shapes library", galaxies=n,
                reference="Mandelbaum et al. 2012 (COSMOS parametric fits); Koekemoer et al. 2007; Leauthaud et al. 2007",
                params=f"{n} COSMOS {k['sample']} galaxies (GalSim selection {cat.nobjects}); F814W {m.min():.1f}-"
                       f"{m.max():.1f}, zphot {z.min():.2f}-{z.max():.2f}; {int(use_bulge[good].sum())} bulge+disk",
                notes=dict(use="draw a shape for a catalogue galaxy without a measured one: the nearest COSMOS "
                               "galaxies in (i magnitude, redshift) (synthstar.galaxy_templates.GalaxyShapes)."))


GENERATORS = [
    Generator("galaxy spectra", "library", "BC03 via bagpipes",
              "a library of galaxy spectra over star-formation history, metallicity, dust and redshift, with "
              "synthetic catalogue-band magnitudes; not a scene object",
              dict(sfh=(SFH, "[label, age Gyr, tau Gyr] exponential star-formation histories"),
                   metallicity=([0.4, 1.0, 2.5], "stellar metallicities (Z/Zsun), BC03's own grid points"),
                   av=([0.0, 0.3, 0.7, 1.2], "Calzetti (2000) dust attenuation A_V (mag)"),
                   logU=(-3.0, "nebular ionisation parameter (bagpipes Cloudy grid)"),
                   z_min=(0.01, "lowest redshift"), z_max=(1.0, "highest redshift"), z_step=(0.01, "redshift step"),
                   wave_min=(3000.0, "observer-frame wavelength start (A)"),
                   wave_max=(11200.0, "observer-frame wavelength end (A)"),
                   wave_step=(5.0, "wavelength step (A)"),
                   bands=(BANDS, "SED_Tools filter ids FACILITY/INSTRUMENT/band of the synthetic magnitudes")),
              make_spectra, "Bruzual & Charlot 2003; Carnall et al. 2018; Calzetti et al. 2000", parallel=False),
    Generator("galaxy shapes", "library", "COSMOS parametric fits",
              "Sersic and bulge+disk shapes of real HST COSMOS galaxies with magnitude and photo-z; not a scene object",
              dict(sample=("23.5", "GalSim COSMOS training sample: 23.5 (F814W<23.5) or 25.2")),
              make_shapes, "Mandelbaum et al. 2012", prepare=prepare_shapes, parallel=False),
]
