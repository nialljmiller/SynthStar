"""Build the BC03 galaxy-SED template grid that observe.py matches real catalogue galaxies against.

This is the galaxy analogue of the MESA stellar library (synthstar/seds.py Templates): a grid of BC03 composite
spectra (Bruzual & Charlot 2003, via bagpipes, the updated 2016 "miles" models) spanning galaxy type x redshift.
Each real Legacy/PS1 galaxy is later matched to the nearest template by its catalogue colours and photo-z, then
scaled to its catalogue magnitude (synthstar/galaxy_templates.py GalaxyTemplates).

Only this builder imports bagpipes; the loader needs only numpy/h5py, so the engine carries no bagpipes dependency.

Grid axes (kept small and defensible; every value is a BC03 composite, not a hand-drawn SED):
  type        exponential-SFH (tau) composites from old/passive to young/star-forming
  redshift    a coarse grid over the range Argus sees galaxies (z <~ 0.6 dominates a magnitude-limited sample)

Output (one file, built once, then synced like the stellar library):
  data/galaxy_templates/bc03_grid.h5
    wavelength   [n_wave]            observer-frame Angstrom, common grid
    flux         [n_tmpl, n_wave]    f_lambda (erg/s/cm^2/A), observer frame, for massformed = 10^10 Msun
    z            [n_tmpl]
    tau, age, metallicity  [n_tmpl]  the BC03 composite parameters
    mag_des      [n_tmpl, 5]         synthetic DECam grizy AB (Legacy DR10 is DECam) at that normalisation
    mag_beta, mag_rho [n_tmpl]       synthetic ArgusSim AB at that normalisation
    (attrs: the band order, the reference log-mass, provenance)

    python tools/build_galaxy_templates.py            # build the default grid
"""
import argparse
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "galaxy_templates" / "bc03_grid.h5"

# type axis: (age_Gyr, tau_Gyr, label) exponentially-declining-SFR composites, passive -> star-forming
TYPES = [(11.0, 0.3, "passive"), (8.0, 0.6, "red"), (6.0, 1.0, "Sab"),
         (4.0, 2.0, "Sbc"), (2.5, 4.0, "Scd"), (1.0, 8.0, "starforming")]
ZGRID = np.round(np.arange(0.02, 0.601, 0.04), 3)   # 15 redshift steps
METALLICITY = 1.0                                    # solar (Z/Zsun); one value for slice 1
LOGM_REF = 10.0                                      # massformed reference (log Msun)
WAVE = np.arange(3000.0, 11000.0, 5.0)               # observer-frame common grid (A), covers beta+rho+grizy
DES_BANDS = ("desg", "desr", "desi", "desz", "desy")


def argus_ab(ab, zp, band, wav, flam):
    import astropy.units as u
    e = ab.electrons_from_sed(band, wav * u.AA, flam * u.erg / (u.s * u.cm**2 * u.AA), airmass=1.0).value
    return float(-2.5 * np.log10(e / zp[band])) if e > 0 else np.nan


def des_ab(bandpasses, wav, flam):
    # AB mag through a sncosmo bandpass: integral f_lambda*T*lambda dlambda / integral (c/lambda^2 * 3631Jy)*T*lambda dlambda
    out = []
    c_AAs = 2.99792458e18
    for bp in bandpasses:
        T = np.interp(wav, bp.wave, bp.trans, left=0, right=0)
        num = np.trapezoid(flam * T * wav, wav)
        den = np.trapezoid((3631e-23 * c_AAs / wav**2) * T * wav, wav)
        out.append(-2.5 * np.log10(num / den) if num > 0 and den > 0 else np.nan)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    import bagpipes as pipes
    import sncosmo
    import h5py
    from argus_sim import c as cfg, ABPhot, SystemThroughput
    from synthstar.true_calibration import ab_zero_point

    tel = cfg.telescope
    ab = ABPhot(throughput=SystemThroughput(filters={"g": True, "rho": True}, throughput_loss=tel.throughput_loss),
                collecting_area=tel.collecting_area)
    zp = {b: ab_zero_point(ab, b, [1.0])[0] for b in ("g", "rho")}
    des = []
    for b in DES_BANDS:
        try:
            des.append(sncosmo.get_bandpass(b))
        except Exception:
            des.append(sncosmo.get_bandpass(b.replace("des", "sdss") if b != "desy" else "ps1::y"))

    from astropy.cosmology import Planck18

    rows_flux, rows = [], []
    for (age, tau, label) in TYPES:
        for z in ZGRID:
            age_c = min(float(age), float(Planck18.age(z).value) - 0.3)    # a galaxy cannot be older than the universe at z
            mc = {"redshift": float(z),
                  "exponential": {"age": age_c, "tau": float(tau), "massformed": float(LOGM_REF),
                                  "metallicity": float(METALLICITY)}}
            m = pipes.model_galaxy(mc, spec_wavs=WAVE)
            flam = np.asarray(m.spectrum[:, 1], float)                    # erg/s/cm2/A on WAVE
            rows_flux.append(flam)
            mdes = des_ab(des, WAVE, flam)
            rows.append((z, tau, age_c, METALLICITY, label,
                         mdes, argus_ab(ab, zp, "g", WAVE, flam), argus_ab(ab, zp, "rho", WAVE, flam)))
            print(f"  {label:12s} z={z:.2f}  g-r={mdes[0]-mdes[1]:+.2f} r-i={mdes[1]-mdes[2]:+.2f}  "
                  f"beta-rho={rows[-1][6]-rows[-1][7]:+.2f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as f:
        f.create_dataset("wavelength", data=WAVE)
        f.create_dataset("flux", data=np.array(rows_flux), compression="gzip", compression_opts=4)
        f.create_dataset("z", data=np.array([r[0] for r in rows]))
        f.create_dataset("tau", data=np.array([r[1] for r in rows]))
        f.create_dataset("age", data=np.array([r[2] for r in rows]))
        f.create_dataset("metallicity", data=np.array([r[3] for r in rows]))
        f.create_dataset("type", data=np.array([r[4] for r in rows], dtype="S16"))
        f.create_dataset("mag_des", data=np.array([r[5] for r in rows]))
        f.create_dataset("mag_beta", data=np.array([r[6] for r in rows]))
        f.create_dataset("mag_rho", data=np.array([r[7] for r in rows]))
        f.attrs["des_bands"] = ",".join(DES_BANDS)
        f.attrs["logm_ref"] = LOGM_REF
        f.attrs["provenance"] = "BC03 (Bruzual & Charlot 2003, 2016 miles update) via bagpipes; exponential-SFH composites"
    print(f"\nwrote {args.out}: {len(rows)} templates ({len(TYPES)} types x {len(ZGRID)} z)")


if __name__ == "__main__":
    main()
