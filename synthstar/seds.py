"""Pack the per-model SEDs written by the colors module into one HDF5 file per track.

With make_csv + sed_per_model, MESA writes SED/<filter>_SED_<model>.csv for every filter at
every history step. Every one of those files repeats the same interpolated SED next to its
own filter curve. We keep one copy per model, matched to that model's history row:

  runs/<track>/seds.h5
    wavelength            [n_wave]            Angstrom
    flux                  [n_model, n_wave]   f_lambda at 10 pc, erg s^-1 cm^-2 A^-1
    model/<column>        [n_model]           model_number, star_age, star_mass, Teff, log_g,
                                              log_L, log_R, mh (photospheric [M/H]), phase,
                                              inside_atm_grid, Interp_rad, Mag_bol, Flux_bol,
                                              fbol_sed_over_L (integrated SED / (L/4 pi d^2))
    mag/<band>            [n_model]           absolute mags: Gaia/Argus Vega, SDSS AB

phase: -1 pre-main sequence, 0 MS, 1 H-shell burning, 2 core He burning.
The raw CSVs are always kept: in SED/, then (tools/archive_seds.sh, once the track is finished) byte for byte in
SED.tar.zst; restore with tar -I zstd -xf runs/<track>/SED.tar.zst -C runs/<track>. seds.h5 is a convenience
copy, not a replacement (unpack() rebuilds the CSVs from it only to the 7th digit of convolved_flux).
"""

import glob
import re
import subprocess

import numpy as np

from . import config as C
from .history import inside_atmosphere_grid, phases, photospheric_mh, read, zero_point_offsets

PC10_CM = 3.0857e19
LSUN = 3.828e33
REF_FILTER = "G"  # any filter's file holds the full SED; G's has the shortest filter table


# Fortran ES14.6 drops the "E" for 3-digit exponents: "1.180971-100" means 1.180971E-100
_FORTRAN_EXP = re.compile(r"(?<=\d)([+-]\d{3})\b")


def read_sed_csv(path):
    """(wavelength [A], f_lambda at 10 pc) columns of one colors-module SED CSV, padding removed."""
    rows = []
    for line in open(path).read().splitlines()[1:]:
        wv, fl = line.split(",")[:2]
        rows.append((float(_FORTRAN_EXP.sub(r"E\1", wv.strip())), float(_FORTRAN_EXP.sub(r"E\1", fl.strip()))))
    a = np.array(rows)
    return a[a[:, 0] > 0]  # rows past the SED length are zero-padded


def pack(track):
    import h5py

    d = C.RUNS / track["name"]
    sed_dir = d / "SED"
    files = sorted(sed_dir.glob(f"{REF_FILTER}_SED_*.csv"))
    models = np.array([int(re.search(r"_(\d+)\.csv$", f.name).group(1)) for f in files])
    seds = [read_sed_csv(f) for f in files]
    wave = seds[0][:, 0]
    flux = np.array([s[:, 1] for s in seds])

    h = read(track["name"])
    row = {int(m): i for i, m in enumerate(h["model_number"])}
    keep = np.array([m in row for m in models])
    models, flux = models[keep], flux[keep]
    idx = np.array([row[int(m)] for m in models])
    hs = h[idx]

    # pre-MS flag uses the full history (first point where L_H / L >= 0.99 is the ZAMS)
    frac = 10.0 ** (h["log_LH"] - h["log_L"])
    zams_model = h["model_number"][int(np.argmax(frac >= 0.99))] if np.any(frac >= 0.99) else h["model_number"][0]
    ph = phases(h)[idx]
    ph[hs["model_number"] < zams_model] = -1
    mh = photospheric_mh(hs)
    teff = 10.0 ** hs["log_Teff"]

    # sanity: integrated SED vs L / (4 pi d^2)
    fbol_sed = np.trapezoid(flux, wave, axis=1)
    fbol_L = 10.0 ** hs["log_L"] * LSUN / (4 * np.pi * PC10_CM**2)
    ratio = fbol_sed / fbol_L

    zp = zero_point_offsets()
    out = d / "seds.h5"
    with h5py.File(out, "w") as f:
        f.attrs.update(dict(
            track=track["name"], initial_mass=track["mass"], feh=track["feh"], alpha_mlt=track["alpha"],
            X=track["X"], Y=track["Y"], Z=track["Z"], why=track.get("why", ""),
            distance_pc=10.0, flux_units="erg s^-1 cm^-2 A^-1 (f_lambda at 10 pc)", wavelength_units="Angstrom",
            atmosphere_grid=C.colors_path("stellar_atm"),
            phase_codes="-1 pre-MS, 0 main sequence, 1 H-shell burning, 2 core He burning",
            fbol_sed_over_L_median=float(np.median(ratio)),
        ))
        f.create_dataset("wavelength", data=wave)
        f.create_dataset("flux", data=flux, compression="gzip", compression_opts=6, chunks=(1, len(wave)))
        g = f.create_group("model")
        for name, v in [("model_number", hs["model_number"].astype(int)), ("star_age", hs["star_age"]),
                        ("star_mass", hs["star_mass"]), ("Teff", teff), ("log_g", hs["log_g"]),
                        ("log_L", hs["log_L"]), ("log_R", hs["log_R"]), ("mh", mh), ("phase", ph),
                        ("inside_atm_grid", inside_atmosphere_grid(teff, hs["log_g"], mh).astype(np.int8)),
                        ("Interp_rad", hs["Interp_rad"]), ("Mag_bol", hs["Mag_bol"]), ("Flux_bol", hs["Flux_bol"]),
                        # integrated SED / (L / 4 pi d^2): ~1 normally; deviates where the atmosphere
                        # interpolation is not flux-conserving (e.g. hot, low-g grid edge)
                        ("fbol_sed_over_L", ratio)]:
            g.create_dataset(name, data=v)
        m = f.create_group("mag")
        for b in C.ALL_BANDS:
            ds = m.create_dataset(b, data=hs[b] + (zp[b] if b.startswith("SDSS") else 0.0))
            ds.attrs["system"] = "AB" if b.startswith("SDSS") else "Vega"

    print(f"      {len(models)} SEDs x {len(wave)} wavelengths; integrated SED / (L/4πd²): "
          f"median {np.median(ratio):.4f}, range {ratio.min():.3f}-{ratio.max():.3f}", flush=True)
    return out


# ---------------------------------------------------------------- unpack (seds.h5 -> colors-module CSVs)
def _fortran_es14_6(x):
    """Format like Fortran ES14.6, which drops the "E" for 3-digit exponents."""
    s = f"{x:.6E}"
    m, e = s.split("E")
    return (m + e if abs(int(e)) >= 100 else s).rjust(14)


def _linear_interpolate(x, y, xv):
    """Exact copy of colors' linear_interpolate (clamped ends, binary search, same operation order)."""
    if xv <= x[0]:
        return y[0]
    if xv >= x[-1]:
        return y[-1]
    lo, hi = 0, len(x) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x[mid] <= xv:
            lo = mid
        else:
            hi = mid
    return y[lo] + (y[lo + 1] - y[lo]) / (x[lo + 1] - x[lo]) * (xv - x[lo])


def _filter_columns(instrument, fname, wave):
    fw, ft = [], []
    for line in (instrument / fname).read_text().splitlines()[1:]:
        if line.strip():
            a, b = line.split(",")[:2]
            fw.append(float(a))
            ft.append(float(b))
    fw, ft = np.array(fw), np.array(ft)
    g = np.array([_linear_interpolate(fw, ft, v) for v in wave])
    g[(wave < fw.min()) | (wave > fw.max())] = 0.0  # zero_filter_outside_support
    g[g < 0] = 0.0
    return fw, ft, g


def _unpack_models(args):
    out_dir, instrument, filters, wave, models, flux = args
    wave_s = [_fortran_es14_6(v) for v in wave]
    cols = {}
    for f in filters:
        fw, ft, g = _filter_columns(instrument, f, wave)
        cols[f] = (g, [_fortran_es14_6(v) for v in fw], [_fortran_es14_6(v) for v in ft])
    zero = _fortran_es14_6(0.0)
    for m, fl in zip(models, flux):
        fl_s = [_fortran_es14_6(v) for v in fl]
        for f in filters:
            g, fw_s, ft_s = cols[f]
            conv_s = [_fortran_es14_6(v) for v in fl * g]
            n = max(len(wave), len(fw_s))
            lines = ["wavelengths,fluxes,convolved_flux,filter_wavelengths,filter_trans"]
            for i in range(n):
                a = (wave_s[i], fl_s[i], conv_s[i]) if i < len(wave) else (zero, zero, zero)
                b = (fw_s[i], ft_s[i]) if i < len(fw_s) else (zero, zero)
                lines.append(",".join(a + b))
            (out_dir / f"{f[:-4]}_SED_{m:08d}.csv").write_text("\n".join(lines) + "\n")
    return len(models)


def archive(track):
    """Move a just-finished track's SED/ CSVs into runs/<track>/SED.tar.zst, byte for byte (tools/archive_seds.sh: the
    CSVs are removed only after the archive is checked against them; on any failure they stay). Its log line."""
    r = subprocess.run(["nice", "-n", "10", "ionice", "-c3", str(C.REPO / "tools" / "archive_seds.sh"), "--now",
                        track["name"]], capture_output=True, text=True)
    lines = [ln for ln in r.stdout.splitlines() if track["name"] in ln]
    return lines[-1].split(" ", 2)[-1] if lines else f"archive_seds: no line for {track['name']} (exit {r.returncode})"


def unpack(track, workers=8):
    """Recreate the colors module's per-model SED CSVs (runs/<track>/SED/) from seds.h5.

    wavelengths, fluxes, filter_wavelengths and filter_trans are byte-identical to what MESA
    writes. convolved_flux (= flux x filter) can differ by 1 in the 7th significant digit in a
    few % of lines, because MESA multiplies the unrounded flux.
    """
    import h5py
    from concurrent.futures import ProcessPoolExecutor

    d = C.RUNS / track["name"]
    out_dir = d / "SED"
    out_dir.mkdir(exist_ok=True)
    instrument = C.INSTRUMENT
    filters = [ln.strip() for ln in (instrument / instrument.name).read_text().split() if ln.strip()]
    with h5py.File(d / "seds.h5") as f:
        wave, flux, models = f["wavelength"][:], f["flux"][:], f["model/model_number"][:]
    chunks = np.array_split(np.arange(len(models)), workers * 4)
    jobs = [(out_dir, instrument, filters, wave, models[c], flux[c]) for c in chunks if len(c)]
    with ProcessPoolExecutor(workers) as ex:
        n = sum(ex.map(_unpack_models, jobs))
    print(f"  {track['name']}: wrote {n} models x {len(filters)} filters to {out_dir}")
    return out_dir


# ---------------------------------------------------------------- constant-star templates
class Templates:
    """MESA SEDs (f_lambda at 10 pc) indexed by Gaia BP-RP, metal-poor track preferred."""

    def __init__(self, stride=1):
        import astropy.units as u
        import h5py

        waves, fluxes, colours, g, feh = None, [], [], [], []
        for path in sorted(glob.glob(str(C.RUNS / "*_sed" / "seds.h5"))):
            with h5py.File(path) as f:
                w = f["wavelength"][:]
                waves = w
                sel = slice(None, None, stride)
                fluxes.append(f["flux"][sel])
                colours.append(f["mag/Gbp"][sel] - f["mag/Grp"][sel])
                g.append(f["mag/G"][sel])
                feh.append(np.full(len(g[-1]), float(f.attrs["feh"])))
        self.wave = waves * u.AA
        self.flux, self.colour, self.g, self.feh = (np.concatenate(x) for x in (fluxes, colours, g, feh))

    def index(self, bp_rp):
        """The template matching bp_rp."""
        score = np.abs(self.colour - bp_rp) + 0.004 * np.abs(self.feh + 1.5)  # mild preference for metal-poor (M3 [Fe/H] ~ -1.5)
        return int(np.argmin(score))

    def sed(self, bp_rp, gmag):
        """Observer-frame f_lambda (erg/s/cm2/A) of the template matching bp_rp, scaled to G = gmag."""
        i = self.index(bp_rp)
        return self.flux[i] * 10 ** (-0.4 * (gmag - self.g[i]))
