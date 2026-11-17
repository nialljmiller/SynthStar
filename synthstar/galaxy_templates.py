"""Load the galaxy libraries (synthstar/sources/galaxies.py) and match real catalogue galaxies to them.

The galaxy analogue of seds.Templates: where a constant star is the MESA template nearest its Gaia colour scaled to
its G, a catalogue galaxy is the library spectrum that best fits its catalogue magnitudes (any subset of the
library's bands: Pan-STARRS, DECam, BASS/MzLS) and photometric redshift, scaled to those magnitudes. A galaxy with no
measured shape takes one from the COSMOS shapes library, drawn among the COSMOS galaxies nearest it in i magnitude
and redshift. Survey-agnostic: an instrument's own magnitudes come from the returned spectrum.

Needs only numpy + h5py (the libraries are built once by make_stars.py, examples/stars/galaxy_library.toml).
"""
import numpy as np

from . import config as C

SPECTRA = C.REPO / "stars" / "galaxy_spectra_bc03" / "seds.h5"
SHAPES = C.REPO / "stars" / "galaxy_shapes_cosmos235" / "shapes.h5"


class GalaxyTemplates:
    """The galaxy spectra library, with a magnitude + photo-z fit."""

    def __init__(self, path=SPECTRA):
        import h5py

        with h5py.File(path) as f:
            self.wave = np.asarray(f["wavelength"])                       # observer-frame A
            self.flux = np.asarray(f["flux"])                             # [n, n_wave] f_lambda at logm_ref
            self.params = {k: np.asarray(v) for k, v in f["param"].items()}
            self.params["sfh"] = np.array([s.decode() for s in self.params["sfh"]])
            self.bands = _band_ids(f["mag"])
            self.mags = np.stack([np.asarray(f["mag"][b]) for b in self.bands], axis=1)  # [n, n_band] AB
            self.logm_ref = float(f.attrs["logm_ref"])
        self.z = self.params["z"]

    def fit(self, mags, errs=None, photo_z=np.nan, photo_z_err=np.nan, floor=0.03):
        """Best library spectrum for one galaxy: (index, magnitude offset, chi2).

        mags: {band id: AB mag} in any of self.bands (nan = missing); errs likewise (default 0.05); floor is added in
        quadrature (template-grid and calibration error). The offset is the best common shift (the scaling to the
        galaxy's brightness); a finite photo_z adds ((z - photo_z) / photo_z_err)^2, photo_z_err default
        0.05 (1 + photo_z).
        """
        cols, obs, sig = [], [], []
        for band, m in mags.items():
            if band not in self.bands:
                raise KeyError(f"{band} is not a library band; the library has {self.bands}")
            if np.isfinite(m):
                e = (errs or {}).get(band, 0.05)
                cols.append(self.bands.index(band)), obs.append(m), sig.append(np.hypot(e if np.isfinite(e) else 0.05, floor))
        if not cols and not np.isfinite(photo_z):
            raise ValueError("a galaxy needs magnitudes or a photo-z to be matched")
        prior = 0.0
        if np.isfinite(photo_z):
            pe = photo_z_err if np.isfinite(photo_z_err) and photo_z_err > 0 else 0.05 * (1 + photo_z)
            prior = ((self.z - photo_z) / pe) ** 2
        if not cols:
            i = int(np.argmin(prior))
            return i, np.nan, float(prior[i])
        w = 1 / np.asarray(sig) ** 2
        resid = np.asarray(obs)[None, :] - self.mags[:, cols]               # [n, k]
        offset = (resid * w).sum(axis=1) / w.sum()                           # best common shift per template
        chi2 = (((resid - offset[:, None]) ** 2) * w).sum(axis=1) + prior
        i = int(np.argmin(chi2))
        return i, float(offset[i]), float(chi2[i])

    def observer_sed(self, i, offset):
        """Observer-frame f_lambda (erg/s/cm^2/A on self.wave) of spectrum i shifted by offset magnitudes."""
        return self.flux[i] * 10 ** (-0.4 * offset)

    def describe(self, i):
        return {k: (v[i].item() if hasattr(v[i], "item") else v[i]) for k, v in self.params.items()}


class GalaxyShapes:
    """The COSMOS shapes library: a shape for a galaxy at (i magnitude, redshift)."""

    def __init__(self, path=SHAPES):
        import h5py

        with h5py.File(path) as f:
            self.cols = {k: np.asarray(v) for k, v in f.items()}
        self.mag, self.z = self.cols["mag_f814w"], self.cols["zphot"]

    def draw(self, i_mag, z, rng, k=20, mag_scale=0.25, z_scale=0.05):
        """One shape (dict) chosen at random among the k COSMOS galaxies nearest (i_mag, z), distances in units of
        mag_scale and z_scale; a missing z matches on magnitude alone. 'out_of_range' flags a galaxy brighter or
        fainter than every COSMOS galaxy (its shape is then the nearest COSMOS one, not a like-for-like match)."""
        d2 = ((self.mag - i_mag) / mag_scale) ** 2
        if np.isfinite(z):
            d2 = d2 + ((self.z - z) / z_scale) ** 2
        near = np.argpartition(d2, k)[:k] if len(d2) > k else np.arange(len(d2))
        j = int(rng.choice(near))
        shape = {key: (v[j].item() if hasattr(v[j], "item") else v[j]) for key, v in self.cols.items()}
        shape["out_of_range"] = bool(i_mag < self.mag.min() or i_mag > self.mag.max())
        shape["position_angle"] = float(rng.uniform(0, 180))  # COSMOS orientations carry no information here
        return shape


def _band_ids(group):
    """The band ids stored as mag/FACILITY/INSTRUMENT/band (h5py nests the slashes as groups)."""
    out = []
    group.visititems(lambda name, obj: out.append(name) if hasattr(obj, "shape") else None)
    return out
