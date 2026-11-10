"""Load the BC03 galaxy template grid and match real catalogue galaxies to it.

The galaxy analogue of seds.Templates: where a constant star is the MESA template nearest its Gaia BP-RP scaled to its
G, a catalogue galaxy is the BC03 template (tools/build_galaxy_templates.py) nearest its catalogue colours and photo-z,
scaled to its catalogue magnitude. Morphology is NOT here -- it comes from the catalogue (Legacy Sersic); this is the
spectrum only. The engine turns the returned observer-frame f_lambda into electrons per band exactly as it does a
star's SED (ABPhot.electrons_from_sed), through the full Milky Way sightline column.

Needs only numpy + h5py (no bagpipes): the grid is built once and loaded here.
"""
from pathlib import Path

import numpy as np

GRID = Path(__file__).resolve().parents[1] / "data" / "galaxy_templates" / "bc03_grid.h5"
Z_WEIGHT = 25.0  # how hard the photo-z pulls the match, relative to a magnitude of colour mismatch


class GalaxyTemplates:
    """BC03 template grid with colour+redshift matching and magnitude scaling."""

    def __init__(self, path=GRID):
        import h5py

        with h5py.File(path) as f:
            self.wave = np.asarray(f["wavelength"])              # observer-frame A
            self.flux = np.asarray(f["flux"])                    # [n_tmpl, n_wave] f_lambda at logm_ref
            self.z = np.asarray(f["z"])
            self.tau, self.age = np.asarray(f["tau"]), np.asarray(f["age"])
            self.type = np.array([s.decode() for s in f["type"]])
            self.mag_des = np.asarray(f["mag_des"])              # [n_tmpl, 5] DECam grizy AB at logm_ref
            self.mag_beta, self.mag_rho = np.asarray(f["mag_beta"]), np.asarray(f["mag_rho"])
            self.des_bands = f.attrs["des_bands"].split(",")     # ['desg','desr','desi','desz','desy']
            self.logm_ref = float(f.attrs["logm_ref"])
        self._colours = self.mag_des[:, :-1] - self.mag_des[:, 1:]  # template g-r,r-i,i-z,z-y

    def match(self, grizy, photo_z):
        """Index of the template nearest the observed DECam grizy colours (nan-tolerant) and photo-z."""
        grizy = np.asarray(grizy, float)
        obs = grizy[:-1] - grizy[1:]                             # observed colours (may contain nan)
        ok = np.isfinite(obs)
        if not ok.any():
            return int(np.argmin(np.abs(self.z - photo_z)))      # no colours: fall back to z alone
        dcol = np.nansum((self._colours[:, ok] - obs[ok]) ** 2, axis=1)
        dz = Z_WEIGHT * (self.z - photo_z) ** 2 if np.isfinite(photo_z) else 0.0
        return int(np.argmin(dcol + dz))

    def scale(self, i, ref_mag, ref_band="desr"):
        """Flux multiplier putting template i at ref_mag in ref_band (a DECam band)."""
        b = self.des_bands.index(ref_band)
        return 10 ** (-0.4 * (ref_mag - self.mag_des[i, b]))

    def observer_sed(self, i, ref_mag, ref_band="desr"):
        """Observer-frame f_lambda (erg/s/cm^2/A on self.wave) of template i scaled to ref_mag; feed to the engine."""
        return self.flux[i] * self.scale(i, ref_mag, ref_band)

    def predicted_argus(self, i, ref_mag, ref_band="desr"):
        """(beta, rho) AB mags the scaled template implies -- for the round-trip check, before dust."""
        d = -2.5 * np.log10(self.scale(i, ref_mag, ref_band))   # a scaling is a uniform magnitude shift
        return float(self.mag_beta[i] + d), float(self.mag_rho[i] + d)
