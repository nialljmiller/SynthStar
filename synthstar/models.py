"""Light-curve models evaluated at Argus: a periodic cycle (RSPStar: MESA RSP limit cycle or PHOEBE orbit) and a
transient (TransientSource: observer-frame SEDs against days since t0). Both read a built star's sed.h5, redden it
for the observation's dust, and return f_lambda at a phase or time. No engine state (no field/block globals)."""
import astropy.units as u
import h5py
import numpy as np

from synthstar import dust
FLAM = u.erg / (u.s * u.cm ** 2 * u.AA)  
class RSPStar:
    """One cycle of a periodic model (MESA RSP limit cycle, or a PHOEBE orbit), phased so phase 0 is maximum light in
    rho (binaries: the primary eclipse). The file's SEDs are at 10 pc; they are seen at `distance` (pc; the file's
    distance modulus if not given), through E(B-V) `ebv` (the observation's; the file's own if not given)."""

    def __init__(self, path, ab, ebv=None, distance=None):
        with h5py.File(path) as f:
            self.attrs = dict(f.attrs)
            wave, flux_10pc, t = f["wavelength"][:], f["flux"][:], f["time_days"][:]
        self.wave = wave * u.AA
        distance = 10 ** (float(self.attrs["distance_modulus"]) / 5 + 1) if distance is None else distance
        ebv = float(self.attrs["ebv"]) if ebv is None else ebv
        seen = dust.redden(wave, flux_10pc * (10.0 / distance) ** 2, ebv)
        rho = np.array([float(ab.mag_from_sed("rho", self.wave, f * FLAM)) for f in seen])
        # one full cycle ending at the last maximum, with the model's own period (RRab have a bump, so local maxima
        # alone would mislead)
        self.period = float(self.attrs["nonlinear_period"])
        if t[-1] - t[0] < 2 * self.period - 1e-9:  # stored as exactly one cycle (a PHOEBE orbit): unroll it to two
            one = t < t[0] + self.period - 1e-9
            t = np.r_[t[one], t[one] + self.period]
            seen, rho = np.r_[seen[one], seen[one]], np.r_[rho[one], rho[one]]
        last = t >= t[-1] - self.period
        if self.attrs.get("kind") == "EB":  # phase 0 = the primary eclipse = the file's start (two equal maxima)
            i_zero = int(np.nonzero(last)[0][0])
            cycle = last
        else:
            i_zero = int(np.nonzero(last)[0][np.argmin(rho[last])])
            cycle = (t > t[i_zero] - self.period) & (t <= t[i_zero])
        phase = ((t[cycle] - t[i_zero]) / self.period) % 1.0
        order = np.argsort(phase)
        self.phase = phase[order]
        self.flux = seen[cycle][order]
        self.rho_cycle = rho[cycle][order]

    def sed(self, phase):
        """f_lambda at Argus at this phase: linear between the two stored steps around it (wrapping the cycle)."""
        phase = phase % 1.0
        n = len(self.phase)
        i = int(np.searchsorted(self.phase, phase))
        if i == 0:
            a, b, pa, pb = n - 1, 0, self.phase[-1] - 1.0, self.phase[0]
        elif i == n:
            a, b, pa, pb = n - 1, 0, self.phase[-1], self.phase[0] + 1.0
        else:
            a, b, pa, pb = i - 1, i, self.phase[i - 1], self.phase[i]
        w = 0.0 if pb == pa else (phase - pa) / (pb - pa)
        return (1 - w) * self.flux[a] + w * self.flux[b]


class TransientSource:
    """A transient: observer-frame SEDs against days since t0, through E(B-V) `ebv` (the observation's t0 and dust;
    the file's own if not given). No light before the model's first step; after its last step is an error."""

    def __init__(self, path, t0=None, ebv=None):
        with h5py.File(path) as f:
            self.attrs = dict(f.attrs)
            wave, flux, self.t = f["wavelength"][:], f["flux"][:], f["time_days"][:]
        self.wave = wave * u.AA
        self.flux = dust.redden(wave, flux, float(self.attrs["ebv"] if ebv is None else ebv))
        self.t0 = float(self.attrs["t0_bjd"] if t0 is None else t0)

    def sed_at(self, bjd):
        """f_lambda at Argus at BJD bjd: linear between the two stored steps around it."""
        d = bjd - self.t0
        if d > self.t[-1]:
            raise ValueError(f"{self.attrs['model']}: {d:.2f} d after t0 is past the model's last step ({self.t[-1]:.2f} d)")
        if d < self.t[0]:
            return np.zeros(self.flux.shape[1])
        i = int(np.clip(np.searchsorted(self.t, d), 1, len(self.t) - 1))
        w = (d - self.t[i - 1]) / (self.t[i] - self.t[i - 1])
        return (1 - w) * self.flux[i - 1] + w * self.flux[i]

    def sed_exposure(self, bjd, exptime, n=61):
        """Mean f_lambda over an exposure centred on bjd (a fast transient changes within one exposure)."""
        return np.mean([self.sed_at(b) for b in bjd + np.linspace(-0.5, 0.5, n) * exptime / 86400], axis=0)

