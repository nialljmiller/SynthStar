"""Radial pulsators from MESA RSP (synthstar/rsp.py), from parameters alone (no Gaia, no dust).

Teff is tuned with RSP's linear analysis until the star's mode has `period`, at luminosity L (or from the dust-free
absolute magnitude M_G, bolometric correction ~0); L is stepped down if needed so that mode is the only unstable one.
Then the limit cycle is run and one cycle of SEDs (every <= 25 s, at 10 pc) is written. ~0.5-2 h per star.
"""
import json

import numpy as np

from .. import config as C
from . import REQUIRED, Generator


def rsp_star(kinds):
    def make(folder, k, name):
        import shutil

        from .. import rsp

        rsp.build_sed_instrument()
        rsp.PREFIX = "star_"  # the run: runs/rsp_star_<name>
        kind = kinds[k["mode"]] if isinstance(kinds, dict) else kinds
        dm = float(5 * np.log10(k["distance"]) - 5)
        target = dict(source=0, kind=kind, period=k["period"], epoch_bjd=float("nan"), dm=dm, ebv=0.0, z=k["z"],
                      mass=k["mass"], kap=k["kap"], x=k["hydrogen"])
        if k["L"] is not None:
            target["L"] = k["L"]
        else:  # L from M_G: rsp.luminosity uses G_mean - dm
            target.update(G=k["M_G"] + dm, G_mean=k["M_G"] + dm)
        rsp.model_star(name, target)
        run = C.RUNS / f"rsp_star_{name}"
        if not (run / "pulsation.h5").exists():
            return None  # no acceptable model: runs/rsp_star_<name>/FAILED says why
        shutil.copy(run / "pulsation.h5", folder / "sed.h5")
        q = json.loads((run / "params.json").read_text())
        return dict(kind=kind, reference="MESA RSP (Paxton et al. 2019)", period=k["period"],
                    params=f"P {k['period']:g} d (model {q.get('nonlinear_period', float('nan')):.5f}); L {q['L']:.2f} Lsun "
                           f"(x {q.get('L_factor', 1.0)} the L asked for), Teff {q['teff']:.0f} K; mass {k['mass']:g} Msun, "
                           f"X {k['hydrogen']:g}, Z {k['z']:g}, {k['kap']} opacities; at {k['distance']:g} pc")
    return make


def keys(modes=None):
    k = {}
    if modes:
        k["mode"] = (modes[0], f"pulsation mode / subtype: {', '.join(modes)}")
    k.update(period=(REQUIRED, "period (days); the model is tuned to it"),
             L=(None, "luminosity (Lsun); or give M_G"),
             M_G=(None, "dust-free absolute Gaia G (bolometric correction ~0); used if L is not given"),
             distance=(REQUIRED, "distance (pc)"),
             z=(REQUIRED, "metal mass fraction Z"),
             mass=(REQUIRED, "mass (Msun)"),
             hydrogen=(0.75, "hydrogen mass fraction X (the RSP setup's default, as M3)"),
             kap=("a09", "opacities: a09 (alpha-enhanced) or gs98 (scaled solar)"))
    return k


GENERATORS = [
    Generator("RR Lyrae", "periodic", "MESA RSP", "RR Lyrae star: RRab (fundamental) or RRc (first overtone)",
              keys(("RRab", "RRc")), rsp_star({"RRab": "RRab", "RRc": "RRc"}), "Paxton et al. 2019"),
    Generator("Cepheid", "periodic", "MESA RSP", "classical Cepheid: F (fundamental) or 1O (first overtone)",
              keys(("F", "1O")), rsp_star({"F": "DCEP F", "1O": "DCEP 1O"}), "Paxton et al. 2019"),
    Generator("Type II Cepheid", "periodic", "MESA RSP", "Type II Cepheid: BL Her or W Vir",
              keys(("BL Her", "W Vir")), rsp_star({"BL Her": "BL Her", "W Vir": "W Vir"}), "Paxton et al. 2019"),
    Generator("HADS", "periodic", "MESA RSP", "high-amplitude delta Scuti star, radiative (convection off)",
              keys(), rsp_star("HADS"), "Paxton et al. 2019"),
]
