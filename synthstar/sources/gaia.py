"""Real stars from Gaia DR3: one catalogued variable modelled as itself, or every star around a pointing as a set.

Gaia variable: period and ephemeris from Gaia (ASAS-SN for HADS), distance (parallax or a period-luminosity relation),
the E(B-V) in front of it (the 3D map at that distance; only to get its luminosity from its measured G), composition and mass
(synthstar/gaia.py, physical()); then MESA RSP, or PHOEBE fitted to Gaia's eclipses. The stored SEDs are dust-free at
10 pc like every star's; its sky position, ephemeris and sightline E(B-V) are kept (`real`) so an observation can put
it where and when it really is.

Gaia field: every Gaia DR3 source within 0.12 deg of a pointing, with Gaia's variability tables and distances
(Bailer-Jones et al. 2021; GSP-Phot where they have none). Their magnitudes are as Gaia measured them, i.e. through
their own dust; an observation takes that dust out (the 3D map at each star's distance) and puts its own in.

Gaia queries are cached in data/gaia/<pointing>/ (committed), so stars can be rebuilt without the network.
"""
import json
import shutil
from pathlib import Path

import numpy as np

from .. import config as C
from . import REQUIRED, Generator

CACHE = Path(__file__).resolve().parents[2] / "data" / "gaia"


def cached_pointing_with(source):
    """The cached pointing whose Gaia table holds this source, and the source's RA, Dec (None if none does)."""
    from astropy.table import Table

    for core in sorted(CACHE.glob("*/gaia_dr3_core.ecsv")):
        t = Table.read(core)
        hit = np.nonzero(np.asarray(t["Source"]).astype(np.int64) == int(source))[0]
        if len(hit):
            return core.parent, float(t["RA_ICRS"][hit[0]]), float(t["DE_ICRS"][hit[0]])
    return None


def prepare_variable(k, name):
    """Gaia's data and the physical parameters for one variable (network queries, shared state: one star at a time)."""
    from .. import gaia as Gd

    found = cached_pointing_with(k["source"])
    if found:
        cache, ra, dec = found
    else:
        from astroquery.vizier import Vizier

        r = Vizier(columns=["Source", "RA_ICRS", "DE_ICRS"]).query_constraints(catalog="I/355/gaiadr3", Source=str(k["source"]))
        ra, dec = float(r[0]["RA_ICRS"][0]), float(r[0]["DE_ICRS"][0])
        cache = CACHE / Gd.field_name(ra, dec)
        Gd.fetch(ra, dec, cache)
    Gd.TRY_RRD = k["try_rrd"]
    rows, stars, _, _ = Gd.inventory(cache, ra, dec, 9)
    row = next(r for r in rows if r["source"] == k["source"])
    if row["model"] is None:
        raise SystemExit(f"{name}: Gaia DR3 {k['source']} ({row['kind']}) cannot be modelled: {row['why']}")
    phys, why = Gd.physical(row, stars, ra, dec)
    if phys is None:
        raise SystemExit(f"{name}: Gaia DR3 {k['source']} ({row['kind']}) cannot be modelled: {why}")
    target = dict(source=row["source"], kind=row["kind"], period=row["period"], epoch_bjd=row["epoch_bjd"], G=row["G"],
                  G_mean=row["G_mean"], amp_G=row["amp_G"], **{k_: v for k_, v in phys.items() if k_ != "notes"},
                  notes=phys["notes"])
    if "period2" in row:
        target["period2"] = row["period2"]
    if row["kind"] == "EB":
        target.update(veb=row["veb"], nss=row.get("nss"))
    k.update(target=target, ra=ra, dec=dec, cache=str(cache), parallel=row["kind"] != "EB")  # PHOEBE: one at a time


def make_variable(folder, k, name):
    from .. import binary as B
    from .. import rsp

    target = k["target"]
    run = C.RUNS / f"rsp_gaia_G{k['source']}"
    epoch = target["epoch_bjd"]
    if target["kind"] == "EB":
        if (run / "pulsation.h5").exists() and (run / "binary.json").exists():  # built before
            info = json.loads((run / "binary.json").read_text())
        else:
            ok, info = B.model_binary(dict(source=target["source"], veb=target["veb"], nss=target.get("nss"),
                                           teff_gsp=target["teff_gsp"], logg_gsp=target["logg_gsp"],
                                           G=target["veb"]["magModRef"]), run, target["dm"], target["ebv"])
            if not ok:
                return None  # info["reason"] says why
        veb = target["veb"]
        epoch = veb["TimeRef"] + 2455197.5 + veb["PhaseE1"] * target["period"]  # a primary eclipse = the model's phase 0
        engine = "PHOEBE"
        how = "; ".join(info[key] for key in ("route", "light_curve_check", "distance_check") if key in info)
    else:
        rsp.build_sed_instrument()
        rsp.PREFIX = "gaia_"  # the run: runs/rsp_gaia_G<source>
        rsp.model_star(f"G{k['source']}", target)
        if not (run / "pulsation.h5").exists():
            return None  # runs/rsp_gaia_G<source>/FAILED says why
        q = json.loads((run / "params.json").read_text())
        engine = "MESA RSP"
        how = (f"L {q['L']:.2f} Lsun (x {q.get('L_factor', 1.0)} the L from the distance), Teff {q['teff']:.0f} K, "
               f"non-linear period {q.get('nonlinear_period', float('nan')):.5f} d (Gaia {target['period']:.5f})")
    shutil.copy(run / "pulsation.h5", folder / "sed.h5")
    return dict(kind=target["kind"], engine=engine, reference="Gaia DR3; " + ("PHOEBE 2" if engine == "PHOEBE" else "MESA RSP"),
                period=target["period"],
                params=f"Gaia DR3 {target['source']}, {target['kind']}, P {target['period']:.5f} d; distance modulus "
                       f"{target['dm']}, sightline E(B-V) {target['ebv']}; {how}",
                notes={key: str(v) for key, v in target["notes"].items()},
                real=dict(source=int(target["source"]), ra=k["ra"], dec=k["dec"], kind=target["kind"],
                          period=target["period"], epoch_bjd=float(epoch), amp_G=float(target["amp_G"]),
                          G=float(target["G"]), sightline_ebv=float(target["ebv"]),
                          distance_pc=float(target.get("distance_pc", 10 ** (target["dm"] / 5 + 1))), veb=target.get("veb"),
                          gaia_cache=k["cache"]))


def make_field(folder, k, name):
    from astropy.table import Table

    from .. import gaia as Gd

    cache = CACHE / Gd.field_name(k["ra"], k["dec"])
    Gd.fetch(k["ra"], k["dec"], cache)
    shutil.copytree(cache, folder / "catalogue", ignore=shutil.ignore_patterns("*.json"))
    n = len(Table.read(folder / "catalogue" / "gaia_dr3_core.ecsv"))
    return dict(kind="Gaia field", reference="Gaia DR3 (VizieR I/355, I/357, I/358); ASAS-SN (II/366)",
                params=f"{n} Gaia DR3 sources within {Gd.RADIUS_DEG} deg of RA {k['ra']:.5f}, Dec {k['dec']:+.5f}",
                centre=[k["ra"], k["dec"]], radius_deg=Gd.RADIUS_DEG, n_stars=n)


GENERATORS = [
    Generator("Gaia variable", "periodic", "MESA RSP / PHOEBE",
              "a real Gaia DR3 variable modelled as itself, keeping its own sky position, ephemeris and sightline "
              "E(B-V) (python make_stars.py --gaia-list RA DEC lists those that can be modelled)",
              dict(source=(REQUIRED, "Gaia DR3 source_id"),
                   try_rrd=(False, "attempt a double-mode RR Lyrae (no stable solution found so far)")),
              make_variable, "Gaia DR3; synthstar/gaia.py", prepare=prepare_variable),
    Generator("Gaia field", "set", "Gaia DR3",
              "every Gaia DR3 source around a pointing, as constant stars (and Gaia's variability tables)",
              dict(ra=(REQUIRED, "deg (ICRS)"), dec=(REQUIRED, "deg (ICRS)")),
              make_field, "Gaia DR3", parallel=False),
]
