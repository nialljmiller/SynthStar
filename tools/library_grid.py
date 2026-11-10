#!/usr/bin/env python3
"""The star library: a grid of model stars covering what Gaia DR3 sees, as star files for make_stars.py.

    python tools/library_grid.py rr_lyrae                # -> queue/rr_lyrae_001.toml, _002, ... (50 stars each)
    python tools/library_grid.py rr_lyrae --check        # the grid and its counts; write nothing

RR Lyrae: Gaia DR3's RR Lyrae (vari_rrlyrae, I/358/vrrlyr) with a photometric [M/H], binned in period (2% wide, the
tolerance a model's period may differ from a star's) and [M/H] (0.25 dex), RRab (fundamental period only) and RRc
(first overtone only) apart; double-mode stars are left out. One model per cell with at least --min-stars stars, at
the cell's centre, with L, mass, X, Z and opacities from the recipe the Gaia variables use (gaia.rr_lyrae_recipe).
A pilot (queue/rr_lyrae_000.toml) comes first: for each mode the cells nearest the corners, edge middles and centre of
the grid in period and [M/H], so failures anywhere in the grid show early. Then the rest, most-populated first, so the
stars most needed are built first.

Library stars are intrinsic: their distance is a label (10 pc); an observation gives them a distance and dust.
"""
import argparse
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from synthstar import gaia, rsp  # noqa: E402
from synthstar.tomlfile import table  # noqa: E402

CACHE = REPO / "data" / "library" / "gaia_rrlyrae_all.ecsv"
PERIOD_STEP = 1.02  # adjacent cells' periods differ by 2%
MH_STEP = 0.25
PER_FILE = 50


def gaia_rr_lyrae():
    """Gaia DR3's RR Lyrae: period columns, [M/H], amplitude (cached in data/library/)."""
    from astropy.table import Table
    from astroquery.vizier import Vizier

    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        columns = ["Source", "PF", "P1O", "[M/H]", "ptpG", "Gmagavg"]
        Vizier(row_limit=-1, columns=columns, timeout=900).get_catalogs("I/358/vrrlyr")[0].write(CACHE)
    return Table.read(CACHE)


def column(t, name):
    c = t[name]
    return np.asarray(c.filled(np.nan) if hasattr(c, "filled") else c, float)


def rr_lyrae_grid(min_stars):
    """[(count, mode, period, mh)], most-populated cell first."""
    t = gaia_rr_lyrae()
    pf, p1, mh = column(t, "PF"), column(t, "P1O"), column(t, "[M/H]")
    cells = []
    for mode, period, single in (("RRab", pf, np.isfinite(pf) & ~np.isfinite(p1)),
                                 ("RRc", p1, np.isfinite(p1) & ~np.isfinite(pf))):
        use = single & np.isfinite(mh) & (period > 0)
        i_p = np.round(np.log(period[use]) / np.log(PERIOD_STEP)).astype(int)  # cell centres at PERIOD_STEP**i
        i_m = np.round(mh[use] / MH_STEP).astype(int)  # cell centres at MH_STEP * i
        cell, count = np.unique(np.c_[i_p, i_m], axis=0, return_counts=True)
        cells += [(int(n), mode, float(PERIOD_STEP ** ip), float(MH_STEP * im)) for (ip, im), n in zip(cell, count)
                  if n >= min_stars]
    return sorted(cells, key=lambda c: -c[0])


def pilot(cells):
    """For each mode, the cells nearest the 9 points (2%, 50%, 98% quantiles of log period) x (same of [M/H])."""
    chosen = []
    for mode in ("RRab", "RRc"):
        mine = [c for c in cells if c[1] == mode]
        log_p, mh = np.log([c[2] for c in mine]), np.array([c[3] for c in mine])
        for qp in (2, 50, 98):
            for qm in (2, 50, 98):
                d = (((log_p - np.percentile(log_p, qp)) / np.ptp(log_p)) ** 2
                     + ((mh - np.percentile(mh, qm)) / np.ptp(mh)) ** 2)
                cell = mine[int(np.argmin(d))]
                if cell not in chosen:
                    chosen.append(cell)
    return chosen


def write(path, cells, what):
    lines = [f"# star library: RR Lyrae grid (tools/library_grid.py), {what}; '# n' is the cell's Gaia DR3 star count", ""]
    for n, mode, period, mh in cells:
        lines += [f"# n {n}"] + table("[[star]]", rr_lyrae_star(mode, period, mh))
    path.write_text("\n".join(lines))


def rr_lyrae_star(mode, period, mh):
    r = gaia.rr_lyrae_recipe(mh)
    name = f"lib_{mode.lower()}_p{period:.4f}_mh{mh:+.2f}".replace(".", "").replace("+", "p").replace("-", "m")
    return dict(name=name, generator="RR Lyrae", mode=mode, period=round(period, 5),
                L=round(float(rsp.m_bol_to_lsun(r["M_G"])), 3), distance=10.0, z=round(r["z"], 7), mass=round(r["mass"], 3),
                hydrogen=r["x"], kap=r["kap"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("family", choices=["rr_lyrae"])
    p.add_argument("--min-stars", type=int, default=5, help="Gaia stars a cell needs to get a model")
    p.add_argument("--out", type=Path, default=REPO / "queue")
    p.add_argument("--check", action="store_true", help="print the grid; write nothing")
    a = p.parse_args()

    cells = rr_lyrae_grid(a.min_stars)
    n_stars = sum(c[0] for c in cells)
    print(f"{len(cells)} cells ({sum(c[1] == 'RRab' for c in cells)} RRab, {sum(c[1] == 'RRc' for c in cells)} RRc) "
          f"with >= {a.min_stars} Gaia stars each, holding {n_stars} stars")
    for n, mode, period, mh in cells[:10]:
        print(f"  {n:6d}  {mode:4s} P {period:.4f} d  [M/H] {mh:+.2f}")
    first = pilot(cells)
    print(f"pilot: {len(first)} cells across the grid: " + ", ".join(f"{m} {p:.3f} d {h:+.2f}" for _, m, p, h in first))
    if a.check:
        return
    a.out.mkdir(parents=True, exist_ok=True)
    write(a.out / "rr_lyrae_000.toml", first, f"the pilot: {len(first)} cells across the grid")
    rest = [c for c in cells if c not in first]
    for f in range(0, len(rest), PER_FILE):
        write(a.out / f"rr_lyrae_{f // PER_FILE + 1:03d}.toml", rest[f:f + PER_FILE],
              f"cells {f + 1}-{min(f + PER_FILE, len(rest))} of the {len(rest)} after the pilot, most-populated first")
    print(f"wrote the pilot and {(len(rest) + PER_FILE - 1) // PER_FILE} star files to {a.out}")


if __name__ == "__main__":
    main()
