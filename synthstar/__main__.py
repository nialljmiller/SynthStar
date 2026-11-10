"""MESA_ARGUS command line.  Run from the repo root:  python -m synthstar <command> ...

  setup                     assemble the filter instrument, check paths and the MESA binary
  calibrate                 solar calibration (alpha_MLT, Y) -> data/solar_calibration.json
  zeropoints                AB - Vega offsets from the colors module -> data/zp_offsets.json
  list GRID                 print a grid's tracks
  run GRID [GRID ...]       run all tracks that are not done (resumable; safe to re-run)
                            [--min-mass M] [--max-mass M] to restrict by initial mass
  status GRID [GRID ...]    done / running / failed / pending counts
  cmd3d GRID [--live]       interactive 3D CMD -> outputs/GRID_cmd3d.html (--live: rebuild every 5 min)
  seds GRID                 pack per-model SED CSVs into runs/<track>/seds.h5 (done automatically after each run)
  fisher [GRID] [--live]    discernibility analysis -> outputs/fisher/ (default grid: fisher_stencil;
                            --live: re-run whenever another anchor's stencil completes)
"""

import argparse
import sys

from . import config as C


def cmd_setup(_):
    from . import filters
    print(f"MESA_DIR      {C.mesa_dir()}")
    print(f"colors data   {C.colors_data()}")
    for key in ("stellar_atm", "vega_sed"):
        p = C.colors_path(key)
        print(f"{key:<13s} {p}  {'OK' if __import__('os').path.exists(p) else 'MISSING'}")
    path, names = filters.build()
    print(f"instrument    {path}  ({', '.join(n[:-4] for n in names)})")
    print(f"MESA binary   {C.STAR_BIN}  {'OK' if C.STAR_BIN.exists() else 'MISSING -> make -C mesa_work'}")


def cmd_list(a):
    from .grids import load
    ts = load(a.grid)
    for t in ts:
        print(f"  {t['name']:<28s} Z={t['Z']:.5f} Y={t['Y']:.4f}  {t.get('population', ''):<38s} {t.get('why', '')}")
    print(f"{len(ts)} tracks")


def cmd_run(a):
    from .grids import load
    from .runner import run_tracks
    tracks = [t for g in a.grids for t in load(g)]
    if a.min_mass or a.max_mass:
        lo, hi = a.min_mass or 0.0, a.max_mass or float("inf")
        n = len(tracks)
        tracks = [t for t in tracks if lo <= t["mass"] <= hi]
        print(f"mass filter [{lo}, {hi}] Msun: {len(tracks)}/{n} tracks")
    res = run_tracks(tracks, workers=a.workers, rerun_failed=a.rerun_failed)
    bad = [n for n, ok in res if not ok]
    if bad:
        print("failed:", *bad, sep="\n  ")


def cmd_status(a):
    from .grids import load
    from .runner import status
    for g in a.grids:
        st = {}
        for t in load(g):
            st.setdefault(status(t["name"]), []).append(t["name"])
        print(f"{g}: " + ", ".join(f"{k} {len(v)}" for k, v in sorted(st.items())))
        for n in st.get("failed", []):
            print(f"  failed: {n}  (see runs/{n}/out.txt)")


def cmd_cmd3d(a):
    from . import cmd3d
    cmd3d.live(a.grid) if a.live else cmd3d.build(a.grid)


def cmd_seds(a):
    from .grids import load
    from .runner import status
    from .seds import pack
    for t in load(a.grid):
        if t.get("save_seds") and status(t["name"]) == "done" and (C.RUNS / t["name"] / "SED").exists():
            print(t["name"], "->", pack(t))


def cmd_fisher(a):
    from . import fisher
    fisher.live(a.grid) if a.live else fisher.main(a.grid)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m synthstar", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup").set_defaults(f=cmd_setup)
    sub.add_parser("calibrate").set_defaults(f=lambda a: __import__("synthstar.calibrate", fromlist=["main"]).main())
    sub.add_parser("zeropoints").set_defaults(f=lambda a: __import__("synthstar.zeropoints", fromlist=["main"]).main())
    s = sub.add_parser("list"); s.add_argument("grid"); s.set_defaults(f=cmd_list)
    s = sub.add_parser("run"); s.add_argument("grids", nargs="+")
    s.add_argument("--workers", type=int); s.add_argument("--rerun-failed", action="store_true")
    s.add_argument("--min-mass", type=float, help="only run tracks with initial mass >= this (Msun)")
    s.add_argument("--max-mass", type=float, help="only run tracks with initial mass <= this (Msun)")
    s.set_defaults(f=cmd_run)
    s = sub.add_parser("status"); s.add_argument("grids", nargs="+"); s.set_defaults(f=cmd_status)
    s = sub.add_parser("cmd3d"); s.add_argument("grid"); s.add_argument("--live", action="store_true"); s.set_defaults(f=cmd_cmd3d)
    s = sub.add_parser("seds"); s.add_argument("grid"); s.set_defaults(f=cmd_seds)
    s = sub.add_parser("fisher"); s.add_argument("grid", nargs="?", default="fisher_stencil")
    s.add_argument("--live", action="store_true"); s.set_defaults(f=cmd_fisher)
    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    sys.exit(main())
