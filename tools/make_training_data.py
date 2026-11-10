"""Real/bogus training data, as fast as the machine allows: one-night observations of the real Gaia fields (each
stars/<field>_field background, at its real positions) with [inject] library stars scattered over the readout
window, no movies, run several at once. Each writes <name>_night_<band>.alerts.csv (every hdps candidate, its
features, its truth); tools/rb_training_set.py turns them into a training set, tools/train_rb.py into an hdps bundle.

Each observation is written as an observation file in <out>/observations/ and run with observe.py into
outputs/pseudo_real/rb_<field>_<date>/. A run that fails is logged and the batch goes on. Re-running skips observations
whose alert tables exist.

With --date, observe.py takes the nearest night (in weekly steps) each field is up >= 3 h: the folder keeps the date
asked, the night may be weeks away, at another Moon phase (its observation.toml has the night used). With --plan, each
line names its night: a (field, date) whose field is not up >= 3 h that night is skipped, not moved, so every night is
at the Moon phase planned.

    python tools/make_training_data.py --fields all --date 2026-10-23 --inject 200 --parallel 4
    python tools/make_training_data.py --plan data/rb/plans/moon_phases.csv --stride 2 --write-only

--plan: a CSV with columns field, date (and any others: phase, round, machine; --machine keeps the lines of one).
--write-only: write the observation files and print "<name> <file>" per observation, for tools/rb_queue.py.
"""
import argparse
import re
import subprocess
import sys
import time
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def field_of(name):
    """RA, Dec from a field set's name (f294_784m11_623_field -> 294.784, -11.623)."""
    m = re.match(r"f(\d{3})_(\d{3})([pm])(\d{2})_(\d{3})_field$", name)
    if not m:
        return None
    ra = float(f"{m[1]}.{m[2]}")
    dec = float(f"{m[4]}.{m[5]}") * (1 if m[3] == "p" else -1)
    return ra, dec


def observation(field, ra, dec, date, inject, seed, stride):
    name = f"rb_{field.removesuffix('_field')}_{date.replace('-', '')}"
    return name, f'''# Real/bogus training data (tools/make_training_data.py): {field} for one night, {inject} library stars
# injected over the readout window, no movies.
[field]
name = "{name}"
ra = {ra}
dec = {dec}

[night]
date = "{date}"
nights = 1

[stars]
background = "{field}"
layout = "real"

[inject]
number = {inject}
seed = {seed}

[output]
movie = false
pipeline_movie = false
lightcurves = false
stride = {stride}
'''


def up_that_night(ra, dec, date):
    """Whether the field is up >= 3 h on the night of `date` (observe.py's rule); if not, observe.py would move it to
    another week, so the night is skipped with a note."""
    sys.path.insert(0, str(REPO))
    from synthstar import night

    found, minutes = night.nearest_night(ra, dec, date, night.SITES["argus"], None, None, None, 180)
    if found != date:
        print(f"({ra}, {dec}) on {date}: not up >= 3 h (observe.py would take {found}); skipped")
        return False
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fields", nargs="+", default=["all"], help="field set names (stars/<name>), or all")
    p.add_argument("--plan", type=Path, help="CSV of (field, date) nights, in order (instead of --fields/--date)")
    p.add_argument("--machine", help="with --plan: only its lines whose machine column is this")
    p.add_argument("--write-only", action="store_true", help="write the observation files, print them, run nothing")
    p.add_argument("--date", default="2026-10-23", help="the night to start from (observe.py takes the nearest "
                                                         "night each field is up >= 3 h)")
    p.add_argument("--inject", type=int, default=200, help="library stars per observation")
    p.add_argument("--stride", type=int, default=1, help="every Nth exposure")
    p.add_argument("--parallel", type=int, default=4, help="observations at once (each runs both bands: 2 processes)")
    p.add_argument("--out", type=Path, default=REPO / "data" / "rb")
    p.add_argument("--setup-timeout", type=float, default=1800.0,
                   help="seconds to wait for an observation's scenes before starting the next anyway")
    args = p.parse_args(argv)

    if args.plan:
        import pandas as pd

        plan = pd.read_csv(args.plan, comment="#", skipinitialspace=True)
        if args.machine:
            plan = plan[plan["machine"] == args.machine]
        nights = [(str(r.field), str(r.date)) for r in plan.itertuples()]
    else:
        fields = sorted(d.name for d in (REPO / "stars").glob("*_field")) if args.fields == ["all"] else args.fields
        nights = [(field, args.date) for field in fields]
    (args.out / "observations").mkdir(parents=True, exist_ok=True)
    jobs = []
    for i, (field, date) in enumerate(nights):
        where = field_of(field)
        if where is None:
            print(f"{field}: not a field set name; skipped")
            continue
        if args.plan and not up_that_night(*where, date):
            continue
        # injection seed: --plan, fixed per night whatever the order; else the field's place in the list
        seed = zlib.crc32(f"{field} {date}".encode()) % 100000 if args.plan else i + 1
        name, text = observation(field, *where, date, args.inject, seed=seed, stride=args.stride)
        done = list((REPO / "outputs" / "pseudo_real" / name).glob("*_night_*.alerts.csv"))
        if len(done) >= 2:
            print(f"{name}: done already")
            continue
        if (REPO / "outputs" / "pseudo_real" / name).exists():
            print(f"{name}: a run is there without both alert tables; skipped (move it aside to run it again)")
            continue
        path = args.out / "observations" / f"{name}.toml"
        path.write_text(text)
        jobs.append((name, path))
    if args.write_only:
        for name, path in jobs:
            print(name, path.relative_to(REPO))
        return
    print(f"{len(jobs)} observations, {args.parallel} at a time", flush=True)
    running, results = [], {}
    log_dir = args.out / "logs"
    log_dir.mkdir(exist_ok=True)
    while jobs or running:
        while jobs and len(running) < args.parallel:
            name, path = jobs.pop(0)
            log = open(log_dir / f"{name}.log", "w")
            proc = subprocess.Popen([sys.executable, "observe.py", str(path)], cwd=REPO, stdout=log,
                                    stderr=subprocess.STDOUT)
            running.append((name, log, proc, time.time()))
            print(f"started {name}", flush=True)
            # one scene build at a time: building a scene loads the 3D dust map (~10 GB), twice (the science night,
            # then the reference night); the next observation starts once this one's engines are running ("observing"
            # in its log, the dust map released), or it has ended. Several at once ran newton out of memory.
            t_start = time.time()
            while proc.poll() is None and time.time() - t_start < args.setup_timeout:
                if "observing " in (log_dir / f"{name}.log").read_text(errors="replace"):
                    break
                time.sleep(10)
        for item in list(running):
            name, log, proc, t0 = item
            if proc.poll() is not None:
                log.close()
                running.remove(item)
                results[name] = proc.returncode
                status = "finished" if proc.returncode == 0 else f"FAILED (log {log_dir / name}.log)"
                print(f"{status} {name} ({(time.time() - t0) / 3600:.1f} h)", flush=True)
        time.sleep(20)
    ok = [n for n, rc in results.items() if rc == 0]
    print(f"done: {len(ok)} of {len(results)} observations; build the set with: python tools/rb_training_set.py "
          + " ".join(f"outputs/pseudo_real/{n}" for n in ok) + f" -o {args.out}")


if __name__ == "__main__":
    main()
