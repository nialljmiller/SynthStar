"""Batches: one base file (an observation file or a star file) and lists of values to iterate over.

    python tools/stage_batch.py batches/example_sweep.toml --check     # expand and validate every run; run nothing
    python tools/stage_batch.py batches/example_sweep.toml             # run (detach with nohup / setsid on a node)
    python tools/stage_batch.py batches/example_sweep.toml --status    # progress table
    python tools/stage_batch.py batches/example_sweep.toml --jobs 4 --after 12345   # 4 at once, after PID 12345 exits

Observation batches (base: an observation file, observe.py). Each run is the base with one combination of [vary]
values; it is written out in full as outputs/batch_<batch>/runs/<name>.toml and made with `python observe.py <that
file>`, so it can be repeated on its own. Several run at once, sharing the cores.

    base = "../examples/observations/synthetic_transients.toml"   # relative to this batch file
    combine = "product"                       # "product": every combination of the lists; "zip": side by side
    jobs = 3                                  # observations at once (--jobs overrides)

    [set]                                     # fixed overrides for every run (any key of the base)
    "output.bands" = ["rho"]

    [vary]                                    # each key a list; several keys together: a list of tuples, or a CSV
    "field.ra,field.dec" = [[294.78, -11.62], [273.04, -20.71]]
    place = [                                 # whole [[place]] lists, one per run
      [{star = "dwarf_nova_400pc", t0 = "start+0.02"}],
      [{star = "microlensing_te02", t0 = "start+0.15", x = 30, y = 70}],
    ]

Star batches (base: a star file, make_stars.py): a grid of stars. Each [vary] key star.<key> sets that key of every
[[star]] in the base; each combination gives a copy of every star, named <name>_<number>. All are written to one
star file, outputs/batch_<batch>/stars.toml, and built by one make_stars.py run (which builds them side by side).

    base = "../examples/stars/variables.toml"
    [vary]
    "star.period" = [0.3, 0.35, 0.4]
    "star.incl" = [70, 80, 90]

Everything is expanded and validated before anything runs. Observation run names: <batch>_<number> (a name in the
base or [vary] is kept). Each observation gets an equal share of the cores (config.toml [run] max_star_procs, else
all) through $ARGUS_CORES; a run whose movies (or photometry, with output.movie false) exist is skipped, so
re-running resumes. Per-run logs and a running summary: outputs/batch_<batch>/.
"""
import argparse
import copy
import csv
import itertools
import os
import shlex
import subprocess
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import make_stars  # noqa: E402
from synthstar import runconfig  # noqa: E402
from synthstar.rsp import cores as total_cores  # noqa: E402  ($ARGUS_CORES, else config.toml max_star_procs, else all)
from synthstar.tomlfile import table  # noqa: E402


def axis(keys, values, where):
    """One [vary] entry as a list of {key: value} overrides."""
    names = [k.strip() for k in keys.split(",")]
    if isinstance(values, str):  # a CSV file: one row per run, columns in the keys' order; a header row is skipped
        rows = [row for row in csv.reader((where / values).read_text().splitlines())
                if row and not row[0].lstrip().startswith("#")]
        if [c.strip() for c in rows[0][:len(names)]] == names:
            rows = rows[1:]
        values = [[runconfig.parse_value(c.strip()) for c in row[:len(names)]] for row in rows]
    if len(names) == 1:
        return [{names[0]: v} for v in values]
    return [dict(zip(names, v)) for v in values]


def combos(b, where):
    """Every run's overrides: every combination of the [vary] lists ("product"), or the lists side by side ("zip")."""
    axes = [axis(k, v, where) for k, v in b.get("vary", {}).items()]
    if not axes:
        return [()]
    if b.get("combine", "product") == "product":
        return list(itertools.product(*axes))
    if len({len(a) for a in axes}) > 1:
        raise SystemExit(f"combine = \"zip\" needs every [vary] list the same length, not {[len(a) for a in axes]}")
    return list(zip(*axes))


def expand(batch_path):
    """(tag, kind, jobs, runs): kind "observation" with runs [(name, resolved observation)], or "stars" with runs
    [(name, [[star]] entries)] (one combined star file)."""
    batch_path = Path(batch_path).resolve()
    b = tomllib.loads(batch_path.read_text())
    unknown = sorted(set(b) - {"base", "combine", "jobs", "set", "vary", "name"})
    if unknown or "base" not in b:
        raise SystemExit(f"{batch_path}: unknown keys {unknown} or no base; a batch takes base, combine, jobs, name, [set], [vary]")
    base_path = batch_path.parent / b["base"]
    base = tomllib.loads(base_path.read_text())
    tag = b.get("name", batch_path.stem)
    cs = combos(b, batch_path.parent)
    if "star" in base or "set" in base:  # a star batch
        stars = []
        width = len(str(len(cs)))
        for i, combo in enumerate(cs, 1):
            over = {**b.get("set", {})}
            for part in combo:
                over.update(part)
            bad = sorted(k for k in over if not k.startswith("star."))
            if bad:
                raise SystemExit(f"{batch_path}: a star batch varies star.<key> only, not {bad}")
            for e in base.get("star", []):
                e = copy.deepcopy(e)
                e.update({k[5:]: v for k, v in over.items()})
                e["name"] = f"{e['name']}_{i:0{width}d}" if len(cs) > 1 else e["name"]
                stars.append(e)
        out = REPO / "outputs" / f"batch_{tag}"
        out.mkdir(parents=True, exist_ok=True)
        text = [f"# batch {tag}: {len(stars)} stars from {base_path.name} ({batch_path.name})", ""]
        for e in base.get("set", []):
            text += table("[[set]]", e)
        for e in stars:
            text += table("[[star]]", e)
        (out / "stars.toml").write_text("\n".join(text))
        entries = make_stars.read_star_files([out / "stars.toml"])  # checks every star
        return tag, "stars", 1, [(p["name"], p) for _, _, p in entries]
    runs = []
    for combo in cs:
        over = dict(b.get("set", {}))
        for part in combo:
            over.update(part)
        runs.append(runconfig.resolve(runconfig.merge(base, over)))
    width = len(str(len(runs)))
    names = [r["field"]["name"] or f"{tag}_{i:0{width}d}" for i, r in enumerate(runs, 1)]
    if len(set(names)) < len(names):
        raise SystemExit(f"{batch_path}: runs share a name ({names}); drop field.name from the base or vary it")
    for n, r in zip(names, runs):
        r["field"]["name"] = n
    return tag, "observation", b.get("jobs", 3), list(zip(names, runs))


def finished(name, run):
    out = runconfig.output_folder(name, run["stars"]["layout"])
    files = [f"{name}_night_{b}.{'mp4' if run['output']['movie'] else 'photometry.csv'}" for b in run["output"]["bands"]]
    return all((out / f).exists() for f in files)


def write_summary(path, runs, state):
    lines = [f"# Batch {path.parent.name}", "", f"updated {datetime.now():%Y-%m-%d %H:%M:%S}", "",
             "| run | RA | Dec | background | stars placed | status | start | end | wall (h) | log |",
             "|---|---|---|---|---|---|---|---|---|---|"]

    def fmt(t):
        return datetime.fromtimestamp(t).strftime("%m-%d %H:%M") if t else ""

    for name, r in runs:
        s = state[name]
        wall = f"{(s['end'] - s['start']) / 3600:.2f}" if s.get("end") and s.get("start") else ""
        placed = ", ".join(e["star"] for e in r["place"]) or "-"
        lines.append(f"| {name} | {r['field']['ra']:.4f} | {r['field']['dec']:+.4f} | {r['stars']['background']} "
                     f"({r['stars']['layout']}) | {placed} | {s['status']} | {fmt(s.get('start'))} | {fmt(s.get('end'))} | "
                     f"{wall} | {name}.log |")
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("batch")
    ap.add_argument("--jobs", type=int, help="observations at once (default: the batch's jobs, else 3)")
    ap.add_argument("--after", type=int, help="wait for this PID to exit first")
    ap.add_argument("--status", action="store_true", help="print the summary and exit")
    ap.add_argument("--check", action="store_true", help="expand and validate everything, write the files; run nothing")
    a = ap.parse_args()
    tag, kind, jobs, runs = expand(a.batch)
    out = REPO / "outputs" / f"batch_{tag}"
    if kind == "stars":
        print(f"{len(runs)} stars, all valid; star file {out / 'stars.toml'}", flush=True)
        for name, p in runs:
            print(f"  {name}: {p['generator']} " + ", ".join(f"{k} {v}" for k, v in p.items() if k not in ("name", "generator")))
        if not a.check:
            subprocess.run([sys.executable, str(REPO / "make_stars.py"), str(out / "stars.toml")], cwd=REPO, check=True)
        return
    jobs = a.jobs or jobs
    summary = out / "summary.md"
    if a.status:
        print(summary.read_text() if summary.exists() else "not started")
        return
    (out / "runs").mkdir(parents=True, exist_ok=True)
    for name, r in runs:
        (out / "runs" / f"{name}.toml").write_text(runconfig.dumps(r, f"batch {tag}: observation {name} ({a.batch})"))
    print(f"{len(runs)} observations, all valid; files in {out / 'runs'}", flush=True)
    if a.check:
        for name, r in runs:
            print(f"  {name}: RA {r['field']['ra']:.4f} Dec {r['field']['dec']:+.4f}, {r['stars']['background']} "
                  f"({r['stars']['layout']}), night {r['night']['date'] or 'today'}, placed: "
                  f"{', '.join(e['star'] for e in r['place']) or 'none'}")
        return
    state = {name: dict(status="done (earlier)" if finished(name, r) else "queued") for name, r in runs}
    write_summary(summary, runs, state)
    if a.after:
        print(f"waiting for PID {a.after} …", flush=True)
        while subprocess.run(["kill", "-0", str(a.after)], capture_output=True).returncode == 0:
            time.sleep(30)
    todo = [(n, r) for n, r in runs if state[n]["status"] == "queued"]
    cores = total_cores()
    running = {}
    while todo or running:
        while todo and len(running) < jobs:
            name, r = todo.pop(0)
            share = max(1, cores // min(jobs, len(running) + 1 + len(todo)))
            cmd = [sys.executable, str(REPO / "observe.py"), str(out / "runs" / f"{name}.toml")]
            log = open(out / f"{name}.log", "w")
            log.write(f"$ ARGUS_CORES={share} {shlex.join(cmd)}\n")
            log.flush()
            p = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 env=dict(os.environ, ARGUS_CORES=str(share)))
            running[name] = (p, log)
            state[name].update(status=f"running ({share} cores)", start=time.time())
            print(f"{datetime.now():%H:%M} start {name} ({share} cores)", flush=True)
            write_summary(summary, runs, state)
        time.sleep(20)
        for name, (p, log) in list(running.items()):
            if p.poll() is None:
                continue
            log.close()
            del running[name]
            state[name].update(status="done" if p.returncode == 0 else f"failed (exit {p.returncode})", end=time.time())
            print(f"{datetime.now():%H:%M} {state[name]['status']} {name}", flush=True)
            write_summary(summary, runs, state)
    print(f"batch finished; summary: {summary}", flush=True)


if __name__ == "__main__":
    main()
