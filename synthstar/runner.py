"""Write MESA run directories from track specs and run them in parallel.

A track is a dict with at least: name, mass, Z, Y, alpha, max_age (see grids/).
Each track runs in runs/<name>/ with a fully rendered `inlist`. Runs are resumable:
finished tracks are skipped, and a `.running` lock (holding the PID) marks
tracks that are in progress, including ones started by another runner.
"""

import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import config as C

DEFAULT_MAX_AGE = 13.5e9


def track_name(mass, feh, alpha, max_age=DEFAULT_MAX_AGE):
    m = f"{mass:05.2f}" if abs(mass * 100 - round(mass * 100)) < 1e-9 else f"{mass:.3f}"
    name = f"M{m}_MH{feh:+.1f}_a{alpha:.2f}"
    if abs(max_age - DEFAULT_MAX_AGE) > 1:
        name += f"_t{max_age / 1e9:g}"
    return name


def render_inlist(track, mag_system="Vega"):
    tpl = (C.INLISTS / "physics.inlist").read_text()
    helium_stop = ("" if track.get("continue_past_he_burning") else
                   "      xa_central_lower_limit_species(1) = 'he4'\n"
                   "      xa_central_lower_limit(1) = 1d-4\n")
    star_job_extra = list(track.get("star_job_extra", []))
    star_job_extra += ([f"save_model_filename = '{track['save_model']}'"] if track.get("save_model") else [])
    star_job_extra += (["load_saved_model = .true.", f"load_model_filename = '{track['load_model']}'"]
                       if track.get("load_model") else [])
    return tpl.format(
        mass=f"{track['mass']:.5f}d0", z=f"{track['Z']:.6e}", y=f"{track['Y']:.6f}d0",
        # 10 digits: staged runs stop on windows of ~1e-4 of the age
        alpha=f"{track['alpha']:.4f}d0", max_age=f"{track.get('max_age', DEFAULT_MAX_AGE):.10e}",
        history_columns=C.INLISTS / "history_columns.list", instrument=C.INSTRUMENT,
        stellar_atm=C.colors_path("stellar_atm"), vega_sed=C.colors_path("vega_sed"),
        mag_system=mag_system,
        make_csv=".true." if track.get("save_seds") else ".false.",
        extra_controls="".join(f"      {ln}\n" for ln in track.get("extra_controls", [])),
        helium_stop=helium_stop,
        history_interval=int(track.get("history_interval", 1)),
        photo_interval=int(track.get("photo_interval", 10000000)),
        # staged runs: save the final model, or start from a saved one
        create_pms=".false." if track.get("load_model") else ".true.",
        save_model=".true." if track.get("save_model") else ".false.",
        star_job_extra="".join(f"      {ln}\n" for ln in star_job_extra),
    )


# ---------------------------------------------------------------- status
def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def termination(d):
    out = d / "out.txt"
    if not out.exists():
        return None
    lines = [ln.strip() for ln in out.read_text(errors="replace").splitlines() if "termination code" in ln]
    return lines[-1].split(":", 1)[-1].strip() if lines else None


def status(name):
    d = C.RUNS / name
    if not d.exists():
        return "pending"
    if termination(d):
        return "done"
    lock = d / ".running"
    if lock.exists() and _pid_alive(int(lock.read_text().split()[0])):
        return "running"
    return "failed" if (d / "out.txt").exists() else "pending"


# ---------------------------------------------------------------- running
_launch_lock = threading.Lock()


def _n_star_procs():
    r = subprocess.run(["pgrep", "-x", "star"], capture_output=True, text=True)
    return len(r.stdout.split())


def _wait_for_slot():
    cap = C.config()["run"].get("max_star_procs", 0)
    while cap and _n_star_procs() >= cap:
        time.sleep(15)


def prepare(track, mag_system="Vega", run_dir=None):
    d = run_dir or C.RUNS / track["name"]
    d.mkdir(parents=True, exist_ok=True)
    for f in ("out.txt", ".running"):
        (d / f).unlink(missing_ok=True)
    (d / "inlist").write_text(render_inlist(track, mag_system))
    return d


def run_dir(d, threads=None, resume_from_photo=False):
    """Run MESA in directory d (which must contain `inlist`). Returns (ok, minutes, termination)."""
    threads = threads or C.config()["run"]["omp_threads"]
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    if resume_from_photo:
        photos = d / "photos"
        available = [p for p in photos.iterdir() if p.is_file()] if photos.is_dir() else []
        if available:
            latest = max(available, key=lambda p: (p.stat().st_mtime_ns, p.name))
            shutil.copy2(latest, d / "restart_photo")
        else:
            (d / "restart_photo").unlink(missing_ok=True)
    t0 = time.time()
    with _launch_lock:
        _wait_for_slot()
        out = open(d / "out.txt", "w")
        p = subprocess.Popen([str(C.STAR_BIN)], cwd=d, stdout=out, stderr=subprocess.STDOUT, env=env)
        (d / ".running").write_text(f"{p.pid}\n")
    p.wait()
    out.close()
    (d / ".running").unlink(missing_ok=True)
    term = termination(d)
    return bool(term) and (d / "LOGS" / "history.data").exists(), (time.time() - t0) / 60, term


def run_tracks(tracks, workers=None, threads=None, rerun_failed=False):
    """Run every track that is not done or already running elsewhere."""
    todo = []
    for t in tracks:
        s = status(t["name"])
        if s == "pending" or (s == "failed" and rerun_failed):
            todo.append(t)
    skipped = len(tracks) - len(todo)
    print(f"{len(todo)} to run, {skipped} done/running/failed-skipped", flush=True)
    if not todo:
        return []
    workers = workers or C.config()["run"]["workers"]

    def one(t):
        ok, minutes, term = run_dir(prepare(t), threads, resume_from_photo=t.get("resume_from_photo", False))
        print(f"  {'OK ' if ok else 'BAD'} {t['name']:<30s} {minutes:7.1f} min  {term or 'no termination code'}", flush=True)
        if ok and t.get("save_seds"):
            from .seds import archive, pack
            print(f"      packed SEDs -> {pack(t)}", flush=True)
            print(f"      SED CSVs -> {archive(t)}", flush=True)  # SED.tar.zst, checked, then the CSVs removed
        return t["name"], ok

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, todo))
