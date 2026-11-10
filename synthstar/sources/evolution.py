"""Queue ordinary stellar evolution using the existing track runner.

This produces a *library track* (seds.h5, many ages), not a periodic-star sed.h5.
It cannot be passed directly to observe.py as a variable or a Gaia background.
The population sampler must choose an age and apply model/inside_atm_grid first.
"""
import json
import shutil

from . import Generator, REQUIRED


def make_track(folder, keys, name):
    from .. import config as C
    from ..grids import load
    from ..runner import run_tracks, termination

    tracks = {t["name"]: t for t in load(keys["grid"])}
    track = tracks[keys["track"]]
    run = C.RUNS / track["name"]
    artifact = run / "seds.h5"
    if not artifact.exists():
        attempts = max(1, int(track.get("resume_attempts", 1)))
        for attempt in range(attempts):
            run_tracks([track], workers=1, threads=1, rerun_failed=attempt > 0)
            term = termination(run) or ""
            if term or not track.get("resume_from_photo"):
                break
    if track.get("tag") != "sed":  # the five imported baseline SEDs do not retain run logs
        term = termination(run) or ""
        expected = ("max_age", "xa_central_lower_limit", "power_he_burn_upper_limit",
                    *track.get("accepted_terminations", []))
        if not artifact.exists() or not any(t in term for t in expected):
            print(f"  {name}: evolution not accepted: {term or 'no valid completed run'}", flush=True)
            return None
    if not artifact.exists():
        return None
    import h5py
    import numpy as np

    with h5py.File(artifact) as f:
        valid = np.asarray(f["model/inside_atm_grid"], bool)
        valid &= np.asarray(f["model/phase"]) >= 0  # pre-MS kept, excluded from this usability count
        ratio = np.asarray(f["model/fbol_sed_over_L"])
        valid &= np.isfinite(ratio) & (ratio > 0.9) & (ratio < 1.1)
        if not np.any(valid):
            raise ValueError(f"{artifact}: no post-ZAMS, in-atmosphere-grid SEDs within 10% bolometric closure")
        models, usable = len(valid), int(valid.sum())
    shutil.copy2(artifact, folder / "seds.h5")
    (folder / "track.json").write_text(json.dumps(track, indent=2) + "\n")
    return dict(kind="MESA evolution library track", track=track["name"], models=models,
                usable_models=usable, reference="existing calibrated MESA_ARGUS evolutionary physics",
                params=f"Minit {track['mass']:g} Msun, [M/H] {track['feh']:+g}; {models} ages, {usable} usable SEDs",
                notes=dict(use="Coverage model, not an abundance weight. seds.h5 is an evolutionary sequence; "
                               "sample with age, dwell time and survey selection. Exclude atmosphere-grid failures "
                               "and assess bolometric closure per row.",
                           **({"recovery": f"MESA photos are saved every {track['photo_interval']} models."}
                              if track.get("resume_from_photo") else {})))  # notes: a dict (make_stars writes key: value)


GENERATORS = [Generator("MESA evolution track", "library", "MESA evolution",
                        "an ordinary-star age/SED sequence for the population library, not a scene object",
                        dict(grid=("sky_library", "evolution grid name"),
                             track=(REQUIRED, "track name in that grid")), make_track)]
