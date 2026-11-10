"""Stage 3 on its own: raw Argus camera frames -> hdps data products and alerts, through hdps's real Orchestrator
(the `hdps batch process` path). Nothing of the image maker or the truth/measurement layer is here; the only inputs
are files on disk (a directory of raw frames and, for a science pass, a directory of reference coadds), and the only
outputs are files on disk (hdps's products, coadds, and per-exposure Avro alert packets). So this stage runs on its
own machine, started and ended by file I/O, independent of stages 1, 2 and 4.

This IS hdps: `synthstar.hdps_worker.run` drives hdps's `Orchestrator` (one Ray worker, the [[hdps.tasks]] chain
through hdps's PipelinedExecutor). What makes it the simulation run rather than the telescope run is the task config,
not this code: TrueCalibrationTask in place of the astrometric/photometric solve, an Avro file sink in place of Kafka,
and an in-process fakeredis in place of valkey — each a config/environment choice. Point the same runner at a
production config (the real solve, Kafka, valkey) and it is the telescope pipeline.

The hdps task chain and coadd settings come from an observation TOML's [hdps] section (the one the image maker wrote
beside its frames): a stage-2 run writes <run>/observation.toml next to <run>/raw, so stage 3 needs only that run
directory.

    # a reference pass: raw reference-night frames -> reference coadds (hdps up to CoaddTapTask)
    python tools/run_hdps.py <ref_run>/raw -o <ref_out> --obs <ref_run>/observation.toml --coadd-only

    # a science pass: raw science frames + those reference coadds -> products, coadds and alerts
    python tools/run_hdps.py <sci_run>/raw -o <sci_out> --obs <sci_run>/observation.toml --reference <ref_out>
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _frames_dir(path):
    """Accept either a directory of *.fits or a run directory that holds raw/*.fits."""
    path = Path(path)
    if any(path.glob("*.fits")):
        return path
    if (path / "raw").is_dir() and any((path / "raw").glob("*.fits")):
        return path / "raw"
    raise SystemExit(f"{path}: no raw frames (*.fits here or in raw/)")


def _obs_path(frames, explicit):
    """The observation TOML with the [hdps] task chain: --obs, else observation.toml beside the frames."""
    if explicit is not None:
        return Path(explicit)
    for cand in (frames.parent / "observation.toml", frames / "observation.toml"):
        if cand.exists():
            return cand
    raise SystemExit("no observation.toml found beside the frames; pass --obs <observation.toml>")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("frames", help="a directory of raw *.fits frames, or a run directory holding raw/")
    p.add_argument("-o", "--out", type=Path, required=True, help="hdps's output directory (products, coadds, alerts/)")
    p.add_argument("--obs", help="the observation TOML for the [hdps] task chain and [hdps.coadd]; "
                                 "default: observation.toml beside the frames")
    p.add_argument("--reference", type=Path, help="a directory of reference coadds (a prior --coadd-only run's output); "
                                                  "SFFT subtracts against these. Omit for a reference/coadd pass.")
    p.add_argument("--coadd-only", action="store_true", help="run hdps up to CoaddTapTask only (make reference coadds)")
    p.add_argument("--worker-id", default="hdps_stage")
    args = p.parse_args(argv)

    from synthstar import hdps_worker, runconfig

    frames = _frames_dir(args.frames)
    obs = runconfig.resolve(runconfig.load(_obs_path(frames, args.obs)))
    result = hdps_worker.run(obs, frames, args.out, reference_dir=args.reference,
                             stop_after="CoaddTapTask" if args.coadd_only else None, worker_id=args.worker_id)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    main()
