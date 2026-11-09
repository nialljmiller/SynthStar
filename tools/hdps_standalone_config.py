"""Convert a resolved observation's [hdps] block into an hdps-native config.yaml, so hdps can be run entirely on its
own (`hdps batch process <raw_dir> --config config.yaml`) exactly as on the real telescope, with no simulator code in
the loop. This is the thin "interface layer" (the user's words): a small auxiliary file next to hdps that turns what
our pipeline configured into hdps's own config format.

The simulator today drives hdps in-process through mesa_argus/hdps_worker.py (fake valkey, no Ray, the worker called
directly). The standalone path instead uses hdps's real CLI + a local valkey + Ray, reading this config.yaml. The
[hdps] section of observation.toml is already a re-encoding of hdps's docker/demo/config.yaml (schema.py cites it), so
the conversion is mostly structural.

What it does:
  - [hdps.coadd]            -> coadd:                (same keys)
  - [[hdps.tasks]]          -> tasks:                (same {type, config} shape)
  - TrueCalibrationTask     -> PresolvedWCSTask      (hdps's native "use a pre-solved WCS" task; the sim's
                              TrueCalibrationTask is not in hdps's registry. Needs companion solved files written by
                              tools/hdps_presolved_wcs.py. Observations that already use the real solve chain -- e.g.
                              ReferenceRetrieval/LinearWCS/SourceMatching/TPV -- are left as-is.)
  - cameras:                one entry per CAMID found in the raw frames, each data_source = the raw dir
  - valkey/paths/data_products: filled for a local run

    python tools/hdps_standalone_config.py <observation.toml> <raw_dir> -o config.yaml [--output-dir products]
"""
import argparse
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

import yaml

# the sim's stand-in WCS task -> hdps's native pre-solved-WCS task (reads a companion solved file per frame)
TRUE_CALIBRATION = "TrueCalibrationTask"
PRESOLVED = "PresolvedWCSTask"


def camera_ids(raw_dir):
    """Every distinct CAMID in the raw frames (read from the FITS headers)."""
    from astropy.io import fits

    ids = set()
    for p in sorted(Path(raw_dir).glob("*.fits")):
        with fits.open(p) as h:
            cam = h[0].header.get("CAMID")
            if cam:
                ids.add(str(cam))
    return sorted(ids)


# The resolved observation.toml carries the schema's NONE sentinel as the string "none" and keeps empty-string
# defaults; hdps builds its Tasks straight from the config dict (hdps.orchestrate.pipeline.build_task_list), so the
# same translation the in-process worker does (mesa_argus/hdps_worker.py task_configs) has to happen here, with no
# mesa_argus import: "none" -> null, and a relative output_dir resolved under the products directory.
NONE = "none"
_OUTPUT_DIR_KEYS = ("output_dir", "science_output_dir", "difference_output_dir")


def _translate_config(cfg, output_dir):
    """A Task's [hdps.tasks.config], as hdps's build_task_list needs it: the NONE string -> None, and a relative
    output dir made absolute under the products directory (mirrors hdps_worker.task_configs)."""
    out = {}
    for k, v in (cfg or {}).items():
        v = None if v == NONE else v
        if k in _OUTPUT_DIR_KEYS and isinstance(v, str) and v and not Path(v).is_absolute():
            v = str((Path(output_dir) / v).resolve())
        out[k] = v
    return out


def to_config(obs, raw_dir, output_dir, solved_dir, suffix, stop_after=None, reference_dir=None,
              alert_model_bundle=None, alert_score_cut=None, alert_emit_all=None):
    """The hdps config.yaml dict from a resolved observation's [hdps] block.

    solved_dir / suffix fill the swapped PresolvedWCSTask's config so it finds each frame's companion WCS file.
    stop_after (a task type) truncates the chain after that task, inclusive — the "plumbing-first" subset.
    reference_dir: the earlier (reference) pass's coadd directory, set on SFFTSubtractionTask (reference_mode is
      already "coadd" in the chain) so difference imaging runs — mirrors mesa_argus/hdps_worker's injection.
    alert_model_bundle / alert_score_cut / alert_emit_all: override AlertVettingTask's classifier bundle, operating
      point and emit-all flag (the standalone plumbing test uses a stand-in bundle + score_cut 0 so candidates emit).
    """
    hdps = obs["hdps"]
    tasks = []
    swapped = False
    for t in hdps["tasks"]:
        if t["type"] == TRUE_CALIBRATION:  # the sim's toy WCS task is not in hdps's registry; use the native one,
            # pointed at the companion solved files tools/hdps_presolved_wcs.py writes
            tasks.append({"type": PRESOLVED, "config": {"solved_dir": str(Path(solved_dir).resolve()),
                                                        "suffix": suffix}})
            swapped = True
        else:
            c = _translate_config(t.get("config", {}), output_dir)
            if t["type"] == "SFFTSubtractionTask" and reference_dir is not None:
                c["reference_dir"] = str(Path(reference_dir).resolve())  # reference_mode is "coadd" in the chain
            if t["type"] == "AlertVettingTask":
                if alert_model_bundle is not None:
                    c["model_bundle"] = str(Path(alert_model_bundle).resolve())
                if alert_score_cut is not None:
                    c["score_cut"] = alert_score_cut
                if alert_emit_all is not None:
                    c["emit_all_candidates"] = bool(alert_emit_all)
            tasks.append({"type": t["type"], "config": c})
    if stop_after is not None:
        types = [t["type"] for t in tasks]
        if stop_after not in types:
            raise SystemExit(f"--stop-after {stop_after!r}: not in the chain {types}")
        tasks = tasks[: types.index(stop_after) + 1]
    n_cams = len(camera_ids(raw_dir))  # reported; frames carry their own CAMID, grouped from the header
    cfg = {
        # one data_source scanned once; the orchestrator groups frames by (RATCHNUM, CAMID) from their headers
        "cameras": [{"id": "auto", "data_source": str(Path(raw_dir).resolve())}],
        "valkey": {"valkey_url": "valkey://localhost:6379"},
        "paths": {"checkpoints": "checkpoints", "snapshots": "snapshots", "calibrations": "calibrations"},
        "file_pattern": "*.fits",
        "tasks": tasks,
        "coadd": dict(hdps.get("coadd", {})),
        "data_products": {"backend": "disk", "output_dir": str(Path(output_dir).resolve())},
    }
    return cfg, swapped, n_cams


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("observation", type=Path, help="a resolved observation.toml (observe.py writes it into scene/)")
    ap.add_argument("raw_dir", type=Path, help="the folder of raw FITS frames (the image-maker's --stop-after-render output)")
    ap.add_argument("-o", "--out", type=Path, default=Path("config.yaml"))
    ap.add_argument("--output-dir", type=Path, default=Path("products"), help="hdps data_products output dir")
    ap.add_argument("--solved-dir", type=Path, default=Path("solved"),
                    help="the companion WCS files' directory (PresolvedWCSTask solved_dir); must not be the raw dir")
    ap.add_argument("--suffix", default="_solved", help="companion suffix (PresolvedWCSTask default '_solved')")
    ap.add_argument("--stop-after", default=None,
                    help="truncate the chain after this task type (inclusive), e.g. CoaddTapTask for coadd-only")
    ap.add_argument("--reference-dir", type=Path, default=None,
                    help="the reference pass's coadd dir, set on SFFTSubtractionTask (enables difference imaging)")
    ap.add_argument("--alert-model-bundle", type=Path, default=None,
                    help="AlertVettingTask model_bundle dir (a stand-in RB bundle for the plumbing test)")
    ap.add_argument("--alert-score-cut", type=float, default=None,
                    help="AlertVettingTask score_cut (0.0 accepts every scored candidate)")
    ap.add_argument("--alert-emit-all", action="store_true", default=None,
                    help="AlertVettingTask emit_all_candidates = true (write below-cut candidates too)")
    args = ap.parse_args(argv)

    obs = tomllib.loads(args.observation.read_text())
    cfg, swapped, n_cams = to_config(obs, args.raw_dir, args.output_dir, args.solved_dir, args.suffix, args.stop_after,
                                     reference_dir=args.reference_dir, alert_model_bundle=args.alert_model_bundle,
                                     alert_score_cut=args.alert_score_cut, alert_emit_all=args.alert_emit_all)
    args.out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {args.out}: {len(cfg['tasks'])} tasks, {len(cfg['cameras'])} camera(s) ({n_cams} CAMID(s) in the raw dir)"
          + (f"; swapped {TRUE_CALIBRATION} -> {PRESOLVED} (solved_dir={args.solved_dir}, suffix={args.suffix!r}; "
             "run tools/hdps_presolved_wcs.py to write the companion solved files)" if swapped
             else "; uses the real solve chain as configured"))


if __name__ == "__main__":
    main()
