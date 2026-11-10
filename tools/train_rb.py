"""Train a real/bogus random forest on a training set (tools/rb_training_set.py) and write it as an hdps model bundle,
the directory hdps's AlertVettingTask loads (model_bundle; hdps.alerts.classifier.RFBundleClassifier):

    manifest.json   model_file, feature_names (hdps's FEATURE_NAMES order), feature_spec, model_version,
                    operating_points {label: score cut}, metrics
    model.joblib    a scikit-learn RandomForestClassifier: predict_proba(X)[:, 1] is the real score

Held-out test, whole fields (a field observed on several nights shares its stars and their residuals between them,
so no field is on both sides): --test-fields, those fields, and the bundle is the model measured (trained without
them, so its operating points are measured on fields it never saw); else, with 4 fields or more, --folds-fold
cross-validation by field (each field scored by a model trained without it) and the bundle trained on every field;
with fewer, whole exposures. The
measures are taken on the test candidates made as Argus runs: with --evaluate mask-off (the default), only nights run
with hdps's bright-star mask off (the training set's mask_bright_stars; a mask hides candidates, so its nights would
show fewer false alerts than Argus sends). Operating points: "faN" is the score cut that leaves N false alerts per
square degree per exposure on those test frames (hdps's convention, e.g. score_cut = "fa10"), from the bogus
candidates above the cut over the frames' readout-window area; "p50" etc. keep that fraction of real ones. Each
operating point is also measured per Moon phase (the training set's moon_phase).

    python tools/train_rb.py data/rb/training_set.parquet -o data/rb/bundle_v1
    python tools/train_rb.py data/rb/training_set.parquet -o data/rb/bundle_v2 --test-fields f278_214m16_942_field
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def frame_area_deg2(observation_dirs):
    """Readout-window area (deg^2) of each observation (its resolved [field] window, hdps minipixes of 28.125")."""
    from synthstar import runconfig

    out = {}
    for d in observation_dirs:
        obs = runconfig.load(Path(d) / "observation.toml")
        side_deg = obs["field"]["window"] * 32 * (900.0 / 1024.0) / 3600.0
        out[d] = side_deg ** 2
    return out


def main(argv=None):
    from hdps.alerts.features import FEATURE_NAMES, FEATURE_SPEC_VERSION
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    import joblib

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("training_set", type=Path, help="training_set.parquet (tools/rb_training_set.py)")
    p.add_argument("-o", "--out", type=Path, required=True, help="the bundle directory to write")
    p.add_argument("--trees", type=int, default=500)
    p.add_argument("--min-leaf", type=int, default=5, help="min_samples_leaf: smooths the score")
    p.add_argument("--test-fraction", type=float, default=0.25, help="held-out exposures (under 4 fields)")
    p.add_argument("--test-fields", nargs="+", help="field sets held out whole (the training set's field column)")
    p.add_argument("--folds", type=int, default=5, help="without --test-fields: cross-validation folds, by field")
    p.add_argument("--evaluate", choices=["mask-off", "all"], default="mask-off",
                   help="test candidates measured: those of nights with the bright-star mask off, or all")
    p.add_argument("--fa", type=float, nargs="+", default=[1.0, 10.0, 100.0], help="false alerts / deg^2 / exposure")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    t = pd.read_parquet(args.training_set)
    manifest_in = json.loads(args.training_set.with_suffix(".json").read_text())
    if manifest_in["feature_spec"] != FEATURE_SPEC_VERSION:
        sys.exit(f"training set feature spec {manifest_in['feature_spec']} != hdps's {FEATURE_SPEC_VERSION}")
    names = list(FEATURE_NAMES)
    t["frame"] = t["observation_dir"] + "|" + t["band"] + "|" + t["exposure"].astype(str)
    if "field" not in t.columns:  # a training set made before it carried provenance
        t["field"] = t["observation_dir"]
    for column, value in (("mask_bright_stars", True), ("moon_phase", "unknown")):
        if column not in t.columns:
            t[column] = value
    X, y = t[names].to_numpy(float), t["label"].to_numpy(int)
    fields = sorted(t["field"].unique())
    make = lambda: RandomForestClassifier(n_estimators=args.trees, min_samples_leaf=args.min_leaf,  # noqa: E731
                                          class_weight="balanced", n_jobs=-1, random_state=args.seed)
    t0 = time.time()
    score = np.full(len(t), np.nan)
    if args.test_fields:  # these fields held out whole
        unknown = sorted(set(args.test_fields) - set(fields))
        if unknown:
            sys.exit(f"--test-fields not in the training set: {', '.join(unknown)}")
        test = t["field"].isin(args.test_fields).to_numpy()
        split = f"fields held out: {', '.join(args.test_fields)} ({len(args.test_fields)} of {len(fields)})"
        if y[~test].sum() == 0 or y[test].sum() == 0:
            sys.exit("no real rows in the training or the test part: more (injected) observations are needed")
        held_out_model = make().fit(X[~test], y[~test])
        score[test] = held_out_model.predict_proba(X[test])[:, 1]
    elif len(fields) >= 4:  # cross-validation by field: every field scored by a model that never saw it
        from sklearn.model_selection import GroupKFold

        folds = min(args.folds, len(fields))
        for train_rows, test_rows in GroupKFold(n_splits=folds).split(X, y, groups=t["field"]):
            score[test_rows] = make().fit(X[train_rows], y[train_rows]).predict_proba(X[test_rows])[:, 1]
        test = np.ones(len(t), bool)
        split = f"{folds}-fold cross-validation by field ({len(fields)} fields)"
    else:  # one or a few fields: hold out whole exposures
        rng = np.random.default_rng(args.seed)
        frames = t["frame"].unique()
        held = set(rng.choice(frames, max(1, round(args.test_fraction * len(frames))), replace=False))
        test = t["frame"].isin(held).to_numpy()
        split = f"by exposure: {len(held)} of {len(frames)} held out (fewer than 4 fields)"
        score[test] = make().fit(X[~test], y[~test]).predict_proba(X[test])[:, 1]
    measured = test.copy()
    if args.evaluate == "mask-off":
        off = ~t["mask_bright_stars"].astype(bool).to_numpy()
        if (measured & off).any():
            measured &= off
            split += "; measured on the mask-off nights"
        else:
            split += "; no mask-off night in the test part: measured on all of it"
    print(f"{len(t)} rows ({y.sum()} real, {(1 - y).sum()} bogus); {split}", flush=True)
    if y[measured].sum() == 0 or (1 - y[measured]).sum() == 0:
        sys.exit("the measured test part needs real and bogus rows")
    area = frame_area_deg2(t.loc[measured, "observation_dir"].unique())

    def sky_of(rows):  # deg^2 x exposures of these rows' frames
        frames = t.loc[rows, ["frame", "observation_dir"]].drop_duplicates()
        return float(sum(area[d] for d in frames["observation_dir"]))

    yt, st = y[measured], score[measured]
    sky = sky_of(measured)
    bogus_scores, real_scores = np.sort(st[yt == 0])[::-1], st[yt == 1]
    points = {}
    for fa in args.fa:  # the cut leaving fa false alerts per deg^2 per exposure
        n_allowed = int(np.floor(fa * sky))
        cut = float(bogus_scores[n_allowed]) if n_allowed < len(bogus_scores) else 0.0
        points[f"fa{fa:g}"] = cut
    for keep in (0.5, 0.9):
        points[f"p{round(keep * 100)}"] = float(np.quantile(real_scores, 1 - keep))

    def at_points(rows):
        rsc, bsc, area_here = score[rows & (y == 1)], score[rows & (y == 0)], sky_of(rows)
        return {k: dict(cut=c, real_kept=float(np.mean(rsc >= c)) if len(rsc) else None,
                        false_per_deg2_exposure=float(np.sum(bsc >= c) / area_here) if area_here else None)
                for k, c in points.items()}

    by_phase = {}
    for phase in sorted(t.loc[measured, "moon_phase"].astype(str).unique()):
        rows = measured & (t["moon_phase"].astype(str) == phase).to_numpy()
        auc = float(roc_auc_score(y[rows], score[rows])) if 0 < y[rows].sum() < rows.sum() else None
        by_phase[phase] = dict(fields=int(t.loc[rows, "field"].nunique()),
                               nights=int(t.loc[rows, "observation_dir"].nunique()), real=int(y[rows].sum()),
                               bogus=int((1 - y[rows]).sum()), deg2_exposures=sky_of(rows), roc_auc=auc,
                               at_points=at_points(rows))
    metrics = dict(train_rows=int((~test).sum()) if not test.all() else int(len(t)), test_rows=int(test.sum()),
                   measured_rows=int(measured.sum()), test_real=int(yt.sum()), test_bogus=int((1 - yt).sum()),
                   test_deg2_exposures=sky, split=split, evaluate=args.evaluate,
                   roc_auc=float(roc_auc_score(yt, st)), average_precision=float(average_precision_score(yt, st)),
                   at_points=at_points(measured), by_moon_phase=by_phase,
                   fit_seconds=round(time.time() - t0, 1))
    # the bundle: with --test-fields, the model measured above; else trained on everything (the split only measures it)
    model = held_out_model if args.test_fields else make().fit(X, y)
    metrics["bundle_trained_on"] = "the training fields (not --test-fields)" if args.test_fields else "every row"
    metrics["feature_importance"] = {n: float(v) for n, v in sorted(zip(names, model.feature_importances_),
                                                                     key=lambda kv: -kv[1])}
    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out / "model.joblib", compress=3)
    version = f"mesa-argus-rf-{time.strftime('%Y%m%d')}-{len(t)}"
    manifest = dict(model_file="model.joblib", feature_names=names, feature_spec=FEATURE_SPEC_VERSION,
                    model_version=version, operating_points=points, metrics=metrics,
                    training=dict(set=str(args.training_set), label_rule=manifest_in["label_rule"],
                                  counts=manifest_in["counts"], trees=args.trees, min_samples_leaf=args.min_leaf,
                                  class_weight="balanced", note="trained on simulated data (MESA_ARGUS injection); "
                                                                "measure it on real data before trusting its cuts"))
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"ROC AUC {metrics['roc_auc']:.4f}, average precision {metrics['average_precision']:.4f}")
    for k, v in metrics["at_points"].items():
        print(f"  {k}: cut {v['cut']:.3f}: real kept {v['real_kept']:.3f}, false {v['false_per_deg2_exposure']:.2f} /deg2/exposure")
    for phase, m in by_phase.items():
        line = ", ".join(f"{k} {v['real_kept']:.3f} / {v['false_per_deg2_exposure']:.1f}"
                         for k, v in m["at_points"].items()
                         if v["real_kept"] is not None and v["false_per_deg2_exposure"] is not None)
        print(f"  {phase} ({m['nights']} nights, {m['fields']} fields): real kept / false per deg2 per exposure: "
              f"{line}")
    print(f"bundle -> {args.out} (model_version {version})")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
