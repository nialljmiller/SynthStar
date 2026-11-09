"""Build a minimal STAND-IN real/bogus bundle so hdps's AlertVettingTask can emit candidates standalone, without a
trained model. This is NOT a real classifier: paired with AlertVettingTask score_cut=0.0 (accept every scored
candidate) and emit_all_candidates=true, it exists only so the pipeline's alert step runs and writes its parquet
products in the standalone plumbing test. It must never be used for real vetting.

hdps.alerts.classifier.RFBundleClassifier(bundle_dir) loads:
  - manifest.json: {model_file, feature_names, feature_spec (must == hdps FEATURE_SPEC_VERSION), model_version,
                    operating_points, metrics}
  - the joblib model named by model_file, with a scikit-learn predict_proba -> (n, 2).

Zero simulator dependency: imports hdps (the system under test) + scikit-learn + joblib only; NO mesa_argus.

    python tools/hdps_make_standin_rb.py <bundle_dir>
"""
import argparse
import json
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle_dir", type=Path, help="directory to write manifest.json + model.joblib into")
    args = ap.parse_args(argv)

    import numpy as np
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    from hdps.alerts.features import FEATURE_NAMES, FEATURE_SPEC_VERSION

    bundle = args.bundle_dir
    bundle.mkdir(parents=True, exist_ok=True)
    n_features = len(FEATURE_NAMES)

    # A trivial two-class forest: enough to give a valid predict_proba of shape (n, 2). The downstream score_cut=0.0
    # makes the actual scores irrelevant (everything is accepted), so the training data is only a formality.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, n_features))
    y = np.array([0, 1] * 20)
    model = RandomForestClassifier(n_estimators=4, max_depth=2, random_state=0).fit(X, y)

    model_file = "model.joblib"
    joblib.dump(model, bundle / model_file)

    manifest = {
        "model_file": model_file,
        "feature_names": list(FEATURE_NAMES),
        "feature_spec": FEATURE_SPEC_VERSION,
        "model_version": "standin-allpass",
        # score_cut is passed numerically (0.0) in the standalone test, so these labels are a fallback only.
        "operating_points": {"fa10": 0.0, "fa1": 0.0, "fa100": 0.0},
        "metrics": {"note": "STAND-IN bundle, not a trained classifier"},
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote stand-in RB bundle -> {bundle} ({n_features} features, spec {FEATURE_SPEC_VERSION})")


if __name__ == "__main__":
    main()
