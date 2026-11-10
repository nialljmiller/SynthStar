#!/usr/bin/env python3
"""SEP's results on the page's test exposures (demo/validate/sep.mjs make), for demo/validate/sep.mjs compare.

    python demo/validate/sep_reference.py   # demo/build/validate_sep/<case>.f64 -> <case>.ref.json, .back, .seg
"""

import json
from pathlib import Path
import sys

import numpy as np
import sep

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from synthstar import sep_photometry as P  # noqa: E402

DIR = REPO / "demo" / "build" / "validate_sep"


def main():
    for meta_path in sorted(DIR.glob("*.json")):
        if meta_path.name.endswith(".ref.json"):
            continue
        name = meta_path.stem
        meta = json.loads(meta_path.read_text())
        n = meta["n"]
        image = np.fromfile(DIR / f"{name}.f64", "<f8").reshape(n, n)
        # the steps of photometry.extract, kept to compare each one
        bkg = sep.Background(image, bw=64, bh=64, fw=3, fh=3)
        sub = image - bkg.back()
        sep.set_extract_pixstack(10_000_000)
        sep.set_sub_object_limit(8192)
        obj, seg = sep.extract(sub, 1.5, err=bkg.globalrms, minarea=5, deblend_nthresh=32, deblend_cont=0.005,
                               segmentation_map=True)
        ids = np.arange(1, len(obj) + 1, dtype=np.int32)
        th = np.clip(np.asarray(obj["theta"], float), -np.pi / 2, np.pi / 2)
        kr, _ = sep.kron_radius(sub, obj["x"], obj["y"], obj["a"], obj["b"], th, 6.0, seg_id=ids, segmap=seg)
        x, y, flux, ferr, peak = P.extract(image, meta["fwhm"])
        nan = lambda a: [None if not np.isfinite(v) else float(v) for v in np.asarray(a, float)]  # noqa: E731
        x_, y_ = np.asarray(meta["x"], float), np.asarray(meta["y"], float)
        mag, flux_c, colour = (np.array([np.nan if v is None else v for v in meta[k]], float) for k in ("mag", "flux", "colour"))
        phot = P.SExtractorPhotometry(x_, y_, mag, flux_c, n, meta["fullwell"], meta["targets"], colour, meta["fwhm"])
        r = phot.measure(image)
        ref = dict(globalrms=float(bkg.globalrms),
                   obj={k: nan(obj[k]) for k in ("x", "y", "a", "b", "theta", "peak", "npix", "cflux")},
                   det=dict(x=nan(x), y=nan(y), flux=nan(flux), ferr=nan(ferr), peak=nan(peak), kron=nan(kr)),
                   cal=[int(i) for i in phot.cal], measure={k: nan(r[k]) for k in ("mag", "flux", "sigma")})
        (DIR / f"{name}.ref.json").write_text(json.dumps(ref))
        np.asarray(bkg.back(), "<f8").tofile(DIR / f"{name}.back")
        np.asarray(seg, "<i4").tofile(DIR / f"{name}.seg")
        print(f"{name}: {len(obj)} detections, {len(phot.cal)} calibration stars")


if __name__ == "__main__":
    main()
