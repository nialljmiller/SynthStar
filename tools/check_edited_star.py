"""The live notebook's edited star checked against hdps itself.

For each transient injected into a pack's run and shown in it (followed, inside the block, measured by hdps), the
same library light is put on empty sky beside it (the faintest spot 25-40" away, within 100" of the centre) and
measured by the app's stand-in (synthstar.interactive: compose, measure). Per frame, the two are compared: detected,
measured - true (hdps's catalogue for the real star; the stand-in for E), and alerted (the real star's hdps candidates
and E's, scored by the same real/bogus model and cut). The engine leaves transients out of the reference, so E's
reference holds none of it here either.

--features: the stand-in's alert features checked on hdps's own candidates: in every 5th frame, each real candidate
(a transient or variable) and up to 8 others, at least 15" inside the cut-out, featurized again by the app's code on
the frame's cut-outs; both feature sets scored by the same model and cut, and the accept/reject decisions compared.

    python tools/check_edited_star.py f294test --band beta
    python tools/check_edited_star.py f294test --band beta --features
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def empty_spot(p, dx0, dy0, radius=12.0, limit=100.0):
    """The offset position near (dx0, dy0) whose surroundings are faintest in the pack's science cut-outs."""
    from synthstar.interactive import measure

    best = None
    for ox, oy in [(25, 0), (-25, 0), (0, 25), (0, -25), (30, 30), (-30, 30), (30, -30), (-30, -30), (40, 0), (0, 40)]:
        x, y = dx0 + ox, dy0 + oy
        if abs(x) > limit or abs(y) > limit:
            continue
        values = []
        for c in p["cutouts"][::10]:
            if c is None:
                continue
            px, py, step = measure._pixel(c["extent"], c["science"].shape, x, y)
            r = int(radius / step)
            sub = c["science"][max(int(py) - r, 0):int(py) + r, max(int(px) - r, 0):int(px) + r]
            values.append(np.nanmax(sub) - np.nanmedian(c["science"]))
        if values and (best is None or np.median(values) < best[0]):
            best = (np.median(values), x, y)
    return best[1], best[2]


def check(pack_name, band):
    import pandas as pd
    from hdps.alerts.features import FEATURE_NAMES

    from synthstar.interactive import compose, measure
    from synthstar.interactive import templates as T
    from synthstar.interactive.night import Night

    night = Night(pack_name, band)
    p = night.p
    phot_path = REPO / "outputs" / "pseudo_real" / p["observation"] / f"{p['observation']}_night_{band}.photometry.csv"
    if not phot_path.exists():
        raise SystemExit(f"{phot_path}: the run's photometry is needed (the pack's run, on this machine)")
    phot = pd.read_csv(phot_path)
    clf = measure.classifier(night.model)
    cut = clf.cut_for(night.cut)
    mad = lambda a: 1.4826 * np.median(np.abs(a - np.median(a)))  # noqa: E731
    exptime = p["args"]["exptime"]
    for k in p["shown"]:
        entry = p["pulsators"][p["source"][k]]
        if "transient" not in entry:
            continue
        v = entry["transient"]
        label, t0 = v["label"], float(v["t0_bjd"])
        light = T.light(v["star"], round(float(v["ebv"]), 3))
        ex, ey = empty_spot(p, float(p["stars"]["dx"][k]), float(p["stars"]["dy"][k]))
        rows = []
        for n, j in enumerate(night.frames):
            c = p["cutouts"][n]
            h = phot[(phot["exposure"] == j) & (phot["star"] == label)]
            hdps_d = float(h["measured_mag"].iloc[0] - h["true_mag"].iloc[0]) if len(h) else np.nan
            cand = p["candidates"][n]
            at_star = [i for i, key in enumerate(cand["key"]) if key == k]
            hdps_alert = bool(at_star) and bool((clf.score_features(
                {name: cand["features"][at_star, i].astype(float) for i, name in enumerate(FEATURE_NAMES)}) >= cut).any())
            if c is None:
                rows.append((False, hdps_d, np.nan, hdps_alert, None))
                continue
            X = float(p["airmass"][j])
            days = float(p["bjd"][j]) - t0
            e = float(np.interp(days, light["x"], [np.interp(X, T.AIRMASS, r) for r in light["electrons"][band]])) * exptime
            true = float(np.interp(days, light["x"], np.nan_to_num(light["mag"][band], nan=99.0)))
            comp = compose.frame(c, band, float(p["seeing"][j]), float(p["survey"]["field_angle"][j]), X, e, 0.0, ex, ey,
                                 seed=n, sky=night.sky[n], q=night.q[n], wing=compose.wing_key(p.get("image")))
            m, _, _, _ = measure.photometry(comp, ex, ey, p["zp"][n])
            features = measure.candidate(comp, ex, ey, night.hdps_fwhm[n], night.hdps_plane[n], mask=night.mask)
            rows.append((True, hdps_d, m - true, hdps_alert, features))
        scored = measure.alerts([r[4] for r in rows], night.model, night.cut)
        valid = np.array([r[0] for r in rows])
        hd, ed = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
        ha, ea = np.array([r[3] for r in rows]), np.array([a for _, a in scored])
        f1, f2 = np.isfinite(hd) & valid, np.isfinite(ed) & valid
        print(f"{label} ({v['kind']}, {v['star']}); E at ({ex:.0f}\", {ey:.0f}\")")
        print(f"   detected:        hdps {f1.sum()}/{valid.sum()}   E {f2.sum()}/{valid.sum()}")
        if f1.any() and f2.any():
            print(f"   measured - true: hdps {np.median(hd[f1]):+.3f} (scatter {mad(hd[f1]):.3f})   "
                  f"E {np.median(ed[f2]):+.3f} (scatter {mad(ed[f2]):.3f})")
        print(f"   alerted:         hdps {ha[valid].sum()}/{valid.sum()}   E {ea[valid].sum()}/{valid.sum()}   "
              f"same decision in {np.mean(ha[valid] == ea[valid]):.0%} of frames")


def features(pack_name, band):
    from hdps.alerts.features import FEATURE_NAMES

    from synthstar.interactive import measure
    from synthstar.interactive.night import Night

    night = Night(pack_name, band)
    p = night.p
    clf = measure.classifier(night.model)
    cut = clf.cut_for(night.cut)
    rng = np.random.default_rng(1)
    rows = []
    for n in range(0, len(p["frames"]), 5):
        c, cand = p["cutouts"][n], p["candidates"][n]
        if c is None or not len(cand["key"]):
            continue
        ext = c["extent"]
        inside = ((cand["dx"] > ext[0] + 15) & (cand["dx"] < ext[1] - 15) & (cand["dy"] > ext[2] + 15)
                  & (cand["dy"] < ext[3] - 15))
        real = np.isin(cand["truth"], ["transient", "variable"])
        other = np.nonzero(inside & ~real)[0]
        rows += [(n, i) for i in list(np.nonzero(inside & real)[0]) + list(rng.choice(other, min(8, len(other)),
                                                                                      replace=False))]
    hd = np.array([p["candidates"][n]["features"][i] for n, i in rows], float)
    s_hd = clf.score_features({name: hd[:, k] for k, name in enumerate(FEATURE_NAMES)})
    mine = [measure.candidate(p["cutouts"][n], float(p["candidates"][n]["dx"][i]), float(p["candidates"][n]["dy"][i]),
                              night.hdps_fwhm[n], night.hdps_plane[n], mask=night.mask) for n, i in rows]
    found = np.array([f is not None for f in mine])
    s_me = np.full(len(rows), np.nan)
    if found.any():
        s_me[found] = clf.score_features({name: np.array([f[name] for f in mine if f is not None])
                                          for name in FEATURE_NAMES})
    real = np.array([p["candidates"][n]["truth"][i] in ("transient", "variable") for n, i in rows])
    same = (s_me[found] >= cut) == (s_hd[found] >= cut)
    print(f"{len(rows)} hdps candidates ({real.sum()} real) in {len({n for n, _ in rows})} frames; found again "
          f"{found.sum()}; same accept/reject as hdps {same.mean():.0%}; score difference median "
          f"{np.median(np.abs(s_me[found] - s_hd[found])):.3f}; accepted: hdps {np.sum(s_hd[found] >= cut)}, "
          f"stand-in {np.sum(s_me[found] >= cut)}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pack", help="a field pack whose run injected transients (e.g. f294test)")
    p.add_argument("--band", default="beta")
    p.add_argument("--features", action="store_true", help="check the stand-in's alert features on hdps's candidates")
    args = p.parse_args(argv)
    if args.features:
        features(args.pack, args.band)
    else:
        check(args.pack, args.band)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()
