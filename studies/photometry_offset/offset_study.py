"""Where does measured - true come from? A decomposition of the movie's SExtractor photometry, star by star.

    python observe.py examples/observations/synthetic_variables.toml --scene-only   # the scenes (observe.py)
    python observe.py examples/observations/f294.toml --scene-only
    python studies/photometry_offset/offset_study.py synthetic_variables f294 --stride 3

For each observation (its scene, outputs/<category>/<name>/scene/), the movie (night_movie.main) is run unchanged, with --no-render, and every measured
exposure's pixels and per-star electron rates are captured. Each exposure is then measured again exactly as
synthstar.photometry does (SEP background, detection, Kron aperture, MAG_AUTO, the per-exposure zero point with a
colour term), and, because the scene is known, the measured flux of each star is split into what lies in its
aperture:

    F = own light + neighbours' light + (true sky - estimated sky) + noise

and its zero point into the colour model's error and the calibration stars' own losses. With e the star's total
electrons and m_true its true magnitude (ArgusSim, above the atmosphere), measured - true is the exact sum of

    aperture    -2.5 log10(own / e)                       light of the star outside its Kron aperture
    neighbours  -2.5 log10((own + nb) / own)              other stars' light inside it (deblending masks only
                                                          detected neighbours)
    sky         -2.5 log10((own + nb + dsky) / (own + nb))   SEP's background mesh vs the true sky + dark
    noise       -2.5 log10(F / (own + nb + dsky))         Poisson + read noise
    zp_colour   zp_fit_ideal(c) - zp_ideal                the colour term's error for this star: the line fitted
                                                          to the calibration stars' ideal zero points (m + 2.5
                                                          log10 e: exact flux, no aperture), at its catalogue colour
    zp_cal      zp_used - zp_fit_ideal(c)                 the calibration stars' own aperture / neighbour / sky /
                                                          noise losses, which the zero point absorbs

so that measured - true = sum of the six. A point source's aperture loss is the same for every star of one exposure,
so for an isolated star "aperture" and "zp_cal" cancel; what is left is the star's neighbours, the sky estimate and
the colour model.

Outputs (outputs/offset_study/): per-field CSVs (targets: the followed sources; field: every catalogue star measured
in every captured exposure) and figures.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = REPO / "outputs" / "offset_study"
TERMS = ("aperture", "neighbours", "sky", "noise", "zp_colour", "zp_cal")


# ---------------------------------------------------------------- capture
def capture(field, stride, band="rho"):
    """Run the movie for `field` without drawing it; return what it measured from."""
    import night_movie as M
    from synthstar import sep_photometry as P

    cap = dict(obs={}, phot=None, scene=None)
    observe = M.Scene.observe

    def observe_(self, name, rates, j, rng):
        r = observe(self, name, rates, j, rng)
        cap["scene"] = self
        if name == band:  # the measured exposure (the reference stack's exposures come first)
            cap["obs"][j] = dict(image=r["image"].copy(), rates=np.array(rates, float), background=float(r["background"]))
        return r

    init = P.SExtractorPhotometry.__init__

    def init_(self, *a, **k):
        init(self, *a, **k)
        cap["phot"] = self

    M.Scene.observe, P.SExtractorPhotometry.__init__ = observe_, init_
    out = OUT / "work" / f"{field}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    from argus_image.runconfig import find_output

    M.main(["--scene", str(find_output(field) / "scene"), "--field", field, "--band", band, "--stride", str(stride),
            "-o", str(out), "--no-render"])
    M.Scene.observe, P.SExtractorPhotometry.__init__ = observe, init
    import pandas as pd

    cap["csv"] = pd.read_csv(out.with_suffix(".photometry.csv"))
    cap["exptime"] = cap["scene"].exptime
    return cap


# ---------------------------------------------------------------- re-measurement
def extract(image, fwhm, background=None):
    """synthstar.photometry.extract, also returning the apertures and the segmentation; background: subtract this
    constant instead of SEP's mesh (the true-sky variant)."""
    import sep

    data = np.ascontiguousarray(image, dtype=np.float64)
    bkg = sep.Background(data, bw=64, bh=64, fw=3, fh=3)
    back = bkg.back() if background is None else np.full_like(data, background)
    sub = data - back
    sep.set_extract_pixstack(10_000_000)
    sep.set_sub_object_limit(8192)
    obj, seg = sep.extract(sub, 1.5, err=bkg.globalrms, minarea=5, deblend_nthresh=32, deblend_cont=0.005,
                           segmentation_map=True)
    ids = np.arange(1, len(obj) + 1, dtype=np.int32)
    x, y = np.asarray(obj["x"], float), np.asarray(obj["y"], float)
    xw, yw, fl = sep.winpos(sub, x, y, fwhm / 2.3548)
    a, b, th = np.asarray(obj["a"], float), np.asarray(obj["b"], float), np.clip(np.asarray(obj["theta"], float), -np.pi / 2, np.pi / 2)
    kr, _ = sep.kron_radius(sub, x, y, a, b, th, 6.0, seg_id=ids, segmap=seg)
    r = np.maximum(2.5 * np.nan_to_num(kr, nan=0.0), 3.5)
    flux, _, _ = sep.sum_ellipse(sub, x, y, a, b, th, r, err=bkg.globalrms, subpix=1, seg_id=ids, segmap=seg)
    good = fl == 0
    return dict(x=np.where(good, xw, x), y=np.where(good, yw, y), xc=x, yc=y, a=a, b=b, theta=th, r=r, ids=ids, seg=seg,
                flux=np.asarray(flux, float), peak=np.asarray(obj["peak"], float), back=back)


def aperture_sum(img, det, j, pad=3):
    """SEP's sum in detection j's aperture (same centre, ellipse, segmentation masking) of image img, on a cutout."""
    import sep

    n = img.shape[0]
    half = int(np.ceil(det["r"][j] * det["a"][j])) + pad
    cx, cy = int(round(det["xc"][j])), int(round(det["yc"][j]))
    x0, x1, y0, y1 = max(cx - half, 0), min(cx + half + 1, n), max(cy - half, 0), min(cy + half + 1, n)
    cut = np.ascontiguousarray(img[y0:y1, x0:x1], dtype=np.float64)
    segc = np.ascontiguousarray(det["seg"][y0:y1, x0:x1])
    s, _, _ = sep.sum_ellipse(cut, np.array([det["xc"][j] - x0]), np.array([det["yc"][j] - y0]), det["a"][j:j + 1],
                              det["b"][j:j + 1], det["theta"][j:j + 1], det["r"][j:j + 1], subpix=1,
                              seg_id=det["ids"][j:j + 1], segmap=segc)
    return float(s[0])


def decompose(cap, exposures=None, field_stars=True):
    """Rows (one per star per captured exposure) with measured - true and its six terms."""

    phot, scene, ex = cap["phot"], cap["scene"], cap["exptime"]
    labels = {}  # target index -> label, from the movie's CSV order
    star_names = list(dict.fromkeys(cap["csv"]["star"]))
    for i, name in enumerate(star_names):
        labels[phot.targets[i]] = name
    truth = {(r.exposure, r.star): r.true_mag for r in cap["csv"].itertuples()}
    cal = set(int(i) for i in phot.cal)
    rows, mismatched = [], []
    measured = sorted(set(cap["csv"]["exposure"]))  # (reference-stack exposures are captured too)
    for jx in measured if exposures is None else exposures:
        o = cap["obs"][jx]
        A = scene.design_at("rho", jx)
        e = o["rates"] * ex
        model = (A @ e).reshape(scene.n, scene.n)
        det = extract(o["image"], phot.fwhm)
        dsky = o["background"] - det["back"]
        sat = 0.98 * phot.fullwell

        def nearest(i):
            d = np.hypot(det["x"] - phot.x[i], det["y"] - phot.y[i])
            k = int(np.argmin(d))
            return k if d[k] <= phot.cfg["match_fwhm"] * phot.fwhm else -1

        # the pipeline's zero point (as photometry.measure), and the ideal one fitted the same way
        cy, cc, ci, ideal = [], [], [], []
        for i in phot.cal:
            k = nearest(i)
            if k >= 0 and det["flux"][k] > 0 and det["peak"][k] < sat:
                cy.append(phot.cat_mag[i] + 2.5 * np.log10(det["flux"][k]))
                cc.append(phot.colour[i])
                ci.append(i)
                ideal.append(phot.cat_mag[i] + 2.5 * np.log10(e[i]))
        cy, cc, ideal = np.array(cy), np.array(cc), np.array(ideal)
        use = np.ones(cy.size, bool)
        for _ in range(10):
            b_, a_ = np.polyfit(cc[use], cy[use], 1)
            r_ = cy - (a_ + b_ * cc)
            new = np.abs(r_) <= 3.0 * 1.4826 * np.median(np.abs(r_[use] - np.median(r_[use])))
            if np.array_equal(new, use):
                break
            use = new
        bi, ai = np.polyfit(cc[use], ideal[use], 1)  # the colour line through the exact zero points, same stars
        med_c = np.median(cc[use])

        who = list(phot.targets) + ([i for i in range(len(phot.x)) if i not in phot.targets and np.isfinite(phot.cat_mag[i])
                                      and phot.cat_mag[i] < 19.0] if field_stars else [])
        for i in who:
            k = nearest(i)
            if k < 0 or det["flux"][k] <= 0 or e[i] <= 0:
                continue
            is_t = i in labels
            m_true = truth.get((jx, labels[i])) if is_t else phot.cat_mag[i]
            if m_true is None or not np.isfinite(m_true):
                continue
            c = phot.colour[i] if np.isfinite(phot.colour[i]) else med_c
            own_img = (A[:, i].toarray().ravel() * e[i]).reshape(scene.n, scene.n)
            own = aperture_sum(own_img, det, k)
            nb = aperture_sum(model - own_img, det, k)
            dsk = aperture_sum(dsky, det, k)
            F = det["flux"][k]
            if own <= 0 or own + nb + dsk <= 0:  # matched to another star's detection: not this star's light
                mismatched.append((jx, i))
                continue
            zp_used, zp_fit, zp_ideal = a_ + b_ * c, ai + bi * c, m_true + 2.5 * np.log10(e[i])
            meas = zp_used - 2.5 * np.log10(F)
            t = dict(aperture=-2.5 * np.log10(own / e[i]), neighbours=-2.5 * np.log10((own + nb) / own),
                     sky=-2.5 * np.log10((own + nb + dsk) / (own + nb)), noise=-2.5 * np.log10(F / (own + nb + dsk)),
                     zp_colour=zp_fit - zp_ideal, zp_cal=zp_used - zp_fit)
            rows.append(dict(exposure=jx, star=labels.get(i, f"#{i}"), target=is_t, calibrator=i in cal, true_mag=m_true,
                             measured=meas, offset=meas - m_true, colour=c, colour_known=bool(np.isfinite(phot.colour[i])),
                             electrons=e[i], own_frac=own / e[i], nb_frac=nb / e[i], dsky_frac=dsk / e[i],
                             saturated=bool(det["peak"][k] >= sat), colour_slope=b_, colour_slope_ideal=bi,
                             ap_radius_px=float(det["r"][k] * np.sqrt(det["a"][k] * det["b"][k])),
                             kron_scale=float(det["r"][k]), iso_a_px=float(det["a"][k]), **t))
        print(f"  exposure {jx}: {len(rows)} rows ({len(mismatched)} matched to another star's detection so far)",
              flush=True)
    import pandas as pd

    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("fields", nargs="+", help="observation names (their scene/ written by observe.py)")
    p.add_argument("--stride", type=int, default=3, help="measure every Nth exposure (the movie's --stride)")
    p.add_argument("--field-every", type=int, default=5, help="field stars: every Nth captured exposure")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for f in a.fields:
        cap = capture(f, a.stride)
        targets = decompose(cap, field_stars=False)
        targets.to_csv(OUT / f"{f}_targets.csv", index=False)
        js = sorted(set(cap["csv"]["exposure"]))[::a.field_every]
        field = decompose(cap, js, field_stars=True)
        field.to_csv(OUT / f"{f}_field.csv", index=False)
        check = (targets.offset - targets[list(TERMS)].sum(axis=1)).abs().max()
        movie = cap["csv"].assign(r=cap["csv"].sextractor_mag - cap["csv"].true_mag)
        mine = targets.set_index(["exposure", "star"]).offset
        same = movie.set_index(["exposure", "star"]).r.reindex(mine.index)
        print(f"{f}: terms sum to the offset within {check:.1e} mag; re-measured vs the movie's own CSV: max "
              f"|diff| {np.nanmax(np.abs(mine - same)):.1e} mag", flush=True)


if __name__ == "__main__":
    main()
