"""Figures for the offset study (offset_study.py writes the CSVs this reads).

    python studies/photometry_offset/plots.py synthetic_variables f294
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from argus_image.argus import apply_style  # noqa: E402

OUT = REPO / "outputs" / "offset_study"
TERMS = ("aperture", "sky", "zp_cal", "neighbours", "noise", "zp_colour")
NAMES = {"aperture": "aperture loss", "sky": "sky estimate", "zp_cal": "zero point: calibrators' losses",
         "neighbours": "neighbours' light", "noise": "noise", "zp_colour": "zero point: colour model"}
COL = {"aperture": "#cc5940", "sky": "#e0a030", "zp_cal": "#2078b4", "neighbours": "#7853a6", "noise": "#9aa5ad",
       "zp_colour": "#3b9a5a"}
FIELD_NAME = {"synthetic_variables": "synthetic field, hand-placed variables", "f294": "real f294 field (Gaia)",
              # the names of the first run of this study (2026-10-13), before observe.py
              "f294_784m11_623_vars": "synthetic field, hand-placed variables", "f294_784m11_623": "real f294 field (Gaia)"}
BINS = np.arange(11.0, 19.51, 0.5)
ORDER = ["f294_784m11_623_vars", "f294_784m11_623"]  # the fields plotted (synthetic first), set by main()


def load(fields):
    t, f = [], []
    for name in fields:
        a = pd.read_csv(OUT / f"{name}_targets.csv").assign(field=name)
        movie = pd.read_csv(OUT / "work" / f"{name}.photometry.csv")[["exposure", "star", "ut_hours", "airmass", "seeing"]]
        t.append(a.merge(movie, on=["exposure", "star"]))
        f.append(pd.read_csv(OUT / f"{name}_field.csv").assign(field=name))
    t, f = pd.concat(t), pd.concat(f)
    return t, f[~f.saturated & ~f.target]


def binned(x, y, bins=BINS):
    idx = np.digitize(x, bins)
    c, m, lo, hi, n = [], [], [], [], []
    for i in range(1, len(bins)):
        v = y[idx == i]
        if len(v) >= 8:
            c.append(0.5 * (bins[i - 1] + bins[i]))
            m.append(np.median(v))
            lo.append(np.percentile(v, 16))
            hi.append(np.percentile(v, 84))
            n.append(len(v))
    return map(np.array, (c, m, lo, hi, n))


def fig_decomposition(t):
    """Stacked medians of the six terms per target, with the net offset."""
    stars = list(t.groupby(["field", "star"]).size().index)
    med = t.groupby(["field", "star"])[list(TERMS) + ["offset", "true_mag", "colour"]].median().loc[stars]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    x = np.arange(len(stars))
    pos, neg = np.zeros(len(stars)), np.zeros(len(stars))
    for term in TERMS:
        v = med[term].to_numpy()
        base = np.where(v >= 0, pos, neg)
        ax.bar(x, v, 0.62, bottom=base, color=COL[term], label=NAMES[term], edgecolor="white", linewidth=0.6)
        pos, neg = pos + np.where(v >= 0, v, 0), neg + np.where(v < 0, v, 0)
    ax.scatter(x, med["offset"], marker="D", s=70, color="black", zorder=5, label="measured − true (net)")
    for xi, (key, r) in zip(x, med.iterrows()):
        ax.text(xi, max(pos[xi], 0) + 0.006, f"{r.offset:+.3f}", ha="center", fontsize=10, weight="bold")
    ax.axhline(0, color="#455868", lw=0.8)
    ax.set_xticks(x, [f"{s}\n{FIELD_NAME[f].split(',')[0]}\nρ {r.true_mag:.1f}, BP−RP {r.colour:.2f}"
                      for (f, s), (_, r) in zip(stars, med.iterrows())], fontsize=9)
    ax.set_ylabel("contribution to measured − true (mag)\n(+: measured too faint)")
    ax.set_title("Where each target's offset comes from (median over the night)", loc="left", weight="bold")
    ax.legend(loc="lower left", fontsize=9, ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    fig.savefig(OUT / "1_decomposition.png", dpi=130)
    plt.close(fig)


def fig_aperture(t, f):
    """MAG_AUTO's aperture loss (and the sky term) against magnitude; what the zero point removes; the targets."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    ax = axes[0]
    for name, mk in zip(ORDER, ("o", "s")):
        g = f[f.field == name]
        c, m, lo, hi, _ = binned(g.true_mag.to_numpy(), (g.aperture + g.sky).to_numpy())
        ax.plot(c, m, mk + "-", color="#cc5940" if mk == "o" else "#2078b4", ms=4, lw=1.5,
                label=f"field stars: {FIELD_NAME[name]}")
        cal = g[g.calibrator]
        ax.axhline(-cal.groupby("exposure").zp_cal.median().median(), ls="--", lw=1.0,
                   color="#cc5940" if mk == "o" else "#2078b4", alpha=0.8)
    for (fld, s), r in t.groupby(["field", "star"]):
        ax.scatter(r.true_mag.median(), (r.aperture + r.sky).median(), s=90, marker="*", color="black", zorder=6)
        ax.annotate(s, (r.true_mag.median(), (r.aperture + r.sky).median()), xytext=(5, 5), textcoords="offset points",
                    fontsize=9, weight="bold")
    ax.text(0.02, 0.97, "lines: field stars, median per 0.5 mag; dashed: what the zero point corrects\n"
            "for (median over the calibration stars); stars: the targets; grey: calibration stars",
            transform=ax.transAxes, va="top", fontsize=8.5, color="#333")
    ax.set(xlabel="true ρ (AB mag)", ylabel="aperture loss + sky-estimate loss (mag)", ylim=(0, 0.2))
    ax.set_title("Light lost from the Kron aperture grows for fainter stars", loc="left", weight="bold", fontsize=11.5)
    ax.legend(loc="center left", fontsize=8.5, frameon=False)
    ax2 = ax.twinx()
    cal = f[f.calibrator & (f.field == ORDER[0])].drop_duplicates("star")
    ax2.hist(cal.true_mag, bins=BINS, color="#2b2b2b", alpha=0.10)
    ax2.set_ylabel("calibration stars (synthetic field)", color="#777", fontsize=9)
    ax2.tick_params(colors="#777", labelsize=8)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    ax = axes[1]
    for term, ls in (("aperture", "-"), ("sky", "--")):
        g = f[f.field == ORDER[0]]
        c, m, lo, hi, _ = binned(g.true_mag.to_numpy(), g[term].to_numpy())
        ax.plot(c, m, ls, color=COL[term], lw=1.8, label=NAMES[term])
        ax.fill_between(c, lo, hi, color=COL[term], alpha=0.12)
    ax.set(xlabel="true ρ (AB mag)", ylabel="mag")
    ax.set_title("The two parts", loc="left", weight="bold", fontsize=11.5)
    ax.legend(fontsize=9, frameon=False)
    ax.text(0.02, 0.97, "sky: SEP's 64 px background mesh includes\nstars' wings, so it is slightly too high\n"
            "(over-subtraction scales with aperture area)", transform=ax.transAxes, va="top", fontsize=8.5, color="#333")

    ax = axes[2]
    g = f[f.field == ORDER[0]]
    ax.scatter(g.true_mag, g.ap_radius_px, s=3, color="#9aa5ad", alpha=0.4)
    c, m, lo, hi, _ = binned(g.true_mag.to_numpy(), g.ap_radius_px.to_numpy())
    ax.plot(c, m, "-", color="black", lw=1.6, label="median")
    ax.set(xlabel="true ρ (AB mag)", ylabel="Kron aperture radius, √(ab) × k (pixels)")
    ax.set_title("Why: the Kron aperture shrinks", loc="left", weight="bold", fontsize=11.5)
    ax.text(0.02, 0.03, "its size scales with the isophotal shape above the 1.5σ\ndetection threshold, which is smaller "
            "for faint stars", transform=ax.transAxes, fontsize=8.5, color="#333")
    for a in axes:
        a.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(OUT / "2_aperture_vs_magnitude.png", dpi=130)
    plt.close(fig)


def fig_neighbours(t, f):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    ax = axes[0]
    for name, colr in zip(ORDER, ("#cc5940", "#2078b4")):
        g = f[f.field == name]
        ax.scatter(g.nb_frac * 100, g.neighbours, s=4, alpha=0.35, color=colr, label=f"field stars: {FIELD_NAME[name]}")
    for (fld, s), r in t.groupby(["field", "star"]):
        ax.scatter(r.nb_frac.median() * 100, r.neighbours.median(), s=110, marker="*", color="black", zorder=5)
        ax.annotate(s, (r.nb_frac.median() * 100, r.neighbours.median()), xytext=(6, 4), textcoords="offset points",
                    fontsize=9, weight="bold")
    xx = np.linspace(0, max(30, float(np.percentile(f.nb_frac * 100, 99))), 100)
    ax.plot(xx, -2.5 * np.log10(1 + xx / 100 / 0.94), color="#455868", lw=1, ls=":", label="-2.5 log(1 + nb / 0.94)")
    ax.set(xscale="symlog", xlabel="neighbours' light inside the aperture (% of the star's total)",
           ylabel="neighbours term (mag)")
    ax.set_xlim(0, None)
    ax.set_title("Unmasked neighbour light makes stars too bright", loc="left", weight="bold", fontsize=11.5)
    ax.legend(fontsize=8.5, frameon=False, loc="lower left")
    ax = axes[1]
    for name, colr in zip(ORDER, ("#cc5940", "#2078b4")):
        g = f[f.field == name]
        c, m, lo, hi, _ = binned(g.true_mag.to_numpy(), g.neighbours.to_numpy())
        ax.plot(c, m, "-", color=colr, lw=1.6, label=FIELD_NAME[name])
        ax.fill_between(c, lo, hi, color=colr, alpha=0.12)
    ax.set(xlabel="true ρ (AB mag)", ylabel="neighbours term (mag)")
    ax.set_title("Typical neighbour bias by magnitude (median, 16-84%)", loc="left", weight="bold", fontsize=11.5)
    ax.legend(fontsize=8.5, frameon=False)
    for a in axes:
        a.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(OUT / "3_neighbours.png", dpi=130)
    plt.close(fig)


def fig_timeseries(t):
    groups = list(t.groupby(["field", "star"]))
    fig, axes = plt.subplots(len(groups), 1, figsize=(12, 2.3 * len(groups)), sharex=True)
    for ax, ((fld, s), r) in zip(np.atleast_1d(axes), groups):
        r = r.sort_values("ut_hours")
        ax.plot(r.ut_hours, r.offset, "o-", color="black", ms=3, lw=1.2, label="measured − true")
        for term in TERMS:
            ax.plot(r.ut_hours, r[term], color=COL[term], lw=1.1, label=NAMES[term])
        ax.axhline(0, color="#455868", lw=0.6)
        ax.set_ylabel("mag")
        ax.text(0.01, 0.92, f"{s} — {FIELD_NAME[fld]} (true ρ {r.true_mag.min():.2f}–{r.true_mag.max():.2f})",
                transform=ax.transAxes, fontsize=9.5, weight="bold", va="top")
        ax.grid(alpha=0.15)
    np.atleast_1d(axes)[0].legend(ncol=4, fontsize=8, frameon=False, loc="upper right", bbox_to_anchor=(1, 1.45))
    np.atleast_1d(axes)[-1].set_xlabel("UT (hours)")
    fig.suptitle("Through the night: the offset and its terms", x=0.01, ha="left", weight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / "4_through_the_night.png", dpi=130)
    plt.close(fig)


def fig_whatif(t, f):
    """What other zero-point models would do, refitted on the same calibration stars per exposure:
    as now (a + b (BP-RP)); + a linear magnitude term (a + b (BP-RP) + d m); local (as now, shifted by the median
    residual of the calibration stars within 0.75 mag of the star)."""
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), gridspec_kw=dict(width_ratios=[1, 1.15]))
    rows = []
    for (fld, jx), g in pd.concat([f, t[t.exposure.isin(f.exposure)]]).groupby(["field", "exposure"]):
        cal = g[g.calibrator]
        # offset = zp(c) - 2.5 log F - m. The zero point a calibration star implies, relative to the one used, is
        # -offset; refitting that with another model gives the change it would make to every star's offset
        z = -cal.offset.to_numpy()
        X = np.c_[np.ones(len(cal)), cal.colour, cal.true_mag]
        use = np.ones(len(cal), bool)
        for _ in range(10):
            coef, *_ = np.linalg.lstsq(X[use], z[use], rcond=None)
            r_ = z - X @ coef
            new = np.abs(r_) <= 3 * 1.4826 * np.median(np.abs(r_[use] - np.median(r_[use])))
            if np.array_equal(new, use):
                break
            use = new
        coef0, *_ = np.linalg.lstsq(X[use][:, :2], z[use], rcond=None)
        resid0 = z - X[:, :2] @ coef0
        m_cal = cal.true_mag.to_numpy()
        for _, s_ in g.iterrows():
            d_used = coef0[0] + coef0[1] * s_.colour  # ~0: the pipeline's own model, refitted on these rows
            d_mag = coef[0] + coef[1] * s_.colour + coef[2] * s_.true_mag
            near = use & (np.abs(m_cal - s_.true_mag) <= 0.75)
            d_local = d_used + (np.median(resid0[near]) if near.sum() >= 5 else 0.0)
            rows.append(dict(field=fld, star=s_.star, target=s_.target, true_mag=s_.true_mag, offset=s_.offset,
                             nb_frac=s_.nb_frac, neighbours=s_.neighbours,
                             offset_magterm=s_.offset + (d_mag - d_used), offset_local=s_.offset + (d_local - d_used)))
    w = pd.DataFrame(rows)
    ax = axes[0]
    iso = w[~w.target & (w.nb_frac < 0.005)]
    for col, ls, lab in (("offset", "-", "as now: a + b (BP−RP)"), ("offset_magterm", "--", "+ linear magnitude term"),
                         ("offset_local", ":", "local: calibrators within ±0.75 mag")):
        for name, colr in zip(ORDER, ("#cc5940", "#2078b4")):
            g = iso[iso.field == name]
            c, m, lo, hi, _ = binned(g.true_mag.to_numpy(), g[col].to_numpy())
            ax.plot(c, m, ls, color=colr, lw=1.8, label=f"{lab} ({FIELD_NAME[name].split(',')[0]})")
    ax.axhline(0, color="#455868", lw=0.6)
    ax.set(xlabel="true ρ (AB mag)", ylabel="median measured − true (mag)", ylim=(-0.08, 0.08))
    ax.set_title("Isolated field stars (< 0.5% neighbour light): a zero-point\nmodel shifts the trend but does not remove it", loc="left", weight="bold", fontsize=11.5)
    ax.legend(fontsize=7.8, frameon=False, ncol=1, loc="lower left")
    ax = axes[1]
    tt = w[w.target].groupby(["field", "star"])[["offset", "offset_magterm", "offset_local", "neighbours"]].median()
    tt["local_without_neighbours"] = tt.offset_local - tt.neighbours
    x = np.arange(len(tt))
    for dx, col, colr, lab in ((-0.3, "offset", "#9aa5ad", "as now"), (-0.1, "offset_magterm", "#e0a030", "+ magnitude term"),
                               (0.1, "offset_local", "#2078b4", "local zero point"),
                               (0.3, "local_without_neighbours", "#3b9a5a", "local, minus the star's neighbours")):
        ax.bar(x + dx, tt[col], 0.2, color=colr, label=lab)
    ax.set_xticks(x, [s_ for _, s_ in tt.index])
    ax.axhline(0, color="#455868", lw=0.6)
    ax.set_ylabel("median measured − true (mag)")
    ax.set_title("Targets: no zero-point model fixes all five; their own\nneighbours (P2 −0.05, P1 −0.02, V1 −0.01) remain", loc="left", weight="bold", fontsize=11.5)
    ax.legend(fontsize=8.5, frameon=False, loc="lower left")
    for a_ in axes:
        a_.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(OUT / "5_what_if_zero_point.png", dpi=130)
    plt.close(fig)
    return w, tt


def fig_colour(f, t):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for name, colr in zip(ORDER, ("#cc5940", "#2078b4")):
        g = f[(f.field == name) & f.colour_known]
        ax.scatter(g.colour, g.zp_colour, s=3, alpha=0.3, color=colr, label=FIELD_NAME[name])
    for (fld, s), r in t.groupby(["field", "star"]):
        ax.scatter(r.colour.median(), r.zp_colour.median(), s=110, marker="*", color="black", zorder=5)
        ax.annotate(s, (r.colour.median(), r.zp_colour.median()), xytext=(5, 4), textcoords="offset points", fontsize=9)
    ax.axhline(0, color="#455868", lw=0.6)
    ax.set(xlabel="catalogue BP−RP", ylabel="colour-model term (mag)", ylim=(-0.06, 0.06))
    ax.set_title("Colour model: < 0.012 mag for the targets, but the linear\nterm misses red stars (BP−RP > 1.8) by up to 0.06 mag", loc="left", weight="bold", fontsize=11.5)
    ax.legend(fontsize=8.5, frameon=False, markerscale=3)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(OUT / "6_colour_model.png", dpi=130)
    plt.close(fig)


def main():
    apply_style()
    global ORDER
    fields = ORDER = sys.argv[1:] or ORDER
    t, f = load(fields)
    fig_decomposition(t)
    fig_aperture(t, f)
    fig_neighbours(t, f)
    fig_timeseries(t)
    w, tt = fig_whatif(t, f)
    fig_colour(f, t)
    print(t.groupby(["field", "star"])[["offset", *TERMS, "true_mag", "nb_frac"]].median().round(4).to_string())
    print(tt.round(4).to_string())


if __name__ == "__main__":
    main()
