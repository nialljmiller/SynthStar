"""How much stellar physics can each photometric system recover? Fisher-information analysis.

For every anchor star in a stencil grid (grids/fisher_stencil.py) and every EEP:

  J[b, p] = d m_b / d p       central differences between the +/- stencil tracks at the same EEP,
                              for p in ([M/H], ln M, alpha_MLT); plus two nuisance columns:
                              the EEP tangent (moving along the track = age) and the distance
                              modulus mu (shifts every band equally).
  F = J^T diag(1/sigma_b^2) J + prior(mu)
  sigma(p) = sqrt((F^-1)_pp)  Cramer-Rao bound: the best precision any unbiased fit
                              to that star's photometry can reach.

Questions (sets of free parameters):
  feh, lnM, alpha   that parameter, marginalised over evolutionary state and distance
                    (the other physics parameters held fixed);
  feh|alpha         [M/H] with alpha_MLT also free: needs >= 3 bands, so with 2 bands
                    (Argus) it is unconstrained from a single star.

Gaia and SDSS use the fixed per-band errors in analysis.toml; Argus is evaluated at each
trial error. One figure per trial error, plus an equal-noise figure (bandpass information
only) and a summary vs Argus error.
"""

import csv
import math
from collections import defaultdict

import numpy as np

from . import config as C
from .eep import N_EEP, SEGMENTS, resample
from .grids import load
from .history import load_track

PARAMS = ("feh", "lnM", "alpha")
QUESTIONS = {  # key: (free physics parameters, target, label)
    "feh": (("feh",), "feh", "σ([M/H])  [dex]"),
    "lnM": (("lnM",), "lnM", "σ(M)/M"),
    "alpha": (("alpha",), "alpha", "σ(α$_{MLT}$)"),
    "feh|alpha": (("feh", "alpha"), "feh", "σ([M/H]), α$_{MLT}$ also free  [dex]"),
}
SYSTEMS = {  # history column -> key used in analysis.toml
    "Gaia": {"G": "G", "Gbp": "BP", "Grp": "RP"},
    "Argus": {"beta": "beta", "rho": "rho"},
    "SDSS": {f"SDSS_{b}": b for b in "ugriz"},
}
SYS_COLOURS = {"Gaia": "#2a78d6", "Argus": "#eb6834", "SDSS": "#1baf7a"}
COLUMNS = C.ALL_BANDS + ("ok", "log_Teff", "log_L", "star_age")


# ------------------------------------------------------------------ derivatives
def anchor_derivatives(group):
    """group: stencil tracks of one anchor. Returns (dm[band] -> {p: array[N_EEP]}, tangent, valid, centre)."""
    eeps = {}
    for t in group:
        d = load_track(t["name"])
        d["ok"] = d["ok"].astype(float)
        eeps[(t["param"], t["sign"])] = (t, resample(d, COLUMNS))
    centre_t, centre = eeps[(None, 0)]
    steps = {
        "feh": eeps[("feh", 1)][0]["feh"] - eeps[("feh", -1)][0]["feh"],
        "alpha": eeps[("alpha", 1)][0]["alpha"] - eeps[("alpha", -1)][0]["alpha"],
        "lnM": math.log(eeps[("lnM", 1)][0]["mass"] / eeps[("lnM", -1)][0]["mass"]),
    }
    valid = np.ones(N_EEP, bool)
    for _, e in eeps.values():
        valid &= np.isfinite(e["G"]) & (e["ok"] > 0.999)
    deriv, tangent = {}, {}
    seg = centre["segment"]
    for b in C.ALL_BANDS:
        deriv[b] = {p: (eeps[(p, 1)][1][b] - eeps[(p, -1)][1][b]) / steps[p] for p in PARAMS}
        tg = np.full(N_EEP, np.nan)
        for s in np.unique(seg):  # do not difference across segment gaps
            idx = np.flatnonzero(seg == s)
            tg[idx] = np.gradient(centre[b][idx])
        tangent[b] = tg
    # drop the 2 EEPs at each segment edge (one-sided tangents, segment-boundary transients)
    edge = np.zeros(N_EEP, bool)
    start = 0
    for _, n in SEGMENTS:
        edge[start:start + 2] = edge[start + n - 2:start + n] = True
        start += n
    return deriv, tangent, valid & np.isfinite(tangent["G"]) & ~edge, centre


def crlb(deriv, tangent, e, bands, sigmas, free, sigma_mu):
    """Cramer-Rao bound on free[0] at EEP e, marginalising over free[1:], EEP and distance."""
    cols = [[deriv[b][p][e] for b in bands] for p in free]
    cols.append([tangent[b][e] for b in bands])   # along-track (age) nuisance, flat prior
    cols.append([1.0] * len(bands))               # distance modulus
    J = np.array(cols).T / np.asarray(sigmas)[:, None]
    F = J.T @ J
    F[-1, -1] += 1.0 / sigma_mu**2
    if np.linalg.matrix_rank(F, tol=1e-10 * np.abs(F).max()) < F.shape[0]:
        return np.inf
    return float(np.sqrt(max(np.linalg.inv(F)[0, 0], 0.0)))


def system_sigmas(argus_sigma, equal=None):
    a = C.analysis()
    out = {}
    for sysname, bands in SYSTEMS.items():
        if equal is not None:
            s = [equal] * len(bands)
        elif sysname == "Argus":
            s = [argus_sigma] * len(bands)
        else:
            s = [a["fixed"][sysname.lower()][k] for k in bands.values()]
        out[sysname] = (list(bands), s)
    return out


def evaluate(anchors, argus_sigma, equal=None):
    """{(anchor, question, system): array[N_EEP] of sigma}."""
    sigma_mu = C.analysis()["sigma_mu"]
    syss = system_sigmas(argus_sigma, equal)
    res = {}
    for name, (deriv, tangent, valid, _) in anchors.items():
        for q, (free, _, _) in QUESTIONS.items():
            for sysname, (bands, s) in syss.items():
                out = np.full(N_EEP, np.nan)
                for e in np.flatnonzero(valid):
                    out[e] = crlb(deriv, tangent, e, bands, s, free, sigma_mu)
                res[(name, q, sysname)] = out
    return res


# ------------------------------------------------------------------ figures
def _phase_axis(ax, show_labels):
    start = 0
    for k, (seg, n) in enumerate(SEGMENTS):
        if k % 2 == 1:
            ax.axvspan(start, start + n, color="0.5", alpha=0.07, lw=0)
        if show_labels:
            ax.text(start + n / 2, 1.0, seg, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=8, color="0.35")
        start += n
    ax.set_xlim(0, N_EEP)
    ax.set_xticks([])


def figure(res, anchors, labels, title, path, ylabel_suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(anchors)
    qs = list(QUESTIONS)
    fig, axes = plt.subplots(len(qs), len(names), figsize=(max(3.1 * len(names), 7.0), 2.3 * len(qs) + 1.6),
                             sharex=True, sharey="row", squeeze=False)
    for i, q in enumerate(qs):
        for j, a in enumerate(names):
            ax = axes[i, j]
            _phase_axis(ax, show_labels=(i == 0))
            for sysname, col in SYS_COLOURS.items():
                y = res[(a, q, sysname)]
                finite = np.where(np.isfinite(y), y, np.nan)
                ax.plot(np.arange(N_EEP), finite, color=col, lw=1.6, label=sysname)
                yv = y[anchors[a][2]]
                if q == "feh|alpha" and yv.size and np.all(np.isinf(yv)):
                    ax.text(0.5, 0.5, f"{sysname}: unconstrained\n(too few bands)", transform=ax.transAxes,
                            ha="center", va="center", fontsize=7.5, color=col)
            ax.set_yscale("log")
            ax.grid(True, which="major", color="0.88", lw=0.6)
            if i == 0:
                ax.set_title(labels[a], fontsize=9.5, pad=14)
            if j == 0:
                ax.set_ylabel(QUESTIONS[q][2] + ylabel_suffix, fontsize=9)
            if i == len(qs) - 1:
                ax.set_xlabel("evolutionary stage (EEP)", fontsize=8.5)
    handles, labs = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle(title.replace("   |   ", "\n"), fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def summary_figure(anchors, path):
    """Median precision per phase vs Argus per-band error; Gaia/SDSS (fixed errors) as references."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.logspace(-4, 0, 17)
    qs = [q for q in QUESTIONS if q != "feh|alpha"]
    seg = np.repeat(np.arange(len(SEGMENTS)), [n for _, n in SEGMENTS])

    def median_by_phase(res, q, sysname):
        out = []
        for k in range(len(SEGMENTS)):
            v = np.concatenate([res[(a, q, sysname)][seg == k] for a in anchors])
            v = v[np.isfinite(v)]
            out.append(np.median(v) if v.size else np.nan)
        return out

    curves = defaultdict(list)
    for s in grid:
        r = evaluate(anchors, s)
        for q in qs:
            curves[q].append(median_by_phase(r, q, "Argus"))
    ref = evaluate(anchors, 0.01)
    fig, axes = plt.subplots(len(qs), len(SEGMENTS), figsize=(10, 2.6 * len(qs) + 0.6), squeeze=False, sharex=True)
    for i, q in enumerate(qs):
        c = np.array(curves[q])
        for k, (segname, _) in enumerate(SEGMENTS):
            ax = axes[i, k]
            ax.loglog(grid, c[:, k], color=SYS_COLOURS["Argus"], lw=2, label="Argus (vs its per-band error)")
            for sysname in ("Gaia", "SDSS"):
                ax.axhline(median_by_phase(ref, q, sysname)[k], color=SYS_COLOURS[sysname], lw=1.6, ls="--",
                           label=f"{sysname} (fixed errors)")
            for s in C.analysis()["argus"]["trial_sigmas"]:
                ax.axvline(s, color="0.8", lw=0.6, zorder=0)
            ax.grid(True, which="major", color="0.9", lw=0.6)
            if i == 0:
                ax.set_title(segname, fontsize=10)
            if k == 0:
                ax.set_ylabel(QUESTIONS[q][2], fontsize=9)
            if i == len(qs) - 1:
                ax.set_xlabel("Argus per-band error (mag)", fontsize=9)
    axes[0, 0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle("Median achievable precision (all anchor stars) vs Argus photometric error;\n"
                 "grey lines = trial errors; where the orange line drops below a dashed line, Argus beats that survey",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------ driver
def complete_anchors(grid="fisher_stencil"):
    from .runner import status
    groups = defaultdict(list)
    for t in load(grid):
        groups[t["anchor"]].append(status(t["name"]))
    return sorted(a for a, st in groups.items() if all(s == "done" for s in st)), \
        any(s in ("running", "pending") for st in groups.values() for s in st)


def live(grid="fisher_stencil", every=600):
    """Re-run the analysis each time another anchor's stencil completes."""
    import time
    seen = None
    while True:
        done, active = complete_anchors(grid)
        if done and done != seen:
            print(time.strftime("%H:%M"), f"{len(done)} complete anchor(s): re-running analysis", flush=True)
            main(grid)
            seen = done
        if not active:
            break
        time.sleep(every)


def main(grid="fisher_stencil", sigmas=None):
    tracks = load(grid)
    groups = defaultdict(list)
    for t in tracks:
        groups[t["anchor"]].append(t)
    anchors, labels, skipped = {}, {}, []
    complete, _ = complete_anchors(grid)
    for name, group in groups.items():
        if name not in complete:
            skipped.append(f"{name} (stencil not finished)")
            continue
        anchors[name] = anchor_derivatives(group)
        labels[name] = group[0]["population"].replace("Stencil: ", "")
    if skipped:
        print("skipping anchors with missing/unfinished tracks:", ", ".join(skipped))
    if not anchors:
        raise SystemExit("no complete anchors yet")
    outdir = C.OUTPUTS / "fisher"
    outdir.mkdir(parents=True, exist_ok=True)
    a = C.analysis()
    fixed = "; ".join(f"{s}: " + ", ".join(f"{k} {v * 1e3:g}" for k, v in a["fixed"][s].items()) + " mmag"
                      for s in ("gaia", "sdss"))
    rows = []
    for s in sigmas or a["argus"]["trial_sigmas"]:
        res = evaluate(anchors, s)
        tag = f"{s * 1e3:g}mmag" if s < 0.1 else f"{s:g}mag"
        figure(res, anchors, labels, f"Argus per-band error {tag}   |   fixed: {fixed}   |   σ(μ) = {a['sigma_mu']} mag",
               outdir / f"argus_sigma_{tag}.png")
        for (an, q, sysname), v in res.items():
            for e in np.flatnonzero(np.isfinite(v) | np.isinf(v)):
                rows.append([s, an, q, sysname, e, SEGMENTS[np.searchsorted(np.cumsum([n for _, n in SEGMENTS]), e, side="right")][0], v[e]])
        print(f"  wrote {outdir / f'argus_sigma_{tag}.png'}")
    eq = a["equal"]["sigma"]
    figure(evaluate(anchors, None, equal=eq), anchors, labels,
           f"Bandpass information only: every band in every system has σ = {eq} mag   |   σ(μ) = {a['sigma_mu']} mag",
           outdir / "equal_noise.png")
    summary_figure(anchors, outdir / "summary_vs_argus_sigma.png")
    with open(outdir / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["argus_sigma", "anchor", "question", "system", "eep", "segment", "sigma"])
        w.writerows(rows)
    print(f"  wrote {outdir / 'equal_noise.png'}\n  wrote {outdir / 'summary_vs_argus_sigma.png'}\n  wrote {outdir / 'results.csv'}")
