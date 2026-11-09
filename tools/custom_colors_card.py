#!/usr/bin/env python3
"""Homepage card animation for MESA Custom Colors.

Five MESA tracks (0.8 Msun at [Fe/H] = -2; 1, 3, 5 and 10 Msun at solar)
evolve together.  Top: Gaia colour-magnitude diagram from the colors
module's own synthetic magnitudes, each star drawn in the true colour of its
SED and sized by its radius.  Bottom: the live SEDs over the 11 passbands,
with band-averaged fluxes marking the synthetic photometry.

Everything comes from runs/*/seds.h5 written by the colors module; models
outside the atmosphere grid are dropped.  The clock runs in log(age) but
spends screen time in proportion to how much is changing, so the long main
sequence wait does not stall the loop.

    python tools/custom_colors_card.py --out custom_colors_card.mp4
    python tools/custom_colors_card.py --frame 0.55 --out frame.png
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import h5py
import matplotlib
import matplotlib.colors
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

ROOT = Path(__file__).resolve().parents[1]
# (run, label, label side); the metal-poor star stays bluer than 1 Msun, so its
# label goes on the left to keep the two apart.
RUNS = [("M10.00_MH+0.0_a1.98_sed", "10 M$_\\odot$", "right"),
        ("M05.00_MH+0.0_a1.98_sed", "5 M$_\\odot$", "right"),
        ("M03.00_MH+0.0_a1.98_sed", "3 M$_\\odot$", "right"),
        ("M01.00_MH+0.0_a1.98_sed", "1 M$_\\odot$", "right"),
        ("M00.80_MH-2.0_a1.98_sed", "0.8 M$_\\odot$ [Fe/H]=$-$2", "left")]
FILTERS = ROOT / "filters/ArgusGaiaSDSS/ArgusGaiaSDSS"
FILTER_NAMES = ["SDSS_u", "SDSS_g", "Gbp", "beta", "SDSS_r", "G", "rho", "SDSS_i", "Grp", "SDSS_z"]

BG = "#070b16"
FG = "#c9d3e6"
GRID = "#1b2438"
W, H, DPI = 1280, 960, 100
FPS, SECONDS, FADE, HOLD = 30, 16, 0.6, 1.2   # HOLD: seconds on the last state before fading
LOG_AGE0 = 4.0
# Star halo: (marker area multiple, opacity) from the outside in.
GLOW = ((9.0, 0.025), (5.5, 0.04), (3.4, 0.07), (2.2, 0.12), (1.5, 0.28), (1.0, 1.0))


# --- true colour from an SED (CIE 1931 2-degree, Wyman, Sloan & Shirley 2013 fit) ---

def _g(x, mu, s1, s2):
    return np.exp(-0.5 * ((x - mu) / np.where(x < mu, s1, s2)) ** 2)


def cie_xyz(nm):
    x = 1.056 * _g(nm, 599.8, 37.9, 31.0) + 0.362 * _g(nm, 442.0, 16.0, 26.7) - 0.065 * _g(nm, 501.1, 20.4, 26.2)
    y = 0.821 * _g(nm, 568.8, 46.9, 40.5) + 0.286 * _g(nm, 530.9, 16.3, 31.1)
    z = 1.217 * _g(nm, 437.0, 11.8, 36.0) + 0.681 * _g(nm, 459.0, 26.0, 13.8)
    return np.stack([x, y, z])


SRGB = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])


def sed_rgb(wl_aa, flam):
    """sRGB (0-1) of an SED, brightness normalised so the brightest channel is 1."""
    nm = wl_aa / 10.0
    sel = (nm > 360) & (nm < 830)
    xyz = np.trapezoid(cie_xyz(nm[sel]) * flam[sel], nm[sel], axis=1)
    rgb = np.clip(SRGB @ xyz, 0, None)
    rgb = rgb / rgb.max()
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)


def band_colour(pivot_aa):
    """Display colour for a passband: its own hue in the visible, violet/deep red outside."""
    nm = np.clip(pivot_aa / 10.0, 400, 720)
    hue = np.interp(nm, [400, 450, 500, 550, 600, 650, 720], [0.78, 0.66, 0.50, 0.30, 0.12, 0.02, 0.97])
    return matplotlib.colors.hsv_to_rgb([hue, 0.55, 0.95])


# --- data ---

class Track:
    def __init__(self, run, label, side):
        with h5py.File(ROOT / "runs" / run / "seds.h5") as f:
            m = {k: f["model"][k][:] for k in ("star_age", "log_R", "inside_atm_grid", "fbol_sed_over_L")}
            good = m["inside_atm_grid"].astype(bool) & (np.abs(m["fbol_sed_over_L"] - 1) < 0.1) & (m["star_age"] > 10 ** (LOG_AGE0 - 0.5))
            idx = np.flatnonzero(good)
            self.wl = f["wavelength"][:]
            self.flux = f["flux"][idx, :]
            self.G = f["mag/G"][idx]
            self.bprp = f["mag/Gbp"][idx] - f["mag/Grp"][idx]
        self.label, self.side = label, side
        self.log_age = np.log10(m["star_age"][idx])
        self.log_R = m["log_R"][idx]
        self.end = self.log_age[-1]

    def at(self, la):
        """Interpolated state at log age `la` (clamped to the track)."""
        la = np.clip(la, self.log_age[0], self.end)
        i = int(np.clip(np.searchsorted(self.log_age, la) - 1, 0, len(self.log_age) - 2))
        t = (la - self.log_age[i]) / max(self.log_age[i + 1] - self.log_age[i], 1e-12)
        lerp = lambda a: a[i] + t * (a[i + 1] - a[i])
        flux = np.exp(np.log(np.maximum(self.flux[i], 1e-300)) * (1 - t) + np.log(np.maximum(self.flux[i + 1], 1e-300)) * t)
        return lerp(self.bprp), lerp(self.G), lerp(self.log_R), flux


def load_filters():
    out = []
    for name in FILTER_NAMES:
        d = np.genfromtxt(FILTERS / f"{name}.dat", delimiter=",", names=True)
        wl, tr = d["Wavelength"], d["Transmission"]
        tr = tr / tr.max()
        pivot = np.sqrt(np.trapezoid(tr * wl, wl) / np.trapezoid(tr / wl, wl))
        out.append((name, wl, tr, pivot))
    return out


def activity_clock(tracks, n_frames):
    """Log age for each frame, giving more frames where the CMD is changing."""
    grid = np.linspace(LOG_AGE0, max(t.end for t in tracks) + 0.02, 4000)
    speed = np.zeros_like(grid)
    for t in tracks:
        alive = grid <= t.end
        x = np.interp(grid, t.log_age, t.bprp)
        y = np.interp(grid, t.log_age, t.G)
        v = np.hypot(np.gradient(x * 3.0, grid), np.gradient(y, grid))
        # The fastest star sets the pace, so one giant climbing gets as much
        # time as five stars contracting together.
        speed = np.maximum(speed, np.where(alive, v, 0.0))
    weight = speed ** 0.6 + 0.35 * np.median(speed ** 0.6)
    cum = np.concatenate([[0], np.cumsum(0.5 * (weight[1:] + weight[:-1]) * np.diff(grid))])
    return np.interp(np.linspace(0, cum[-1], n_frames), cum, grid)


def age_label(la):
    a = 10 ** la
    if a < 1e6:
        return f"{a / 1e3:,.0f} kyr"
    if a < 1e9:
        return f"{a / 1e6:,.1f} Myr" if a < 1e8 else f"{a / 1e6:,.0f} Myr"
    return f"{a / 1e9:.2f} Gyr"


# --- figure ---

def build(tracks, filters):
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": FG, "axes.labelcolor": FG,
                         "xtick.color": FG, "ytick.color": FG, "axes.edgecolor": GRID})
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
    cmd = fig.add_axes([0.105, 0.395, 0.865, 0.515], facecolor=BG)
    sed = fig.add_axes([0.105, 0.085, 0.865, 0.235], facecolor=BG)
    band = sed.twinx()

    x_all = np.concatenate([t.bprp for t in tracks]); y_all = np.concatenate([t.G for t in tracks])
    cmd.set_xlim(x_all.min() - 0.25, x_all.max() + 0.25)
    cmd.set_ylim(y_all.max() + 0.8, y_all.min() - 0.9)
    cmd.set_xlabel(r"$G_{\rm BP}-G_{\rm RP}$", fontsize=26, labelpad=4)
    cmd.set_ylabel(r"$M_G$", fontsize=26, labelpad=6)
    for ax in (cmd, sed):
        ax.tick_params(labelsize=19, length=5, width=1.2)
        ax.grid(color=GRID, lw=1.0)
        for s in ax.spines.values():
            s.set_color(GRID)
    for t in tracks:
        cmd.plot(t.bprp, t.G, color="#4a5d85", lw=1.8, alpha=0.7, zorder=1, solid_capstyle="round")

    sed.set_xscale("log"); sed.set_yscale("log")
    sed.set_xlim(2000, 25000)
    sed.set_xticks([3000, 5000, 10000, 20000]); sed.set_xticklabels(["0.3", "0.5", "1", "2 µm"])
    sed.minorticks_off()
    lam = tracks[0].wl
    sel = (lam > 1800) & (lam < 27000)
    peaks = [np.nanmax(t.flux[:, sel] * lam[sel]) for t in tracks]
    floors = [np.nanpercentile(np.nanmax(t.flux[:, sel] * lam[sel], axis=1), 2) for t in tracks]
    sed.set_ylim(min(floors) / 80, max(peaks) * 3)
    sed.set_yticks([]); sed.set_ylabel(r"$\lambda F_\lambda$", fontsize=24, labelpad=8)
    # The passbands sit on a twin axis underneath the SEDs; its background must
    # be dark too, since the SED axis above it is transparent.
    band.set_ylim(0, 4.5); band.set_yticks([]); band.set_zorder(0); band.set_facecolor(BG)
    sed.set_zorder(1); sed.patch.set_visible(False)
    for s in band.spines.values():
        s.set_color(GRID)
    for name, wl, tr, pivot in filters:
        c = band_colour(pivot)
        band.fill_between(wl, 0, tr, color=c, alpha=0.16, lw=0)
        band.plot(wl, tr, color=c, alpha=0.5, lw=1.1)

    fig.text(0.105, 0.945, "MESA Custom Colors", fontsize=34, weight="bold", color="#eef3ff", va="center")
    age_txt = fig.text(0.97, 0.945, "", fontsize=28, color="#eef3ff", ha="right", va="center", family="DejaVu Sans Mono")

    art = []
    for t in tracks:
        trail = LineCollection([], linewidths=3.2, zorder=3, capstyle="round")
        cmd.add_collection(trail)
        glow = [cmd.scatter([], [], s=1, lw=0, zorder=4 + k) for k in range(len(GLOW))]
        lbl = cmd.text(0, 0, t.label, fontsize=18, zorder=8, va="center", ha="left" if t.side == "right" else "right")
        sed_line, = sed.plot([], [], lw=2.6, zorder=3)
        dots = sed.scatter([], [], s=34, zorder=4, edgecolors=BG, linewidths=0.8)
        art.append(dict(trail=trail, glow=glow, lbl=lbl, sed=sed_line, dots=dots, hist=[]))
    return fig, cmd, sed, age_txt, art


def draw_frame(la, alpha_all, tracks, filters, cmd, age_txt, art, trail_len=26):
    age_txt.set_text(age_label(la))
    age_txt.set_alpha(alpha_all)
    lam = tracks[0].wl
    sel = (lam > 1800) & (lam < 27000)
    for t, a in zip(tracks, art):
        x, y, logR, flux = t.at(la)
        dead = la > t.end
        fade = alpha_all * (np.clip(1 - (la - t.end) / 0.08, 0, 1) if dead else 1.0)
        rgb = sed_rgb(lam, flux)
        a["hist"].append((x, y))
        a["hist"] = a["hist"][-trail_len:]
        pts = np.array(a["hist"])
        if len(pts) > 1:
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            al = np.linspace(0.05, 0.9, len(segs)) * fade
            a["trail"].set_segments(segs)
            a["trail"].set_color([(*rgb, v) for v in al])
        size = 90 + 900 * np.clip((logR + 0.15) / 2.6, 0, 1) ** 1.3
        for k, (mult, alp) in enumerate(GLOW):
            a["glow"][k].set_offsets([[x, y]])
            a["glow"][k].set_sizes([size * mult])
            a["glow"][k].set_color([(*rgb, alp * fade)])
        offset = 0.09 + 0.012 * np.sqrt(size)
        a["lbl"].set_position((x + offset if t.side == "right" else x - offset, y))
        a["lbl"].set_color((*rgb, 0.95 * fade))
        lf = flux[sel] * lam[sel]
        a["sed"].set_data(lam[sel], lf)
        a["sed"].set_color((*rgb, 0.95 * fade))
        bx, by = [], []
        for name, wl, tr, pivot in filters:
            f_b = np.interp(wl, lam, flux)
            bx.append(pivot); by.append(pivot * np.trapezoid(f_b * tr * wl, wl) / np.trapezoid(tr * wl, wl))
        a["dots"].set_offsets(np.column_stack([bx, by]))
        a["dots"].set_color([(*rgb, fade)])
        a["dots"].set_edgecolor([(*matplotlib.colors.to_rgb(BG), fade)])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="custom_colors_card.mp4")
    ap.add_argument("--frame", type=float, help="render a single PNG at this fraction of the loop")
    ap.add_argument("--crf", type=int, default=25)
    ap.add_argument("--width", type=int, default=1024, help="output width; rendered at 1280x960 and scaled")
    args = ap.parse_args()

    tracks = [Track(*run) for run in RUNS]
    filters = load_filters()
    n = FPS * SECONDS
    n_hold = int(HOLD * FPS)
    clock = np.concatenate([activity_clock(tracks, n - n_hold), np.full(n_hold, max(t.end for t in tracks))])
    fade_n = int(FADE * FPS)
    alpha = np.ones(n); alpha[:fade_n] = np.linspace(0, 1, fade_n); alpha[-fade_n:] = np.linspace(1, 0, fade_n)
    fig, cmd, sed, age_txt, art = build(tracks, filters)

    if args.frame is not None:
        i = int(args.frame * (n - 1))
        for j in range(max(0, i - 26), i + 1):
            draw_frame(clock[j], alpha[i] if j == i else 1.0, tracks, filters, cmd, age_txt, art)
        fig.savefig(args.out, facecolor=BG)
        return

    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-an", "-vf", f"scale={args.width}:-2:flags=lanczos",
                            "-c:v", "libx264", "-preset", "veryslow",
                            "-crf", str(args.crf), "-pix_fmt", "yuv420p", "-profile:v", "high",
                            "-movflags", "+faststart", args.out], stdin=subprocess.PIPE)
    for i in range(n):
        draw_frame(clock[i], alpha[i], tracks, filters, cmd, age_txt, art)
        fig.canvas.draw()
        enc.stdin.write(fig.canvas.buffer_rgba().tobytes())
    enc.stdin.close(); enc.wait()


if __name__ == "__main__":
    main()
