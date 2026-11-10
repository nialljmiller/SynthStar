"""The pipeline movie: how each exposure goes through the chain. Per frame: where the readout window falls on its
telescope's sensor (ArgusSim's geometry), the raw camera frame with hdps's detections, hdps's solved WCS against the
true star positions over the night, and the detections per frame. Drawn from the engine's per-frame records
(night_movie.py), each raw frame as its preview (block means of preview_bin pixels), so it can be drawn after the run.
"""
import numpy as np


def render(records, output, field, band, fps, dpi, ut, airmass, seeing):
    import matplotlib.pyplot as plt
    from argus_sim import c
    from matplotlib.animation import FFMpegWriter
    from matplotlib.colors import AsinhNorm
    from matplotlib.patches import Rectangle

    from argus_image import camera
    from argus_image.argus import COLORS, NAVY, RED, add_logo, apply_style

    t = c.telescope
    apply_style()
    fig = plt.figure(figsize=(16, 9))
    writer = FFMpegWriter(fps=fps, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p",
                                               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])
    history = dict(t=[], err=[], ndet=[])
    print(f"pipeline movie: {len(records)} frames -> {output}", flush=True)
    with writer.saving(fig, str(output), dpi=dpi):
        for n, r in enumerate(records):
            j, win = r["j"], r["window"]
            raw = np.asarray(r["preview"], np.float32)  # electrons, block means of preview_bin x preview_bin pixels
            cat = r["catalogue"]
            err = np.nan
            if r["wcs"] is not None and len(r["truth"][0]):
                sx, sy, ra, dec = r["truth"]
                r2, d2 = r["wcs"].all_pix2world(sx, sy, 0)
                err = float(np.median(np.hypot((r2 - ra) * np.cos(np.radians(dec)), d2 - dec)) * 3600)
            history["t"].append(ut[j])
            history["err"].append(err)
            history["ndet"].append(len(cat) if cat is not None else 0)

            fig.clear()
            grid = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.6, 1.25], height_ratios=[1, 1], wspace=0.28,
                                    hspace=0.38, left=0.05, right=0.97, bottom=0.08, top=0.86)
            ax_s = fig.add_subplot(grid[0, 0])
            ax_s.add_patch(Rectangle((0, 0), t.n_pixels_x, t.n_pixels_y, fill=False, ec=NAVY, lw=1.2))
            ax_s.add_patch(Rectangle(win[:2], win[2], win[3], fc=COLORS[band], ec=COLORS[band], alpha=0.8))
            ax_s.plot(*r["field_xy"], "+", color=NAVY, ms=10)
            ax_s.plot((t.n_pixels_x - 1) / 2, (t.n_pixels_y - 1) / 2, "x", color="0.6", ms=6)
            ax_s.set(xlim=(-300, t.n_pixels_x + 300), ylim=(-300, t.n_pixels_y + 300), aspect="equal",
                     xlabel="sensor x (px)", ylabel="sensor y (px)")
            ax_s.set_title(f"Telescope {r['tel_id']}: where the window falls", loc="left", weight="bold", fontsize=11)
            ax_s.text(0.02, 0.97, f"field angle {r['field_angle']:.2f}°", transform=ax_s.transAxes, va="top", fontsize=9)

            ax_i = fig.add_subplot(grid[:, 1])
            norm = AsinhNorm(linear_width=40, vmin=np.percentile(raw, 5), vmax=np.percentile(raw, 99.8))
            ax_i.imshow(raw, origin="lower", cmap="gray", norm=norm, interpolation="nearest",
                        extent=(-0.5, raw.shape[1] * r["preview_bin"] - 0.5, -0.5, raw.shape[0] * r["preview_bin"] - 0.5))
            if cat is not None and len(cat):
                ax_i.scatter(cat["xcentroid"], cat["ycentroid"], s=26, facecolors="none", edgecolors=RED, linewidths=0.6)
            status = "hdps: " + (f"{history['ndet'][-1]} detections (red); WCS solved; "
                                 f"zero point {r['zp']['zp']:.2f}" if r["zp"] else
                                 f"{history['ndet'][-1]} detections (red); WCS solved" if cat is not None else
                                 f"not processed ({(r['failed'] or 'no WCS')[:60]})")
            ax_i.set_title(f"Raw frame, {win[2]} x {win[3]} px readout window ({band})", loc="left", weight="bold",
                           fontsize=11)
            ax_i.set(xlabel="window x (px)", ylabel="window y (px)")
            ax_i.text(0.02, 0.98, status, transform=ax_i.transAxes, va="top", fontsize=9, color="white",
                      bbox=dict(facecolor=NAVY, alpha=0.75, edgecolor="none", pad=3))

            th, eh, nh = np.array(history["t"]), np.array(history["err"]), np.array(history["ndet"])
            ax_w = fig.add_subplot(grid[0, 2])
            ax_w.plot(th, eh, "o", ms=3, color=COLORS[band])
            ax_w.set(xlabel="UT (hours)", ylabel="median |WCS - truth| (\")",
                     ylim=(0, max(0.5, np.nanmax(eh) * 1.2 if np.isfinite(eh).any() else 0.5)))
            ax_w.set_title("hdps WCS solve against truth", loc="left", weight="bold", fontsize=11)
            ax_n = fig.add_subplot(grid[1, 2])
            ax_n.plot(th, nh, "o", ms=3, color=COLORS[band])
            ax_n.set(xlabel="UT (hours)", ylabel="detections per frame")
            ax_n.set_title("hdps source detection", loc="left", weight="bold", fontsize=11)

            ax_t = fig.add_subplot(grid[1, 0])
            ax_t.axis("off")
            ax_t.text(0, 1, "\n".join([f"exposure {j + 1} (frame {n + 1} of {len(records)})",
                                       f"telescope {r['tel_id']}, ratchet {r['ratchet']}", f"band {band}",
                                       f"seeing {seeing[j]:.2f}\"  airmass {airmass[j]:.2f}",
                                       f"window origin ({win[0]}, {win[1]})", "",
                                       "ArgusSim: survey, telescope, filter, time,", "tracking angle, seeing, geometry, PSF",
                                       f"toy camera: gain {camera.GAIN:.3f} e-/ADU, no bias",
                                       "hdps: reduction -> detection -> WCS ->", "reprojection -> coadd -> SFFT -> alerts"]),
                      va="top", fontsize=10, family="monospace")
            add_logo(fig, (0.012, 0.9, 0.14, 0.08))
            fig.suptitle(f"Raw frames through hdps — {field} — {band}", x=0.58, fontsize=14)
            writer.grab_frame()
            if (n + 1) % max(1, len(records) // 10) == 0:
                print(f"  {n + 1}/{len(records)}", flush=True)
    plt.close(fig)
