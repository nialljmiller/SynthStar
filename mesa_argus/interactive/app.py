"""The interactive Argus night (Jupyter): the field movie, live, with one star you edit.

    %matplotlib widget
    from mesa_argus.interactive.app import App
    App().show()

The window is the field movie (night_movie.MovieFrame, the movie's own drawing): CMD, light curves, hdps's exposure and difference image with
its alerts, SEDs. Below it: the field (a precomputed hdps night), the band, the star's type and its sliders, the
airmass offset, and the frame and Play controls. Click the exposure or the difference image to move the star.

Everything but the edited star is a real hdps run (field packs, mesa_argus.interactive.pack). The edited star is
added to hdps's images and measured with hdps's own code (night.Night); the stand-ins are listed under the controls.
"""
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from mesa_argus.interactive import night as N
from mesa_argus.interactive import pack, templates
from mesa_argus.interactive.star import Settings
from mesa_argus.interactive.worker import Workers, quiet_argus_sim

quiet_argus_sim()

FIELD_NAMES = {"m3": "M3 (globular cluster)", "m67": "M67 (old open cluster)", "m45": "M45, the Pleiades (young open cluster)",
               "m45stamp": "M45, the Pleiades: ArgusSim's 28 px PSF stamp, no halo (for comparison)",
               "m67stamp": "M67: ArgusSim's 28 px PSF stamp, no halo (for comparison)",
               "m3stamp": "M3: ArgusSim's 28 px PSF stamp, no halo (for comparison)",
               "f294test": "f294, test field (injected stars)"}  # the field dropdown's order
# Star E's starting place per field (arcsec from the block centre): hdps's detector takes the mean of a 13-22" annulus
# as a source's background, so a bright star there makes E faint, and a bright star's difference residual under E
# makes the real/bogus model reject it. 2026-10-27, 16.5 mag RRab / 15 mag supernova accepted under bundle_v1 (fa10):
# M45 detected 11/11 (-0.15 mag) / 11 of 11; M67 8/8 (-0.11) / 6 of 8 (at (30, -20) 0/8 detected, 4" from a 12.5 mag
# star; at (36, -26) -0.33 mag, 0 of 8 on that star's residual; (60, 80) as good but under the panel's label); M3 9/9
# (+0.48, crowding) / 1 of 9 (the least crowded spot within 100" of the core; at (30, -20) 0/10 detected; no spot
# tried in M3 above 1 of 9)
E_HOME = {"m3": (98.0, -86.0), "m67": (38.0, -66.0), "m45": (30.0, -20.0)}


def home(field):
    """Star E's starting place in a field (its pack variants share it)."""
    return list(next((xy for k, xy in E_HOME.items() if field.startswith(k)), (30.0, -20.0)))
BANDS = (("β", "beta"), ("ρ", "rho"))
WORDS = {  # each type's slider names: magnitude, amplitude, period/timescale
    "periodic": ("Mean mag", "Amplitude", "Period (d)"),
    "dwarf nova": ("Quiescent mag", "Outburst (mag)", "Timescale (×)"),
    "supernova (Ia)": ("Peak mag", "Amplitude", "Timescale (×)"),
    "microlensing": ("Baseline mag", "Peak gain (mag)", "t_E (d)"),
}
def stand_ins(masked, true_calibration=False, image=None):
    """The note under the controls. masked: the night's hdps run had its bright-star detection mask on;
    true_calibration: its frames' WCS and zero point were the simulation's (TrueCalibrationTask), not hdps's solve;
    image: the run's [image] settings (how its stars were drawn: the PSF's outer wing, a wider box, the halo)."""
    if true_calibration:
        night = ("Everything except star E is one real hdps night (reduction, source detection, reprojection, SFFT "
                 "difference, detection, alert features), with each frame's WCS and zero point taken from the "
                 "simulation instead of hdps's astrometric solve (the true WCS; the throughput's AB zero point with "
                 "the share of a star's light in hdps's aperture).")
        zero_point = "zero point from hdps's catalogue of the frame, calibrated with the simulation's"
    else:
        night = ("Everything except star E is one real hdps night (reduction, WCS solve, reprojection, SFFT "
                 "difference, detection, alert features).")
        zero_point = "zero point from hdps's own catalogue of the frame"
    image = image or {}
    if image.get("psf_wing"):
        psf = "ArgusSim's PSF continued by a power-law outer wing fitted to it (as the night's stars)"
    elif int(image.get("psf_box", 28)) > 28:
        psf = (f"ArgusSim's PSF in a {int(image['psf_box'])} px box" + (", with the halo measured on sky"
                                                                         if image.get("psf_halo") else "")
               + " (as the night's stars)")
    else:
        psf = "ArgusSim's PSF"
    return (night + f" Star E is added to hdps's images afterwards: {psf} at each frame's seeing on hdps's tile "
            "grid, with its photon noise, up to the sensor's full well; in the difference image as (now − reference) "
            "× PSF (SFFT's match taken as perfect for it, so E has none of the small residuals hdps leaves at the "
            f"field's bright stars). It is measured with hdps's source detector ({zero_point}) "
            "and alerted by hdps's alert features" + (", hdps's bright-star mask" if masked else "") + " and the "
            "real/bogus model. Checked against injected stars hdps measured itself (on a night hdps processed with "
            "its bright-star mask on and its own WCS solve): the same photometry within 0.01 mag and the same alerts "
            "(97% and 100% of frames)." + (" This night was processed with hdps's bright-star mask on (not Argus's "
                                            "production setting): as in hdps then, a source brighter than about β 15 "
                                            "does not alert." if masked else "")
            + " Airmass +: every image dimmed by the extra extinction, blurred to the worse seeing (∝ X^0.6) and "
            "given the extra sky noise; the field's own alerts and light curves stay as observed.")


NOTE_STYLE = "font-size:11px;color:#555;max-width:1150px"


def available():
    """{pack name: [bands]} of the packs built (data/interactive/packs)."""
    out = {}
    for path in sorted(pack.PACKS.glob("*_*.joblib")):
        name, band = path.stem.rsplit("_", 1)
        out.setdefault(name, []).append(band)
    order = list(FIELD_NAMES)
    return dict(sorted(out.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else len(order)))


def _start(coro):
    """Run a coroutine in the kernel's event loop (Jupyter's), or to its end where there is none (a plain script)."""
    try:
        return asyncio.get_running_loop().create_task(coro)
    except RuntimeError:
        return asyncio.run(coro)


def words(kind):
    timing = templates.TYPES[kind]["timing"]
    return WORDS["periodic"] if timing == "periodic" else WORDS[kind]


class App:
    """dpi: the live figure's (screen pixels per inch); render_dpi: the played frames' (sharper on HiDPI screens);
    workers: worker processes (0: none, everything in this process)."""

    def __init__(self, dpi=72, render_dpi=108, seed=0, workers=None):
        import ipywidgets as W
        import matplotlib
        import matplotlib.pyplot as plt

        backend = matplotlib.get_backend()  # the notebook's (%matplotlib widget)
        import night_movie as nm  # the engine selects Agg when imported

        if matplotlib.get_backend() != backend:
            plt.switch_backend(backend)

        self.W, self.seed, self.render_dpi = W, seed, render_dpi
        self.workers = None if workers == 0 else Workers(workers)
        packs = available()
        if not packs:
            raise SystemExit(f"no field packs in {pack.PACKS}: python -m mesa_argus.interactive.pack <observation> --pack <name>")
        self.packs = packs
        nm.apply_style()
        with plt.ioff():
            self.fig = plt.figure(figsize=(16, 10), dpi=dpi)
        canvas = self.fig.canvas
        for attr, value in (("header_visible", False), ("toolbar_visible", False), ("footer_visible", False),
                            ("resizable", False)):
            if hasattr(canvas, attr):
                setattr(canvas, attr, value)
        canvas.mpl_connect("button_press_event", self._click)
        size = W.Layout(width=f"{16 * dpi}px", height=f"{10 * dpi}px")
        self.screen = W.Image(format="jpeg", layout=size)  # the played frames
        self.screen.layout.display = "none"

        layout = W.Layout(width="420px")
        style = {"description_width": "120px"}
        first = next(iter(packs))
        self.field = W.Dropdown(options=[(FIELD_NAMES.get(k, k), k) for k in packs], value=first,
                                description="Field", layout=layout, style=style)
        self.band = W.ToggleButtons(options=[b for b in BANDS if b[1] in packs[first]], description="Band",
                                    style={"description_width": "120px", "button_width": "60px"})
        self.kind = W.Dropdown(options=list(templates.TYPES), value="RR Lyrae (RRab)", description="Star type",
                               layout=layout, style=style)
        self.mag = W.FloatSlider(value=16.5, min=11.0, max=20.0, step=0.1, description="", layout=layout, style=style,
                                 continuous_update=False, readout_format=".1f")
        self.amp = W.FloatSlider(value=0.8, min=0.05, max=3.0, step=0.05, description="", layout=layout, style=style,
                                 continuous_update=False, readout_format=".2f")
        self.time = W.FloatLogSlider(value=0.55, base=10, min=-1, max=0, step=0.01, description="", layout=layout,
                                     style=style, continuous_update=False, readout_format=".3f")
        self.airmass = W.FloatSlider(value=0.0, min=0.0, max=1.0, step=0.1, description="Airmass +", layout=layout,
                                     style=style, continuous_update=False, readout_format=".1f")
        self.zoom = W.FloatLogSlider(value=1.0, base=2, min=0, max=3, step=0.25, description="Zoom (×)", layout=layout,
                                     style=style, continuous_update=False, readout_format=".2g")  # 1-8x, about star E
        self.frame = W.IntSlider(value=0, min=0, max=1, description="Frame", layout=W.Layout(width="640px"),
                                 style=style, continuous_update=False)
        self.play = W.ToggleButton(value=False, description="Play", icon="play", layout=W.Layout(width="100px"))
        self.status = W.HTML()
        self.notes = W.HTML(f"<div style='{NOTE_STYLE}'>{stand_ins(False)}</div>")
        self.where = home(first)
        self.night = None
        self._busy = False
        self._quiet = False  # set while the app changes its own controls: their handlers do nothing
        self._task = None
        self._frames = {}  # rendered frames (n -> JPEG) for the current night and star
        self._renders = []  # their futures
        self._generation = 0  # changes made; a measurement for an older one is dropped
        self._measuring = ThreadPoolExecutor(max_workers=1)  # measurements run here, off the kernel's own thread

        self.field.observe(self._field_changed, "value")
        self.band.observe(self._quietly(self._load), "value")
        self.kind.observe(self._kind_changed, "value")
        for w in (self.mag, self.amp, self.time, self.airmass):
            w.observe(self._quietly(self._recompute), "value")
        self.zoom.observe(self._quietly(self._zoomed), "value")
        self.frame.observe(self._quietly(self._draw), "value")
        self.play.observe(self._play_changed, "value")
        self._kind_changed(None, recompute=False)
        self._load()

    # ---------------------------------------------------------------- layout
    def show(self):
        from IPython.display import display

        W = self.W
        controls = W.VBox([
            W.HBox([self.field, self.band]),
            W.HBox([self.kind, self.mag]),
            W.HBox([self.amp, self.time]),
            W.HBox([self.airmass, self.zoom]),
            W.HBox([self.play, self.frame]),
            self.status, self.notes,
        ])
        display(W.VBox([W.Box([self.fig.canvas, self.screen]), controls]))

    # ---------------------------------------------------------------- events
    def _quietly(self, action):
        """A control's handler: the action, unless the app itself is setting controls."""
        def handler(_):
            if not self._quiet:
                action()
        return handler

    def _set(self, *steps):
        """Set controls, in order, without their handlers running: (widget, trait, value) steps."""
        self._quiet = True
        try:
            for widget, trait, value in steps:
                setattr(widget, trait, value)
        finally:
            self._quiet = False

    def _field_changed(self, _):
        bands = self.packs[self.field.value]
        current = self.band.value
        self._set((self.band, "options", [b for b in BANDS if b[1] in bands]),
                  (self.band, "value", current if current in bands else bands[0]))
        self.where = home(self.field.value)  # a place in one field means nothing in another
        self._load()

    def _kind_changed(self, _, recompute=True):
        kind = self.kind.value
        spec = templates.TYPES[kind]
        lo, hi, default = spec["period"] if spec["timing"] == "periodic" else spec["timescale"]
        m, a, t = words(kind)
        self._set((self.time, "min", -3.0), (self.time, "max", np.log10(hi)), (self.time, "min", np.log10(lo)),
                  (self.time, "value", default))  # widened first: max may never fall below min
        band = dict((v, k) for k, v in BANDS).get(self.band.value, "β")
        self.mag.description = f"{m} ({band})"
        self.amp.description = a
        self.time.description = t
        self.amp.disabled = kind == "supernova (Ia)"
        if recompute:
            self._recompute()

    def _click(self, event):
        ax = event.inaxes
        if ax is None or event.xdata is None:
            return
        if not ax.get_title(loc="left").startswith(("Exposure", "Difference")):
            return
        self.where = [float(event.xdata), float(event.ydata)]
        self._recompute()

    def _zoomed(self):
        """The image panels magnified about star E: nothing measured again; frames rendered for Play are dropped."""
        self._stop_playing()
        self._draw()

    def view(self):
        """(zoom, x, y): the image panels' magnification and the point (arcsec, star E) they are magnified about."""
        return float(self.zoom.value), self.where[0], self.where[1]

    def _play_changed(self, change):
        self.play.icon = "pause" if change["new"] else "play"
        self.play.description = "Pause" if change["new"] else "Play"
        if change["new"] and (self.night is None or self.night.d is None):  # nothing measured yet
            self._set((self.play, "value", False))
            return
        if change["new"]:
            if self.frame.value >= self.frame.max:
                self._set((self.frame, "value", 0))
            self._task = _start(self._run())

    async def _run(self):
        """The night from the current frame at the movie's frame rate: frames drawn by the workers (rendered once
        per star setting, kept for replays), shown as they arrive; without workers, drawn here one by one."""
        try:
            if self.workers is None or not self.workers.ready():
                while self.play.value and self.frame.value < self.frame.max:
                    self.frame.value += 1  # draws it
                    await asyncio.sleep(0.02)  # let the browser take the frame and the controls be read
                return
            start = int(self.frame.value)
            missing = [n for n in range(start, self.frame.max + 1) if n not in self._frames]
            if missing:
                r = self.night.result
                try:
                    self._renders = self.workers.render(self.night.state, r["meas"], r["cands"], r["scored"],
                                                        missing, self.render_dpi, self.view())
                except Exception:  # the pool broke: play by drawing here, and restart the workers meanwhile
                    self.workers.use(*self.workers.key)
                    while self.play.value and self.frame.value < self.frame.max:
                        self.frame.value += 1
                        await asyncio.sleep(0.02)
                    return
            fps = float(self.night.args.fps)
            futures, frames = list(self._renders), self._frames
            for n in range(start, self.frame.max + 1):
                while n not in frames:  # not drawn yet: take the next finished batch
                    if not futures:
                        return
                    for m, jpeg in await asyncio.wrap_future(futures.pop(0)):
                        frames[m] = jpeg
                if not self.play.value or frames is not self._frames:  # paused, or the star changed meanwhile
                    return
                shown = time.time()
                self.screen.value = self._frames[n]
                self._playing(True)  # (from the first frame drawn: the live figure stays until then)
                self._set((self.frame, "value", n))
                await asyncio.sleep(max(0.0, 1.0 / fps - (time.time() - shown)))
        except Exception as error:  # a worker failed mid-way: say so, and stop
            self.status.value += f"<br><b>playback stopped:</b> {type(error).__name__}: {error}"
        finally:
            self.play.value = False
            self._playing(False)
            self._draw()

    def _playing(self, on):
        """Show the played frames (on) or the live figure (off) in the same place."""
        self.screen.layout.display = None if on else "none"
        layout = getattr(self.fig.canvas, "layout", None)  # the live figure is a widget under %matplotlib widget
        if layout is not None:
            layout.display = "none" if on else None

    # ---------------------------------------------------------------- work
    def _load(self):
        band = self.band.value
        self.status.value = f"<b>loading {FIELD_NAMES.get(self.field.value, self.field.value)} ({band}) …</b>"
        t = time.time()
        self.night = N.Night(self.field.value, band)
        p = self.night.p
        note = stand_ins(self.night.mask is not None, "TrueCalibrationTask" in p.get("chain", []), p.get("image"))
        self.notes.value = f"<div style='{NOTE_STYLE}'>{note}</div>"
        if self.workers is not None:
            self.workers.use(self.field.value, band, self.night.model, self.night.cut)
        last = len(self.night.frames) - 1
        self._set((self.frame, "value", min(self.frame.value, last)), (self.frame, "max", last))
        self._kind_changed(None, recompute=False)
        self._loaded = time.time() - t
        self._recompute()

    def settings(self):
        timing = templates.TYPES[self.kind.value]["timing"]
        return Settings(kind=self.kind.value, band=self.band.value, mag=float(self.mag.value),
                        amplitude=float(self.amp.value), period=float(self.time.value) if timing == "periodic" else 1.0,
                        timescale=float(self.time.value) if timing != "periodic" else 1.0,
                        dx=self.where[0], dy=self.where[1])

    def _recompute(self):
        """A change: the night measured again without holding up the controls (in the worker processes, or threads
        while they load), then drawn; the newest change wins."""
        if self.night is None:
            return
        self._stop_playing()
        self._generation += 1
        self.status.value = f"<b>Star E</b> at ({self.where[0]:+.0f}″, {self.where[1]:+.0f}″) · measuring …"
        _start(self._measure(self._generation))

    async def _measure(self, generation):
        t = time.time()
        night, s, d_airmass = self.night, self.settings(), float(self.airmass.value)
        workers = self.workers if self.workers is not None and self.workers.ready() else None

        def measure():  # (skipped if a newer change came while it waited)
            return None if generation != self._generation else night.measure(s, d_airmass, self.seed, workers)

        measured = await asyncio.get_running_loop().run_in_executor(self._measuring, measure)
        if measured is None or generation != self._generation or night is not self.night:
            return
        self._stop_playing()  # (frames played meanwhile were of the previous star)
        r = night.assemble(*measured)
        took = time.time() - t
        self._report(r, took, workers is not None)
        self._draw()

    def _stop_playing(self):
        """Stop the playback and drop the frames rendered for it: the star is changing."""
        self._set((self.play, "value", False))
        for f in self._renders:
            f.cancel()
        self._frames, self._renders = {}, []

    def _report(self, r, took, with_workers):
        found = np.isfinite(r["mag"])
        resid = r["mag"][found] - r["true"][found]
        band = dict((v, k) for k, v in BANDS)[self.band.value]
        self.status.value = (
            f"<b>Star E</b> at ({self.where[0]:+.0f}″, {self.where[1]:+.0f}″) · template {r['template']} · "
            f"detected by hdps in {int(found.sum())}/{len(found)} frames"
            + (f" (measured − true {np.median(resid):+.2f} mag, scatter {1.4826 * np.median(np.abs(resid - np.median(resid))):.3f})"
               if found.any() else "")
            + (f" · saturated in {int(r['saturated'].sum())} frames" if r["saturated"].any() else "")
            + f" · alerted in {int(r['alerted'].sum())} frames · field alerts per frame {np.mean(r['n_field_alerts']):.1f} "
            f"(real/bogus model {N.model_label(self.night.model) if self.night.model else 'none'}, cut {self.night.cut}) · "
            f"{band} · recomputed in {took:.1f} s" + ("" if with_workers else " (worker processes loading)"))

    def _draw(self):
        if self.night is None or self.night.d is None or self._busy:
            return
        self._busy = True
        try:
            self.night.draw(self.fig, int(self.frame.value), self.view())  # into the canvas's own image
            self.fig.canvas.blit(self.fig.bbox)  # shown (an interactive canvas sends it to the browser)
        finally:
            self._busy = False
