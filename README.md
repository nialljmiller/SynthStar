# MESA_ARGUS

Stellar models from MESA, seen through the **Argus Array**. The repo does two things:

1. **Simulated Argus data through the real pipeline.** Nights (or weeks) of Argus 60 s exposures of any field, every
   star an SED (MESA pulsation models, PHOEBE eclipsing binaries, supernovae, microlensing, dwarf novae, ...), each
   exposure made as the camera would read it out and put through hdps itself: photometry, difference images, alerts
   (Avro packets), light curves, movies, and labelled real/bogus training data.
2. **How much stellar physics Argus photometry can recover**, compared with Gaia and SDSS: MESA tracks with synthetic
   photometry in all three systems, an interactive 3D CMD, and a Fisher-information analysis.

Both use the same MESA install and the same Argus throughputs, from ArgusSim.

## What is simulated, and how

**Argus.** 1,190 telescopes (ArgusSim's layout), each with one filter (599 β, 591 ρ) and a 3.25° × 2.44° field
(11,648 × 8,742 px of 1.007″). They point at different parts of the sky, side by side like tiles: together about
8,000 deg² at once, each spot seen by one telescope in one band (neighbouring fields overlap along their edges,
~14% of a field). Each telescope tracks for a 15-minute ratchet, then resets; as the sky drifts, a spot passes from
telescope to telescope through the night, alternating β and ρ.

**One patch of sky, followed through the night.** An observation is a field (RA, Dec). ArgusSim's survey says which
telescope sees it at each 60 s exposure, in which band, with which tracking angle. Every exposure is rendered on that
telescope's sensor with ArgusSim's geometry and PSF (at the field's position on that sensor), sky and noise.

**Each frame is a cut-down camera readout.** Not the whole 3.3° × 2.4° sensor: a readout window of N × N minipixes
around the field (`field.window`, default 56 ≈ 26′), written as that camera's raw frame (its ID, the window's place
on its sensor). Whole-frame quantities (WCS solve, zero point) are fitted from the window alone.

**The pipeline is hdps.** All frames go through one hdps worker (`GPUWorker`, the production Task chain, on the CPU
via the `hdps_full_cpu` branch), as real frames would: each frame reduced, detected, astrometrically solved and
reprojected onto hdps's fixed sky tiles; each camera's frames stacked per ratchet into coadds (the only combining);
each frame's tiles differenced (SFFT) against a reference coadd (the night before, transients left out) and its
candidates vetted into alerts. Where hdps has no piece, the simplest stand-in: fakeredis for valkey, Avro files for
Kafka, every real/bogus score 1 until a model bundle is given (`tools/train_rb.py`).

| Term | Size | What it is |
|---|---|---|
| sensor pixel / tile pixel | 1.007″ / 0.879″ | a camera pixel / a pixel of hdps's sky grid |
| minipix | 32 tile px = 28.1″ | hdps's basic sky cell |
| hdps tile | 64 × 64 minipix = 30′ | hdps's fixed sky tiling; coadds, references, SFFT and alerts are per tile |
| readout window / raw frame | 56 minipix ≈ 26′ | the part of one telescope's sensor simulated per exposure |
| block | 9 × 9 minipix = 253″ | the movie's image panels only |
| science / difference image | per tile | a frame reprojected onto a tile; minus the matched reference |

The whole flow, step by step and with the code for each step: [docs/PIPELINE.md](docs/PIPELINE.md) (flow charts).

## Setup

Requirements: a working MESA install with `$MESA_DIR` set and its colors data, ArgusSim installed in the same
Python (≥ 3.11), and `pip install -r requirements.txt`. The movies also need `ffmpeg`, and the dust maps
`bash tools/fetch_dust.sh` (once per machine). The pipeline needs hdps (argus-hdps/hdps; without a GPU, branch
`hdps_full_cpu`) with its photometry, astrometry, diffimg and alerts packages installed editable in the same Python,
and ATLAS-Refcat2 at `~/.rico_cache/atlas_refcat2` (~83 GB).

```bash
$EDITOR config.toml               # machine settings: colors-data path, parallelism (usually only the path)
make -C mesa_work                 # build MESA's star executable
python -m synthstar setup        # assemble the filter set (Gaia + Argus from your colors data, SDSS from data/)
python -m synthstar run argus_sim_seds   # the stellar SEDs every movie uses for its constant stars (once)
```

On macOS, wrap long runs in `caffeinate -dims` so sleep does not suspend MESA.

## 1. Night movies: stars, then observations

Two scripts, two kinds of file you write:

| Step | Command | Reads | Writes |
|---|---|---|---|
| simulate stars | `python make_stars.py <star file> [...]` | a **star file**: `[[star]]` and `[[set]]` entries | `stars/<name>/`, one folder per star |
| observe them | `python observe.py <observation file>` | an **observation file**: pointing, night, stars placed, measurement, output | `outputs/pseudo_real/<name>/` or `outputs/fully_synthetic/<name>/`: photometry, alerts, movie, FITS |

A star knows nothing about observing: no dust, no position in an image, no night, no instrument. An observation never
simulates a star: it names stars that are already built and adds everything about observing.

### Quick start

```bash
python make_stars.py examples/stars/f294_field.toml examples/stars/transients.toml   # build stars (seconds)
python observe.py examples/observations/synthetic_transients.toml --set output.stride=30   # quick look: every 30th exposure
python observe.py examples/observations/synthetic_transients.toml                          # the whole night (~30 min per band)
```

The first command builds a background (every Gaia star around RA 294.78, Dec −11.62, as constant stars) and four
transients. The second observes a synthetic field (those stars scattered at random) with a microlensing event and a
dwarf nova on the night of 2026-07-19, and writes `outputs/synthetic_transients/synthetic_transients_night_rho.mp4`
and its photometry.

The examples:

| Star file (`examples/stars/`) | Builds | Time |
|---|---|---|
| `f294_field.toml` | the background set for every example observation | seconds |
| `transients.toml` | a microlensing event, a dwarf nova, a SN Ia | seconds |
| `variables.toml` | a W UMa and a detached eclipsing binary (PHOEBE), an RRab (MESA RSP) | minutes; ~1 h for the RRab |
| `f294.toml` | the real f294 field's set and its two Gaia variables (MESA RSP) | ~1–4 h |

| Observation file (`examples/observations/`) | Needs | What it shows |
|---|---|---|
| `synthetic_transients.toml` | f294_field, transients | a short microlensing event and a dwarf nova rising |
| `synthetic_variables.toml` | f294_field, variables | the three hand-made periodic variables |
| `f294.toml` | f294 | the real field with its RRab and HADS on their real ephemerides |
| `template.toml` | — | every key, with its default and meaning |

### Star files

```toml
[[star]]
name = "wuma_035"            # its folder: stars/wuma_035/
generator = "W UMa"          # python -m synthstar.sources lists every generator and its keys
period = 0.35
q = 0.4
incl = 80
teff1 = 5800
teff2 = 5600
fillout = 0.2
distance = 620

[[set]]
name = "f294_784m11_623_field"
generator = "Gaia field"     # every Gaia star around a pointing, as constant stars
ra = 294.7844
dec = -11.6235
```

Each star is built once, into:

```
stars/<name>/
  star.toml     the definition it was built from (every key)
  meta.json     what the build produced: kind, engine, model parameters; real stars also position, ephemeris,
                sightline E(B-V)
  sed.h5        the model: SEDs against time (transients, before any dust) or one cycle at 10 pc (periodic)
  catalogue/    (sets) the catalogue files
  summary.md    the same, to read
```

Running a star file again skips stars already built. A changed definition under an existing name is refused unless
named with `--rebuild NAME`; `--only NAME` builds just those; `python make_stars.py --list` lists every built star.
`stars/` is not in git: each machine builds its own from the star files.

### Observation files

```toml
[field]
ra = 294.7844
dec = -11.6235
block = 9                    # minipixes across the block shown (9 = 253")

[night]
date = "2026-07-19"          # or "auto": the night the placed real stars change most

[stars]
background = "f294_784m11_623_field"   # a built star set
layout = "random"            # "real": its own positions; "random": scattered around the pointing (a synthetic field)

[[place]]                    # any number
star = "wuma_035"
x = 30                       # percent of the image from its lower left (default 50, 50; -1: random)
y = 30
epoch = "start+0.05"         # periodic: a primary eclipse / maximum; transients: t0 = ...
                             # ISO UTC, "start+D" days from the start of the night (or the UT window), or BJD_TDB

[camera]                     # the toy camera electronics (gain, offset) where ArgusSim has none
[argussim]                   # observatory, telescope, survey, schedule: read from ArgusSim (override with care)
[hdps]                       # the hdps chain: [[hdps.tasks]] (the production Task list, every setting), [hdps.coadd]
[lightcurves]  [truth]       # linking detections into objects; matching them to the simulated stars (scoring)
[output]                     # bands, movies, catalogues, alerts, light curves, raw frames; stride, fps, noise seed
```

[`examples/observations/template.toml`](examples/observations/template.toml) lists every key with its default and
meaning. Anything can be overridden on the command line: `--set section.key=value`, `--place '{star = "...", ...}'`
(replaces the file's placements), `--date`, `--ra`, `--dec`, `--name`, `--bands`. Everything is
checked before any work starts. Outputs, in `outputs/pseudo_real/<name>/` (real Gaia stars where they are, `layout = "real"`)
or `outputs/fully_synthetic/<name>/` (stars at random positions, `layout = "random"`); `<name>`: `[field] name`, else the
file's name:

- `catalogue.csv`: every object detected (both bands' detections linked by position, `link_arcsec`): ID, RA, Dec,
  detections, median magnitude, scatter and error per band, the nearest ATLAS-Refcat2 star, its light curve file;
  `truth_*` columns (the simulated star it is) for scoring only.
- `lightcurves/NNNNNN.csv`: one per object, one row per detection: time (UTC, MJD, BJD_TDB), band, exposure,
  telescope, airmass, seeing, zero point, magnitude and error, flux and error, S/N, position, flags.
- `catalogues/`: every exposure's detections (`<name>_<band>_NNNN.parquet`): hdps's `SOURCES` catalogue from the
  solved frame (position, flux and error, S/N, saturation), RA/Dec from hdps's WCS, magnitude from hdps's zero point
  (`coadd_magzp − 2.5 log10 flux`, electron-referenced), and the simulated star each one is (`star`, for scoring).
- `hdps/<band>/`: hdps's own products, as its worker writes them: `transients_*.parquet` (difference-image candidates,
  `TransientOutputTask`), `alerts_*.parquet` and `alerts_stamps_*.npz` (`AlertVettingTask`), `ratchet_*_coadd.fits`
  (the ratchet coadds), and `alerts/<exposure>.avro`: the alert packets, hdps's own, Avro with the public `argus.alert`
  schema (read from hdps at run time), one file per exposure in place of the Kafka stream.
- `reference/`: the reference night (the night before, transients left out) through the same chain; its coadds in
  `reference/hdps/<band>/coadds/` are SFFT's references.
- `<name>_night_<band>.photometry.csv`: the placed stars' measurements, with the true model magnitude beside it.
- `<name>_night_<band>.alerts.csv`: every candidate hdps's AlertVettingTask found, its features, its Avro `alertId`,
  where its stamps are, and the truth (the simulated star there: transient / variable / constant / none, its true
  flux now and in the reference): a real/bogus training set.
- `<name>_night_<band>.mp4`: the field movie (CMD, exposure, hdps's difference image, light curves, SEDs, alerts);
  `<name>_night_<band>_pipeline.mp4`: the pipeline movie (each raw frame on its sensor, hdps's detections and WCS).
- `output.raw = true` keeps the raw camera frames (`hdps/<band>/raw/`) and hdps's bulky per-frame products (the solved
  frames and tile images); otherwise they are deleted once read.
- `observation.toml`: the observation as made (every key, the night as used); running it makes the same data.
- `summary.md`; `scene/`: exactly what the engine (`night_movie.py`) read.

The dust is added here: for every star, the E(B−V) in front of it at its distance along its sightline (the Bayestar19
3D map, capped at SFD's whole column; `synthstar/dust.py`, docs/ASSUMPTIONS.md). Gaia's constant stars have their
own measured dust taken out first with the same law. Network: Gaia queries the first time a pointing is used (then
cached in `data/gaia/`). The dust maps are local: `bash tools/fetch_dust.sh` once per machine (~860 MB, into
`~/dustmaps_data`).

### Real fields

```bash
python make_stars.py --gaia-list 294.7844 -11.6235 --write my_field.toml --write-observation my_obs.toml
python make_stars.py my_field.toml        # the field's stars as a set + each variable that can be modelled
python observe.py my_obs.toml             # layout "real"; each real variable placed by name only; date "auto"
```

A real star (generator `Gaia variable`) is modelled from Gaia DR3: its period and ephemeris, distance, the E(B−V) in
front of it at that distance (used only to infer its luminosity), composition and mass, then MESA RSP or PHOEBE
fitted to Gaia's eclipses. Its SEDs are dust-free like every star's; an observation puts it at its real position and
ephemeris, seen through the same dust. A Gaia field set's constant stars carry their distances (Bailer-Jones et al.
2021; GSP-Phot where they have none); an observation takes each one's measured dust out and puts the dust where it is
observed back in. Gaia queries are cached in `data/gaia/<pointing>/`.

### Generators

| Generator | Timing | Engine | What it is |
|---|---|---|---|
| `RR Lyrae` (`mode` RRab, RRc) | periodic | MESA RSP | radial pulsator tuned to your period, at `L` or absolute `M_G`, at `distance` |
| `Cepheid` (F, 1O), `Type II Cepheid` (BL Her, W Vir), `HADS` | periodic | MESA RSP | as RR Lyrae |
| `W UMa` | periodic | PHOEBE | contact binary from period, q, inclination, Teffs, fillout factor, distance |
| `detached EB` | periodic | PHOEBE | detached binary from period, inclination, Teffs, radii (masses optional), distance |
| `Gaia variable` | periodic | MESA RSP / PHOEBE | a real Gaia DR3 variable, modelled as itself |
| `supernova` | transient | sncosmo | any sncosmo model at a redshift and peak absolute magnitude |
| `microlensing` | transient | point lens | point or uniform-disc source, a MESA template SED |
| `dwarf nova` | transient | steady disc | accretion disc with a rising and decaying accretion rate, plus the white dwarf |
| `Gaia field` | set | Gaia DR3 | every Gaia star around a pointing, as constant stars |

Microlensing and dwarf nova are parametric models written in this repo (equations and references in
`synthstar/transients.py`). MESA RSP models take ~0.5–2 h each and PHOEBE a few minutes.

**Adding a generator:** a module in `synthstar/sources/` with a list `GENERATORS` of `Generator(name, timing,
engine, about, keys, make)`, added to `MODULES` in `all_generators()` (`synthstar/sources/__init__.py`). Its keys
are then checked in star files and listed by `python -m synthstar.sources`, and observations place it like every
other star. A star file names its generator exactly as listed (e.g. `"W UMa"`); an unknown generator or key, or a
missing required key, stops the run with the names.

### How it fits together

```
make_stars.py star file          → stars/<name>/   (generators in synthstar/sources/: MESA RSP, PHOEBE, sncosmo,
                                                     parametric transients, Gaia)
observe.py observation file      → outputs/<category>/<name>/scene/  (field, background stars, placed stars with their file,
                                                            position, timing and dust)
  ├─ the reference night        → the night before, same stars, no transients, through hdps up to its coadds
  └─ night_movie.py --scene      → ArgusSim's survey of the field: each exposure on its telescope's sensor
                                     (synthstar/sensor.py: ArgusSim geometry and PSF) in an N x N-minipix readout
                                     window, written as a raw camera frame (synthstar/camera.py, toy electronics)
                                   → hdps's own worker over the raw frames (synthstar/hdps_worker.py: GPUWorker, the
                                     production Task chain: reduction → detection → WCS → reprojection → coadd →
                                     SFFT → difference detection → alert vetting)
                                   → catalogues, photometry CSV, alerts, both movies, from hdps's products
  └─ after every band            → detections linked into objects → catalogue.csv, lightcurves/
```

Every assumption (what comes from ArgusSim, what is written here, what is assumed) is in
[docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md).

### Real/bogus training data

`[inject] number = N` scatters N library stars (supernovae, microlensing, dwarf novae, eclipsing and contact binaries,
RR Lyrae from `stars/lib*`) over the readout window: transients timed into the night, periodic stars at a random
phase and a mean G in `mag_min..mag_max`. Their candidates are the labelled real ones; residuals at the field's own
stars the bogus ones.

```bash
python tools/make_training_data.py --fields all --inject 200 --parallel 4   # one night per field, no movies
python tools/rb_training_set.py outputs/pseudo_real/rb_* -o data/rb             # -> data/rb/training_set.parquet
python tools/train_rb.py data/rb/training_set.parquet -o data/rb/bundle_v1      # -> an hdps model bundle
```

The alerts should be the ones Argus sends, and Argus observes every night of the lunar month, so the training nights
span it: `examples/rb/moon_phases_2026.csv` lists each field's nights at new moon, the quarters and full moon (a date
the field is up >= 3 h, so observe.py does not move it to another week and another phase). `--plan` writes those
observations, `tools/rb_queue.py` runs them one at a time (on Linux each in its own memory-limited systemd unit):

```bash
python tools/make_training_data.py --plan examples/rb/moon_phases_2026.csv --machine mac --stride 2 --write-only
python tools/rb_queue.py outputs/rb_queue/mac.queue        # lines "<name> python observe.py data/rb/observations/<name>.toml"
```

Every training row carries its night (field, night_date, mask_bright_stars, moon_phase, moon_illumination, moon_alt).
`train_rb.py` holds out whole fields (`--test-fields`, else cross-validation by field), measures on the nights run
with hdps's bright-star mask off (as Argus runs it; nights before 2026-10-25 had it on, which hides candidates) and
gives every operating point per Moon phase.

The bundle is what hdps's `RFBundleClassifier` loads: set `model_bundle` for AlertVettingTask in `[[hdps.tasks]]`
and `score_cut = "fa10"` (10 false alerts per deg² per exposure, as hdps's production config).

### Batches

`tools/stage_batch.py` takes a base file and lists to iterate over, as one TOML file:

- **Observation batches** (base: an observation file): one observation per combination of the `[vary]` lists (`combine
  = "product"` or `"zip"`; values inline or from a CSV), each written out as its own observation file and run with
  `observe.py`, several at once.
- **Star batches** (base: a star file): `[vary]` keys `star.<key>` make a grid of every star in the base, built by one
  `make_stars.py` run.

```toml
base = "../examples/observations/synthetic_transients.toml"
[vary]
"field.ra,field.dec" = [[294.7844, -11.6235], [273.0414, -20.7099]]
place = [
  [{star = "dwarf_nova_400pc", t0 = "start+0.02"}],
  [{star = "microlensing_te02", t0 = "start+0.15", x = 30, y = 70}],
]
```

```bash
python tools/stage_batch.py batches/example_sweep.toml --check       # expand and validate: 4 observations
python tools/stage_batch.py batches/example_sweep.toml               # run them; --status for progress
python tools/stage_batch.py batches/example_star_grid.toml --check   # 9 W UMa stars (without --check: builds them)
```

An observation batch only names stars, so build them first (`example_sweep.toml` needs `f294_field.toml` and
`transients.toml`). Each observation is written to `outputs/batch_<batch>/runs/<name>.toml` before it runs, and can be
rerun on its own with `observe.py`; a rerun of the batch skips observations already done.

`make_stars.py --gaia-list RA DEC [--gaia-list RA DEC ...] --write stars.toml --write-batch batch.toml` writes the
star file and the observation batch for any number of real fields.

### Live notebook

`notebooks/argus_night_live.ipynb` is the field movie, live, with one star you edit. The window is the movie itself
(`night_movie.MovieFrame`): CMD, light curves, hdps's exposure and difference image with the alerts, SEDs. Below it:

- the field: M67, M45 (the Pleiades) or the f294 test field (injected stars), each one precomputed hdps night (M3's
  night is to be rerun); the band, β or ρ;
- star E's type (RRab, RRc, contact binary, detached binary, dwarf nova, SN Ia, microlensing) and its sliders:
  magnitude, amplitude (a supernova: none), period or timescale; a click on either image moves it;
- an airmass offset (the same night observed lower);
- Play, at the movie's frame rate, and a frame slider.

Everything except star E keeps its photometry and alerts from the real hdps run. E is a library template whose
sliders are exact arithmetic on its integrated tables. It is added to hdps's cut-outs with ArgusSim's PSF and its
photon noise, and measured with hdps's own source detector and alert features. Its zero point comes from hdps's own
catalogue of each frame. The real/bogus model then scores E and the field's candidates.

The feature inputs that the cut-outs lack (variance plane, FWHM, window) are fitted to reproduce hdps's features.
Fitted this way, the same code gives hdps's own accept/reject on 96% of hdps's candidates in a real run. E itself was
checked against stars injected into a real run and measured by hdps: the same light, on empty sky next to them, gives
the same photometry within 0.01 mag (median and scatter) and the same alert decisions (97% and 100% of frames).
Two things that check found, both now reproduced:
- hdps's tiles carry surface brightness, so a star's flux on a tile is its electrons × 1.31 (the pixel-area ratio).
- hdps's bright-star mask compares tiles with the sky still in them. Anything brighter than about β 15 cannot alert.

The packs built so far (M67, M45, the f294 test field) and that check were made with hdps's bright-star mask on. Our
configuration has it off since 2026-10-25, as Argus's production configuration, so these nights are to be remade
with it off (M3's too), and the check repeated on a night processed without it (rb_model_test_f294_maskoff). The note
under the controls says which way the night shown was processed; E follows the night's own setting.

The note under the controls lists every simplification.

How it keeps up (timings on a 10-core Mac also running other work, 140-frame night; a night of up to 350 frames scales
in proportion):
- **Drawing.** `night_movie.MovieFrame` builds the figure once per star setting. Each frame draws only what changes over
  a kept background, in matplotlib's own order, so a frame is the image a full redraw gives. A frame takes about
  0.15 s, against about 0.6 s before (the engine's movies use the same code).
- **Measuring.** Six worker processes measure E in every frame and draw the frames for Play.
  - A change shows its new frame after about 1.5 s, without holding up the controls; a setting seen before after under
    1 s.
  - Play shows its first frame after about 2 s, then about 16 frames per second, and replays at the movie's 24.
- **Caches.** Kept in `data/interactive/`:
  - each pack uncompressed, memory-mapped by every process;
  - per pack, the sky, noise and field alerts;
  - ArgusSim's PSF stamps;
  - the measurements of each setting, keyed to the code that made them.

```bash
python3 -m ipykernel install --user --name mesa-argus --display-name "MESA_ARGUS (Python 3.14)"   # once
python observe.py examples/observations/interactive_m3.toml        # one night through hdps, every candidate kept
python -m synthstar.interactive.pack outputs/pseudo_real/interactive_m3 --pack m3   # -> data/interactive/packs/
python -m synthstar.interactive.precompute_templates               # star E's templates (else made when chosen)
python tools/check_edited_star.py f294test                          # star E against stars hdps measured itself
jupyter lab notebooks/argus_night_live.ipynb
```

A pack is captured from the engine's own movie setup, one band at a time (about 15 minutes). The observation must
have kept its movie data (`output.movie_data`) and every alert candidate (AlertVettingTask `emit_all_candidates`).

### Interactive page

`demo/` builds a single offline HTML page of one observation (real Gaia stars, for now) with a control panel,
simulating and measuring the exposures in the browser (`output.page = true`, or by hand):

```bash
python observe.py examples/observations/f294.toml --scene-only
python demo/export_demo.py --scene outputs/pseudo_real/f294/scene && python demo/build_html.py f294   # -> demo/argus_demo.html
```

`demo/validate/` checks the page's simulator and SExtractor port against the Python ones.

## 2. Argus vs Gaia vs SDSS: the 3D CMD and Fisher analysis

MESA tracks with the colors module give synthetic photometry in all three systems at every timestep.

```bash
python -m synthstar run population fisher_stencil      # resumable: re-running skips finished and running tracks
python -m synthstar status population fisher_stencil
python -m synthstar cmd3d population                   # -> outputs/population_cmd3d.html (one self-contained file)
python -m synthstar fisher                             # -> outputs/fisher/*.png, results.csv
```

Add `--live` to `cmd3d` or `fisher` to rebuild as runs finish.

**The analysis.** Tracks are resampled onto equivalent evolutionary points (EEPs), so index *e* is the same stage on
every track. At each EEP, central differences across the stencil tracks give ∂m_band/∂θ for θ = [M/H], ln M and
α_MLT; age (along the track) and the distance modulus (with the prior in `analysis.toml`) are nuisance directions.
The Cramér–Rao bound √(F⁻¹)_θθ, with F = Jᵀ Σ⁻¹ J, is the best precision any unbiased fit to that star's photometry
can reach. Gaia and SDSS use fixed per-band errors (`analysis.toml`); Argus is evaluated at a range of trial errors
(1 mmag to 0.5 mag). Two bands constrain at most two quantities per star (plus the distance prior), so separating
[M/H] from α_MLT needs ≥ 3 bands.

**Physics.** Kurucz2003 (ATLAS9) atmospheres; magnitudes absolute (Vega for Gaia and Argus, AB for SDSS, with AB − Vega
offsets measured from the module). Scaled-solar GS98 composition, no diffusion; Y from the solar calibration
(`data/solar_calibration.json`, α_MLT⊙ = 1.980). MIST-like overshoot (f = 0.016), Reimers/Blöcker winds. Stops at
core He depletion or 13.5 Gyr (25 Gyr for the stencils).

**Caveats.** Scaled-solar models (for α-enhanced halo stars read [M/H] ≈ [Fe/H] + 0.3); no extinction in the Fisher
analysis; atmospheres outside the grid (Teff < 3500 K giants, hot low-g corners) are clamped by MESA and excluded.

### MESA SEDs

`grids/argus_sim_seds.py` tracks write the SED at every model; `python -m synthstar seds GRID` packs them into
`runs/<track>/seds.h5` (wavelength, f_λ at 10 pc per model, model properties, magnitudes).

## Layout

```
make_stars.py           simulate stars (star files) -> stars/<name>/
observe.py              observe built stars (observation files) -> outputs/<category>/<name>/
night_movie.py          the engine observe.py runs: simulate and measure the exposures, draw the movie
synthstar/             the Python package (python -m synthstar ...)
  argus.py                Argus bands, units and plot style shared by the movies
  dust.py                 dust: the 3D map (Bayestar19, SFD), CCM89, dereddening Gaia stars
  schema.py               every observation parameter: default, unit, limits, justification (template.toml from it)
  sensor.py, camera.py    an exposure on its telescope's sensor (ArgusSim geometry, PSF); the raw camera frame
  hdps_worker.py          the raw frames through hdps's own worker (GPUWorker); the toys it needs
  alert_files.py          hdps's alert packets written as Avro files (in place of Kafka)
  pipeline_movie.py       the pipeline movie
  pipeline/               what is added after hdps: lightcurves (detections linked into objects), truth (scoring)
  interactive/            the live notebook: templates, the edited star, its measurement, field packs, the app
  sep_photometry.py       the earlier SExtractor (SEP) photometry (the demo's validation, the offset study)
  rsp.py                  MESA RSP pulsator models
  binary.py               PHOEBE eclipsing-binary models
  sources/                the star generators (one module per family; python -m synthstar.sources)
  transients.py           the transient physics: sncosmo, microlensing, dwarf nova
  gaia.py                 Gaia DR3 queries, a field's variables, a variable's physical parameters
  night.py                when a pointing is observable; BJD at the site
  runconfig.py            observation files: schema, defaults, validation, overrides
  seds.py                 MESA SED packing; the constant-star SED templates
  runner.py, grids.py     resumable parallel MESA runs; grid definitions → track specs
  history.py, eep.py      history reading, phases, equivalent evolutionary points
  cmd3d.py, fisher.py     the 3D CMD page; Fisher matrices and figures
  calibrate.py, zeropoints.py, filters.py, composition.py, config.py
grids/                  MESA grid definitions, one file per grid
inlists/                the single MESA physics template
mesa_work/              MESA work directory (make -C mesa_work)
data/                   committed inputs: Gaia query caches (data/gaia/<pointing>/), calibrations, filter curves
batches/                batch files for tools/stage_batch.py
examples/               star files (examples/stars/) and observation files (examples/observations/, template.toml)
tools/                  batch runner, star-library grid, newton job queue, dust-map download
studies/                characterisation studies (the photometric offset study)
demo/                   the interactive HTML page and its validation
notebooks/              the live notebook (argus_night_live.ipynb)
docs/ASSUMPTIONS.md     what the night movies model and assume
docs/LIBRARY.md         the star library: what newton builds (MESA and non-MESA), coverage, limits
docs/PIPELINE.md        flow charts of the whole pipeline, from the stars to the alerts and the real/bogus model
config.toml             machine settings          analysis.toml   Fisher-analysis inputs
templates/, vendor/     the 3D CMD page's template and its bundled plotly
stars/                  built stars (not in git; rebuilt from the star files)
runs/, outputs/         generated (not in git): MESA / PHOEBE working runs; observations, figures, CSVs
```

The hand-made fields `m3`, `m15`, `rr308108` and `rr416445` are the first movies, from before the star /
observation split (run with `night_movie.py --field m3`). `data/sparse/` is the search that found the two RR Lyrae fields.
