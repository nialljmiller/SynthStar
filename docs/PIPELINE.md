# The pipeline, as flow charts

How one observation is made, from the stars to the alerts, and how the real/bogus model is trained. Plain-text
Mermaid: GitHub, VS Code and Obsidian draw it.

**How to read them**

- A box is a step: what it does in plain words, then (small) the code that does it.
- An arrow carries what is handed on; its label names that data.
- A dashed arrow is optional or later (a setting turns it on, or it happens in another run).
- Circled numbers give the order things happen in.

| Colour | Who owns the step |
|---|---|
| blue | ArgusSim (the Argus instrument, site and survey model) |
| green | MESA_ARGUS (this repository) |
| amber | hdps (the Argus pipeline), run unchanged |
| red | a stand-in, where hdps or Argus has no real piece yet |
| grey | a file written |

Charts:

1. [Overview: one observation, start to finish](#1-overview-one-observation-start-to-finish)
2. [The scene: stars on the sky and the night's exposures](#2-the-scene-stars-on-the-sky-and-the-nights-exposures)
3. [One exposure: from the sky to a raw camera frame](#3-one-exposure-from-the-sky-to-a-raw-camera-frame)
4. [hdps on every frame](#4-hdps-on-every-frame)
5. [The reference night](#5-the-reference-night)
6. [Measurements and outputs](#6-measurements-and-outputs)
7. [Real/bogus: training the model and using it](#7-realbogus-training-the-model-and-using-it)

## 1. Overview: one observation, start to finish

Four stages, each **started and ended by files on disk**, so each can run on its own machine: the star library on one,
the synthetic images on another, hdps on a third (e.g. longleaf), the measurements on a fourth. Stage 3 *is* hdps —
its real pipeline, the one the telescope runs; the simulation differs from the live run only in its task config (a
simulated calibration in place of the astrometric solve, Avro files in place of the Kafka stream, an in-process
fakeredis in place of valkey).

`python observe.py <observation file>` drives stages 2–4 of a whole observation at once (each band β, ρ its own
process). `python tools/run_hdps.py <frames> -o <out>` runs **stage 3 alone** — hdps's real Orchestrator over a
directory of raw frames — so it can live on its own compute.

```mermaid
flowchart TD
  classDef argus fill:#dbeafe,stroke:#1e40af,color:#0b1d4a
  classDef ours fill:#dcfce7,stroke:#166534,color:#052e16
  classDef hdps fill:#fde68a,stroke:#92400e,color:#3b1d02
  classDef out fill:#e5e7eb,stroke:#374151,color:#111827

  OBS["Observation file: field, night, stars, every setting<br/><small>examples/observations/*.toml · schema.py · runconfig.py</small>"]:::ours

  subgraph S1["① Star maker"]
    LIB["Each star's light over time (SED), built once<br/><small>make_stars.py · MESA, PHOEBE, sncosmo, Gaia</small>"]:::ours
  end
  F1["stars/ library<br/><small>sed.h5 · meta.json</small>"]:::out

  subgraph S2["② Synthetic image maker, per band"]
    SCENE["The scene: stars on the sky, dust, the exposure list<br/><small>observe.py → scene/ · chart 2</small>"]:::ours
    EXP["Every exposure rendered as a raw Argus camera frame<br/><small>night_movie.py · chart 3</small>"]:::ours
  end
  F2["raw FITS frames (science + reference)<br/><small>hdps/&lt;band&gt;/raw/*.fits · chart 5</small>"]:::out

  subgraph S3["③ hdps — the real pipeline"]
    HDPS["hdps's real Orchestrator over every frame<br/><small>run_hdps.py / hdps_worker.py · hdps Orchestrator + GPUWorker (Ray, 1 worker) · chart 4</small>"]:::hdps
  end
  F3["hdps products: coadds, catalogues, Avro alert packets<br/><small>hdps/&lt;band&gt;/ · alerts/&lt;exposure&gt;.avro</small>"]:::out

  subgraph S4["④ Truth, measurement, light curves"]
    MEAS["Measurements + simulation truth from hdps's products<br/><small>night_movie.py · chart 6</small>"]:::ours
    LINK["Both bands' detections linked into light curves<br/><small>pipeline/lightcurves.py</small>"]:::ours
  end
  OUT["catalogues · photometry · alerts.csv · light curves · movies"]:::out
  TRAIN["Real/bogus training, over many observations<br/><small>chart 7</small>"]:::ours

  OBS --> LIB
  OBS --> SCENE
  LIB --> F1
  F1 -.->|"SEDs"| SCENE
  SCENE --> EXP --> F2
  SCENE -.->|"reference night, chart 5"| F2
  F2 -->|"raw frames (+ reference coadds)"| HDPS --> F3
  F3 -->|"hdps products"| MEAS --> LINK --> OUT
  OUT -.->|"labelled alerts"| TRAIN
  TRAIN -.->|"trained model bundle"| HDPS
```

## 2. The scene: stars on the sky and the night's exposures

`observe.py`, before anything is rendered. Everything it decides is written to `scene/`, which the engine reads.

```mermaid
flowchart TD
  classDef argus fill:#dbeafe,stroke:#1e40af,color:#0b1d4a
  classDef ours fill:#dcfce7,stroke:#166534,color:#052e16
  classDef out fill:#e5e7eb,stroke:#374151,color:#111827

  OBS["Observation file<br/><small>field · night · stars · place entries · inject section</small>"]:::ours

  DATE["The night: the nearest one the field is up ≥ 3 h<br/><small>observe.py choose_night · night.py · array minimum altitude 32.84°</small>"]:::ours
  SURVEY["Every exposure of the field that night: time, telescope,<br/>tracking angle, band, field angle; seeing drawn per ratchet<br/><small>survey.py · ArgusSim coverage_exact.covering · draw_seasonal_seeing</small>"]:::argus

  FIELD["Field stars: every Gaia DR3 source within 0.52° of the field<br/><small>stars/&lt;field&gt;_field · observe.py refuses a set built smaller</small>"]:::ours
  BG["Constant stars: a template SED from each one's Gaia colour, at its distance,<br/>at their real positions (layout real) or scattered (layout random)<br/><small>observe.py write_background</small>"]:::ours
  PLACE["Placed stars: the variables and transients the file names<br/><small>place entries</small>"]:::ours
  INJ["Injected stars: random library stars over the readout window,<br/>transients timed into the night, periodic stars at a drawn brightness<br/><small>inject section · observe.py inject (training data)</small>"]:::ours
  LIB["Star library<br/><small>stars/</small>"]:::ours

  DUST["Dust: E(B−V) in front of each star, at its distance<br/><small>dust.py · Bayestar19 3D map, capped at SFD</small>"]:::ours
  SCN["scene/: field stars, placed and injected stars, exposure list<br/><small>gaia_dr3_core.ecsv · sources.json · exposures.ecsv · field.json</small>"]:::out

  OBS --> DATE --> SURVEY
  OBS --> FIELD --> BG
  OBS --> PLACE
  OBS -.->|"if inject number > 0"| INJ
  LIB --> PLACE
  LIB --> INJ
  BG --> DUST
  PLACE --> DUST
  INJ --> DUST
  DUST --> SCN
  SURVEY -->|"exposures.ecsv"| SCN
```

## 3. One exposure: from the sky to a raw camera frame

`night_movie.py`, once per exposure in the band. Only a readout window of the telescope's sensor around the field is
made, not the whole 11,648 × 8,742 px sensor.

```mermaid
flowchart TD
  classDef argus fill:#dbeafe,stroke:#1e40af,color:#0b1d4a
  classDef ours fill:#dcfce7,stroke:#166534,color:#052e16
  classDef toy fill:#fecaca,stroke:#991b1b,color:#450a0a
  classDef out fill:#e5e7eb,stroke:#374151,color:#111827

  ROW["This exposure: time, telescope, tracking angle, band, seeing, field angle<br/><small>scene/exposures.ecsv</small>"]:::ours

  SED["Each star's SED at this moment, reddened by its dust<br/><small>pulsation phase · binary phase · transient light curve · constant template</small>"]:::ours
  PHOT["Electrons from each star: SED × band throughput × atmosphere at this airmass<br/><small>ArgusSim ABPhot · SystemThroughput</small>"]:::argus
  SKY["Sky and dark current per pixel: airglow, zodiacal light, starlight, moon<br/><small>ArgusSim sky_components_at · SpectralSky</small>"]:::argus

  GEOM["Where each star lands on this telescope's sensor:<br/>sky → telescope frame (tracking angle) → focal plane → pixel<br/><small>sensor.py Projector · ArgusSim Layout</small>"]:::ours
  WIN["The readout window: N × N minipixes around the field, kept on the sensor<br/><small>sensor.window · field.window (default 56 ≈ 26′)</small>"]:::ours
  PSF["Each star's PSF at its own field angle, with this seeing<br/><small>ArgusSim ZemaxPSF.with_seeing · sensor.PSFs</small>"]:::argus

  RENDER["The image in electrons: stars + sky, Poisson noise, full-well clip, read noise<br/><small>sensor.render</small>"]:::ours
  CAM["The raw camera frame: 16-bit ADU, real AEGIS header keys,<br/>toy gain (full well / 58,000), no bias or overscan<br/><small>camera.py · night_movie.py write_raw_frame</small>"]:::toy
  RAW["hdps/&lt;band&gt;/raw/&lt;camera&gt;_&lt;UTC&gt;.fits"]:::out
  PREV["4 × 4-binned preview, for the pipeline movie<br/><small>movie data · chart 6</small>"]:::out

  ROW --> SED --> PHOT -->|"electrons per star"| RENDER
  ROW --> SKY -->|"electrons per pixel"| RENDER
  ROW --> GEOM -->|"sensor pixels"| WIN -->|"window pixels"| RENDER
  ROW --> PSF --> RENDER
  RENDER -->|"electrons"| CAM -->|"ADU"| RAW
  RENDER -.-> PREV
```

## 4. hdps on every frame

hdps's **real Orchestrator** — the path `hdps batch process` runs: it spawns one Ray worker actor (hdps's own
`GPUWorker`, which builds hdps's `PipelinedExecutor`) and runs hdps's production Task chain, unchanged. Driven by
`run_hdps.py` (stage 3 on its own) or by `hdps_worker.py` inside `observe.py`; nothing of hdps is reimplemented or
monkey-patched — the alert sink, the classifier and the calibration are all chosen through task config. On a CPU host
the compute runs through the `no_cupy` shim while Ray still schedules the one worker on a (logical) GPU. The chain and
every setting are `[[hdps.tasks]]` in the observation file. A frame that a Task fails on is skipped; the rest carry on.

```mermaid
flowchart TD
  classDef hdps fill:#fde68a,stroke:#92400e,color:#3b1d02
  classDef toy fill:#fecaca,stroke:#991b1b,color:#450a0a
  classDef out fill:#e5e7eb,stroke:#374151,color:#111827

  RAW["Raw frames<br/><small>chart 3</small>"]:::out
  MAN["Frames grouped by camera and ratchet<br/><small>group_images_by_ratchet_camera · ManifestSource</small>"]:::hdps
  CACHE["WCS cache: each camera's solution, reused within a ratchet<br/><small>worker · valkey, run offline on an in-process fakeredis (per Ray actor)</small>"]:::toy

  subgraph CHAIN["hdps's Task chain, in order: each Task hands the frame, then its tiles, to the next"]
    T1["Clean and reduce: ADU → electrons, saturated pixels flagged<br/><small>DataCleanup · ImageReduction</small>"]:::hdps
    T2["Find sources on the frame<br/><small>SourceDetection · SourceFilter</small>"]:::hdps
    T3["Load reference stars around the frame<br/><small>ReferenceRetrieval · ATLAS-Refcat2</small>"]:::hdps
    T4["Solve the WCS, with distortion; fit the zero point<br/><small>LinearWCSSolver · SourceMatching · TPVWCSSolver</small>"]:::hdps
    T5["Write the solved frame<br/><small>LightweightFITSOutput</small>"]:::hdps
    T6["Resample onto hdps's fixed sky tiles (30′ each; a frame touches 4–9)<br/><small>Reprojection</small>"]:::hdps
    T7["Add each tile to this camera's ratchet coadd<br/><small>CoaddTap</small>"]:::hdps
    T8["Subtract the reference coadd: fit a matching kernel and the sky,<br/>keeping no-data pixels out of the fit;<br/>bright-star footprints hidden from detection<br/><small>SFFTSubtraction</small>"]:::hdps
    T9["Detect candidates on the difference<br/><small>DiffDetection · TransientOutput</small>"]:::hdps
    T10["Vet alerts: its own detection on the difference, features,<br/>real/bogus score, keep those above the cut<br/><small>AlertVetting</small>"]:::hdps
    T11["Write the movie tile's science and difference images<br/><small>TileImageOutput</small>"]:::hdps
    T1 --> T2 --> T3 --> T4 --> T5 --> T6 --> T7 --> T8 --> T9 --> T10 --> T11
  end

  REFC["Reference coadds<br/><small>chart 5</small>"]:::out
  RB["Real/bogus, chosen by model_bundle: no bundle → hdps's<br/>PassthroughClassifier (every score 1); a trained bundle → RFBundleClassifier<br/><small>hdps.alerts.classifier · AlertVetting model_bundle · chart 7</small>"]:::hdps
  FIN["Ratchet coadds closed at each ratchet change and at the end<br/><small>worker finalize_ratchet · CoaddWriter</small>"]:::hdps
  KAFKA["Alert packets, accepted alerts only: Avro, the argus.alert schema,<br/>one file per exposure — hdps's offline sink in place of the Kafka stream<br/><small>AlertVetting emit_avro · hdps.alerts.avro_file</small>"]:::hdps

  subgraph PRODUCTS["Written to hdps/&lt;band&gt;/"]
    P1["solved frames: sources, WCS, zero point<br/><small>solved/*_solved.fits</small>"]:::out
    P2["ratchet coadds<br/><small>ratchet_*_coadd.fits</small>"]:::out
    P3["transient_catalog<br/><small>transients_*.parquet</small>"]:::out
    P4["alert_catalog + alert stamps<br/><small>alerts_*.parquet · alerts_stamps_*.npz</small>"]:::out
    P5["science + difference tiles<br/><small>epoch_*_tile_*.fits</small>"]:::out
    P6["alert packets<br/><small>alerts/&lt;exposure&gt;.avro</small>"]:::out
  end

  RAW --> MAN --> T1
  CACHE <-.-> T4
  REFC -->|"reference_dir"| T8
  RB -.-> T10
  T5 --> P1
  T7 --> FIN --> P2
  T9 --> P3
  T10 --> P4
  T10 --> KAFKA --> P6
  T11 --> P5
```

## 5. The reference night

What SFFT subtracts against. Made only when `output.alerts` is on (the default), before the science night.

```mermaid
flowchart TD
  classDef ours fill:#dcfce7,stroke:#166534,color:#052e16
  classDef hdps fill:#fde68a,stroke:#92400e,color:#3b1d02
  classDef out fill:#e5e7eb,stroke:#374151,color:#111827

  RN["The night(s) before: same field and stars;<br/>transients left out; injected variables kept, no new injection<br/><small>observe.py make_reference · night.reference_nights (default 1)</small>"]:::ours
  RX["Exposures rendered as raw frames<br/><small>chart 3</small>"]:::ours
  RH["hdps up to the coadd step, nothing after<br/><small>run_hdps.py --coadd-only · hdps_worker.run(stop_after=CoaddTapTask)</small>"]:::hdps
  RC["Ratchet coadds, one per camera, ratchet and tile<br/><small>reference/hdps/&lt;band&gt;/coadds/</small>"]:::out
  PICK["For each tile, SFFT takes one coadd: the deepest<br/><small>hdps CoaddReferenceProvider</small>"]:::hdps
  SCI["Science night's subtraction<br/><small>chart 4, SFFTSubtraction</small>"]:::hdps

  RN --> RX --> RH --> RC --> PICK -->|"reference tile"| SCI
```

## 6. Measurements and outputs

Per band, `night_movie.py` reads hdps's products once its worker has finished; after both bands, `observe.py` links
the light curves. hdps's bulky per-frame files (raw frames, solved frames, tile images) are deleted once read, unless
`output.raw` is on.

```mermaid
flowchart TD
  classDef ours fill:#dcfce7,stroke:#166534,color:#052e16
  classDef out fill:#e5e7eb,stroke:#374151,color:#111827

  subgraph IN["From hdps (chart 4) and the render (chart 3)"]
    SOLVED["solved frames"]:::out
    ALERTCAT["alert_catalog · alert stamps · Avro packets"]:::out
    TILES["science + difference tiles"]:::out
    PREV["raw-frame previews"]:::out
  end
  TRUTH["Truth: which simulated star is where, its true brightness and change<br/><small>pipeline/truth.py · the scene</small>"]:::ours

  CAL["Each frame's catalogue: RA/Dec from hdps's WCS,<br/>mag = hdps zero point − 2.5 log10 flux, each detection's simulated star<br/><small>night_movie.py measure_frame</small>"]:::ours
  ATAB["Every alert candidate with its features, alertId, stamps<br/>location and truth class (transient, variable, constant, none)<br/><small>night_movie.py alert_table · truth.label_alerts</small>"]:::ours
  MD["Movie data: block cut-outs, raw previews, detections, WCS,<br/>measurements, alert marks<br/><small>output.movie_data</small>"]:::ours
  LC["Both bands linked by position into objects<br/><small>pipeline/lightcurves.py · after both bands</small>"]:::ours

  O1["catalogues/&lt;name&gt;_&lt;band&gt;_NNNN.parquet<br/>&lt;name&gt;_night_&lt;band&gt;.photometry.csv"]:::out
  O2["&lt;name&gt;_night_&lt;band&gt;.alerts.csv"]:::out
  O3["&lt;name&gt;_night_&lt;band&gt;.movie.joblib"]:::out
  O4["field movie + pipeline movie (.mp4)<br/><small>now if output.movie; later with tools/render_movies.py</small>"]:::out
  O5["catalogue.csv + lightcurves/NNNNNN.csv"]:::out

  SOLVED --> CAL
  TRUTH --> CAL
  CAL --> O1
  ALERTCAT --> ATAB
  TRUTH --> ATAB
  ATAB --> O2
  TILES --> MD
  PREV --> MD
  CAL --> MD
  ALERTCAT -->|"accepted alerts"| MD
  MD --> O3 -.-> O4
  O1 --> LC --> O5
```

## 7. Real/bogus: training the model and using it

Without a model every candidate scores 1, so every star that subtracts imperfectly becomes an alert. The simulation
knows what is real, so it can make the training set.

```mermaid
flowchart TD
  classDef ours fill:#dcfce7,stroke:#166534,color:#052e16
  classDef hdps fill:#fde68a,stroke:#92400e,color:#3b1d02
  classDef out fill:#e5e7eb,stroke:#374151,color:#111827

  RUNS["Many training observations: one night per real Gaia field,<br/>200 injected library stars each, no movies, several at once<br/><small>tools/make_training_data.py → observe.py</small>"]:::ours
  AT["Every candidate with its truth, per band<br/><small>&lt;name&gt;_night_&lt;band&gt;.alerts.csv · chart 6</small>"]:::out
  LABEL["Label each candidate. Real: a transient or variable whose true change<br/>is ≥ 3 flux errors. Bogus: a constant-star residual, or nothing there.<br/>Anything else: left out<br/><small>tools/rb_training_set.py</small>"]:::ours
  SET["training_set.parquet + .json<br/><small>hdps's feature names and order; missing values as −999</small>"]:::out
  TRAIN["Train a random forest; test it on whole fields it never saw<br/>(whole exposures if fewer than 4 fields); set the cuts: fa1 / fa10 / fa100 false alerts per deg² per exposure<br/><small>tools/train_rb.py · scikit-learn</small>"]:::ours
  BUNDLE["hdps model bundle: manifest.json + model.joblib<br/><small>data/rb/…/bundle</small>"]:::out
  USE["In an observation file: AlertVetting model_bundle and score_cut = fa10<br/><small>hdps RFBundleClassifier scores every candidate · chart 4</small>"]:::hdps

  RUNS --> AT --> LABEL --> SET --> TRAIN --> BUNDLE --> USE
```

## Editing

- **A new step:** add a box with its colour class, and arrows in and out. Write what it does first, then the code in
  `<small>`; label each arrow with the data handed on.
- **A stand-in replaced by the real thing** (a trained model, real camera electronics, a valkey server, Kafka): change
  its class from `toy` to `hdps` or `ours`.
- **Keep the overview small.** Detail belongs in the stage charts.
- **Check it renders** before committing: `npx -y @mermaid-js/mermaid-cli -i docs/PIPELINE.md -o /tmp/pipeline.md`
  writes one image per chart.
