# Hand-off: the interactive demo (2026-10-26)

From 2026-10-26 the MESA_ARGUS work runs in two Claude Code chats:

- **Alerts pipeline chat** (the existing one): the hdps runs, the alert samples for BOOM, the real/bogus training, the
  run queues on the Mac, newton and vega.
- **Interactive demo chat** (new): this document is its starting point.

## Direction

Hank (2026-10-26): shift focus to making the interactive demo much better. The user decides what "better" means
first; ask them before building.

## What exists

The live notebook `notebooks/argus_night_live.ipynb` (kernel `mesa-argus`), code in `mesa_argus/interactive/`:

| File | Role |
|---|---|
| `templates.py`, `precompute_templates.py` | library light-curve/SED templates, integrated once, cached in `data/interactive/templates/` |
| `star.py` | the edited star's slider arithmetic (type, magnitude, amplitude, period or timescale), exact |
| `compose.py` | adds the edited star's ArgusSim PSF stamp, with photon noise, to hdps's cut-outs |
| `measure.py` | measures it with hdps's own source detector and alert features, scores it with the real/bogus bundle |
| `pack.py` | captures `night_movie`'s draw data from a `--render-only` run into a field pack (about 20+ min per band) |
| `night.py` | pack + edited star -> draw dict; calls `night_movie.draw` unchanged |
| `worker.py` | worker processes that measure and draw frames for Play |
| `app.py` | ipywidgets + ipympl controls |

Agreed design (user, 2026-10-23): the screen looks like the field movie (CMD, exposure, difference, light curves, SED,
alert marks); controls below (field, star type, magnitude / amplitude / period sliders, click to place, airmass,
beta/rho, Play). Everything except the edited star E comes from a real hdps run. Simplifications must be stated on
screen. User, 2026-10-23: "as an initial state, this is very good"; next was to finish the functionality, then make it
faster (a0243d4 made it faster: about 16 frames/s on Play).

Validation: `tools/check_edited_star.py` (E against injected stars hdps measured itself): photometry within 0.01 mag,
alerts 97% / 100% the same.

Background notes: memory `interactive-notebook.md` (the design and history) and `interactive-html-demo.md` (the older,
parked plan for a browser-only HTML demo for the Schmidt Sciences symposium; open questions there: offline file or
hosted link, which photometry methods, the symposium date).

## Field packs (data/interactive/packs/, not in git)

| Pack | State |
|---|---|
| `m45_beta`, `m45_rho` | built 2026-10-24, bright-star mask ON |
| `m67_beta`, `m67_rho` | built 2026-10-24 from the new-moon night (2026-02-17), mask ON |
| `f294test_beta` | development pack, mask ON |
| M3 | **none.** The M3 new-moon night (2026-05-16) on newton was OOM-killed at its 24 GB cap on 2026-10-24 17:42 (scope `rerun-m3`, 52 SFFT steps into the beta band) and never reported. Its partial folder is `~/MESA_Argus/outputs/pseudo_real/interactive_m3` on newton. Needs a rerun with a larger cap. |

hdps's bright-star mask has been off in our configuration since 2026-10-25, so every pack is to be remade without it
(the notebook's note already says which packs had it on), and the edited-star check repeated on
`rb_model_test_f294_maskoff`.

M3's observation uses `field.window_offset = [0, -7]` and `tpv_terms = 7`; M67's linear WCS fails on about a third of
frames (hdps-side, weaker solves than M45's).

## Known simulation issues that show in the demo

- Bright stars sit in a visible square: ArgusSim's Zemax PSF stamp ends at 29 x 29 sensor pixels, so the wings stop
  at about 50 e- (figure: `outputs/pseudo_real/alert_sample_v1_f359_539p61_214_20260910_mac_beta/psf_wings_check.png`).
  A fix (a fitted outer wing, renormalised) is planned for the alerts v2 runs; it is an engine change both chats share.
- Saturated cores are flat at the full well, 21,657 e-, in ArgusSim's 60 s exposure: physical for its camera model.

## Compute (shared with the alerts chat)

As of 2026-10-26 10:30:

- **Mac:** the alerts v1t7 nights run one at a time (`outputs/rb_queue/mac.queue`, nice 10) until about 20:45.
- **newton:** the alerts v1t7 f286 pair until about 16:00; then the v2 training nights queue.
- **vega:** takes no new jobs until the user says.

Demo jobs (pack rebuilds, the M3 rerun) must not stop or slow the alerts runs without the user's word; agree the
machine time through the user. On newton, every long job runs in its own `systemd-run --user --scope -p MemoryMax=...`
unit, watched for failures (memory `newton-job-monitoring.md`). Measured peaks: an M3 night > 24 GB.

## Standing rules (both chats)

- hdps is the pipeline: run it, never re-implement it (memory `pipeline-is-hdps.md`). The v4 worktree
  `/Users/njm/argus_repos/hdps_v4` stays local; never post to or push to argus-hdps without asking.
- Every parameter lives in the observation TOML with default, limits and justification (`one-config-file.md`).
- Videos: beta only, whole night, every frame, unless asked.
- Never use or store the newton password; no SSH keys on newton; user `argus@newton`. Nothing on newton is deleted.
- No new venvs. Commit or push only when asked; no attribution lines in commits.
- Plain, precise language; direct questions get the bare answer.
