"""The engine, split into file-bounded stages (see docs/PIPELINE.md and three-repo plan):

  state   shared field/block state and constants (use_field, set_block)
  models  periodic (RSPStar) and transient (TransientSource) light-curve models — no engine state
  scene   the scene and its stars, their rates and per-exposure conditions (stage 1/2 input)
  render  the camera: a scene -> raw Argus frames (stage 2, destined for argus_image)
  measure hdps's products -> photometry, alerts, cut-outs (stage 4)
  movie   the movie drawing (stage 4 presentation)

night_movie.py is the CLI + orchestration; it re-exports these names so existing importers keep working while
the stages are pulled apart. Nothing here imports hdps."""
