"""argus_image — the Argus-specific image maker (and the measurement/movie layer): takes synthstar's star library +
ArgusSim's instrument model and renders raw Argus camera frames, then measures hdps's products. Depends on synthstar
(star light: dust, SED templates, variability models, synthetic photometry) and argus_sim (the instrument). Destined
for the ArgusSim image-maker branch. Nothing in synthstar imports this package (one-directional)."""
