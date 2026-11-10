# Night movies: what is modelled, what is taken from ArgusSim, and what is assumed

This applies to every observation made by `observe.py` (and the hand-made fields M3, M15 and two RR Lyrae fields).
Stars are simulated by `make_stars.py` with no dust and nothing about observing; each built star's own choices are in
`stars/<name>/summary.md`, and each observation's (the night, background, placements, dust) in
`outputs/<name>/summary.md` and `observation.toml`.

## 1. Argus, taken directly from ArgusSim (not chosen here)

| What | Source in ArgusSim |
|---|---|
| β and ρ throughput: filter × optics × QE, throughput loss | `SystemThroughput` (config defaults) |
| Collecting area | `config.telescope.collecting_area` |
| Plate scale 1.007″/px, full well 21,657 e⁻, read noise 1.2 e⁻, dark 0.0022 e⁻/s/px | `config.telescope` |
| PSF: Zemax optical PSF with the delivered EE50 profile, convolved with seeing, block-summed to pixels | `ZemaxPSF.from_config(...).with_seeing(...)` |
| Atmospheric transmission, raised to the airmass | `SystemThroughput` atmosphere curve; `ABPhot.electrons_from_sed` |
| Sky per exposure: airglow (solar activity, airmass), zodiacal light (ecliptic position), integrated starlight (Galactic latitude), moonlight (Krisciunas & Schaefer 1991), site artificial light | `Observatory.sky_components_at` + `SpectralSky.sky_electrons` |
| Seeing: the site's seasonal median from the DIMMs, scaled to airmass as X^0.6 | `seasonal_seeing_params`, `seeing_at_airmass` |
| The night's exposures: 60 s cadence in 15-min ratchets on UTC quarter hours, the last minute a reset, the Sun below −18° | `coverage_exact.Schedule` built from `config.survey` (as `Survey` builds it) |
| When a field can be observed: at or above the array's minimum altitude, 32.84° (airmass ≈ 1.84) | `config.observatory.min_alt` |
| Airmass: Pickering (2002), for the stars' extinction and seeing as for the sky | `pickering_airmass` |
| Site | `config.observatory` (a field may move it, e.g. `--site sro`; the instrument stays Argus) |
| Synthetic AB magnitudes (filter only) and detected electrons (full system) from an SED | `ABPhot.mag_from_sed`, `ABPhot.electrons_from_sed` |
| HDPS minipix geometry (32 × 0.879″ = 28.1″) | `skymap_shim` |

## 2. Written here (ArgusSim has no equivalent): images

ArgusSim supplies the survey (which telescope takes the field when, in which band, its tracking angle; `Survey`,
`coverage_exact`), the geometry (each telescope's array frame and focal plane, `coverage_exact.Layout`) and one star's
PSF at a time (`ZemaxPSF.with_seeing`). It does not make images. The following is this repo's own code, built on those.

- **Sensor** (`synthstar/sensor.py`). Every star's sensor pixel in every exposure, as ArgusSim's geometry places it:
  ICRS → CIRS once per night, then the telescope's array frame with its tracking angle → gnomonic focal plane →
  pixels. Its ArgusSim PSF is taken at its own field angle (the nearest tabulated one) and sub-pixel phase (1/8 px),
  scaled by its ArgusSim electrons for the exposure (its SED at that moment, at that airmass), and summed with
  ArgusSim's sky and dark current. Only an N × N-minipix readout window around the field is made (`field.window`,
  default 56 minipix, ≈ 26′; at least 28, hdps's detector minimum; at most the sensor).
- **Noise.** A Poisson draw per pixel, clipped at ArgusSim's full well, plus Gaussian read noise.
- **Camera** (`synthstar/camera.py`, a toy). The window is written as a raw camera frame with the real AEGIS header
  keys (UTCSTART, RATCHNUM, CAMID, TARGRA/TARGDEC: the window centre, ...): uint16 with gain = full well / 58,000
  e⁻/ADU (so hdps's 58,000 ADU saturation is the full well), no bias, no overscan, the real frames' readout parity.

## 2b. Measurement: hdps itself, with the toys it needs

The raw frames go through hdps's own worker (`synthstar/hdps_worker.py`: `GPUWorker`, its ratchet manifests, WCS
cache, coadd finalize and product writer) running the production Task chain (`docker/demo/config.yaml`, every setting
in the observation file's `[[hdps.tasks]]`): reduction → source detection → Refcat2 → linear and TPV WCS → solved
frame (`LightweightFITSOutputTask`) → reprojection onto hdps sky tiles → ratchet coadd → SFFT against reference
coadds → difference detection → transient catalogue → alert vetting → tile images (`TileImageOutputTask`). hdps runs
on the CPU through the `hdps_full_cpu` branch, whose CPU paths mirror the GPU code. What stands in for what does not
exist here:

- **valkey**: an in-process fakeredis server (as hdps's own tests use). **Kafka**: hdps's alert packets written as
  Avro files, one per exposure, with the public `argus.alert` schema read from hdps at run time.
- **Real/bogus**: not run; every candidate scores 1 (no trained model). Anything left in a difference image that
  hdps's alert detector finds is an alert (bright stars leave residuals that would be alerts too).
- **References**: the night before, through the same chain to its ratchet coadds (transients left out). hdps picks
  one coadd per tile; with the readout window smaller than a tile it covers part of it.
- **`hdps_full_cpu` changes beyond CPU mirroring**: SFFT keeps pixels with no reference or no science data out of
  its kernel fit (`mask_no_coverage_in_fit`; hdps main fits across them, which gave the kernel a flat pedestal, a
  kernel-sized square around every star and a gradient on part-covered tiles); the CPU SFFT does no whole-tile median
  subtraction (the GPU code has none).
- **Photometry**: hdps's source detector's matched-filter flux on the reduced frame, calibrated by hdps's zero point
  (`coadd_magzp`, electron-referenced, fitted against ATLAS g for β and r for ρ; the scalar tier, no colour term, so
  β reads ≈ 0.3 mag off truth for these stars' colours). No PSF or aperture photometry, as the real pipeline.
- **Ours after hdps** (`synthstar/pipeline/`): linking each band's detections by position into objects and light
  curves (hdps has no light-curve product yet), and the truth labels for scoring.
- **Colour exposure** (display only). β and ρ, sky subtracted; ρ → red, β → blue, their mean → green, with β scaled
  so the field's median star is grey. Brightness is ρ electrons per pixel above sky on an asinh stretch; colour is the
  pixel's channel ratios (Lupton et al. 2004); a pixel at full well in either band is white.

## 3. Argus-related choices made here (assumptions, not from ArgusSim)

Items 3–5 are conditions ArgusSim's own survey simulation draws (Survey, ForwardMC) and these movies do not use yet.
(Earlier assumptions 1 and 2, a fixed field angle and continuous coverage in both bands, are gone: the field angle,
telescope, band and coverage of every exposure now come from ArgusSim's survey and geometry, section 2.)

3. **A median-seeing night.** The seasonal median zenith seeing for the whole night, varying only with airmass.
   ArgusSim's survey draws it per ratchet (`draw_atmosphere`: lognormal about the seasonal median, correlated with
   transparency).
4. **Photometric sky, every night clear.** Transparency 1. ArgusSim draws a grey cloud transparency per exposure
   (`draw_cloud_transparency`) and whether a night is clear (`observable_weather`).
5. **Solar activity 0.8** (ArgusSim's config default, solar minimum), which sets the airglow level. ArgusSim's survey
   uses a per-night 2028–2033 prior (`solar_activity_by_ratchet`).
6. **PSF details.** The PSF stamp ends at 28 × 28 pixels (ArgusSim's Zemax grid limit), so very bright stars lose
   their far wings (≈ 1% of the light beyond 12 px). The PSF is the nearest tabulated field angle's. The same seeing FWHM is used in β and
   ρ (real seeing scales as λ^−0.2, so β's would be ~8% broader; ArgusSim has no chromatic seeing either).
7. **Not modelled at all:** flat-field and calibration errors, detector non-linearity, blooming beyond full-well
   clipping, cosmic rays, persistence, gain and ADC quantisation, scintillation, tracking jitter and image rotation.

## 4. Astrophysics: modelled vs assumed

Every source is an SED; ArgusSim turns it into electrons in each band at each exposure's airmass. Milky Way dust is
CCM89 (R_V = 3.1; `synthstar/dust.py`), added at observation, with E(B−V) the dust in front of each star: Bayestar19
(Green et al. 2019) at the star's distance along its sightline, never more than SFD's whole column (beyond a
sightline's reliable distances Bayestar19 follows its prior); SFD where Bayestar19 has no data (dec < −30°) or the star
has no distance (supernovae, microlensing sources: behind all the dust). E(B−V) = 0.884 × the map value (both maps are
in SFD units; with CCM89 this gives Schlafly & Finkbeiner's 2011 A_V = 2.742 per SFD unit). Distances: real stars,
Bailer-Jones et al. (2021) geometric, else Gaia GSP-Phot, else none (the whole column); Gaia variables, their own
(parallax or period-luminosity, iterated with the dust); placed periodic stars and dwarf novae, their model's.

- **Constant stars.** Gaia DR3 measured each through the dust in front of it; that dust is taken out with the same
  law: the MESA SED (the `argus_sim_seds` tracks: 0.8 M☉ [M/H] = −2; 1, 3, 5 and 10 M☉ solar) whose BP−RP, seen
  through that E(B−V), is the measured one gives the intrinsic colour, and its A_G the dust-free G. At observation that
  SED, scaled to the dust-free G, is seen through the dust where the star is observed: at its real position, exactly as
  Gaia measured it; in a synthetic field, through the new sightline. Only colour is matched (not gravity, metallicity
  or mass). Above E(B−V) ≈ 0.9 neighbouring templates of other metallicities redden slightly differently (≤ 0.05 mag
  in BP−RP at E(B−V) = 4); the inversion uses their running maximum. Sources with no or blended BP−RP take a median
  colour.
- **Pulsators** (`synthstar/rsp.py`): MESA RSP models whose linear period of the star's own mode is tuned to the
  Gaia (or ASAS-SN) period, run to the limit cycle, then one cycle with an SED every ≤ 25 s from MESA's colors module.
  Placed on the night with the catalogue period and epoch, so the timing of each maximum is a prediction with the
  catalogue's propagated uncertainty. Amplitudes and shapes are the models' predictions, not fitted. Mass assumptions
  by class: RR Lyrae from [M/H]; classical Cepheids from the Bono et al. (2000) mass–luminosity relation; Type II
  Cepheids 0.55 M☉; HADS from the Gaia FLAME mass, radiative (convection off).
- **Eclipsing binaries** (`synthstar/binary.py`): PHOEBE 2 with MESA-atmosphere SEDs, timed and shaped by Gaia's
  eclipsing-binary model (and its NSS orbit when it has one); masses from a main-sequence relation; scaled to Gaia's
  out-of-eclipse G.
- **Stars made from parameters** (`make_stars.py`; `synthstar/sources/`, `python -m synthstar.sources`): defined
  entirely by the star file's keys, with no Gaia input and no dust.
  - Periodic: MESA RSP pulsators (RR Lyrae, Cepheids, Type II Cepheids, HADS) tuned to the given period, at the given L
    (or dust-free absolute M_G) and distance; PHOEBE binaries (W UMa from period, q, inclination, Teffs and fillout
    factor; detached from radii and Teffs) at the given distance, masses from the main-sequence Teff–mass relation
    unless given. Placed by an observation (position, epoch), they are in the reference night
    (so in hdps's reference coadds) as they are on that night, as a real variable would be.
- **Transients** (`synthstar/transients.py`): SEDs against time since t0.
  - sncosmo models (supernova templates built from observed spectra; no physics run), placed at a redshift and peak
    absolute magnitude with sncosmo's default cosmology.
  - Two parametric models written in this repo, not from sncosmo: point-lens microlensing (MESA template source), and
    a dwarf-nova outburst (steady accretion disc
    of changing accretion rate plus the white dwarf; no donor, bright spot or lines).
  - Transients are left out of the reference night, so they are not in hdps's reference coadds.
- **Variables not modelled** (held constant at their Gaia mean): double-mode pulsators, non-radial pulsators,
  long-period, rotational, eruptive and stochastic variables, and any star whose model failed to reproduce its
  catalogue period or eclipses. `make_stars.py --gaia-list RA DEC` gives the reason per star.
- **Synthetic fields** (`[stars] layout = "random"`): every star of a Gaia field set, placed uniformly at random
  within 0.12 deg of the pointing at its own distance, its own dust taken out and the dust in front of it at the new
  position put in; every star constant; the pointing's night and sky.
- **Positions.** Gaia DR3 (fine for one night). Gaia misses faint stars in crowded fields, so unresolved light is
  underestimated there.

### The original field: M3

The first movie (`--field m3`, night 2027-05-05, new Moon) follows four RRab and two RRc in an 84″ block of M3's core
(about 1,550 Gaia sources). Their RSP models use X = 0.75, Z = 0.001, α-enhanced (a09) opacities, 0.68 M☉ and
convection set A. Matching each period with only that mode unstable needed L lowered from M3's horizontal-branch
level (~50 L☉) by 20% for the RRab and 5–10% for the RRc, so the RRab models are ~0.25 mag fainter than the real
stars and their ρ amplitudes ~35% smaller. M3 is at 10.18 kpc (Baumgardt & Vasiliev 2021) with E(B−V) = 0.01
(Harris). The block's 15.3 d Type II Cepheid could not be modelled (every RSP setup crashed or diverged) and is
constant. In the core, 1,424 of 2,702 sources have missing or blended BP−RP and take the median colour of outer-M3
stars at the same G.

## 5. What the panels show

- **CMD:** grey, the constant stars at their true (template) magnitudes; coloured tracks, each followed source's model
  (one cycle, or the night for a transient); coloured points, its **true** model value now, not a measurement.
- **Exposure:** the β/ρ colour composite of this exposure (section 2).
- **Difference:** hdps's SFFT difference image of the tile holding the block (`TileImageOutputTask`), in electrons
  per pixel; a red exclamation mark where hdps raised an alert.
- **Light curves:** hdps's calibrated source-detector magnitude on each exposure (points) and the model (line),
  relative to the night's mean; shaded bands where the source alerted; below, measured − true.
- **Pipeline movie** (`*_pipeline.mp4`): where the readout window falls on the telescope's sensor, the raw frame with
  hdps's detections, hdps's WCS against the true positions, and detections per frame.
- **SED:** each followed source's SED now (solid), and the same SED times the β and ρ system responses at this
  airmass (dotted).
