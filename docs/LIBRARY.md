# The star library

Reusable stars that observations draw from (`stars/lib_*`, `stars/libnm_*`): the MESA library (evolutionary tracks, Cepheids, HADS, RR Lyrae; built on newton by `tools/newton_queue.sh` from `queue/`) and the non-MESA templates (eclipsing and contact binaries, microlensing, dwarf novae, supernovae; built by `tools/nonmesa_library.py`, run on newton as `tools/argus-nonmesa-library.service`). The number of templates per type is coverage, not the abundance of that type on the sky. `[inject]` (README) draws from them for real/bogus training data.

## General Argus stellar library: Newton expansion, 2026-10-19

Goal: reusable intrinsic MESA models from which to synthesize observed Argus fields.
The library's number of templates per type is **not the abundance of that type on the sky**.

### Queue additions

`python tools/expand_library_queue.py` validates without enqueuing. Add `--enqueue`
to publish 19 queue files atomically, without replacing existing RR Lyrae files.
Newton's existing `tools/newton_queue.sh` picks these up after its existing RR
files, using its current worker limits. No second runner is required.

- 62 new evolutionary tracks: 59 ordinary-star tracks and 3 helium-flash/HB
  pilots. Initial mass 0.6--20 Msun; [M/H] -2.5--+0.3. Six ordinary tracks
  form an initial pilot. The existing five `argus_sim_seds` tracks are reused.
- 36 representative Gaia DR3 Cepheids: 17 classical fundamental, 9 classical
  first-overtone, 5 BL Her and 5 W Vir. Occupied log-period (0.1 dex) and
  metallicity (0.5 dex, unknown separately) cells, with common cells first and
  minimum coverage per supported subtype. A relatively bright source in each
  cell is modelled with the existing Gaia distance/dust/composition recipes.
  This is representative coverage, **not random population sampling**.
- 12 HADS pilot models selected from observed ASAS-SN HADS period cells (0.05
  dex, 0.03--0.3 days), with Gaia DR3 identifiers and the same loose declination
  cut. Common period cells plus short/long-period coverage. These inherit the
  existing RSP radiative-only approximation; amplitudes are not a validated
  prediction of the HADS population. Gaia/ASAS-SN class disagreement can change
  the modelled type; missing physical parameters can reject a candidate at build
  time. Use each built object's actual metadata rather than its selection label.
- Manifest / observed catalogue cell counts:
  `data/library/sky_library_20261019.json`. Input catalogue:
  `data/library/gaia_cepheids_population.ecsv` (VizieR I/358/vcep).
  HADS input: `data/library/asassn_hads_population.ecsv` (II/366/catv2021).

The Cepheid selection excludes declinations below -59 degrees, a loose
ever-visible limit for McDonald, not an Argus footprint/cadence selection.
The classical-Cepheid mass-luminosity prescription, fallback metallicities,
Type II mass assumption and zero bolometric-correction approximation are
inherited from `synthstar/gaia.py`; fitted spectra are not exact stellar truth.

### Outputs and their use

Ordinary tracks: `runs/<track>/seds.h5`, also copied into
`stars/lib_evol_<track>/seds.h5` with `track.json` and normal library metadata.
The `MESA evolution track` generator has timing `library`. Its **plural**
`seds.h5` contains many evolutionary ages; it is not a periodic `sed.h5` and
cannot be placed directly into `observe.py`. A sampler must select an intrinsic
stellar state first. Existing periodic-star outputs keep their existing format.

Quality checks: exclude pre-MS unless deliberately modelling a young population;
apply `model/inside_atm_grid`; assess `model/fbol_sed_over_L` per state. The
library wrapper requires at least one post-ZAMS state within the atmosphere grid
and with bolometric closure within 10%, and reports that usable-state count.
Passing that test does not make every row of the track usable. Only ordinary
physical stopping conditions are accepted for newly run tracks; failed tracks
are not advertised as successfully built library objects.

Bulk low-mass tracks stop at helium ignition. The three separate pilots attempt
the flash through core-He burning to cover red clump/HB. They may fail and are
not a validated HB population model. Existing packed tracks are reused as-is.

### Turning coverage into a realistic sky

For a specific pointing, the existing Gaia field supplies observed positions,
fluxes and colours. Preserve its population rather than distributing library
templates equally. Match templates in intrinsic physical parameters/colour and
luminosity, with distance and dust treated consistently.

A population-weighted synthetic sky is a separate follow-on, not a requirement
for this physical coverage grid. That sampler would need spatial and
age-metallicity distributions/SFH for the disk, thick disk, halo and satellites,
binary/variable incidence, evolutionary dwell times, and Argus throughput,
extinction, limiting magnitude, saturation and cadence selection. It should
weight in stellar age, not by adaptive MESA row count, and restrict old
populations to appropriate ages.

Observed Gaia cell counts are retained as provenance, not promoted to unbiased
all-sky occurrence rates. Gaia selection varies with magnitude and crowding:
https://www.cosmos.esa.int/web/gaia/dr3-data-gaps

### Completeness pilot queue: 2026-10-21

The next queue batches broaden physical coverage; their template counts are not
intended to reproduce the IMF or event rates.

- Seven lower-main-sequence MESA tracks: 0.45--0.8 Msun across solar to
  metal-poor compositions. The 0.45 Msun case approaches the current Kurucz
  floor; spectra below about 3500 K remain unsupported.
- Three 1.5, 2, and 3 Msun solar-composition tracks continue through core-He
  burning to log L/Lsun = 3.8 as AGB / LPV structural pilots. They save restart
  photos every 500 models and retry from the latest photo for up to four total
  attempts. These tracks do not produce Mira pulsation cycles; cool AGB
  atmospheres and a validated LPV pulsation model are still needed.
- Twelve PHOEBE cases: eight detached systems spanning cool through B-type
  components, low mass ratio, near-Roche geometry, evolved radii and eccentric
  orbits, plus four W UMa contact systems spanning period, mass ratio, primary
  temperature and fillout. Semidetached systems remain unsupported.

Queue batches, also kept under `queue/` on Newton:
`sky_20261021_10_faint_star_tracks.toml`,
`sky_20261021_20_lpv_checkpoint_tracks.toml`, and
`sky_20261021_30_phoebe_binaries.toml`.

### Remaining physical coverage gaps

- Cool M dwarfs below the Kurucz atmosphere range: the new track pilot reaches
  0.45 Msun near the grid edge, but cooler spectra need appropriate atmospheres;
  do not silently clamp their spectra.
- White-dwarf cooling spectra; cool TP-AGB atmospheres and LPV pulsation cycles;
  nonradial/multimode pulsators; rotational spots and flares.
- Semidetached and compact-companion binaries, with irradiation, spots and
  mass-transfer physics beyond the current detached/contact PHOEBE models.
- Optional population weights and selection validation, if a realistic synthetic
  star-count distribution is later required. They are separate from model-family
  coverage.

## Non-MESA sky-library coverage: 2026-10-20

The separate non-MESA library contains **5,584 candidate templates**, not 5,584
prevalidated physical systems or an astrophysical population sample. Accepted
objects live in `stars/libnm_*/`, in the existing `sed.h5` / `star.toml` format.
The original MESA queue and its four-attempt policy are unchanged.

### Coverage

| Family | Candidates | Main axes |
|---|---:|---|
| Detached main-sequence binaries | 512 | M1 0.65--12 Msun, M2 0.65--M1; empirical solar-metallicity Teff/radius relation; circular/eccentric orbits; eclipse, grazing and lower-inclination controls |
| Detached evolved binaries | 256 | Coeval components from existing solar-metallicity MESA SED tracks; evolved primaries 3/5/10 Msun initially; lower-mass companions; period <=3000 d |
| W UMa contact binaries | 512 | P 0.22--1 d; q 0.12--1; primary Teff 4400--7500 K; T2/T1 0.88--1.02; fillout 0.02--0.8; inclinations |
| Microlensing | 1024 | tE 0.01--500 d; u0 0.001--3; point/finite source rho 1e-5--0.1; fs 0.01--1; intrinsic BP-RP -0.3--2.3; baseline 8--25 AB |
| Dwarf novae | 512 | WD mass 0.5--1.3 Msun; T 7000--50000 K; disc radius 0.1--0.8 Rsun; accretion, rise, plateau, decay, inclination and distance |
| sncosmo template diversity | 2448 | 204 latest registered source names x 12 redshift/peak-M/host-dust combinations; z 0.001--0.3; host E(B-V) 0--0.7 |
| Additional SALT shape/colour | 320 | 64 samples for each of 5 SALT source names; x1 -3--3; c -0.25--0.4; correlated reference magnitude plus scatter, redshift and host dust |

Ranges are design bounds; deterministic scrambled Sobol samples generally fall
inside the endpoints. Extra dimensions without an implemented physical model
are not varied arbitrarily. The eight registered Pop III sources are explicitly
excluded: the high-redshift/IGM treatment required for their optical observation
is not in the current generator. All included source names, pinned versions,
parameters, references and exclusions are in `data/library/nonmesa_20261020.json`.

The default SNEMO/SUGAR/etc. template is included where supported, but this is
not a dense unconstrained grid of their higher-dimensional empirical basis
coefficients. V19 corrected/uncorrected sources and multiple templates of one
subtype are alternative descriptions, not independent abundance counts.

### Physics and quality limits

Detached components are within both Eggleton Roche lobes **at periastron**.
Evolved pairs share age and metallicity from existing tracks; the main-sequence
grid uses the existing Pecaut--Mamajek relation rather than arbitrary independent
temperatures, masses and radii. Contact models receive a rough MS-density
prefilter, then PHOEBE's actual geometry checks. All accepted binary components
must be inside the supported atmosphere grid. Metal-poor binaries and a complete
evolved mass/age grid await the expanded MESA tracks.

PHOEBE currently uses blackbody intensities, manual/default limb darkening and no
irradiation, with atmosphere SEDs weighted by the existing BP/RP mesh method.
There are no spots, mass transfer, semi-detached systems or accretion luminosity
in these binary generators. Dense parameter coverage does not remove those
approximations. Full orbits retain the existing 401 phase samples / 3000-triangle
mesh setting; very narrow eclipses still need cadence/resolution validation.

Microlensing is a point lens with a uniform finite source; blends share the
template colour, without binary lenses, parallax or limb darkening. Dwarf-nova
outbursts are the existing variable steady-disc + blackbody WD approximation,
not a solved disc-instability model; donor/spot/line and eclipsing-disc physics
are absent. GRB afterglows were previously removed from the current generator;
an old stored GRB model does not mean an active GRB generator is available.

All candidates use existing source generators. A model must have finite,
ordered axes, positive flux and full Argus optical wavelength coverage. Small
negative samples in noisy empirical SN spectra are retained and counted, never
silently clipped; appreciably negative broad optical integrals are rejected.
At template phase edges, a contiguous valid window around optical maximum may
be retained if it spans at least half of the supplied phase interval. The
original bounds and trimming flag are recorded in the HDF5 attributes, and
accepted bounds in the quality record; no flux is clipped or extrapolated.
No extrapolation is used to make an unsupported sncosmo wavelength range fit.
Coverage counts and validated completions are reported separately.

Builds happen in a staging area. Only models passing validation are moved into
`stars/`; failures and logs stay under `outputs/nonmesa_library_20261020/`.
Each forward model has a 15-minute budget. Failed candidates are recorded and
skipped on resume, rather than being retried forever. Network/source failures
may require explicit retry after fixing the underlying issue.

### Resources and operation on Newton

Newton has 32 logical CPUs; MESA was already using about 30. Free RAM alone
cannot speed CPU-bound models. The new independent service has three isolated
single-thread workers: up to two PHOEBE jobs and one transient job, sharing a
total two-core CPU quota. This fills otherwise idle capacity when one job waits
for data. The transient worker retains quality-filtered stellar templates in
RAM instead of reloading them for every microlensing event. They run concurrently
with MESA. Pilot runs use one worker per lane; the full run waits on the same
exclusive lock until the pilots finish, then resumes all remaining candidates.

- `CPUQuota=200%`: at most two CPU cores' worth in aggregate.
- `MemoryHigh=24G`, `MemoryMax=28G`; `Nice=10`, `IOWeight=10`.
- PHOEBE multiprocessing/MPI and BLAS/OpenMP threading are disabled in these
  workers, preventing nested oversubscription.
- New jobs pause when system-available memory is below 8 GiB or free disk below
  100 GiB. The output budget is tens of GB, not a second copy of the whole sky.
- Resume is per-model; the single-runner lock prevents duplicate dispatch.

Newton's configured Python is **`/usr/local/bin/python3` (3.13)**, which has the
science packages. `/usr/bin/python3` is a different interpreter without them.

```bash
cd ~/MESA_Argus
python3 tools/nonmesa_library.py status
systemctl --user status argus-nonmesa-library
journalctl --user -u argus-nonmesa-library -n 30
```

Definitions are also exported as normal star TOMLs in
`examples/stars/nonmesa_20261020/`. `tools/nonmesa_grid.py` reproduces the manifest
and refuses to replace it with a changed design. The systemd unit source is
`tools/argus-nonmesa-library.service`.

These template counts are **not occurrence rates**. A sky sampler still needs
binary/transient incidence, correlated physical population priors, distances,
extinction, event epochs, instrument noise, saturation/depth and cadence. This
library broadens supported phenomenology; it does not cover every physically
possible transient or binary system.
