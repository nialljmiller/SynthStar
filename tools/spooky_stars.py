"""Spooky stars: a Halloween field for the live demo. A jack-o-lantern of short-period, high-amplitude variables
(W UMa contact binaries and RRc pulsators) placed among the real stars of a dense Milky Way field (f359, Cassiopeia,
b = -1 deg), with magnitudes spread like the field's, so the exposure is an ordinary star field and the jack-o-lantern
shows in the difference image, where only what changes is left: all of them at maximum light at once mid-night (GLOW),
so the whole shape lights up together. The field's stars brighter than the jack-o-lantern's brightest are left out
around it (BACKGROUND), so no constant star's residual competes with it there.

    python tools/spooky_stars.py          # writes stars/spooky_stars_field/ and examples/observations/spooky_stars.toml
    python observe.py examples/observations/spooky_stars.toml --name spooky_stars --bands beta

The settings other than the field, the night and the stars are interactive_m45.toml's (the live demo's: Gaussian PSF,
the pack's movie data). Each star's distance is set so its mean Gaia G, through the dust on the sightline at that
distance (the 3D map, as observe.py applies it), is the G drawn for it.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from matplotlib.path import Path as Polygon

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUT = REPO / "examples" / "observations" / "spooky_stars.toml"
TEMPLATE = REPO / "examples" / "observations" / "interactive_m45.toml"
RA, DEC = 359.539, 61.214  # f359's centre: 645 Gaia stars in the 9x9 block, 296 of them G < 19
FIELD = "f359_539p61_214_field"  # f359's Gaia stars (make_stars.py), 0.52 deg about the centre
BACKGROUND = "spooky_stars_field"  # FIELD without its stars brighter than G_RANGE[0], the jack-o-lantern's brightest,
# within CLEAR of the centre: in the first run's difference images (2026-10-28, the block) those left the largest
# residuals, G 11-13 ~13000-32000 e- and G 13-14.5 ~500-670 e- each frame, against the jack-o-lantern's 800-2500 e-;
# fainter constant stars ~200 e-
CLEAR = 300.0  # arcsec from the centre: the block's cut-out (+-130", corners 184") and a bright star's residual beyond
# it. Farther out the bright stars stay, so every frame keeps saturated pixels: with no master calibrations (the
# simulation's), hdps's ImageReducer gives a frame its noise model only when a pixel saturates; without one SFFT has no
# Var(D) and AlertVettingTask skips every tile (v3, every star brighter than G 14.5 left out: 0 alerts, 2026-10-29)
HALF = 126.5  # arcsec: half the 9x9 block (253")
SPACING = 9.0  # arcsec between the jack-o-lantern's stars (FWHM ~2.3-3": each star stands alone)
JITTER = 2.0  # arcsec: each star moved at random by up to this, so no line is ruler-straight in the exposure
G_RANGE = (14.5, 17.5)  # mean G, drawn uniformly: among the field's own (126 stars G < 17, 296 G < 19 in the block);
# the first run's 15-18, at random phases, left each frame's residuals faint beside the bright stars' (2026-10-28)
MIN_AMPLITUDE = {"W UMa": 0.5, "RRc": 0.4}  # G mag, peak to peak; RRc reach 0.63, contact binaries 1.36
MAX_PERIOD = 0.42  # days: the night (~10 h) holds a cycle or more of every star
GLOW = 0.14  # days after the first exposure (mid-night of the 6.7 h on 2027-10-31): every star at maximum light at once,
# so the whole jack-o-lantern lights up together in the difference image, then drifts apart (periods 0.21-0.42 d)
SEED = 31


def resample(points, spacing, closed=False):
    """Points every `spacing` arcsec along a polyline (closed: back to its start)."""
    p = np.asarray(points, float)
    if closed:
        p = np.vstack([p, p[:1]])
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    n = max(int(round(s[-1] / spacing)), 1)
    at = np.linspace(0, s[-1], n, endpoint=not closed) if not closed else np.arange(n) * s[-1] / n
    return np.column_stack([np.interp(at, s, p[:, 0]), np.interp(at, s, p[:, 1])])


def filled(polygon, spacing):
    """A shape filled: points along its edge, and a hexagonal grid inside kept clear of them."""
    edge = resample(polygon, spacing, closed=True)
    (x0, y0), (x1, y1) = np.min(polygon, axis=0), np.max(polygon, axis=0)
    rows = np.arange(y0, y1, spacing * np.sqrt(3) / 2)
    grid = np.array([(x + (i % 2) * spacing / 2, y) for i, y in enumerate(rows) for x in np.arange(x0, x1, spacing)])
    inside = grid[Polygon(polygon).contains_points(grid)] if len(grid) else grid
    clear = [q for q in inside if np.min(np.hypot(*(edge - q).T)) >= 0.8 * spacing]
    return np.vstack([edge] + ([np.array(clear)] if clear else []))


def jack_o_lantern():
    """{part: points (arcsec from the block centre, x right, y up)}."""
    t = np.linspace(0, 2 * np.pi, 721)[:-1]
    cy, a, b = -14.0, 100.0, 80.0
    dent = 12 * np.exp(-((t - np.pi / 2) / 0.22) ** 2) - 6 * np.exp(-((t - 3 * np.pi / 2) / 0.3) ** 2)
    body = np.column_stack([a * np.cos(t), cy + b * np.sin(t) - dent])
    top = cy + b - 12
    stem = np.vstack([resample([(-6, top + 2), (-2, top + 14), (4, top + 26)], SPACING),
                      resample([(6, top + 2), (9, top + 13), (14, top + 23)], SPACING)])
    eye = np.array([(-58, 6), (-22, 6), (-40, 38)])
    nose = np.array([(-11, -14), (11, -14), (0, 5)])
    mouth = np.array([(-64, -26), (-40, -34), (-30, -34), (-30, -44), (-18, -44), (-18, -36), (8, -36), (8, -46),
                      (20, -46), (20, -36), (40, -34), (64, -26), (48, -46), (24, -60), (0, -64), (-24, -60),
                      (-48, -46)])
    return {"body": resample(body, SPACING, closed=True), "stem": stem, "left eye": filled(eye, SPACING),
            "right eye": filled(eye * [-1, 1], SPACING), "nose": filled(nose, SPACING), "mouth": filled(mouth, SPACING)}


def pool():
    """The library's short-period, high-amplitude variables: [(name, kind, period, G at 10 pc, amplitude)]."""
    import observe

    out = []
    for d in sorted((REPO / "stars").glob("lib_rrc_*")) + sorted((REPO / "stars").glob("libnm_contact_*")):
        if not ((d / "meta.json").exists() and (d / "sed.h5").exists()):
            continue
        meta = json.loads((d / "meta.json").read_text())
        kind = "RRc" if d.name.startswith("lib_rrc") else meta["kind"]
        if meta["period"] > MAX_PERIOD:
            continue
        G, _, amplitude = observe.catalogue_entry(d / "sed.h5", 0.0, 10.0)
        if amplitude >= MIN_AMPLITUDE[kind]:
            out.append((d.name, kind, float(meta["period"]), G, amplitude))
    return out


def distance_for(name, G_want):
    """The distance (pc) at which the star's mean G, through the sightline's dust at that distance, is G_want."""
    import observe
    from synthstar import dust

    sed, d = REPO / "stars" / name / "sed.h5", 1000.0
    for _ in range(8):
        ebv = float(dust.ebv(RA, DEC, d)[0][0])
        G = observe.catalogue_entry(sed, ebv, d)[0]
        if abs(G - G_want) < 0.005:
            break
        d *= 10 ** ((G_want - G) / 5)
    return d, G, ebv


def field_stars():
    """stars/BACKGROUND/: FIELD's star set without its stars brighter than G_RANGE[0] within CLEAR of the centre (those
    with no Gaia G kept); its other tables as they are (they join on Source). Returns (stars kept, stars left out)."""
    from astropy.table import Table

    src, out = REPO / "stars" / FIELD, REPO / "stars" / BACKGROUND
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(src / "catalogue", out / "catalogue")
    gaia = Table.read(src / "catalogue" / "gaia_dr3_core.ecsv")
    g = gaia["Gmag"]
    ra, dec = np.radians(np.asarray(gaia["RA_ICRS"], float)), np.radians(np.asarray(gaia["DE_ICRS"], float))
    ra0, dec0 = np.radians(RA), np.radians(DEC)
    sep = np.degrees(np.arccos(np.clip(np.sin(dec) * np.sin(dec0) + np.cos(dec) * np.cos(dec0) * np.cos(ra - ra0),
                                       -1, 1))) * 3600
    bright = np.asarray(g.filled(np.nan) if hasattr(g, "filled") else g, float) < G_RANGE[0]  # nan: no G, kept
    keep = ~(bright & (sep < CLEAR))
    gaia[keep].write(out / "catalogue" / "gaia_dr3_core.ecsv", overwrite=True)
    meta = json.loads((src / "meta.json").read_text())
    meta.update(name=BACKGROUND, n_stars=int(keep.sum()),
                params=f"{meta['params']}, without the {int((~keep).sum())} brighter than G {G_RANGE[0]:g} within "
                       f"{CLEAR:g}\" of the centre (tools/spooky_stars.py, from {FIELD})")
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    (out / "summary.md").write_text(f"# {BACKGROUND}\n\n`{FIELD}` without its stars brighter than Gaia G "
                                    f"{G_RANGE[0]:g} within {CLEAR:g}\" of the centre: {int(keep.sum())} of "
                                    f"{len(gaia)} kept. Made by "
                                    f"tools/spooky_stars.py (run it again; do not edit).\n")
    return int(keep.sum()), int((~keep).sum())


def main():
    kept, cut = field_stars()
    rng = np.random.default_rng(SEED)
    parts = jack_o_lantern()
    stars = pool()
    n = sum(len(p) for p in parts.values())
    if n > len(stars):
        raise SystemExit(f"{n} positions, {len(stars)} library stars")
    picks = [stars[i] for i in rng.permutation(len(stars))[:n]]
    template = TEMPLATE.read_text()
    tail = template[template.index("[image]"):]
    lines = [
        "# Spooky stars (user, 2026-10-28): a Halloween field for the live demo. A jack-o-lantern of short-period,",
        "# high-amplitude variables (W UMa contact binaries and RRc pulsators) among the real stars of f359 (Cassiopeia,",
        "# b = -1 deg), their magnitudes spread like the field's: the exposure is an ordinary star field, and the",
        "# jack-o-lantern shows in the difference image. Made by tools/spooky_stars.py (do not edit the [[place]] list by",
        "# hand: run it again). The settings from [image] on are interactive_m45.toml's.",
        "",
        "[field]",
        'name = "spooky_stars"',
        f"ra = {RA}",
        f"dec = {DEC}",
        "block = 9",
        "",
        "[night]",
        "reference_nights = 5  # as interactive_m45.toml: each telescope's sharpest ratchet of 5 nights",
        "reference_ratchets = 1",
        'date = "2027-10-31"  # Halloween with a new Moon (user: no Moon): 8% lit and below the horizon all night at '
        "the site; on 2026-10-31 it is 56% lit and up for 6.2 of the 10.2 dark hours",
        "nights = 1",
        "",
        "[stars]",
        f'background = "{BACKGROUND}"  # f359\'s Gaia stars ({FIELD}) without the {cut} brighter than G '
        f"{G_RANGE[0]:g} (the jack-o-lantern's brightest) within {CLEAR:g}\" of the centre (made by tools/spooky_stars.py): "
        "their residuals would compete with it in the difference image; those farther out keep each frame's saturated "
        "pixels, which give hdps its noise model",
        'layout = "real"',
        "",
        tail.rstrip("\n"),
        "",
        f"# The jack-o-lantern: {n} stars {SPACING:g}\" apart (each moved up to {JITTER:g}\"), x right and y up in "
        f"percent of the block; mean G {G_RANGE[0]:g}-{G_RANGE[1]:g} through the",
        f"# dust at each star's distance; every star at maximum light {GLOW:g} d after the first exposure (an RRc's",
        "# epoch is its maximum, a W UMa's its primary eclipse, a quarter period from a maximum). Library stars, no two",
        "# the same.",
    ]
    k = 0
    for part, points in parts.items():
        lines += ["", f"# {part}: {len(points)} stars"]
        for x, y in points:
            name, kind, period, _, amplitude = picks[k]
            k += 1
            x, y = (np.array([x, y]) + rng.uniform(-JITTER, JITTER, 2)) / (2 * HALF) * 100 + 50
            G_want = rng.uniform(*G_RANGE)
            d, G, ebv = distance_for(name, G_want)
            epoch = GLOW if kind == "RRc" else GLOW + 0.75 * period  # W UMa: maxima a quarter period from the eclipse
            lines += ["[[place]]", f'star = "{name}"  # {kind}, P {period:.3f} d, {amplitude:.2f} mag; G {G:.2f} '
                      f"through E(B-V) {ebv:.2f}", f"x = {x:.2f}", f"y = {y:.2f}", f'epoch = "start+{epoch:.4f}"',
                      f"distance = {d:.0f}"]
    OUT.write_text("\n".join(lines) + "\n")
    count = {kind: sum(p[1] == kind for p in picks) for kind in MIN_AMPLITUDE}
    print(f"stars/{BACKGROUND}: {kept} stars ({cut} brighter than G {G_RANGE[0]:g} within {CLEAR:g}\" left out)")
    print(f"{OUT.relative_to(REPO)}: {n} stars ({', '.join(f'{v} {k}' for k, v in count.items())}), "
          + ", ".join(f"{part} {len(p)}" for part, p in parts.items()))


if __name__ == "__main__":
    main()
