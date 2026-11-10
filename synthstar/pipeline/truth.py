"""Which simulated star each detection is. For scoring the pipeline only: no measuring stage reads this.

The simulated stars in order of expected flux, brightest first, each take the nearest detection not yet taken within
match_radius pixels; a detection belongs to at most one star. A star with no detection was too faint, or merged
into a brighter neighbour's detection.
"""
import numpy as np

SETTINGS = dict(match_radius=2.0)


def associate(det_x, det_y, star_x, star_y, star_flux, radius):
    """For each star, the index of its detection, or -1."""
    match = np.full(len(star_x), -1)
    taken = np.zeros(len(det_x), bool)
    for i in np.argsort(-np.nan_to_num(star_flux, nan=0.0), kind="stable"):
        d = np.hypot(det_x - star_x[i], det_y - star_y[i])
        d[taken] = np.inf
        j = int(np.argmin(d)) if len(d) else -1
        if j >= 0 and d[j] <= radius:
            match[i] = j
            taken[j] = True
    return match


def label(catalogue, star_x, star_y, star_flux, cfg):
    """Adds `star` (the simulated star's index, -1 for none) to the catalogue; returns each star's detection index."""
    radius = cfg.get("match_radius", SETTINGS["match_radius"])
    star_of = associate(np.asarray(catalogue["xcentroid"], float), np.asarray(catalogue["ycentroid"], float),
                        star_x, star_y, star_flux, radius)
    star = np.full(len(catalogue), -1)
    star[star_of[star_of >= 0]] = np.nonzero(star_of >= 0)[0]
    catalogue["star"] = star
    return star_of


def label_candidates(candidates, star_x, star_y, star_flux, followed, labels, cfg):
    """Adds `star` (the nearest simulated star within match_radius, -1 for none) and `truth` (its label if it is a
    followed star: a variable or transient; 'star' for another simulated star; '' for nothing) to the candidates."""
    radius = cfg.get("match_radius", SETTINGS["match_radius"])
    x, y = np.asarray(candidates["x"], float), np.asarray(candidates["y"], float)
    star = np.full(len(candidates), -1)
    for i in range(len(candidates)):
        d = np.hypot(star_x - x[i], star_y - y[i]) - 1e-9 * np.nan_to_num(star_flux)  # ties: the brighter
        k = int(np.argmin(d))
        star[i] = k if d[k] <= radius else -1
    candidates["star"] = star
    candidates["truth"] = [labels[k] if k in followed else ("star" if k >= 0 else "") for k in star]


def label_alerts(ra, dec, star_ra, star_dec, star_flux, followed, transient, labels, radius_arcsec=2.0):
    """What the simulation put where each alert is (for scoring and for training a real/bogus model): the nearest
    simulated star within radius_arcsec (ties: the brighter) and its class: 'transient' (a placed transient),
    'variable' (a followed variable), 'constant' (any other simulated star: the alert is a subtraction residual),
    'none' (nothing simulated there). Returns dict of arrays: truth_star, truth_label, truth_class,
    truth_sep_arcsec."""
    ra, dec = np.asarray(ra, float), np.asarray(dec, float)
    star_ra, star_dec = np.asarray(star_ra, float), np.asarray(star_dec, float)
    n = len(ra)
    star = np.full(n, -1)
    sep_out = np.full(n, np.nan)
    for i in range(n):
        sep = np.hypot((star_ra - ra[i]) * np.cos(np.radians(dec[i])), star_dec - dec[i]) * 3600.0
        sep = sep - 1e-9 * np.nan_to_num(star_flux)  # ties: the brighter
        k = int(np.argmin(sep)) if len(sep) else -1
        if k >= 0 and sep[k] <= radius_arcsec:
            star[i], sep_out[i] = k, sep[k]
    cls = np.array(["transient" if k in transient else "variable" if k in followed else "constant" if k >= 0
                    else "none" for k in star])
    label = np.array([labels.get(k, "") if k >= 0 else "" for k in star])
    return dict(truth_star=star, truth_label=label, truth_class=cls, truth_sep_arcsec=sep_out)
