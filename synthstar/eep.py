"""Equivalent evolutionary points (EEPs): resample tracks so that index e means the same
evolutionary stage on every track (cf. Dotter 2016, MIST).

Segments and the coordinate that defines "the same stage" within each:
  MS      ZAMS -> TAMS          central H fraction, u = 1 - sqrt(Xc / Xc_ZAMS)
                                (denser sampling towards the turnoff hook)
  RGB     TAMS -> RGB tip       fractional path length in (10 log Teff, log L)
                                (subgiant, Hertzsprung gap and giant branch)
  CHeB    ZACHeB -> TACHeB      central He fraction, u = 1 - sqrt(Yc / Yc_ZACHeB)
The He flash / ignition transient between the RGB tip and ZACHeB is skipped:
it is brief, numerically noisy, and essentially never observed.
A segment the track does not finish (e.g. stopped by max_age) is left as NaN.
"""

import numpy as np

SEGMENTS = (("MS", 150), ("RGB", 250), ("CHeB", 150))
N_EEP = sum(n for _, n in SEGMENTS)


def _segment_bounds(d):
    ph, n = d["ph"], len(d["ph"])
    out = {}
    ms_end = np.flatnonzero(ph >= 1)
    if ms_end.size:
        out["MS"] = (0, ms_end[0])
        he = np.flatnonzero(ph == 2)
        h_shell = np.flatnonzero(ph == 1)
        if he.size:
            tip = h_shell[np.argmax(d["log_L"][h_shell])]
            out["RGB"] = (ms_end[0], tip)
            # complete only if the track reached core-He exhaustion
            if d["center_he4"][-1] < 1e-3:
                out["CHeB"] = (he[0], n - 1)
    return out


def _coordinate(d, seg, i0, i1):
    sl = slice(i0, i1 + 1)
    if seg == "MS":
        x = np.clip(d["center_h1"][sl] / d["center_h1"][i0], 0, 1)
        u = 1 - np.sqrt(x)
    elif seg == "CHeB":
        y = np.clip(d["center_he4"][sl] / d["center_he4"][i0], 0, 1)
        u = 1 - np.sqrt(y)
    else:
        step = np.hypot(10 * np.diff(d["log_Teff"][sl]), np.diff(d["log_L"][sl]))
        s = np.concatenate([[0], np.cumsum(step)])
        u = s / s[-1]
    u = np.maximum.accumulate(u)          # tiny non-monotonic wiggles -> monotonic
    return (u - u[0]) / max(u[-1] - u[0], 1e-30)


def resample(d, columns):
    """Return {column: array[N_EEP]} plus 'segment' (index into SEGMENTS) for track dict d."""
    out = {c: np.full(N_EEP, np.nan) for c in columns}
    out["segment"] = np.repeat(np.arange(len(SEGMENTS)), [n for _, n in SEGMENTS])
    bounds = _segment_bounds(d)
    start = 0
    for seg, n in SEGMENTS:
        if seg in bounds:
            i0, i1 = bounds[seg]
            if i1 - i0 >= 5:
                u = _coordinate(d, seg, i0, i1)
                keep = np.concatenate([[True], np.diff(u) > 0])
                target = np.linspace(0, 1, n)
                for c in columns:
                    out[c][start:start + n] = np.interp(target, u[keep], np.asarray(d[c], float)[i0:i1 + 1][keep])
        start += n
    return out
