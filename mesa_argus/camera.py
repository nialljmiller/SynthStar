"""The camera: a simulated exposure written as a raw Argus camera frame, the file hdps's pipeline reads
(hdps.core.images.FullFrameImage.from_fits).

toy: the simplest camera that hdps can process, to be replaced by a model of the real Argus camera. Pixels are
uint16 ADU with BZERO 32768 (the raw layout hdps's loader reads on its fast path); ADU = electrons / CAMGAIN +
CAMOFFST, rounded and clipped to 0..65535. ArgusSim defines the detector in electrons only (no gain, bias level or
overscan), so: the gain puts ArgusSim's full well at hdps's raw saturation level (hdps.calibration.ImageReducer
saturation_value 58000 ADU), 21657 / 58000 = 0.373 e-/ADU (the prototype QHY600M's is 0.33); offset 0; no overscan
columns, no bias or dark structure, no flat-field pattern (ImageReductionTask then runs with this gain and no masters). The
header carries what a real frame's does (hdps's AEGIS test frames): UTCSTART, UTCEND, RATCHNUM, IMGTYPE, TARGET, TARGRA
(hours), TARGDEC, EXPTIME, AEXPTIME, CAMID, CAMMOD, CAMGAIN, CAMOFFST, and the readout window XORIGIN, YORIGIN, XSIZE,
YSIZE (a window on the sensor: the simulated region). Plus, not in real headers, FILTER (the band; real frames get it
from the camera's telescope configuration), TELID and SUBARRAY (ArgusSim's telescope number and its subarray: one
camera, filter and focuser per telescope, none numbered on its own) and TELCONF = TELID (the hardware code of hdps's
frame key, hdps.core.frame_key; real frames get it from the telescope-configuration registry).
"""
import numpy as np
from argus_sim import c as ARGUS
from astropy.io import fits

SATURATION_ADU = 58000.0  # hdps.calibration.ImageReducer's default saturation_value
GAIN = ARGUS.telescope.fullwell / SATURATION_ADU  # e-/ADU
OFFSET = 0  # ADU
MODEL = "MESA_ARGUS toy camera"
# The readout: how pixel rows and columns lie on ArgusSim's focal-plane axes (ArgusSim does not define it). Columns
# along the long axis (et), rows against the short axis (er): the parity of real Argus frames (an AEGIS test frame's
# solved WCS, hdps_astrometry/tests/data/frames, has det(PC) > 0), which hdps's linear WCS solver assumes
# (quad_align parity (1, 1), hdps/astrometry/linear_solver.py).
READOUT_Y_SIGN = -1


def specs():
    """The camera_specs ImageReductionTask takes for this camera: a serial number without overscan geometry."""
    return {"serial_number": MODEL}


def camera_id(tel_id):
    return f"ARGUSSIM-{int(tel_id):04d}"


def write_raw(path, electrons, meta):
    """electrons: the exposure (sensor pixels, electrons). meta: utc_start, utc_end (isot), exptime, ratchet, tel_id,
    subarray, target_ra, target_dec (deg), band, x_origin, y_origin; optional cards: more header cards (the
    simulation's calibration, mesa_argus.true_calibration). Writes a raw frame; returns its path."""
    adu = np.clip(np.round(np.asarray(electrons, float) / GAIN + OFFSET), 0, 65535).astype(np.uint16)
    header = fits.Header()
    header["UTCSTART"] = meta["utc_start"]
    header["UTCEND"] = meta["utc_end"]
    header["RATCHNUM"] = str(meta["ratchet"])
    header["IMGTYPE"] = "sci"
    header["TARGET"] = "MESA_ARGUS"
    header["TARGRA"] = meta["target_ra"] / 15.0  # hours, as the telescope control system writes it
    header["TARGDEC"] = meta["target_dec"]
    header["EXPTIME"] = meta["exptime"]
    header["AEXPTIME"] = meta["exptime"]
    header["CAMID"] = camera_id(meta["tel_id"])
    header["CAMMOD"] = MODEL
    header["CAMGAIN"] = GAIN
    header["CAMOFFST"] = OFFSET
    header["XORIGIN"], header["YORIGIN"] = int(meta["x_origin"]), int(meta["y_origin"])
    header["XSIZE"], header["YSIZE"] = adu.shape[1], adu.shape[0]
    header["FILTER"] = meta["band"]
    header["TELID"] = (int(meta["tel_id"]), "ArgusSim telescope number")
    header["SUBARRAY"] = (int(meta["subarray"]), "ArgusSim subarray number")
    header["TELCONF"] = (int(meta["tel_id"]), "frame-key hardware code: the telescope number")
    for key, value in meta.get("cards", {}).items():
        header[key] = value
    fits.PrimaryHDU(adu, header=header).writeto(path, overwrite=True)  # uint16: astropy writes BITPIX 16, BZERO 32768
    return path
