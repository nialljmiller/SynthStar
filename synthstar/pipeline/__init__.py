"""What MESA_ARGUS adds after hdps's pipeline (synthstar.hdps_worker runs hdps itself on the raw frames):

    lightcurves  every frame's hdps detections linked by position into objects: a catalogue (with the nearest
                 ATLAS-Refcat2 star, from hdps's loader) and one light curve per object. hdps has no light-curve
                 product yet; this is the simplest positional linker, to be replaced when it does.
    truth        which simulated star each detection is: for scoring only; nothing in hdps's chain reads it.
"""
from . import lightcurves, truth
