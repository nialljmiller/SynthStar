"""SED grid for the Argus simulator: five stars spanning the CMD, with every model's SED kept.

Each run writes the interpolated atmosphere SED at every history step (f_lambda at 10 pc),
packed into runs/<track>/seds.h5 together with that step's stellar parameters and
synthetic magnitudes (synthstar.seds).

The two low-mass stars stop at the RGB tip (helium ignition). Following them through the
He flash and horizontal branch costs many hours of run time, and the flash itself is a
numerical transient, not an observable phase.
"""

# stop at He ignition in a degenerate core: L_He runs away at the tip of the RGB
STOP_AT_RGB_TIP = ["power_he_burn_upper_limit = 10d0"]

STARS = [
    # (mass, [M/H], population label, why, extra controls)
    (1.0, 0.0, "Thin disk (solar)", "the Sun: MS -> turnoff -> subgiant -> RGB tip", STOP_AT_RGB_TIP),
    (0.8, -2.0, "Outer halo / M92-M15", "metal-poor halo star: largest [M/H] contrast with the Sun, MS -> RGB tip", STOP_AT_RGB_TIP),
    (3.0, 0.0, "Thin disk (solar)", "late-B star: MS -> Hertzsprung gap -> giant -> core He burning (Fisher anchor)", []),
    (5.0, 0.0, "Thin disk (solar)", "mid-B star: blue loop through the Cepheid instability strip", []),
    (10.0, 0.0, "Thin disk (solar)", "early-B star: upper MS -> red supergiant", []),
]


def tracks(make):
    return [make(mass, feh, make.alpha_sun, population=pop, why=why, kind="base", tag="sed",
                 save_seds=True, extra_controls=extra)
            for mass, feh, pop, why, extra in STARS]
