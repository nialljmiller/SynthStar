"""Initial composition for a given [M/H] with linear helium enrichment."""

# Must match the colors module's z_over_x_ref default (GS98): [M/H] = log10((Z/X) / ZX_SUN)
ZX_SUN = 0.0169 / (1.0 - 0.0169 - 0.2485)
# Primordial helium (Planck 2015 BBN, as adopted by MIST)
Y_P = 0.2485


def from_feh(feh, y_sun, z_sun):
    """(X, Y, Z) for [M/H] = feh, with Y = Y_P + (dY/dZ) Z anchored on the calibrated Sun."""
    dydz = (y_sun - Y_P) / z_sun
    zx = ZX_SUN * 10.0**feh
    z = zx * (1.0 - Y_P) / (1.0 + zx * (1.0 + dydz))
    y = Y_P + dydz * z
    return 1.0 - y - z, y, z


def solar_zy(y):
    """(Z, Y) at the solar surface Z/X for a given Y (used by the solar calibration)."""
    return ZX_SUN * (1.0 - y) / (1.0 + ZX_SUN), y
