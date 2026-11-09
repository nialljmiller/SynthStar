"""Write the companion "solved" files that hdps's PresolvedWCSTask reads, from a folder of raw frames that already
carry a WCS in their primary header (the ones the engine stamps when the chain has TrueCalibrationTask).

hdps's PresolvedWCSTask (hdps.orchestrate.tasks.astrometry.PresolvedWCSTask) skips the refcat/solve chain by loading
a pre-computed WCS from a companion file:

    <solved_dir>/<strip_fits_ext(filename)><suffix>.fits   (default suffix "_solved")

where ``filename`` is the raw frame's basename (hdps.core.images sets metadata["filename"] = os.path.basename(path)).
It builds the WCS with ``astropy.wcs.WCS(hdul[0].header)`` of that companion. So for each raw frame we copy its WCS
cards into a fresh companion primary header. (The task also *optionally* harvests a zero point and seeing from
CROSSMATCH_SOURCES / SOURCES tables in the solved file; both are wrapped in try/except, so a WCS-only companion is
valid — coadd_magzp / coadd_seeing are simply not harvested from it.)

Zero simulator dependency: astropy only. Run it with -I so nothing is imported from the frame directory.

    python -I tools/hdps_presolved_wcs.py <raw_dir> <solved_dir> [--suffix _solved] [--pattern '*.fits']
"""
import argparse
from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS


def strip_fits_ext(name: str) -> str:
    """PresolvedWCSTask._strip_fits_ext, replicated (longest extension first)."""
    for ext in (".fits.fz", ".fits", ".fit"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def companion_path(solved_dir: Path, raw_name: str, suffix: str) -> Path:
    """The path PresolvedWCSTask._solved_path(filename, ...) computes in solved_dir mode."""
    return solved_dir / f"{strip_fits_ext(raw_name)}{suffix}.fits"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_dir", type=Path, help="folder of raw FITS frames whose headers carry a WCS")
    ap.add_argument("solved_dir", type=Path, help="folder to write the companion <stem><suffix>.fits files into "
                    "(must NOT be the raw dir or under it: hdps globs the raw dir recursively for *.fits)")
    ap.add_argument("--suffix", default="_solved", help="companion suffix (PresolvedWCSTask default '_solved')")
    ap.add_argument("--pattern", default="*.fits", help="glob for the raw frames")
    args = ap.parse_args(argv)

    solved_dir = args.solved_dir.resolve()
    solved_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(args.raw_dir.glob(args.pattern))
    if not frames:
        raise SystemExit(f"no frames matching {args.pattern!r} in {args.raw_dir}")

    written, skipped = 0, []
    for frame in frames:
        with fits.open(frame) as hdul:
            header = hdul[0].header
            wcs = WCS(header)
            # A frame without celestial WCS cards has no (RA,Dec) axes: that is a frame the engine did not stamp.
            if not wcs.has_celestial:
                skipped.append(frame.name)
                continue
            wcs_header = wcs.to_header(relax=True)  # the clean set of WCS cards PresolvedWCSTask will re-read
        out = companion_path(solved_dir, frame.name, args.suffix)
        phdu = fits.PrimaryHDU()
        phdu.header.update(wcs_header)
        phdu.writeto(out, overwrite=True)
        written += 1

    print(f"wrote {written} companion solved file(s) -> {solved_dir} (suffix {args.suffix!r})")
    if skipped:
        print(f"skipped {len(skipped)} frame(s) with no celestial WCS: {skipped[:5]}")


if __name__ == "__main__":
    main()
