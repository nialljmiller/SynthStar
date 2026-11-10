"""Measure a finished observation again from the raw frames and hdps's products it left in its hdps/ folder, without
rendering or running hdps: for a run that stopped after hdps (night_movie.py --measure-only), e.g. on an error while
writing its outputs. Then the outputs as usual: photometry, alerts table, catalogues, movie data.

    python tools/remeasure.py outputs/pseudo_real/interactive_m45
    python tools/remeasure.py outputs/pseudo_real/interactive_m45 --bands rho
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main(argv=None):
    import observe
    from argus_image import runconfig

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("observation_dir", type=Path, help="outputs/<category>/<name> of the observation")
    p.add_argument("--bands", nargs="+", help="default: the bands whose raw frames are there")
    args = p.parse_args(argv)

    out = args.observation_dir.resolve()
    obs = runconfig.resolve(runconfig.load(out / "observation.toml"))
    have = [b for b in obs["output"]["bands"] if (out / "hdps" / b / "raw").is_dir()]
    bands = args.bands or have
    missing = sorted(set(bands) - set(have))
    if missing:
        sys.exit(f"no raw frames for {missing} in {out / 'hdps'} (cleaned up after a finished run?)")
    obs["output"]["bands"] = bands
    n_placed = len(json.loads((out / "scene" / "sources.json").read_text()))
    os.chdir(REPO)
    observe.run_engine(obs, out, n_placed, extra=["--measure-only"])
    print(f"measured again: {', '.join(bands)} in {out}")


if __name__ == "__main__":
    main()
