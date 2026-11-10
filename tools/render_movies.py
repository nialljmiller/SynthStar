"""Draw an observation's movies after the run, from the movie data it saved (output.movie_data): the field movie and
the pipeline movie for each band, as observe.py would have drawn them, without rendering or running hdps again. The
engine rebuilds the scene as in the run (night_movie.py --render-only), then draws from
<name>_night_<band>.movie.joblib.

    python tools/render_movies.py outputs/pseudo_real/rb_f065_830p44_237_20261023
    python tools/render_movies.py outputs/pseudo_real/<name> --bands rho --no-pipeline
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
    p.add_argument("observation_dir", type=Path, help="outputs/<category>/<name> of a finished observation")
    p.add_argument("--bands", nargs="+", help="default: the bands with movie data")
    p.add_argument("--no-pipeline", action="store_true", help="the field movie only")
    args = p.parse_args(argv)

    out = args.observation_dir.resolve()
    obs = runconfig.resolve(runconfig.load(out / "observation.toml"))
    name = obs["field"]["name"]
    have = [b for b in obs["output"]["bands"] if (out / f"{name}_night_{b}.movie.joblib").exists()]
    bands = args.bands or have
    missing = sorted(set(bands) - set(have))
    if missing:
        sys.exit(f"no movie data for {missing} in {out} (the run had output.movie_data = false?)")
    obs["output"].update(bands=bands, movie=True, pipeline_movie=not args.no_pipeline)
    n_placed = len(json.loads((out / "scene" / "sources.json").read_text()))
    os.chdir(REPO)
    observe.run_engine(obs, out, n_placed, extra=["--render-only"])
    for b in bands:
        print(f"{b}: {out / f'{name}_night_{b}.mp4'}" + ("" if args.no_pipeline else f", {name}_night_{b}_pipeline.mp4"))


if __name__ == "__main__":
    main()
