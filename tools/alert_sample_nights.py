"""Observation files for an alert-broker sample: nights run as Argus would run them, with the real/bogus model loaded.

Each (field, date) of a plan CSV (examples/alert_sample/<sample>.csv: field,date,phase,machine) becomes
<out>/<sample>_<field>_<YYYYMMDD>.toml: the field's real Gaia stars for one night, both bands, every exposure, hdps's
production chain with the bright-star mask off, and AlertVettingTask with the model bundle loaded at score_cut (an
operating point of the bundle, hdps's convention: "fa10"), so hdps sends exactly the packets production would
(emit_all_candidates only adds the below-cut candidates to the alert catalogue, for the truth table and diagnostics;
it adds no packets). [inject] stars, the one setting the sample's realism depends on, from --inject. No movies.

A plan's nights must be ones the model never trained on and that did not set its cut (tools/train_rb.py's training
and calibration nights): the sample is then a fair draw of what the model sends. Run each file from the repository
(the bundle path is relative to it) with hdps's alert schema of the sample (PYTHONPATH, see the switch steps), then
package with tools/alert_sample.py.

    python tools/alert_sample_nights.py examples/alert_sample/v1.csv --bundle data/rb/bundle_v1 --inject 200
"""
import argparse
import csv
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def field_of(name):
    """RA, Dec from a field set's name (f294_784m11_623_field -> 294.784, -11.623)."""
    m = re.match(r"f(\d{3})_(\d{3})([pm])(\d{2})_(\d{3})_field$", name)
    if not m:
        raise SystemExit(f"{name}: not a field set name (fNNN_NNN[pm]NN_NNN_field)")
    return float(f"{m[1]}.{m[2]}"), float(f"{m[4]}.{m[5]}") * (1 if m[3] == "p" else -1)


def observation(sample, row, inject, seed, bundle, cut, stride, note):
    from synthstar import schema as S

    field, date = row["field"], row["date"]
    ra, dec = field_of(field)
    name = f"{sample}_{field.removesuffix('_field')}_{date.replace('-', '')}"
    chain = []
    for task in S.TASKS:
        chain.append(f'[[hdps.tasks]]\ntype = "{task["type"]}"')
        if task["type"] == "AlertVettingTask":
            chain[-1] += (f'\n[hdps.tasks.config]\nmodel_bundle = "{bundle}"\nscore_cut = {cut!r}'
                          f"\nemit_all_candidates = true").replace("'", '"')
    import textwrap

    head = (f'Alert-broker sample "{sample}" (tools/alert_sample_nights.py): {field} on the night of {date} '
            f'({row.get("phase", "")} Moon), as Argus would run it: every exposure, both bands, bright-star mask off '
            f'(hdps\'s production config), the real/bogus model {bundle} loaded at score_cut "{cut}". {note}')
    return name, "\n".join("# " + line for line in textwrap.wrap(head, 116)) + f'''
[field]
name = "{name}"
ra = {ra}
dec = {dec}

[night]
date = "{date}"
nights = 1

[stars]
background = "{field}"
layout = "real"

[inject]
number = {inject}
seed = {seed}

[output]
movie = false
pipeline_movie = false
movie_data = false
lightcurves = false
catalogues = false
stride = {stride}

''' + "\n\n".join(chain) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("plan", type=Path, help="CSV: field,date[,phase,machine]")
    p.add_argument("--sample", help="the sample's name (default: the plan's file name, e.g. v1 -> alert_sample_v1)")
    p.add_argument("--bundle", required=True, help="the real/bogus bundle, relative to the repository")
    p.add_argument("--cut", default="fa10", help="score_cut: an operating point of the bundle, or a number")
    p.add_argument("--inject", type=int, required=True, help="library stars injected per readout window")
    p.add_argument("--stride", type=int, default=1, help="every Nth exposure (production sends every one)")
    p.add_argument("--seed", type=int, default=1000, help="first [inject] seed (one per night, in plan order)")
    p.add_argument("--out", type=Path, default=REPO / "examples" / "observations" / "alert_sample")
    args = p.parse_args(argv)
    sample = args.sample or f"alert_sample_{args.plan.stem}"
    try:
        cut = float(args.cut)
    except ValueError:
        cut = args.cut
    sys.path.insert(0, str(REPO / "tools"))
    from make_training_data import up_that_night  # observe.py's rule: a night the field is up >= 3 h

    note = (f"{args.inject} library stars injected over the readout window (a training-style density, far above the "
            f"variables and transients Argus would see there)." if args.inject else "Nothing injected.")
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.plan) as f:
        rows = [r for r in csv.DictReader(line for line in f if not line.startswith("#"))]
    for i, row in enumerate(rows):
        ra, dec = field_of(row["field"])
        if not up_that_night(ra, dec, row["date"]):
            continue
        name, text = observation(sample, row, args.inject, args.seed + i, args.bundle, cut, args.stride, note)
        (args.out / f"{name}.toml").write_text(text)
        print(f"{args.out.relative_to(REPO) if args.out.is_relative_to(REPO) else args.out}/{name}.toml "
              f"({row.get('phase', '')}, {row.get('machine', '')})")


if __name__ == "__main__":
    main()
