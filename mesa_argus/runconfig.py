"""Observation files: the one file that defines a run. Every parameter, its default, its allowed values and why are in
mesa_argus/schema.py (examples/observations/template.toml is written from it).

    [field] [night] [stars] [argussim.*] [camera] [hdps] [hdps.coadd] [[hdps.tasks]] [lightcurves] [truth] [output]
    [[place]] (any number)

An observation is the file's values over the defaults, with command-line overrides over both; every value is checked
against its allowed values, and the run stops naming each one that is not. observe.py writes the resolved
observation (every parameter) to observation.toml in its folder, so the same data can be made again.
"""
import copy
import tomllib
from pathlib import Path

from mesa_argus import schema as S


def _fill(where, given, params, errors):
    """given over the defaults of params; unknown keys, missing required keys and disallowed values -> errors."""
    keys = {p.key: p for p in params}
    unknown = sorted(set(given) - set(keys))
    if unknown:
        errors.append(f"{where}: unknown keys {unknown}")
    out = {}
    for p in params:
        v = given.get(p.key, p.default)
        if v == S.REQUIRED:
            errors.append(f"{where}.{p.key}: required")
            continue
        out[p.key] = copy.deepcopy(v)
    return out


def _check(where, params, values, resolved, errors):
    for p in params:
        if p.key in values and values[p.key] is not None:
            err = S.check(f"{where}.{p.key}", p, values[p.key], resolved)
            if err:
                errors.append(f"{err}  ({p.why})")


def _tasks(given, errors):
    """[[hdps.tasks]]: the file's chain (or hdps's production chain), each Task's settings over its defaults."""
    defaults = {t["type"]: t["config"] for t in S.TASKS + S.TOY_TASKS}
    chain = given if given is not None else copy.deepcopy(S.TASKS)
    out = []
    for i, t in enumerate(chain):
        kind = t.get("type")
        if kind not in defaults:
            errors.append(f"[[hdps.tasks]] #{i + 1}: type {kind!r}: one of {list(defaults)}")
            continue
        config = t.get("config", {})
        unknown = sorted(set(config) - set(defaults[kind]))
        if unknown:
            errors.append(f"[[hdps.tasks]] {kind}: unknown settings {unknown}")
        values = {**copy.deepcopy(defaults[kind]), **config}
        for k, v in values.items():
            allowed, why = S.TASK_RULES[(kind, k)]
            err = None if v == S.NONE else S.check(f"[[hdps.tasks]] {kind}.{k}", S.P(k, v, "", allowed, why), v)
            if err:
                errors.append(f"{err}  ({why})")
        out.append(dict(type=kind, config=values))
    return out


def _cross_checks(o, errors):
    """Limits that involve more than one parameter."""
    t, cam = o["argussim"]["telescope"], o["camera"]
    if t["central_obstruction_diameter"] >= t["aperture_diameter"]:
        errors.append("[argussim.telescope] central_obstruction_diameter must be below aperture_diameter")
    if t["fullwell"] / cam["gain"] + cam["offset"] > 65535:
        errors.append(f"[camera] gain {cam['gain']}: the full well ({t['fullwell']} e-) must fit the 16-bit raw "
                      f"frame (fullwell / gain + offset <= 65535)")
    sch = o["argussim"]["schedule"]
    if sch["exptime_s"] > sch["track_min"] * 60:
        errors.append("[argussim.schedule] exptime_s must be at most track_min x 60")
    for task in o["hdps"]["tasks"]:
        if task["type"] == "SourceDetectionTask" and o["field"]["window"]:
            px = o["field"]["window"] * S.minipix_arcsec() / plate_scale(o)
            if px < 3 * task["config"]["bg_region_size"]:
                errors.append(f"[field] window {o['field']['window']} minipixes ({px:.0f} px): hdps's source detector "
                              f"needs 3 x bg_region_size ({task['config']['bg_region_size']}) px across")


def plate_scale(o):
    """Arcsec per sensor pixel from [argussim.telescope] (as ArgusSim's TelescopeUnit.plate_scale)."""
    t = o["argussim"]["telescope"]
    return t["pixel_size"] / 1000 / t["focal_length"] * 206264.80624709636


def resolve(observation):
    """The observation with every default filled and every value checked (SystemExit naming every problem)."""
    errors = []
    sections = set(observation) - {"field", "night", "stars", "inject", "argussim", "camera", "image", "hdps",
                                   "lightcurves", "truth", "output", "place"}
    if sections:
        errors.append(f"unknown sections {sorted(sections)}")
    out = {name: _fill(f"[{name}]", observation.get(name, {}), params, errors) for name, params in S.SECTIONS.items()}
    given = observation.get("argussim", {})
    unknown = sorted(set(given) - set(S.ARGUSSIM))
    if unknown:
        errors.append(f"[argussim]: unknown sections {unknown}")
    out["argussim"] = {sub: _fill(f"[argussim.{sub}]", given.get(sub, {}), ps, errors) for sub, ps in S.ARGUSSIM.items()}
    hdps = observation.get("hdps", {})
    unknown = sorted(set(hdps) - {"coadd", "tasks", *[p.key for p in S.HDPS]})
    if unknown:
        errors.append(f"[hdps]: unknown keys {unknown}")
    out["hdps"] = _fill("[hdps]", {k: v for k, v in hdps.items() if k not in ("coadd", "tasks")}, S.HDPS, errors)
    out["hdps"]["coadd"] = _fill("[hdps.coadd]", hdps.get("coadd", {}), S.HDPS_COADD, errors)
    out["hdps"]["tasks"] = _tasks(hdps.get("tasks"), errors)
    out["place"] = [_fill(f"[[place]] {p.get('star')}", p, S.PLACE, errors) for p in observation.get("place", [])]

    # the defaults that depend on other parameters (skipped where those are themselves wrong: reported below)
    try:
        if out["camera"]["gain"] in (None, S.NONE):
            out["camera"]["gain"] = out["argussim"]["telescope"]["fullwell"] / out["camera"]["saturation_adu"]
        from mesa_argus import camera

        for task in out["hdps"]["tasks"]:
            c = task["config"]
            if task["type"] == "ImageReductionTask":
                c["camera_specs"] = c["camera_specs"] or camera.specs()
                c["gain"] = out["camera"]["gain"] if c["gain"] == S.NONE else c["gain"]
            if task["type"] == "ReferenceRetrievalTask" and c["pixscale"] == S.NONE:
                c["pixscale"] = plate_scale(out)
    except (KeyError, TypeError, ZeroDivisionError):
        pass

    for name, params in S.SECTIONS.items():
        _check(f"[{name}]", params, out[name], out, errors)
    for sub, ps in S.ARGUSSIM.items():
        _check(f"[argussim.{sub}]", ps, out["argussim"][sub], out, errors)
    _check("[hdps]", S.HDPS, out["hdps"], out, errors)
    _check("[hdps.coadd]", S.HDPS_COADD, out["hdps"]["coadd"], out, errors)
    for p in out["place"]:
        _check(f"[[place]] {p['star']}", S.PLACE, p, out, errors)
    if not errors:  # the cross-checks need every single value valid
        _cross_checks(out, errors)
    if errors:
        raise SystemExit("observation file:\n  " + "\n  ".join(errors))
    for p in out["place"]:  # unset optional placement keys as None (what observe.py reads; dumps leaves them out)
        for k, v in p.items():
            if v == S.NONE:
                p[k] = None
    return out


def apply_argussim(observation):
    """Set ArgusSim's configuration to the observation's [argussim] values (argus_sim.c, the configuration ArgusSim
    reads); returns the Schedule to use (coverage_exact.Schedule with [argussim.schedule])."""
    from argus_sim import c
    from argus_sim.coverage_exact import Schedule

    for sub, values in observation["argussim"].items():
        if sub == "schedule":
            continue
        target = getattr(c, sub)
        for k, v in values.items():
            setattr(target, k, None if v == S.NONE else v)
    return Schedule(**observation["argussim"]["schedule"])


def output_folder(name, layout):
    """Where an observation's outputs go: outputs/fully_synthetic/<name> (its stars at random positions) or
    outputs/pseudo_real/<name> (real Gaia stars where they are, with any placed stars)."""
    from mesa_argus.config import OUTPUTS

    return OUTPUTS / ("fully_synthetic" if layout == "random" else "pseudo_real") / name


def find_output(name):
    """An existing observation's folder, in either category."""
    from mesa_argus.config import OUTPUTS

    for category in ("pseudo_real", "fully_synthetic"):
        if (OUTPUTS / category / name).is_dir():
            return OUTPUTS / category / name
    raise SystemExit(f"no observation {name!r} in outputs/pseudo_real/ or outputs/fully_synthetic/")


def load(path):
    return tomllib.loads(Path(path).read_text())


def merge(observation, overrides):
    """overrides {"section.key": value, "argussim.telescope.readnoise": value, ...} (or {"place": [...]}, replacing
    the placed stars) applied to the file's values."""
    out = copy.deepcopy(observation)
    for dotted, v in overrides.items():
        if dotted == "place":
            out["place"] = v
            continue
        *path, key = dotted.split(".")
        node = out
        for part in path:
            node = node.setdefault(part, {})
        node[key] = v
    return out


def parse_value(text):
    """A command-line value as TOML (numbers, true/false, "strings", [lists], {tables}); a bare word is a string."""
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text


def dumps(observation, header):
    """The resolved observation as TOML (every parameter)."""
    from mesa_argus.tomlfile import table

    from mesa_argus.schema import SECTIONS

    written = ("field", "night", "stars", "inject", "camera", "image", "lightcurves", "truth", "output")
    if set(SECTIONS) - set(written):  # a section left out is not made again ([image] until 2026-10-26, [inject] until
        raise ValueError(f"runconfig.dumps does not write {sorted(set(SECTIONS) - set(written))}")  # 2026-10-27)
    lines = [f"# {header}", ""]
    for name in ("field", "night", "stars", "inject"):
        lines += table(f"[{name}]", observation[name])
    for sub, values in observation["argussim"].items():
        lines += table(f"[argussim.{sub}]", values)
    lines += table("[camera]", observation["camera"])
    lines += table("[image]", observation["image"])
    lines += table("[hdps]", {k: v for k, v in observation["hdps"].items() if k not in ("coadd", "tasks")})
    lines += table("[hdps.coadd]", observation["hdps"]["coadd"])
    for task in observation["hdps"]["tasks"]:
        lines += ["[[hdps.tasks]]", f"type = \"{task['type']}\""] + table("[hdps.tasks.config]", task["config"])
    for name in ("lightcurves", "truth", "output"):
        lines += table(f"[{name}]", observation[name])
    for p in observation["place"]:
        lines += table("[[place]]", p)
    return "\n".join(lines)
