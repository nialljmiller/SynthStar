"""Star generators. make_stars.py builds each [[star]] / [[set]] of a star file with the generator it names.

A generator has a name, its timing ("periodic", "transient" or "set"), the engine that computes it, its keys
{key: (default, note)} (REQUIRED: no default) and a function make(folder, keys, name) that writes the star into its
folder (sed.h5, or catalogue/ for a set) and returns its metadata (kind, params, reference, ...).

To add a generator: write it in a module here and add the module to MODULES below.

    python -m synthstar.sources              # every generator and its keys
    python -m synthstar.sources "W UMa"      # one generator
"""
from dataclasses import dataclass
from typing import Callable

REQUIRED = "required"


@dataclass
class Generator:
    name: str
    timing: str  # periodic, transient or set
    engine: str  # MESA RSP, PHOEBE, sncosmo, ...
    about: str
    keys: dict  # {key: (default, note)}
    make: Callable  # make(folder, keys, name) -> metadata
    reference: str = ""
    prepare: Callable = None  # prepare(keys, name): one star at a time, before any make (network queries)
    parallel: bool = True  # make() may run alongside other stars


def all_generators():
    from . import agb, binaries, evolution, gaia, pulsators, transients

    MODULES = (transients, pulsators, binaries, gaia, evolution, agb)
    return {g.name: g for module in MODULES for g in module.GENERATORS}


def check(entry):
    """The generator an entry names, and its keys with defaults filled. Unknown or missing keys stop the run."""
    generators = all_generators()
    if entry.get("generator") not in generators:
        raise SystemExit(f"{entry.get('name')}: unknown generator {entry.get('generator')!r}; there are {sorted(generators)}")
    gen = generators[entry["generator"]]
    unknown = set(entry) - {"name", "generator"} - set(gen.keys)
    missing = [k for k, (default, note) in gen.keys.items() if default == REQUIRED and k not in entry]
    if unknown or missing:
        raise SystemExit(f"{entry['name']} ({gen.name}): unknown keys {sorted(unknown)}, missing keys {missing}")
    keys = {k: default for k, (default, note) in gen.keys.items()}
    keys.update({k: v for k, v in entry.items() if k not in ("name", "generator")})
    return gen, keys


def describe(gen):
    lines = [f"{gen.name}  ({gen.timing}; {gen.engine})", f"    {gen.about}"]
    if gen.reference:
        lines.append(f"    reference: {gen.reference}")
    for k, (default, note) in gen.keys.items():
        lines.append(f"    {k:12s} {note} ({default if default == REQUIRED else f'default {default!r}'})")
    return "\n".join(lines)
