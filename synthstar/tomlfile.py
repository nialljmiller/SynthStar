"""Writing TOML (Python reads it with tomllib but has no writer): enough for star and observation files."""


def value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(value(x) for x in v) + "]"
    return "{" + ", ".join(f"{key(k)} = {value(x)}" for k, x in v.items()) + "}"  # a dict


def key(k):
    return k if k.replace("_", "").replace("-", "").isalnum() else value(k)


def table(header, d):
    """'[header]' (or '[[header]]') and its key = value lines; None values are left out."""
    return [header] + [f"{key(k)} = {value(v)}" for k, v in d.items() if v is not None] + [""]
