#!/usr/bin/env python3
"""Make an existing star library (built by the old `mesa_argus` package) indistinguishable from a fresh `synthstar`
output: rewrite the cosmetic provenance names in meta.json / summary.md. The sed.h5 DATA is format-identical and is
NEVER touched. Idempotent. Usage: migrate_library.py <stars_dir> [--apply]  (default: dry run)."""
import sys, re
from pathlib import Path

SUBS = [("mesa_argus", "synthstar"), ("MESA_Argus", "SynthStar"), ("MESA_ARGUS", "SynthStar")]
root = Path(sys.argv[1]); apply = "--apply" in sys.argv
changed = scanned = 0
for star in sorted(root.glob("*/")):
    for fname in ("meta.json", "summary.md"):
        f = star / fname
        if not f.exists(): continue
        scanned += 1
        s = f.read_text(); o = s
        for a, b in SUBS: o = o.replace(a, b)
        if o != s:
            changed += 1
            if apply: f.write_text(o)
            elif changed <= 3:
                for a, _ in SUBS:
                    for ln in s.splitlines():
                        if a in ln: print(f"  {f.relative_to(root)}: {ln.strip()[:100]}")
print(f"\n{'APPLIED' if apply else 'DRY RUN'}: {changed} files would change of {scanned} scanned; sed.h5 untouched")
