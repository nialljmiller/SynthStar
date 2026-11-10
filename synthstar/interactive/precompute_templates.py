"""Integrate every template the interactive app offers, into the disk cache (data/interactive/templates/):
python -m synthstar.interactive.precompute_templates"""
import time

from synthstar.interactive import templates as T

if __name__ == "__main__":
    for kind in T.TYPES:
        for name in T.precomputed(kind):
            t = time.time()
            T.light(name)
            print(f"{kind:28s} {name:40s} {time.time() - t:5.1f} s", flush=True)
