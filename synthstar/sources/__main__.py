"""python -m synthstar.sources [NAME ...]: the star generators and their keys."""
import sys

from . import all_generators, describe

generators = all_generators()
names = sys.argv[1:] or sorted(generators, key=lambda n: (generators[n].timing, n))
print("\n\n".join(describe(generators[n]) for n in names))
