"""Copy a random sample of map PNGs to a folder for manual inspection.

Usage:
    python sample_for_qc.py --maps-dir data/maps_6inch_2nd_ed --n 20 --out data/qc_sample

Then rsync the output folder to your laptop:
    rsync -av <host>:<path>/data/qc_sample/ ./qc_sample/
"""

import argparse
import random
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--maps-dir", required=True, help="Directory containing map PNGs")
parser.add_argument("--n", type=int, default=20, help="Number of sheets to sample")
parser.add_argument("--out", required=True, help="Output directory for sampled PNGs")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

maps_dir = Path(args.maps_dir)
out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)

pngs = sorted(maps_dir.glob("map_*.png"))
if not pngs:
    raise SystemExit(f"No map_*.png files found in {maps_dir}")

random.seed(args.seed)
sample = random.sample(pngs, min(args.n, len(pngs)))

for src in sample:
    shutil.copy2(src, out_dir / src.name)
    print(f"  copied {src.name}")

print(f"\n{len(sample)} sheets copied to {out_dir}")
print(f"\nTo rsync to laptop:")
print(f"  rsync -av <host>:{out_dir.resolve()}/ ./qc_sample/")
