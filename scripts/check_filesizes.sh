#!/usr/bin/env python3
import sys
from pathlib import Path
from PIL import Image

dir_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
pngs = sorted(dir_path.glob("*.png"))

print(f"PNG pixel dimensions in: {dir_path}")
print("=" * 55)

dims = {}
for f in pngs:
    with Image.open(f) as img:
        w, h = img.size
    dims[f.name] = (w, h)
    print(f"{w:6} x {h:<6}  {f.name}")

print("=" * 55)
print(f"Total files: {len(pngs)}")
unique = set(dims.values())
if len(unique) == 1:
    w, h = list(unique)[0]
    print(f"All images are the same size: {w} x {h}")
else:
    print(f"Unique sizes: {len(unique)}")
    for size in sorted(unique):
        n = sum(1 for v in dims.values() if v == size)
        print(f"  {size[0]} x {size[1]}: {n} file(s)")
