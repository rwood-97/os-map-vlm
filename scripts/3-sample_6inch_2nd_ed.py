"""Sample ~4,000 sheets from the 6-inch 2nd ed. datastore.

The datastore splits maps into three regional subdirectories (ENG, SCO, WAL),
each with a maps/ subfolder containing tiles and a metadata.csv.

Usage:
  python 3-sample_6inch_2nd_ed.py <datastore_dir>

Outputs:
  data/6inch_2nd_sample_filelist.txt        — region-prefixed paths for: rsync --files-from
  data/6inch_2nd_metadata_combined.csv      — combined metadata for all three regions
  data/6inch_2nd_metadata_for_filtering.csv — sampled image names for MapReader filtering
"""

import ast
import sys
from itertools import pairwise
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <datastore_dir>")
    sys.exit(1)

DATASTORE = Path(sys.argv[1])
REGION_DIRS = {
    "ENG": "6inch_2nd_ed_maps_ENG",
    "SCO": "6inch_2nd_ed_maps_SCO",
    "WAL": "6inch_2nd_ed_maps_WAL",
}

# Read and combine metadata from all three regions
dfs = []
for _, region_dir in REGION_DIRS.items():
    df = pd.read_csv(DATASTORE / region_dir / "maps" / "metadata.csv", index_col=0)
    df["region_dir"] = region_dir
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
combined = combined.drop_duplicates(subset="name")
print(f"Combined metadata: {len(combined)} sheets")

out_dir = Path("./data")
combined.to_csv(out_dir / "6inch_2nd_metadata_combined.csv", index=False)
print("Written data/6inch_2nd_metadata_combined.csv")


# Parse "coordinates" col "(minx, miny, maxx, maxy)" into geometry for band sampling
def parse_coords(coord_str):
    minx, miny, maxx, maxy = ast.literal_eval(coord_str)
    return box(minx, miny, maxx, maxy)


metadata = gpd.GeoDataFrame(
    combined,
    geometry=combined["coordinates"].apply(parse_coords),
    crs="EPSG:4326",
)
bounds = metadata.bounds

# Split y (latitude) into 10 bands, sample 400 sheets per band (~4,000 total)
ys = np.linspace(metadata.total_bounds[1], metadata.total_bounds[3], 11)

samples = []
for y_min, y_max in pairwise(ys):
    band = metadata[(bounds["miny"] <= y_max) & (y_min <= bounds["maxy"])]
    sample = band.sample(n=400, random_state=42) if len(band) > 400 else band
    samples.append(sample)

sampled = pd.concat(samples, ignore_index=True)
sampled = sampled.drop_duplicates(subset="name")

print(f"Sampled {len(sampled)} sheets from {len(metadata)} total")

# One filelist per region (filenames only, no path prefix) so all maps land flat in the dst dir.
# Also write a combined filelist for reference.
print("\nWriting per-region filelists...")
for region, region_dir in REGION_DIRS.items():
    region_sample = sampled[sampled["region_dir"] == region_dir]
    filelist_path = out_dir / f"6inch_2nd_sample_filelist_{region}.txt"
    with open(filelist_path, "w") as f:
        for name in region_sample["name"]:
            f.write(f"{name}\n")
    print(f"  Written {filelist_path} ({len(region_sample)} maps)")

# Sampled image names for MapReader filtering
sampled[["name"]].rename(columns={"name": "mapreader_image_id"}).reset_index(
    drop=True
).to_csv(out_dir / "6inch_2nd_metadata_for_filtering.csv", index=False)

print("\nTo transfer files from datastore (runs can be parallelised):")
for _, region_dir in REGION_DIRS.items():
    src = DATASTORE / region_dir / "maps"
    filelist_path = out_dir / f"6inch_2nd_sample_filelist_{region}.txt"
    print(f"  rsync -av --files-from={filelist_path} {src}/ <hpc_maps_dir>/")
