"""Sample ~4,000 sheets from the 6-inch 2nd ed. datastore.

Outputs:
  data/6inch_2nd_sample_filelist.txt       — filenames for: rsync --files-from
"""

from itertools import pairwise
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

# start out same as the download scripts, read metadata, split into 10 latitudinal bands, sample 400 sheets per band (~4,000 total)

metadata = gpd.read_file("./data/metadata_nls_six_inch_2nd_ed.geojson", engine="fiona")
bounds = metadata.bounds

# Split y (latitude) into 10 bands, sample 400 sheets per band (~4,000 total)
ys = np.linspace(metadata.total_bounds[1], metadata.total_bounds[3], 11)

samples = []
for y_min, y_max in pairwise(ys):
    band = metadata[(bounds["miny"] <= y_max) & (y_min <= bounds["maxy"])]
    sample = band.sample(n=400, random_state=42) if len(band) > 400 else band
    samples.append(sample)

sampled = pd.concat(samples, ignore_index=True)
sampled = sampled.drop_duplicates(subset="IMAGE")

print(f"Sampled {len(sampled)} sheets from {len(metadata)} total")

out_dir = Path("./data")

# filename list for: rsync --files-from=data/6inch_2nd_sample_filelist.txt <src>/ <dst>/
filelist_path = out_dir / "6inch_2nd_sample_filelist.txt"
with open(filelist_path, "w") as f:
    for image_id in sampled["IMAGE"]:
        f.write(f"map_{image_id}.png\n")
    # Also get full 6inch 2nd ed metadata from the datastore
    f.write("metadata.csv\n")
print(f"Written {filelist_path}")

# Full metadata will need to be filtered for the sampled sheets.
sampled["mapreader_image_id"] = "map_" + sampled["IMAGE"].astype(str) + ".png"
sampled[["mapreader_image_id"]].reset_index(drop=True).to_csv(
    out_dir / "6inch_2nd_metadata_for_filtering.csv", index=False
)

print("\nTo transfer files from datastore:")
print(f"  rsync -av --files-from={filelist_path} <datastore_dir>/ <hpc_maps_dir>/")
