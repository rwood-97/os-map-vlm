"""
Follow-up 25-inch download pass: retry failed Scotland sheets with scotland_2 URL.

scotland_2 covers northern/eastern Scotland (bounds extend to lon -0.76, lat 60.82)
vs scotland_1 which only covers southern Scotland (lon -2.01, lat 58.53). The original
download script mapped all Scottish counties to scotland_1, so northern/eastern sheets
need retrying against scotland_2.
"""
from itertools import pairwise
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from mapreader.download.sheet_downloader import SheetDownloader

# Reproduce original stratified sample (same parameters as 5-download_25inch.py)
metadata = gpd.read_file("./data/metadata_nls_OS_25_Inch_with_xyz.geojson")
bounds = metadata.bounds
ys = np.linspace(metadata.total_bounds[1], metadata.total_bounds[3], 11)

samples = []
for y_min, y_max in pairwise(ys):
    sample = metadata[(bounds["miny"] <= y_max) & (y_min <= bounds["maxy"])]
    sample = sample.sample(n=400, random_state=42) if len(sample) > 400 else sample
    samples.append(sample)

all_sampled = pd.concat(samples)
print(f"Total in original sample: {len(all_sampled)}")

# Filter to Scotland sheets not yet downloaded
csv_path = Path("./data/maps_25inch/metadata.csv")
already_downloaded = pd.read_csv(csv_path)
downloaded_ids = set(already_downloaded["name"].str.extract(r"map_(\d+)\.png")[0])

remaining = all_sampled[~all_sampled["IMAGE"].isin(downloaded_ids)].copy()
scotland_remaining = remaining[remaining["Layer"].str.contains("scotland_1", na=False)].copy()

print(f"Skipping {len(downloaded_ids)} already-downloaded sheets.")
print(f"Scotland sheets to retry with scotland_2: {len(scotland_remaining)}")

# Swap scotland_1 → scotland_2
scotland_remaining["Layer"] = scotland_remaining["Layer"].str.replace(
    "25_inch/scotland_1/", "25_inch/scotland_2/", regex=False
)

downloader = SheetDownloader(
    "./data/metadata_nls_OS_25_Inch_with_xyz.geojson",
    scotland_remaining["Layer"].iloc[0],
)

for layer_url in scotland_remaining["Layer"].unique():
    downloader.tile_server = [layer_url]
    subset = scotland_remaining[scotland_remaining["Layer"] == layer_url]
    downloader.metadata = subset
    downloader.get_grid_bb(zoom_level=18)
    downloader.download_all_map_sheets(
        "./data/maps_25inch",
        download_in_parallel=True,
        force=True,
        error_on_missing_map=False,
    )
