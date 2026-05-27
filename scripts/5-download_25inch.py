from itertools import pairwise

import geopandas as gpd
import numpy as np
from mapreader.download.sheet_downloader import SheetDownloader

# load NLS metadata
metadata = gpd.read_file(
    "./data/metadata_nls_OS_25_Inch_Eng_Wal_Scot_WFS_2026-05-27_123929_with_xyz.geojson"
)
bounds = metadata.bounds

# Split y (latitude) into 10 bands, sample 1500 sheets per band (~15,000 total)
ys = np.linspace(metadata.total_bounds[1], metadata.total_bounds[3], 11)

samples = []
for y_min, y_max in pairwise(ys):
    sample = metadata[(bounds["miny"] <= y_max) & (y_min <= bounds["maxy"])]
    sample = sample.sample(n=1500, random_state=42) if len(sample) > 1500 else sample
    samples.append(sample)

print(f"Total samples: {sum(len(sample) for sample in samples)}")

# define sheet downloader (tile server is set per county layer below)
downloader = SheetDownloader(
    "./data/metadata_nls_OS_25_Inch_Eng_Wal_Scot_WFS_2026-05-27_123929_with_xyz.geojson",
    "https://mapseries-tilesets.s3.amazonaws.com/25_inch/somerset/{z}/{x}/{y}.png",
)

# download sheets for each sample
for sample in samples:
    for xyz in sample["Layer"].unique():
        downloader.tile_server = [xyz]
        sample_subset = sample[sample["Layer"] == xyz]
        downloader.metadata = sample_subset
        downloader.get_grid_bb(zoom_level=18)
        downloader.download_all_map_sheets(
            "./data/maps_25inch",
            download_in_parallel=False,
            force=True,
            error_on_missing_map=False,
        )
