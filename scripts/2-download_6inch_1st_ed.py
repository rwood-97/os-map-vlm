import os

import geopandas as gpd
import numpy as np
from mapreader.download.sheet_downloader import SheetDownloader

# load NLS metadata
metadata = gpd.read_file("./data/metadata_nls_six_inch_1st_ed_with_xyz.geojson")
bounds = metadata.bounds

# Split y (latitude) into 10 bands, sample 200 sheets per band (~2,000 total)
ys = np.linspace(metadata.total_bounds[1], metadata.total_bounds[3], 11)

samples = []
for y_min, y_max in zip(ys[:-1], ys[1:], strict=True):
    sample = metadata[(bounds["miny"] <= y_max) & (y_min <= bounds["maxy"])]
    sample = sample.sample(n=250, random_state=42) if len(sample) > 250 else sample
    samples.append(sample)

print(f"Total samples: {sum(len(sample) for sample in samples)}")

# define sheet downloader
downloader = SheetDownloader(
    "./data/metadata_nls_six_inch_1st_ed_with_xyz.geojson",
    "https://geo.nls.uk/mapdata3/os/6inchfirst/{z}/{x}/{y}.png", # placeholder, will be replaced county by county
)

# download sheets for each sample
for sample in samples:
    for xyz in sample["Layer"].unique():
        downloader.tile_server = [xyz]
        sample_subset = sample[sample["Layer"] == xyz]
        downloader.metadata = sample_subset
        downloader.get_grid_bb()
        downloader.download_all_map_sheets("./data/maps_6inch_1st_ed", download_in_parrallel=False, force=True, error_on_missing_map=False)
