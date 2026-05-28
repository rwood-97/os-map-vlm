from pathlib import Path

import geopandas as gpd
import pandas as pd
from mapreader.download.sheet_downloader import SheetDownloader

# load NLS metadata, sample 4 sheets per town
metadata = gpd.read_file("./data/metadata_ostowns_500_with_xyz.json")
metadata = metadata.groupby("TWN", group_keys=False).apply(lambda x: x.sample(min(4, len(x)), random_state=42)).reset_index(drop=True)

# filter out already-downloaded sheets (after sampling so it doesn't affect selection)
csv_path = Path("./data/maps_os_town_plans_500/metadata.csv")
if csv_path.exists():
    already_downloaded = pd.read_csv(csv_path)
    downloaded_ids = set(already_downloaded["name"].str.extract(r"map_(\d+)\.png")[0])
    metadata = metadata[~metadata["IMAGE"].isin(downloaded_ids)]
    print(f"Skipping {len(downloaded_ids)} already-downloaded sheets, {len(metadata)} remaining.")

# define sheet downloader (tile server is set per layer below)
downloader = SheetDownloader(
    "./data/metadata_ostowns_500_with_xyz.json",
    "https://mapseries-tilesets.s3.amazonaws.com/os/town-england/Wales/{z}/{x}/{y}.png",
)

for xyz in metadata["Layer"].unique():
    downloader.tile_server = [xyz]
    downloader.metadata = metadata[metadata["Layer"] == xyz]
    downloader.get_grid_bb(zoom_level=20)
    downloader.download_all_map_sheets(
        "./data/maps_os_town_plans_500",
        download_in_parallel=True,
        force=True,
        error_on_missing_map=False,
    )
