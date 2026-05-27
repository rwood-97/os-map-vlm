import geopandas as gpd
from mapreader.download.sheet_downloader import SheetDownloader

# load NLS metadata, sample 4 sheets per town
metadata = gpd.read_file("./data/metadata_ostowns_500_with_xyz.json")
metadata = metadata.groupby("TWN", group_keys=False).apply(lambda x: x.sample(min(4, len(x)), random_state=42)).reset_index(drop=True)

# define sheet downloader (tile server is set per layer below)
downloader = SheetDownloader(
    "./data/metadata_ostowns_500_with_xyz.json",
    "https://mapseries-tilesets.s3.amazonaws.com/os/town-england/Wales/{z}/{x}/{y}.png"
)

for xyz in metadata["Layer"].unique():
    downloader.tile_server = [xyz]
    downloader.metadata = metadata[metadata["Layer"] == xyz]
    downloader.get_grid_bb(zoom_level=20)
    downloader.download_all_map_sheets("./data/maps_os_town_plans_500", download_in_parallel=True, force=True, error_on_missing_map=False)
