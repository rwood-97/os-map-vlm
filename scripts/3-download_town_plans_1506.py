import geopandas as gpd
from mapreader.download.sheet_downloader import SheetDownloader

# load NLS metadata
metadata = gpd.read_file("./data/metadata_nls_OS_Town_Plans_Eng_non500_WFS_2026-05-27_084755_with_xyz.geojson")

# define sheet downloader (tile server is set per county layer below)
downloader = SheetDownloader(
    "./data/metadata_nls_OS_Town_Plans_Eng_non500_WFS_2026-05-27_084755_with_xyz.geojson",
    "https://mapseries-tilesets.s3.amazonaws.com/town_england/Accrington/{z}/{x}/{y}.png"
)

for xyz in metadata["Layer"].unique():
    downloader.tile_server = [xyz]
    downloader.metadata = metadata[metadata["Layer"] == xyz]
    downloader.get_grid_bb(zoom_level=20)
    downloader.download_all_map_sheets("./data/maps_os_town_plans_1056", download_in_parrallel=True, force=True, error_on_missing_map=False)
