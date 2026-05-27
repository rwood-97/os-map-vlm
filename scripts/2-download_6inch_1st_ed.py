import geopandas as gpd
from mapreader.download.sheet_downloader import SheetDownloader

# load NLS metadata
metadata = gpd.read_file("./data/metadata_nls_six_inch_1st_ed_with_xyz.geojson")

# define sheet downloader (tile server is set per county layer below)
downloader = SheetDownloader(
    "./data/metadata_nls_six_inch_1st_ed_with_xyz.geojson",
    "https://geo.nls.uk/mapdata3/os/6inchfirst/{z}/{x}/{y}.png",
)

for xyz in metadata["Layer"].unique():
    downloader.tile_server = [xyz]
    downloader.metadata = metadata[metadata["Layer"] == xyz]
    downloader.get_grid_bb(zoom_level=17)
    downloader.download_all_map_sheets(
        "./data/maps_6inch_1st_ed",
        download_in_parallel=True,
        force=True,
        error_on_missing_map=False,
    )
