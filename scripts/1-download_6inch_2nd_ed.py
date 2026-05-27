import os

from mapreader.download.sheet_downloader import SheetDownloader

# get maptiler key
maptiler_key = os.getenv("MAPTILER_KEY")

# define sheet downloader
downloader = SheetDownloader(
    "./data/metadata_nls_six_inch_2nd_ed.geojson",
    "https://api.maptiler.com/tiles/uk-osgb10k1888/{z}/{x}/{y}.jpg?key=" + maptiler_key
)

downloader.get_grid_bb(zoom_level=17)
downloader.download_all_map_sheets("./data/maps_6inch_2nd_ed", download_in_parallel=False, force=True, error_on_missing_map=False)
