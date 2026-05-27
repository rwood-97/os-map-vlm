import re

import geopandas as gpd
import pandas as pd

# read input data
metadata = gpd.read_file("./data/metadata_nls_OS_Town_Plans_Eng_non500_WFS_2026-05-27_084755.geojson")
tilelayers = pd.read_csv("../nls_tilelayers.csv", index_col=0)

# expected layers
layers = ['cupar1854', 'dunfermline1854', 'dalkeith1852', 'dumfries1850', 'edin1849', 'haddington1853', 'kirkcaldy1855', 'kirkcudbright1850', 'musselburgh1853', 'standrews1854', 'stranraer1847', 'wigtown1848', 'OStownsAccrington', 'OStownsAlnwick2640', 'OStownsAshton', 'OStownsBacup', 'OStownsBarnsley', 'OStownsBeverley', 'OStownsBingley', 'OStownsBlackburn', 'OStownsBlyth', 'OStownsBolton', 'OStownsBradford', 'OStownsBridlington', 'OStownsBurnley', 'OStownsBury', 'OStownsChorley', 'OStownsClitheroe', 'OStownsColne', 'OStownsDarlington', 'OStownsDewsbury', 'OStownsDoncaster', 'OStownsFleetwood', 'OStownsHalifax', 'OStownsHaslingden', 'OStownsHeywood', 'OStownsHowden', 'OStownsHuddersfield', 'OStownsKeighley', 'OStownsKingstonuponHull', 'OStownsKingstonuponThames', 'OStownsKnaresborough', 'OStownsLancaster1056', 'OStownsLeeds1056', 'OStownsLiverpool', 'OStownsLondon1056', 'OStownsManchester', 'OStownsMalton', 'OStownsMiddlesbrough', 'OStownsMiddleton', 'OStownsOrmskirk', 'OStownsPontefract', 'OStownsPrescot', 'OStownsPreston', 'OStownsRichmond', 'OStownsRipon', 'OStownsRochdale', 'OStownsRotherham', 'OStownsScarborough', 'OStownsSelby', 'OStownsSheffield', 'OStownsSkipton', 'OStownsSt_Helens', 'OStownsStockport', 'OStownsTodmorden', 'OSTownsTyneside', 'OStownsUlverston', 'OStownsWakefield', 'OStownsWarrington', 'OStownsWhitby', 'OStownsWigan', 'OStownsWindsor', 'OStownsYork']


def get_xyz_url(layer):
    if layer in tilelayers["Name"].values:
        return tilelayers[tilelayers["Name"]==layer]["XYZ URL"].values[0]
    elif f"{layer}_tileset" in tilelayers["Name"].values:
        return tilelayers[tilelayers["Name"]==f"{layer}_tileset"]["XYZ URL"].values[0]
    else:
        raise ValueError(f"Layer {layer} not found in tilelayers.csv")

# layer names to XYZ URLs
layers_xyz = {layer: get_xyz_url(layer) for layer in layers}

def get_layer_strings_to_match(layer):
    if layer.lower().startswith("ostowns"):
        layer = layer[len("ostowns"):]
    if re.match(r"[a-zA-Z]+[0-9]+$", layer):
        return re.match(r"([a-zA-Z]+)([0-9]+)$", layer).groups()[0]
    else:
        return layer

# layer strings to match against WFS_TITLE
layer_strs = [get_layer_strings_to_match(layer) for layer in layers]
assert len(layer_strs) == len(layers)

# function to get the layer name for a given WFS_TITLE
def get_metadata_layer_name(wfs_title):
    loc = wfs_title.split(",")[0].strip()
    # remove spaces
    loc = loc.replace(" ", "")

    for i, layer_str in enumerate(layer_strs):
        if layer_str.lower() in loc.lower():
            return layers_xyz[layers[i]]

    if "StHelens" in loc: # special case for St Helens
        return layers_xyz["OStownsSt_Helens"]

    print(f"No match found for WFS_TITLE: {wfs_title}")

metadata["Layer"] = metadata["WFS_TITLE"].apply(get_metadata_layer_name)
metadata.dropna(inplace=True, subset=["Layer"], ignore_index=True)

# save output data
metadata.to_file("./data/metadata_nls_OS_Town_Plans_Eng_non500_WFS_2026-05-27_084755_with_xyz.geojson", driver="GeoJSON", engine="pyogrio")
