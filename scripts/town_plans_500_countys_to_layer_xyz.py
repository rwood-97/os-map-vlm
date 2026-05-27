import re

import geopandas as gpd
import pandas as pd

# read input data
metadata = gpd.read_file("./data/metadata_ostowns_500.json")
tilelayers = pd.read_csv("../nls_tilelayers.csv", index_col=0)

# most expected layers have typename nls:OS_Town_Plans_Eng_500_WFS
town_plans_500_tilelayers = tilelayers[
    tilelayers["Typename"] == "nls:OS_Town_Plans_Eng_500_WFS"
]
town_plans_500_tilelayers = town_plans_500_tilelayers[
    ~town_plans_500_tilelayers["Name"].str.startswith("R_")
]
layers = town_plans_500_tilelayers["Name"].tolist()

# Additional layers not in the 500 WFS typename
layers.extend(
    ["OSTowns_Guildford", "OSTowns_Hitchin", "edin1849", "edin1893", "glas1857"]
)


def get_xyz_url(layer):
    if layer in tilelayers["Name"].values:
        return tilelayers[tilelayers["Name"] == layer]["XYZ URL"].values[0]
    else:
        raise ValueError(f"Layer {layer} not found in tilelayers.csv")


# layer names to XYZ URLs
layers_xyz = {layer: get_xyz_url(layer) for layer in layers}

# Add regional layers that have no typename in the CSV to layers_xyz
regional_layers = [
    "OSTowns_Wales",
    "OSTowns_Cornwall",
    "OSTowns_Isle_of_Man",
    "OSTowns_North",
    "OSTowns_South",
    "OSTowns_Midlands_east",
    "OSTowns_Midlands_west",
    "TownsBirmimgham1855",
]
for name in regional_layers:
    if name in tilelayers["Name"].values and name not in layers_xyz:
        layers_xyz[name] = get_xyz_url(name)


def get_layer_strings_to_match(layer):
    if layer.lower().startswith("ostowns_"):
        layer = layer[len("ostowns_") :]
    if layer.lower().startswith("ostowns"):
        layer = layer[len("ostowns") :]
    if re.match(r"[a-zA-Z]+[0-9]+s$", layer):
        return re.match(r"([a-zA-Z]+)([0-9]+s)$", layer).groups()[0]
    if re.match(r"[a-zA-Z]+[0-9]+$", layer):
        return re.match(r"([a-zA-Z]+)([0-9]+)$", layer).groups()[0]
    else:
        return layer


# layer strings to match against name
layer_strs = [get_layer_strings_to_match(layer) for layer in layers]
assert len(layer_strs) == len(layers)

# Regional layers mapping
regional_layers = {}

for name, layer_name in [
    (
        [
            "glamorgan",
            "monmouth",
            "caernarvon",
            "carnarvon",
            "carmarthen",
            "pembroke",
            "denbigh",
            "flint",
            "merioneth",
            "cardigan",
            "brecon",
            "brecknock",
            "radnor",
            "anglesey",
            "anglesea",
            "montgomery",
        ],
        "OSTowns_Wales",
    ),
    (["cornwall"], "OSTowns_Cornwall"),
    (["isle of man"], "OSTowns_Isle_of_Man"),
    (
        [
            "lancashire",
            "yorkshire",
            "cheshire",
            "cumberland",
            "westmorland",
            "durham",
            "northumberland",
        ],
        "OSTowns_North",
    ),
    (
        [
            "lincolnshire",
            "derbyshire",
            "leicestershire",
            "nottinghamshire",
            "northamptonshire",
            "rutland",
            "huntingdonshire",
        ],
        "OSTowns_Midlands_east",
    ),
    (
        [
            "staffordshire",
            "warwickshire",
            "worcestershire",
            "shropshire",
            "herefordshire",
            "gloucestershire",
        ],
        "OSTowns_Midlands_west",
    ),
    (
        [
            "kent",
            "sussex",
            "hampshire",
            "surrey",
            "essex",
            "suffolk",
            "norfolk",
            "dorset",
            "devon",
            "somerset",
            "wiltshire",
            "berkshire",
            "oxfordshire",
            "buckinghamshire",
            "hertfordshire",
            "cambridgeshire",
            "bedfordshire",
            "middlesex",
            "no county",
        ],
        "OSTowns_South",
    ),
]:
    for k in name:
        regional_layers[k] = layer_name

# Some towns need to be matched manually due to naming differences
town_layers = {
    "birmingham": "TownsBirmimgham1855",  # typo in layer name
    "campbelton": "campbeltown",  # spelling variant
    "rutherglen": "glas1857",  # Lanarkshire town, no dedicated layer
}


def get_metadata_layer_name(row):
    town = row["TWN"].replace(" ", "") if pd.notna(row["TWN"]) else ""
    county = row["CNTY"] or ""

    # First try direct name match against layer strings
    for i, layer_str in enumerate(layer_strs):
        if layer_str.lower() in town.lower():
            return layers_xyz[layers[i]]

    # Check manual town layers
    for town_key, layer_name in town_layers.items():
        if town_key in town.lower() and layer_name in layers_xyz:
            return layers_xyz[layer_name]

    # Otherwise regional layers
    for county_key, layer_name in regional_layers.items():
        if county_key in county.lower() and layer_name in layers_xyz:
            return layers_xyz[layer_name]

    print(f"Could not find layer for TWN: {town}, CNTY: {county}")


metadata["Layer"] = metadata.apply(get_metadata_layer_name, axis=1)
metadata.dropna(inplace=True, subset=["Layer"], ignore_index=True)

# save output data
metadata.to_file(
    "./data/metadata_ostowns_500_with_xyz.json", driver="GeoJSON", engine="pyogrio"
)
