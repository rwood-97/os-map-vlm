import re

import geopandas as gpd
import pandas as pd

# read input data
metadata = gpd.read_file(
    "./data/metadata_nls_OS_25_Inch_Eng_Wal_Scot_WFS_2026-05-27_123929.geojson"
)
tilelayers = pd.read_csv("../nls_tilelayers.csv", index_col=0)

# expected layers
layers = [
    "ostwentyfiveinchparishabercorn",
    "ostwentyfiveinchparishbathgate",
    "ostwentyfiveinchparishboness",
    "ostwentyfiveinchparishcarriden",
    "ostwentyfiveinchparishdalmeny",
    "ostwentyfiveinchparishkirkliston",
    "ostwentyfiveinchparishlinlithgow",
    "ostwentyfiveinchparishlivingston",
    "ostwentyfiveinchparishtorphichen",
    "ostwentyfiveinchparishuphall",
    "ostwentyfiveinchparishwhitburn",
    "ostwentyfiveinchparishavondale",
    "ostwentyfiveinchparishbarony",
    "ostwentyfiveinchparishbiggar",
    "ostwentyfiveinchparishblantyre",
    "ostwentyfiveinchparishbothwell",
    "ostwentyfiveinchparishcadder",
    "ostwentyfiveinchparishcalton",
    "ostwentyfiveinchparishcambuslang",
    "ostwentyfiveinchparishcambusnethan",
    "ostwentyfiveinchparishcarluke",
    "ostwentyfiveinchparishcarmichael",
    "ostwentyfiveinchparishcarmunnock",
    "ostwentyfiveinchparishcarnwath",
    "ostwentyfiveinchparishcarstairs",
    "ostwentyfiveinchparishcity_of_glasgow",
    "ostwentyfiveinchparishcovington",
    "ostwentyfiveinchparishcrawfordjohn",
    "ostwentyfiveinchparishcrawford",
    "ostwentyfiveinchparishculter",
    "ostwentyfiveinchparishdalserf",
    "ostwentyfiveinchparishdalziel",
    "ostwentyfiveinchparishdolphinton",
    "ostwentyfiveinchparishdouglas",
    "ostwentyfiveinchparishdunsyre",
    "ostwentyfiveinchparisheast_kilbride",
    "ostwentyfiveinchparishglassford",
    "ostwentyfiveinchparishgovan",
    "ostwentyfiveinchparishhamilton",
    "ostwentyfiveinchparishlamington_and_wandell",
    "ostwentyfiveinchparishlanark",
    "ostwentyfiveinchparishlesmahagow",
    "ostwentyfiveinchparishliberton",
    "ostwentyfiveinchparishmaryhill",
    "ostwentyfiveinchparishnew_monkland",
    "ostwentyfiveinchparishold_monkland",
    "ostwentyfiveinchparishpettinain",
    "ostwentyfiveinchparishrutherglen",
    "ostwentyfiveinchparishshettleston",
    "ostwentyfiveinchparishshotts",
    "ostwentyfiveinchparishspringburn",
    "ostwentyfiveinchparishstonehouse",
    "ostwentyfiveinchparishsymington",
    "ostwentyfiveinchparishwalston",
    "ostwentyfiveinchparishwiston_roberton",
    "ostwentyfiveinchparishairth",
    "ostwentyfiveinchparishbaldernock",
    "ostwentyfiveinchparishbalfron",
    "ostwentyfiveinchparishbothkennar",
    "ostwentyfiveinchparishbuchanan",
    "ostwentyfiveinchparishcampsie",
    "ostwentyfiveinchparishdenny",
    "ostwentyfiveinchparishdrymen",
    "ostwentyfiveinchparishdunipace",
    "ostwentyfiveinchparishfalkirk",
    "ostwentyfiveinchparishfintry",
    "ostwentyfiveinchparishgargunnock",
    "ostwentyfiveinchparishkillearn",
    "ostwentyfiveinchparishkilsyth",
    "ostwentyfiveinchparishkippen",
    "ostwentyfiveinchparishlarbert",
    "ostwentyfiveinchparishlecropt",
    "ostwentyfiveinchparishlogie",
    "ostwentyfiveinchparishmuiravonside",
    "ostwentyfiveinchparishpolmont",
    "ostwentyfiveinchparishslamannan",
    "ostwentyfiveinchparishst_ninians",
    "ostwentyfiveinchparishstirling",
    "ostwentyfiveinchparishstrathblane",
    "ostwentyfiveinchparishabbey",
    "ostwentyfiveinchparishcathcart",
    "ostwentyfiveinchparisheaglesham",
    "ostwentyfiveinchparisheastwood",
    "ostwentyfiveinchparisherskine",
    "ostwentyfiveinchparishgreenock",
    "ostwentyfiveinchparishhouston",
    "ostwentyfiveinchparishinchinnan",
    "ostwentyfiveinchparishinnerkip",
    "ostwentyfiveinchparishkilbarchan",
    "ostwentyfiveinchparishkilmalcolm",
    "ostwentyfiveinchparishlochwinnoch",
    "ostwentyfiveinchparishmearns",
    "ostwentyfiveinchparishneilston",
    "ostwentyfiveinchparishport_glasgow",
    "ostwentyfiveinchparishrenfrew",
]

layers.extend(
    [
        "OStwentyfiveinchnewcastleadds",
        "OStwentyfiveinchholes",
        "OStwentyfiveinchholes_103676684",
        "OStwentyfiveinchholes_103683170",
        "OStwentyfiveinchholes_104194119",
        "OStwentyfiveinchholes_104194125",
        "OStwentyfiveinchholes_128075868",
        "OStwentyfiveinchholes_132280016",
        "OStwentyfiveinch114581170",
        "OStwentyfiveinchbedfordshire",
        "OStwentyfiveinchberkshire",
        "OStwentyfiveinchcambridge",
        "OStwentyfiveinchcheshire",
        "OStwentyfiveinchcornwall",
        "OStwentyfiveinchcumberland",
        "OStwentyfiveinchdevon",
        "OStwentyfiveinchdorset",
        "OStwentyfiveinchyorkshire",
        "OStwentyfiveinch125623684_fix",
        "OStwentyfiveinchdurham",
        "OStwentyfiveinchhampshire",
        "OStwentyfiveinchbuckingham",
        "OStwentyfiveinchessex",
        "OStwentyfiveinch_raywilson",
        "OStwentyfiveinchgloucestershire",
        "OStwentyfiveinchherefordshire",
        "OStwentyfiveinchhuntingdon",
        "OStwentyfiveinchlancashire",
        "OStwentyfiveinchleicestershire",
        "OStwentyfiveinchlincolnshire",
        "OStwentyfiveinchmiddlesex",
        "OStwentyfiveinchnorfolk",
        "OStwentyfiveinchnorthampton",
        "OStwentyfiveinchnorthumberland",
        "OStwentyfiveinchnottinghamshire",
        "OStwentyfiveinchkent",
        "OStwentyfiveinchrutland",
        "OStwentyfiveinchshropshire_derby",
        "OStwentyfiveinchstaffordshire",
        "OStwentyfiveinchsurrey",
        "OStwentyfiveinchsussex",
        "OStwentyfiveinchlondon",
        "OStwentyfiveinchhertfordshire",
        "OStwentyfiveinchoxford",
        "OStwentyfiveinchsomerset",
        "OStwentyfiveinchsuffolk",
        "OStwentyfiveinchwarwick",
        "OStwentyfiveinchwestmorland",
        "OStwentyfiveinchwiltshire",
        "OStwentyfiveinchworcestershire",
        "OStwentyfiveinchwales",
        "OStwentyfiveinchholes_135198775",
        "OStwentyfiveinchcarmarthenadds",
        "OStwentyfiveinchgtyarmouthnadds",
        "os25scotland",
        "os25scotland2",
        "os25scotland2_lauder",
        "os25scotland2_banff",
        "os25scotland2_elgin_banff",
        "os25scotland2_84241480",
        "os25scotland2_82863069",
        "edinburgh_west",
        "guernsey_25inch",
    ]
)


def get_xyz_url(layer):
    if layer in tilelayers["Name"].values:
        return tilelayers[tilelayers["Name"] == layer]["XYZ URL"].values[0]
    elif f"{layer}_tileset" in tilelayers["Name"].values:
        return tilelayers[tilelayers["Name"] == f"{layer}_tileset"]["XYZ URL"].values[0]
    else:
        raise ValueError(f"Layer {layer} not found in tilelayers.csv")


# layer names to XYZ URLs
layers_xyz = {layer: get_xyz_url(layer) for layer in layers}


def get_layer_strings_to_match(layer):
    if layer.lower().startswith("ostwentyfiveinchparish"):
        layer = layer[len("ostwentyfiveinchparish") :]
    if layer.lower().startswith("ostwentyfiveinch"):
        layer = layer[len("ostwentyfiveinch") :]
    if re.match(r"[a-zA-Z]+[0-9]+$", layer):
        return re.match(r"([a-zA-Z]+)([0-9]+)$", layer).groups()[0]
    else:
        return layer


# layer strings to match against WFS_TITLE
layer_strs = [get_layer_strings_to_match(layer) for layer in layers]
assert len(layer_strs) == len(layers)

# County-name to layer mapping for counties not directly embedded in layer names
regional_layers = {
    # Wales
    "anglesey": "OStwentyfiveinchwales",
    "brecknockshire": "OStwentyfiveinchwales",
    "caernarvonshire": "OStwentyfiveinchwales",
    "cardiganshire": "OStwentyfiveinchwales",
    "carmarthenshire": "OStwentyfiveinchwales",
    "denbighshire": "OStwentyfiveinchwales",
    "flintshire": "OStwentyfiveinchwales",
    "glamorgan": "OStwentyfiveinchwales",
    "merionethshire": "OStwentyfiveinchwales",
    "monmouthshire": "OStwentyfiveinchwales",
    "montgomeryshire": "OStwentyfiveinchwales",
    "pembrokeshire": "OStwentyfiveinchwales",
    "radnorshire": "OStwentyfiveinchwales",
    # Shropshire and derbyshire
    "shropshire": "OStwentyfiveinchshropshire_derby",
    "derbyshire": "OStwentyfiveinchshropshire_derby",
    # Scotland
    "aberdeenshire": "os25scotland",
    "argyllshire": "os25scotland",
    "ayrshire": "os25scotland",
    "banffshire": "os25scotland2_banff",
    "berwickshire": "os25scotland",
    "buteshire": "os25scotland",
    "caithness": "os25scotland",
    "clackmannanshire": "os25scotland",
    "dumbartonshire": "os25scotland",
    "dumfriesshire": "os25scotland",
    "edinburghshire": "os25scotland",
    "elginshire": "os25scotland2_elgin_banff",
    "fifeshire": "os25scotland",
    "forfarshire": "os25scotland",
    "haddingtonshire": "os25scotland",
    "inverness-shire": "os25scotland",
    "kincardineshire": "os25scotland",
    "kinross-shire": "os25scotland",
    "kirkcudbrightshire": "os25scotland",
    "nairnshire": "os25scotland",
    "orkney": "os25scotland",
    "peebles-shire": "os25scotland",
    "perth": "os25scotland",
    "ross-shire": "os25scotland",
    "roxburghshire": "os25scotland",
    "selkirkshire": "os25scotland",
    "sutherland": "os25scotland",
    "wigtownshire": "os25scotland",
    "zetland": "os25scotland",
}


# function to get the layer name for a given WFS_TITLE
def get_metadata_layer_name(wfs_title):
    loc = wfs_title.split(",")[0].strip()
    loc_nospace = loc.replace(" ", "")

    # First try direct substring match against layer strings
    for i, layer_str in enumerate(layer_strs):
        if layer_str.lower() in loc_nospace.lower():
            return layers_xyz[layers[i]]

    # Then check regional layers + strip sheet number (Roman numeral + digits) from loc
    county = re.sub(r"\s+[IVXLCDM]+[A-Za-z]?\.\d+.*$", "", loc).strip().lower()
    for county_key, layer_name in regional_layers.items():
        if county_key in county and layer_name in layers_xyz:
            return layers_xyz[layer_name]

    print(f"No match found for WFS_TITLE: {wfs_title!r}")
    return None


metadata["Layer"] = metadata["WFS_TITLE"].apply(get_metadata_layer_name)

# save output data
metadata.to_file(
    "./data/metadata_nls_OS_25_Inch_with_xyz.geojson",
    driver="GeoJSON",
    engine="pyogrio",
)
