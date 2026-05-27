import geopandas as gpd
import pandas as pd

# read input data
metadata = gpd.read_file("./data/metadata_nls_six_inch_1st_ed.geojson")
tilelayers = pd.read_csv("../nls_tilelayers.csv", index_col=0)

# layers to match
layers = [
    "ossixinchanglesey",
    "ossixinchbedfordshire",
    "ossixinchberkshire",
    "ossixinchbrecknockshire",
    "ossixinchbuckinghamshire",
    "ossixinchcaernarvonshire",
    "ossixinchcambridgeshire",
    "ossixinchcardiganshire",
    "ossixinchcarmarthenshire",
    "ossixinchcheshire",
    "ossixinchcornwall",
    "ossixinchcumberland",
    "ossixinchdenbighshire",
    "ossixinchderbyshire",
    "ossixinchdevonshire",
    "ossixinchdorset",
    "ossixinchdurham",
    "ossixinchessex",
    "ossixinchflintshire",
    "ossixinchglamorgan",
    "ossixinchgloucestershire",
    "ossixinchhampshire",
    "ossixinchherefordshire",
    "ossixinchhertfordshire",
    "ossixinchhuntingdonshire",
    "ossixinchisleofman",
    "ossixinchkent",
    "ossixinchlancashire",
    "ossixinchleicestershire",
    "ossixinchlincolnshire",
    "ossixinchmerionethshire",
    "ossixinchmiddlesex",
    "ossixinchmonmouthshire",
    "ossixinchmontgomeryshire",
    "ossixinchnorfolk",
    "ossixinchnorthumberland",
    "ossixinchnottinghamshire",
    "ossixinchnorthamptonshire",
    "ossixinchoxfordshire",
    "ossixinchpembrokeshire",
    "ossixinchradnorshire",
    "ossixinchrutland",
    "ossixinchshropshire",
    "ossixinchstaffordshire",
    "ossixinchsomerset",
    "ossixinchsuffolk",
    "ossixinchsurrey",
    "ossixinchsussex",
    "ossixinchwestmorland",
    "ossixinchwarwickshire",
    "ossixinchwiltshire",
    "ossixinchworcestershire",
    "ossixinchyorkshire",
    "sixinchscotlayer",
    "ossixincharmagh",
    "ossixinchantrim",
    "ossixinchcarlow",
    "ossixinchcavan",
    "ossixinchclare",
    "ossixinchcork",
    "ossixinchdonegal",
    "ossixinchdown",
    "ossixinchdublin",
    "ossixinchfermanagh",
    "ossixinchgalway",
    "ossixinchkerry",
    "ossixinchkildare",
    "ossixinchkilkenny",
    "ossixinchkings",
    "ossixinchleitrim",
    "ossixinchlimerick",
    "ossixinchlondonderry",
    "ossixinchlongford",
    "ossixinchlouth",
    "ossixinchmayo",
    "ossixinchmeath",
    "ossixinchmonaghan",
    "ossixinchqueens",
    "ossixinchroscommon",
    "ossixinchsligo",
    "ossixinchtipperary",
    "ossixinchtyrone",
    "ossixinchwaterford",
    "ossixinchwestmeath",
    "ossixinchwexford",
    "ossixinchwicklow",
]

# layer names to XYZ URLs
layers_xyz = {
    layer: tilelayers[tilelayers["Name"] == layer]["XYZ URL"].values[0]
    for layer in layers
}

# get the part of the layer name after "ossixinch" for layers that start with "ossixinch"
layer_ends = [
    layer[len("ossixinch") :] for layer in layers if layer.startswith("ossixinch")
]

# only sixinchscotlayer doesn't start with "ossixinch", so this should be the only layer in this list
layers_non_matching = [layer for layer in layers if not layer.startswith("ossixinch")]
assert len(layers_non_matching) == 1


# function to get the layer name for a given WFS_TITLE
def get_metadata_layer_name(wfs_title):
    loc = wfs_title.split(",")[0].strip()

    for layer_end in layer_ends:
        if layer_end.lower() in loc.lower():
            return layers_xyz["ossixinch" + layer_end]
    return layers_xyz["sixinchscotlayer"]


metadata["Layer"] = metadata["WFS_TITLE"].apply(get_metadata_layer_name)
assert metadata["Layer"].isna().sum() == 0, (
    "Some WFS_TITLE values did not match any layer"
)

# save output data
metadata.to_file(
    "./data/metadata_nls_six_inch_1st_ed_with_xyz.geojson",
    driver="GeoJSON",
    engine="pyogrio",
)
