"""
Align MapReader railspace predictions (LwM MapReader_Outputs_SIGSPATIAL_2022) to
6-inch 2nd ed. patch tiles.

The LwM predictions are made on a 100m x 100m geographic grid that does not
align with our 512x512 pixel patch grid. For each of our patches, this script
finds MapReader "railspace" squares whose polygon overlaps the patch bounding
box, keeps only detections above a confidence threshold and an area-overlap
threshold (to avoid injecting a detection from a sliver of overlap at a patch
edge), and computes the tile-relative pixel position of each detection's
overlap centroid.

Output: JSONL where each line is one patch with its building detections:
  {
    "patch_id": "...",
    "parent_id": "...",
    "detections": [
      {"label": "building", "conf": ..., "tile_x": ..., "tile_y": ...,
       "overlap_frac": ...}
    ]
  }

Usage:
  uv run python scripts/8c-align_mapreader_railspace.py \\
      --patches-dir data/patches_6inch_2nd_ed \\
      --predictions predictions_railspace_geo.csv \\
      --output data/patches_6inch_2nd_ed/mapreader_railspace.jsonl
"""

import argparse
import json
from ast import literal_eval
from pathlib import Path

import geopandas as gpd
import pandas as pd

LABEL = "railspace"


def parse_coordinates(coord_str: str) -> tuple[float, float, float, float]:
    """Parse '(lon_min, lat_min, lon_max, lat_max)' string."""
    return literal_eval(coord_str)


def compute_tile_pixel(lon: float, lat: float, coords: tuple) -> tuple[int, int]:
    """Convert lon/lat to pixel position within a 512x512 tile."""
    lon_min, lat_min, lon_max, lat_max = coords
    tile_x = int((lon - lon_min) / (lon_max - lon_min) * 512)
    tile_y = int((lat_max - lat) / (lat_max - lat_min) * 512)
    return max(0, min(511, tile_x)), max(0, min(511, tile_y))


def main():
    parser = argparse.ArgumentParser(
        description=f"Align MapReader {LABEL} predictions to 6-inch 2nd ed. patches"
    )
    parser.add_argument(
        "--patches-dir", required=True, help="Dir containing patch_df.csv"
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to predictions_railspace_geo.csv",
    )
    parser.add_argument(
        "--output",
        help=f"Output JSONL path (default: <patches-dir>/mapreader_{LABEL}.jsonl)",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.8,
        help="Minimum prediction confidence to keep (default: 0.8)",
    )
    parser.add_argument(
        "--min-overlap",
        type=float,
        default=0.3,
        help="Minimum fraction of a MapReader square's area that must fall "
        "within our patch to count as a detection (default: 0.3)",
    )
    args = parser.parse_args()

    patches_dir = Path(args.patches_dir)
    output_path = (
        Path(args.output) if args.output else patches_dir / f"mapreader_{LABEL}.jsonl"
    )

    # --- Load patch_df ---
    patch_csv = patches_dir / "patch_df.csv"
    print(f"Loading patch_df from {patch_csv}...")
    patch_df = pd.read_csv(patch_csv)
    print(f"  {len(patch_df):,} patches")

    patch_df["coords"] = patch_df["coordinates"].apply(parse_coordinates)
    patch_df["lon_min"] = patch_df["coords"].apply(lambda c: c[0])
    patch_df["lat_min"] = patch_df["coords"].apply(lambda c: c[1])
    patch_df["lon_max"] = patch_df["coords"].apply(lambda c: c[2])
    patch_df["lat_max"] = patch_df["coords"].apply(lambda c: c[3])

    patch_gdf = gpd.GeoDataFrame(
        patch_df,
        geometry=gpd.GeoSeries.from_wkt(patch_df["geometry"]),
        crs="EPSG:4326",
    )

    # --- Load MapReader predictions ---
    print(f"Loading MapReader {LABEL} predictions from {args.predictions}...")
    preds = pd.read_csv(
        args.predictions,
        usecols=["image_id", "parent_id", "predicted_label", "conf", "polygon"],
    )
    preds = preds[
        (preds["predicted_label"] == LABEL) & (preds["conf"] >= args.min_conf)
    ]
    print(f"  {len(preds):,} {LABEL} predictions with conf >= {args.min_conf}")

    preds_gdf = gpd.GeoDataFrame(
        preds,
        geometry=gpd.GeoSeries.from_wkt(preds["polygon"]),
        crs="EPSG:4326",
    )
    preds_gdf["pred_area"] = preds_gdf.geometry.area

    # --- Spatial join: keep MapReader squares overlapping one of our patches ---
    print("Building spatial index and joining...")
    joined = gpd.sjoin(
        preds_gdf,
        patch_gdf[
            [
                "image_id",
                "parent_id",
                "lon_min",
                "lat_min",
                "lon_max",
                "lat_max",
                "geometry",
            ]
        ],
        how="inner",
        predicate="intersects",
        lsuffix="pred",
        rsuffix="patch",
    )
    print(f"  {len(joined):,} candidate overlaps")

    # --- Compute overlap fraction and tile-relative centroid position ---
    patch_geom_lookup = patch_gdf.set_index("image_id")["geometry"]
    records_by_patch: dict[tuple[str, str], list[dict]] = {}
    for row in joined.itertuples():
        patch_geom = patch_geom_lookup[row.image_id_patch]
        intersection = row.geometry.intersection(patch_geom)
        overlap_frac = intersection.area / row.pred_area if row.pred_area else 0.0
        if overlap_frac < args.min_overlap:
            continue
        centroid = intersection.centroid
        tile_x, tile_y = compute_tile_pixel(
            centroid.x,
            centroid.y,
            (row.lon_min, row.lat_min, row.lon_max, row.lat_max),
        )
        records_by_patch.setdefault(
            (row.image_id_patch, row.parent_id_patch), []
        ).append(
            {
                "label": LABEL,
                "conf": round(float(row.conf), 4),
                "tile_x": tile_x,
                "tile_y": tile_y,
                "overlap_frac": round(overlap_frac, 3),
            }
        )

    # --- Write JSONL output, one line per patch ---
    print(f"Writing detections to {output_path}...")
    n_written = 0
    with open(output_path, "w") as f:
        for (image_id, parent_id), detections in records_by_patch.items():
            f.write(
                json.dumps(
                    {
                        "patch_id": image_id,
                        "parent_id": parent_id,
                        "detections": detections,
                    }
                )
                + "\n"
            )
            n_written += 1

    print(
        f"Done. {n_written:,} patches with {LABEL} detections written to {output_path}."
    )


if __name__ == "__main__":
    main()
