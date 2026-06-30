"""
Align GB1900 gazetteer entries to 6-inch 2nd ed. patch tiles.

For each GB1900 point, finds which 512x512 patch it falls within (using the
patch bounding box), then computes tile-relative pixel coordinates.

Output: JSONL where each line is one patch with its GB1900 annotations:
  {
    "patch_id": "patch-...",
    "parent_id": "map_....png",
    "annotations": [
      {"pin_id": "...", "text": "...", "lat": ..., "lon": ...,
       "tile_x": ..., "tile_y": ..., "nation": "...", "parish": "..."}
    ]
  }

Usage:
  uv run python scripts/8-align_gb1900.py \
      --patches-dir data/patches_6inch_2nd_ed \
      --gb1900 data/GB1900_gazetteer_complete_july_2018/gb1900_gazetteer_complete_july_2018.csv \
      --output data/patches_6inch_2nd_ed/gb1900_annotations.jsonl
"""

import argparse
import json
from ast import literal_eval
from pathlib import Path

import geopandas as gpd
import pandas as pd


def parse_coordinates(coord_str: str) -> tuple[float, float, float, float]:
    """Parse '(lon_min, lat_min, lon_max, lat_max)' string."""
    return literal_eval(coord_str)


def compute_tile_pixel(lon: float, lat: float, coords: tuple) -> tuple[int, int]:
    """Convert lon/lat to pixel position within a 512x512 tile."""
    lon_min, lat_min, lon_max, lat_max = coords
    tile_x = int((lon - lon_min) / (lon_max - lon_min) * 512)
    # y=0 is top of image, lat_max is top
    tile_y = int((lat_max - lat) / (lat_max - lat_min) * 512)
    tile_x = max(0, min(511, tile_x))
    tile_y = max(0, min(511, tile_y))
    return tile_x, tile_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--patches-dir", required=True, help="Dir containing patch_df.csv"
    )
    parser.add_argument(
        "--gb1900",
        required=True,
        help="Path to gb1900_gazetteer_complete_july_2018.csv",
    )
    parser.add_argument(
        "--output",
        help="Output JSONL path (default: <patches-dir>/gb1900_annotations.jsonl)",
    )
    parser.add_argument(
        "--edge-margin",
        type=int,
        default=32,
        help="Exclude GB1900 entries within this many pixels of tile edge (default: 32)",
    )
    args = parser.parse_args()

    patches_dir = Path(args.patches_dir)
    output_path = (
        Path(args.output) if args.output else patches_dir / "gb1900_annotations.jsonl"
    )

    # --- Load patch_df ---
    patch_csv = patches_dir / "patch_df.csv"
    print(f"Loading patch_df from {patch_csv}...")
    patch_df = pd.read_csv(patch_csv)
    print(f"  {len(patch_df):,} patches")

    # Parse coordinate bounding boxes
    patch_df["coords"] = patch_df["coordinates"].apply(parse_coordinates)
    patch_df["lon_min"] = patch_df["coords"].apply(lambda c: c[0])
    patch_df["lat_min"] = patch_df["coords"].apply(lambda c: c[1])
    patch_df["lon_max"] = patch_df["coords"].apply(lambda c: c[2])
    patch_df["lat_max"] = patch_df["coords"].apply(lambda c: c[3])

    # Build GeoDataFrame from patch bounding boxes
    patch_gdf = gpd.GeoDataFrame(
        patch_df,
        geometry=gpd.GeoSeries.from_wkt(patch_df["geometry"]),
        crs="EPSG:4326",
    )

    # --- Load GB1900 ---
    print(f"Loading GB1900 from {args.gb1900}...")
    gb1900 = pd.read_csv(
        args.gb1900,
        encoding="utf-16",
        usecols=["pin_id", "final_text", "nation", "parish", "latitude", "longitude"],
        dtype={"pin_id": str, "final_text": str, "nation": str, "parish": str},
    )
    gb1900 = gb1900.dropna(subset=["latitude", "longitude"])
    print(f"  {len(gb1900):,} entries")

    # Build GeoDataFrame for GB1900 points
    print("Building spatial index and joining...")
    gb1900_gdf = gpd.GeoDataFrame(
        gb1900,
        geometry=gpd.points_from_xy(gb1900["longitude"], gb1900["latitude"]),
        crs="EPSG:4326",
    )

    # Spatial join: for each GB1900 point, find the patch it falls within
    joined = gpd.sjoin(
        gb1900_gdf,
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
        predicate="within",
    )
    print(f"  {len(joined):,} GB1900 entries matched to patches")

    # Apply edge margin filter
    if args.edge_margin > 0:
        pixel_margin = args.edge_margin / 512
        lon_range = joined["lon_max"] - joined["lon_min"]
        lat_range = joined["lat_max"] - joined["lat_min"]
        joined = joined[
            ((joined["longitude"] - joined["lon_min"]) / lon_range > pixel_margin)
            & ((joined["lon_max"] - joined["longitude"]) / lon_range > pixel_margin)
            & ((joined["latitude"] - joined["lat_min"]) / lat_range > pixel_margin)
            & ((joined["lat_max"] - joined["latitude"]) / lat_range > pixel_margin)
        ]
        print(
            f"  {len(joined):,} entries after {args.edge_margin}px edge margin filter"
        )

    # Compute tile pixel coords
    joined["tile_x"] = (
        (
            (joined["longitude"] - joined["lon_min"])
            / (joined["lon_max"] - joined["lon_min"])
            * 512
        )
        .astype(int)
        .clip(0, 511)
    )
    joined["tile_y"] = (
        (
            (joined["lat_max"] - joined["latitude"])
            / (joined["lat_max"] - joined["lat_min"])
            * 512
        )
        .astype(int)
        .clip(0, 511)
    )

    # --- Write JSONL output, one line per patch ---
    print(f"Writing annotations to {output_path}...")
    n_patches_written = 0
    with open(output_path, "w") as f:
        for (image_id, parent_id), group in joined.groupby(["image_id", "parent_id"]):
            annotations = []
            for _, row in group.iterrows():
                annotations.append(
                    {
                        "pin_id": row["pin_id"],
                        "text": row["final_text"],
                        "lat": round(row["latitude"], 7),
                        "lon": round(row["longitude"], 7),
                        "tile_x": int(row["tile_x"]),
                        "tile_y": int(row["tile_y"]),
                        "nation": row["nation"] if pd.notna(row["nation"]) else None,
                        "parish": row["parish"] if pd.notna(row["parish"]) else None,
                    }
                )
            record = {
                "patch_id": image_id,
                "parent_id": parent_id,
                "annotations": annotations,
            }
            f.write(json.dumps(record) + "\n")
            n_patches_written += 1

    print(
        f"Done. {n_patches_written:,} patches with GB1900 annotations written to {output_path}."
    )

    # Summary stats
    counts = joined.groupby("image_id").size()
    print(
        f"\nAnnotations per patch: min={counts.min()}, median={counts.median():.0f}, max={counts.max()}, mean={counts.mean():.1f}"
    )


if __name__ == "__main__":
    main()
