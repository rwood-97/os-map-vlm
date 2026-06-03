"""
Convert patchified OS map tiles into WebDataset shards for MAE training.

Each shard is a .tar file. Each sample in the tar has:
    {key}.png  — raw PNG bytes (may be RGBA; converted to RGB in dataloader)
    {key}.json — metadata: image_id, parent_id, coordinates, pixel_bounds, series, scale, edition

Usage
-----
Full series (all patches):
    python scripts/11-create_shards.py \\
        --series town_plans_500 \\
        --patches-dir data/patches_town_plans_500 \\
        --output-dir data/shards_town_plans_500

Subsampled (e.g. for smoke-test or balanced MAE training set):
    python scripts/11-create_shards.py \\
        --series 6inch_2nd_ed \\
        --patches-dir data/patches_6inch_2nd_ed \\
        --output-dir data/shards_6inch_2nd_ed \\
        --max-patches 500000

Shards can be fed to webdataset.WebDataset via a braceexpand glob:
    "data/shards_town_plans_500/shard-{000000..001234}.tar"
or as a list from glob.glob("data/shards_town_plans_500/shard-*.tar").
"""

import argparse
import ast
import json
from pathlib import Path

import pandas as pd
import webdataset as wds
from tqdm import tqdm

SERIES_META: dict[str, dict[str, str]] = {
    "town_plans_500": {"scale": "1:500", "edition": "OS Town Plans"},
    "town_plans_1056": {"scale": "1:1056", "edition": "OS Town Plans"},
    "25inch": {"scale": "1:2500", "edition": "OS 25-inch County Series"},
    "6inch_1st_ed": {"scale": "1:10560", "edition": "OS 6-inch 1st Edition"},
    "6inch_2nd_ed": {"scale": "1:10560", "edition": "OS 6-inch 2nd Edition"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create WebDataset shards from patchified OS map tiles"
    )
    parser.add_argument(
        "--series",
        required=True,
        choices=list(SERIES_META),
        help="Map series name",
    )
    parser.add_argument(
        "--patches-dir",
        required=True,
        help="Directory containing patch PNG files and patch_df.csv",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for shard .tar files",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=5000,
        help="Patches per shard (default: 5000, ~1 GB at ~200 KB/patch)",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=None,
        help="Maximum patches to write; randomly sampled when set (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling when --max-patches is set (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patches_dir = Path(args.patches_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    series_meta = SERIES_META[args.series]

    # Load patch manifest
    patch_csv = patches_dir / "patch_df.csv"
    print(f"Reading {patch_csv} …")
    df = pd.read_csv(
        patch_csv,
        usecols=["image_id", "parent_id", "image_path", "pixel_bounds", "coordinates"],
    )
    print(f"  {len(df):,} patches in manifest")

    if args.max_patches is not None and len(df) > args.max_patches:
        df = df.sample(n=args.max_patches, random_state=args.seed).reset_index(drop=True)
        print(f"  Sampled {len(df):,} patches (seed={args.seed})")

    # Write shards
    shard_pattern = str(output_dir / "shard-%06d.tar")
    n_written = 0
    n_failed = 0

    with wds.ShardWriter(shard_pattern, maxcount=args.shard_size) as writer:
        for row in tqdm(df.itertuples(index=False), total=len(df), desc="Writing shards"):
            image_path = Path(row.image_path)
            if not image_path.exists():
                n_failed += 1
                continue

            try:
                png_bytes = image_path.read_bytes()
            except OSError as e:
                print(f"\n  Failed to read {image_path}: {e}")
                n_failed += 1
                continue

            # Numeric key — WebDataset groups tar entries by the part before the
            # last dot, so keys must not contain dots. Original image_id is in JSON.
            key = f"{args.series}_{n_written:010d}"

            # Coordinate strings in patch_df are "(lon_min, lat_min, lon_max, lat_max)"
            coords = list(ast.literal_eval(row.coordinates))
            pixel_bounds = list(ast.literal_eval(row.pixel_bounds))

            writer.write({
                "__key__": key,
                "png": png_bytes,
                "json": json.dumps({
                    "image_id": str(row.image_id),
                    "parent_id": str(row.parent_id),
                    "coordinates": coords,
                    "pixel_bounds": pixel_bounds,
                    "series": args.series,
                    "scale": series_meta["scale"],
                    "edition": series_meta["edition"],
                }).encode(),
            })
            n_written += 1

    n_shards = (n_written + args.shard_size - 1) // args.shard_size
    print(f"\nDone: {n_written:,} patches across {n_shards} shards")
    if n_failed:
        print(f"WARNING: {n_failed:,} patches skipped (missing or unreadable)")

    manifest = {
        "series": args.series,
        "scale": series_meta["scale"],
        "edition": series_meta["edition"],
        "total_patches": n_written,
        "n_shards": n_shards,
        "shard_size": args.shard_size,
        "shard_pattern": shard_pattern,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
