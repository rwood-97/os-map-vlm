"""
Convert patchified OS map tiles into WebDataset shards for MAE training.

Each shard is a .tar file. Each sample in the tar has:
    {key}.png  — raw PNG bytes (may be RGBA; converted to RGB in dataloader)
    {key}.json — metadata: image_id, parent_id, coordinates, pixel_bounds, series, scale, edition,
                 survey_date_start, survey_date_end, revised_date_start, revised_date_end,
                 pub_date_start, pub_date_end

Usage
-----
Full series (all patches):
    python scripts/7-create_shards.py \\
        --series town_plans_500 \\
        --patches-dir data/patches_town_plans_500 \\
        --output-dir data/shards_town_plans_500

Subsampled (e.g. for smoke-test or balanced MAE training set):
    python scripts/7-create_shards.py \\
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
import re
from pathlib import Path

import pandas as pd
import webdataset as wds
from tqdm import tqdm

SERIES_META: dict[str, dict[str, str]] = {
    "town_plans_500": {
        "scale": "1:500",
        "edition": "OS Town Plans",
        "nls_metadata": "data/metadata_ostowns_500.geojson",
    },
    "town_plans_1056": {
        "scale": "1:1056",
        "edition": "OS Town Plans",
        "nls_metadata": "data/metadata_nls_OS_Town_Plans_Eng_1056.geojson",
    },
    "25inch": {
        "scale": "1:2500",
        "edition": "OS 25-inch County Series",
        "nls_metadata": "data/metadata_nls_OS_25_Inch.geojson",
    },
    "6inch_1st_ed": {
        "scale": "1:10560",
        "edition": "OS 6-inch 1st Edition",
        "nls_metadata": "data/metadata_nls_six_inch_1st_ed.geojson",
    },
    "6inch_2nd_ed": {
        "scale": "1:10560",
        "edition": "OS 6-inch 2nd Edition",
        "nls_metadata": "data/metadata_nls_six_inch_2nd_ed.geojson",
    },
}


def _parse_wfs_date(wfs_title: str, keyword: str) -> tuple[str | None, str | None]:
    """Extract start/end date strings for a keyword (Surveyed|Revised|Published) from WFS_TITLE.

    Returns (start, end) where values may include a 'ca. ' prefix.
    Both start and end are set to the same value for single approximate dates.
    Returns (None, None) if no match.
    """
    # regex is: keyword: (optional "ca. ") year (optional "to" year)
    # match groups are: 1=ca. , 2=start year, 3=end year (if present)
    match = re.search(
        rf"{keyword}:\s*(ca\.\s*)?(\d{{4}})(?:\s+to\s+(\d{{4}}))?",
        wfs_title,
        flags=re.IGNORECASE,
    )
    if not match:
        return None, None

    prefix = "ca. " if match.group(1) else ""
    start = prefix + match.group(2)
    end = prefix + (match.group(3) if match.group(3) else match.group(2))
    return start, end


def load_nls_dates(geojson_path: Path) -> dict[str, dict[str, str | None]]:
    """Creates a dictionary from NLS image ID to survey/revised/published date strings.

    Uses SUR_STA/SUR_END/REV_STA/REV_END/PUB_STA/PUB_END fields in nls metadata when present and falls back to parsing WFS_TITLE if those fields not found.
    """
    with geojson_path.open() as f:
        data = json.load(f)
    lookup = {}
    for feature in data["features"]:
        props = feature["properties"]
        image_id = str(props["IMAGE"])
        wfs_title = props.get("WFS_TITLE") or ""

        sur_sta = str(props["SUR_STA"]) if props.get("SUR_STA") else None
        sur_end = str(props["SUR_END"]) if props.get("SUR_END") else None
        rev_sta = str(props["REV_STA"]) if props.get("REV_STA") else None
        rev_end = str(props["REV_END"]) if props.get("REV_END") else None
        pub_sta = str(props["PUB_STA"]) if props.get("PUB_STA") else None
        pub_end = str(props["PUB_END"]) if props.get("PUB_END") else None

        if sur_sta is None:
            sur_sta, sur_end = _parse_wfs_date(wfs_title, "Surveyed")

        if rev_sta is None:
            rev_sta, rev_end = _parse_wfs_date(wfs_title, "Revised")

        if pub_sta is None:
            pub_sta, pub_end = _parse_wfs_date(wfs_title, "Published")

        lookup[image_id] = {
            "survey_date_start": sur_sta,
            "survey_date_end": sur_end,
            "revised_date_start": rev_sta,
            "revised_date_end": rev_end,
            "pub_date_start": pub_sta,
            "pub_date_end": pub_end,
        }
    return lookup


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
    if not patches_dir.exists():
        raise FileNotFoundError(f"Patches directory not found: {patches_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.series not in SERIES_META:
        raise ValueError(
            f"Unknown series: {args.series}. Valid options: {list(SERIES_META)}"
        )
    series_meta = SERIES_META[args.series]

    # Load NLS date metadata
    nls_metadata_path = Path(series_meta["nls_metadata"])
    if nls_metadata_path.exists():
        nls_dates = load_nls_dates(nls_metadata_path)
        print(
            f"Loaded NLS date metadata for {len(nls_dates):,} sheets from {nls_metadata_path}"
        )
    else:
        nls_dates = {}
        print(
            f"WARNING: NLS metadata not found at {nls_metadata_path} — survey/revision/pub dates will be null"
        )

    # Load patch manifest
    patch_csv = patches_dir / "patch_df.csv"
    print(f"Reading {patch_csv} …")
    df = pd.read_csv(
        patch_csv,
        usecols=["image_id", "parent_id", "image_path", "pixel_bounds", "coordinates"],
    )
    print(f"  {len(df):,} patches in manifest")

    if args.max_patches is not None:
        if len(df) > args.max_patches:
            df = df.sample(n=args.max_patches, random_state=args.seed).reset_index(
                drop=True
            )
            print(f"  Sampled {len(df):,} patches (seed={args.seed})")
        else:
            print(
                f"  max_patches={args.max_patches} is greater than total patches; using all {len(df):,}"
            )

    # Write shards
    shard_pattern = str(output_dir / "shard-%06d.tar")
    n_written = 0
    n_failed = 0

    with wds.ShardWriter(shard_pattern, maxcount=args.shard_size) as writer:
        for row in tqdm(
            df.itertuples(index=False), total=len(df), desc="Writing shards"
        ):
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

            # WebDataset groups tar entries by the part before the last dot.
            # Original image_id will be in the JSON metadata, so we can use a simple numeric key here.
            key = f"{args.series}_{n_written:010d}"

            # Coordinate strings in patch_df are "(lon_min, lat_min, lon_max, lat_max)"
            coords = list(ast.literal_eval(row.coordinates))
            pixel_bounds = list(ast.literal_eval(row.pixel_bounds))

            # parent_id is e.g. "map_103680731.png" - strip prefix/suffix to get NLS image ID
            nls_image_id = str(row.parent_id).removeprefix("map_").removesuffix(".png")
            dates = nls_dates.get(nls_image_id, {})

            writer.write(
                {
                    "__key__": key,
                    "png": png_bytes,
                    "json": json.dumps(
                        {
                            "image_id": str(row.image_id),
                            "parent_id": str(row.parent_id),
                            "coordinates": coords,
                            "pixel_bounds": pixel_bounds,
                            "series": args.series,
                            "scale": series_meta["scale"],
                            "edition": series_meta["edition"],
                            "survey_date_start": dates.get("survey_date_start"),
                            "survey_date_end": dates.get("survey_date_end"),
                            "revised_date_start": dates.get("revised_date_start"),
                            "revised_date_end": dates.get("revised_date_end"),
                            "pub_date_start": dates.get("pub_date_start"),
                            "pub_date_end": dates.get("pub_date_end"),
                        }
                    ).encode(),
                }
            )
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
