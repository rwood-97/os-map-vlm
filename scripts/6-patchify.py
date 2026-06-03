import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from mapreader import loader


def patchify_sheet(args: tuple[str, str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    png_path, metadata_csv, patches_dir = args
    f = loader(png_path)
    f.add_metadata(metadata_csv, ignore_mismatch=True)
    f.patchify_all(
        method="pixel",
        patch_size=512,
        path_save=patches_dir,
        skip_blank_patches=True,
    )
    return f.convert_images()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps-dir", required=True)
    parser.add_argument("--patches-dir")
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    maps_dir = args.maps_dir
    patches_dir = args.patches_dir or maps_dir.replace("maps_", "patches_")
    metadata_csv = f"{maps_dir}/metadata.csv"

    Path(patches_dir).mkdir(parents=True, exist_ok=True)
    sheets = [str(p) for p in Path(maps_dir).glob("*.png")]
    print(
        f"Patchifying {len(sheets)} sheets from {maps_dir} with {args.workers} workers..."
    )

    all_parent, all_patch, failed = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(patchify_sheet, (s, metadata_csv, patches_dir)): s for s in sheets
        }
        for i, fut in enumerate(as_completed(futures), 1):
            sheet = futures[fut]
            try:
                parent_df, patch_df = fut.result()
                all_parent.append(parent_df)
                all_patch.append(patch_df)
            except Exception as e:
                failed.append(sheet)
                print(f"  FAILED {sheet}: {e}")
            if i % 100 == 0:
                print(f"  {i}/{len(sheets)} sheets done")

    if failed:
        failed_log = Path(patches_dir) / "failed_sheets.txt"
        failed_log.write_text("\n".join(failed))
        print(f"{len(failed)} sheets failed — see {failed_log}")

    parent_df = pd.concat(all_parent)
    patch_df = pd.concat(all_patch)
    parent_df.to_csv(f"{patches_dir}/parent_df.csv")
    patch_df.to_csv(f"{patches_dir}/patch_df.csv")
    print(f"Done! {len(patch_df)} patches saved to {patches_dir}.")
