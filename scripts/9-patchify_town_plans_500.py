from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from mapreader import loader

MAPS_DIR = "./data/maps_os_town_plans_500"
PATCHES_DIR = "./data/patches_town_plans_500"
METADATA_CSV = f"{MAPS_DIR}/metadata.csv"
NUM_WORKERS = 64


def patchify_sheet(png_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = loader(png_path)
    f.add_metadata(METADATA_CSV)
    f.patchify_all(
        method="pixel",
        patch_size=512,
        path_save=PATCHES_DIR,
        skip_blank_patches=True,
    )
    return f.convert_images()


if __name__ == "__main__":
    Path(PATCHES_DIR).mkdir(parents=True, exist_ok=True)
    sheets = [str(p) for p in Path(MAPS_DIR).glob("*.png")]
    print(f"Patchifying {len(sheets)} sheets with {NUM_WORKERS} workers...")

    all_parent, all_patch, failed = [], [], []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futures = {ex.submit(patchify_sheet, s): s for s in sheets}
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
        failed_log = Path(PATCHES_DIR) / "failed_sheets.txt"
        failed_log.write_text("\n".join(failed))
        print(f"{len(failed)} sheets failed — see {failed_log}")

    parent_df = pd.concat(all_parent)
    patch_df = pd.concat(all_patch)
    parent_df.to_csv(f"{PATCHES_DIR}/parent_df.csv")
    patch_df.to_csv(f"{PATCHES_DIR}/patch_df.csv")
    print(f"Done! {len(patch_df)} patches for town plans 1:500.")
