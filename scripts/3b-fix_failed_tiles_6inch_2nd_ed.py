"""Re-download failed (black) tiles and patch them back into sheet PNGs.
Uses same download method as MapReader sheet downloader.

Usage:
    python fix_failed_tiles.py --maps-dir <maps_dir> --tile-url <tile_url_template/{z}/{x}/{y}.png>
"""

import argparse
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ALPHA_THRESHOLD = 5  # alpha value below which a pixel is considered transparent
RETRY_DELAY = 1.0  # seconds between retries on rate-limit


def parse_grid_bb(grid_bb: str) -> tuple[int, int, int, int, int]:
    """Parse '[(z, x_min, y_min)x(z, x_max, y_max)]' -> (z, x_min, y_min, x_max, y_max)."""
    nums = list(map(int, re.findall(r"\d+", grid_bb)))
    z, x_min, y_min, _, x_max, y_max = nums
    return z, x_min, y_min, x_max, y_max


def get_tile_size(
    img_w: int, img_h: int, x_min: int, x_max: int, y_min: int, y_max: int
) -> int:
    """Derive tile size from image dimensions and grid extent, matching MapReader's TileMerger."""
    n_x = x_max - x_min + 1  # grid_bb range is inclusive
    n_y = y_max - y_min + 1
    tile_w = img_w // n_x
    tile_h = img_h // n_y
    if tile_w != tile_h:
        raise ValueError(f"Non-square tiles inferred: {tile_w}x{tile_h}")
    return tile_w


def find_transparent_blocks(arr: np.ndarray, tile_size: int) -> list[tuple[int, int]]:
    """Return list of (col_i, row_j) for blocks containing any transparent pixels."""
    h, w = arr.shape[:2]
    failed = []
    for row_j in range(h // tile_size):
        for col_i in range(w // tile_size):
            block = arr[
                row_j * tile_size : (row_j + 1) * tile_size,
                col_i * tile_size : (col_i + 1) * tile_size,
            ]
            if (block[:, :, 3] < ALPHA_THRESHOLD).any():
                failed.append((col_i, row_j))
    return failed


def fetch_tile(url: str, tile_size: int, retries: int = 3) -> Image.Image | None:
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                img = Image.open(BytesIO(r.content)).convert("RGBA")
                arr = np.array(img)
                if (arr[:, :, 3] < ALPHA_THRESHOLD).all():
                    return None  # server returned a fully transparent tile
                return img
            elif r.status_code == 429:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                return None
        except Exception:
            time.sleep(RETRY_DELAY)
    return None


def fix_sheet(png_path: Path, grid_bb: str, tile_url_template: str) -> int:
    z, x_min, y_min, x_max, y_max = parse_grid_bb(grid_bb)
    img = Image.open(png_path).convert("RGBA")
    arr = np.array(img)
    h, w = arr.shape[:2]

    tile_size = get_tile_size(w, h, x_min, x_max, y_min, y_max)

    transparent_blocks = find_transparent_blocks(arr, tile_size)
    if not transparent_blocks:
        return 0

    patched = 0
    for col_i, row_j in transparent_blocks:
        x = x_min + col_i
        y = y_min + row_j
        url = tile_url_template.format(z=z, x=x, y=y)
        tile = fetch_tile(url, tile_size)
        if tile is not None:
            tile_arr = np.array(tile)
            arr[
                row_j * tile_size : (row_j + 1) * tile_size,
                col_i * tile_size : (col_i + 1) * tile_size,
            ] = tile_arr
            patched += 1

    if patched:
        Image.fromarray(arr).save(png_path)

    return patched


def fix_sheet_worker(args: tuple[Path, str, str]) -> tuple[str, int]:
    png_path, grid_bb, tile_url_template = args
    n = fix_sheet(png_path, grid_bb, tile_url_template)
    return str(png_path), n


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps-dir", required=True)
    parser.add_argument(
        "--tile-url",
        required=True,
        help="Tile URL template with {z}, {x}, {y} placeholders",
    )
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    import pandas as pd

    maps_dir = Path(args.maps_dir)
    metadata = pd.read_csv(maps_dir / "metadata.csv")

    tasks = []
    for _, row in metadata.iterrows():
        png_path = maps_dir / row["name"]
        if not png_path.exists():
            print(f"  SKIP {row['name']} (file not found)")
            continue
        tasks.append((png_path, row["grid_bb"], args.tile_url))

    print(f"Checking {len(tasks)} sheets with {args.workers} workers...")

    total_patched = 0
    failed = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(fix_sheet_worker, t): t[0] for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            png_path = futures[fut]
            try:
                _, n = fut.result()
                total_patched += n
                if n:
                    print(f"  patched {n} tiles in {Path(png_path).name}")
            except Exception as e:
                failed.append(str(png_path))
                print(f"  FAILED {Path(png_path).name}: {e}")
            if i % 100 == 0:
                print(f"  {i}/{len(tasks)} sheets done")

    if failed:
        failed_log = maps_dir / "fix_failed_sheets.txt"
        failed_log.write_text("\n".join(failed))
        print(f"{len(failed)} sheets failed — see {failed_log}")

    print(f"\nDone. Total tiles patched: {total_patched}")
