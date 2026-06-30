"""Download OS characteristic sheets from NLS.

Supports two source types:
  - IIIF Image API endpoints: fetched at full resolution by tiling and stitching
  - Direct PDF URLs: downloaded as-is

Usage:
    uv run python scripts/download_characteristic_sheets.py \\
        --urls https://map-view.nls.uk/iiif/2/12807%2F128076891 \\
               https://maps.nls.uk/os/characteristic-sheets/Notes_on_Archaeology.pdf \\
        --output-dir data/characteristic_sheets

Or pass a text file with one URL per line:
    uv run python scripts/download_characteristic_sheets.py \\
        --url-file data/characteristic_sheet_urls.txt \\
        --output-dir data/characteristic_sheets
"""

import argparse
import io
import re
import time
from pathlib import Path
from urllib.parse import unquote

import requests
from PIL import Image


def fetch_info(base_url: str) -> dict:
    resp = requests.get(f"{base_url}/info.json", timeout=30)
    resp.raise_for_status()
    return resp.json()


def download_sheet(base_url: str, output_dir: Path) -> Path:
    info = fetch_info(base_url)
    width = info["width"]
    height = info["height"]
    tile_w = info["tiles"][0]["width"]
    tile_h = info["tiles"][0].get("height", tile_w)

    # Derive filename from URL identifier
    identifier = unquote(base_url.rstrip("/").split("/iiif/2/")[-1])
    safe_name = re.sub(r"[^\w\-]", "_", identifier)
    out_path = output_dir / f"{safe_name}.png"

    if out_path.exists():
        print(f"  Already exists, skipping: {out_path.name}")
        return out_path

    print(f"  Downloading {width}x{height}px in {tile_w}x{tile_h} tiles...")
    canvas = Image.new("RGB", (width, height))

    x = 0
    while x < width:
        y = 0
        while y < height:
            region_w = min(tile_w, width - x)
            region_h = min(tile_h, height - y)
            url = f"{base_url}/{x},{y},{region_w},{region_h}/full/0/default.jpg"

            for attempt in range(3):
                try:
                    resp = requests.get(url, timeout=30)
                    resp.raise_for_status()
                    tile = Image.open(io.BytesIO(resp.content))
                    canvas.paste(tile, (x, y))
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    print(f"    Retry {attempt + 1}/3 for tile ({x},{y}): {e}")
                    time.sleep(1)

            y += tile_h
        x += tile_w

    canvas.save(out_path, format="PNG")
    print(f"  Saved: {out_path.name} ({out_path.stat().st_size // 1024}KB)")
    return out_path


def download_pdf(url: str, output_dir: Path) -> Path:
    filename = url.rstrip("/").split("/")[-1]
    out_path = output_dir / filename

    if out_path.exists():
        print(f"  Already exists, skipping: {out_path.name}")
        return out_path

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    print(f"  Saved: {out_path.name} ({out_path.stat().st_size // 1024}KB)")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Download OS characteristic sheets from NLS IIIF endpoints"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--urls", nargs="+", help="IIIF image base URLs (no /info.json)")
    group.add_argument("--url-file", help="Text file with one IIIF base URL per line")
    parser.add_argument(
        "--output-dir",
        default="data/characteristic_sheets",
        help="Directory to save downloaded sheets (default: data/characteristic_sheets)",
    )
    args = parser.parse_args()

    if args.url_file:
        urls = [
            line.strip()
            for line in Path(args.url_file).read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    else:
        urls = args.urls

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(urls)} characteristic sheet(s) → {output_dir}/")
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] {url}")
        if url.lower().endswith(".pdf"):
            download_pdf(url, output_dir)
        else:
            download_sheet(url.rstrip("/"), output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
