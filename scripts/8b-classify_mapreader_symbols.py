"""
Classify MapReader building/railspace symbols directly on our own 512x512 patch grid.

This script slices each 512x512 patch into a 4x4 grid of 128px sub-crops (about
96x96m) and classifies each sub-crop directly with the published MapReader models.

We reproduce MapReader's own documented "test"/"val"
preprocessing (mapreader.classify.datasets.PatchDataset._default_transform: Resize((224,224)),
ToTensor, ImageNet normalize) in memory rather than using PatchDataset/ClassifierContainer,
since those expect patches already materialised as separate files on disk -- unnecessary here
and expensive at our scale (733k patches x 16 sub-crops = ~11.7M files).

Output: the same JSONL schema as the scripts this replaces, so 10-gb1900_captions.py needs
no changes:
  {"patch_id": "...", "parent_id": "...",
   "detections": [{"label": "...", "conf": ..., "tile_x": ..., "tile_y": ...}]}

Usage:
  uv run python scripts/8b-classify_mapreader_symbols.py \\
      --patches-dir data/patches_6inch_2nd_ed \\
      --label building \\
      --output data/patches_6inch_2nd_ed/mapreader_building.jsonl
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PATCH_SIZE = 512
SUB_PATCH_SIZE = 128
GRID = PATCH_SIZE // SUB_PATCH_SIZE  # 4x4 = 16 sub-crops per patch

HF_REPO = {
    "building": "hf_hub:Livingwithmachines/mr_resnest101e_finetuned_OS_6inch_2nd_ed_building",
    "railspace": "hf_hub:Livingwithmachines/mr_resnest101e_finetuned_OS_6inch_2nd_ed_railspace",
}
POSITIVE_CLASS_INDEX = (
    1  # labels_map = {0: "no", 1: label}, per the published model cards
)

# MapReader's documented "test"/"val" preprocessing, reproduced here for in-memory sub-crops.
_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# (row, col) -> (tile_x, tile_y) centre, fixed for every patch since the grid is uniform.
_SUB_CROP_CENTRES = [
    (
        col * SUB_PATCH_SIZE + SUB_PATCH_SIZE // 2,
        row * SUB_PATCH_SIZE + SUB_PATCH_SIZE // 2,
    )
    for row in range(GRID)
    for col in range(GRID)
]


class PatchSubCropDataset(Dataset):
    """Yields, per patch, a stack of its 16 128px sub-crops -- one file open per patch."""

    def __init__(self, patch_df: pd.DataFrame):
        self.patch_df = patch_df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.patch_df)

    def __getitem__(self, idx: int):
        record = self.patch_df.iloc[idx]
        image = Image.open(record["image_path"]).convert("RGB")
        crops = []
        for row in range(GRID):
            for col in range(GRID):
                box = (
                    col * SUB_PATCH_SIZE,
                    row * SUB_PATCH_SIZE,
                    (col + 1) * SUB_PATCH_SIZE,
                    (row + 1) * SUB_PATCH_SIZE,
                )
                crops.append(_TRANSFORM(image.crop(box)))
        return torch.stack(crops), record["image_id"], record["parent_id"]


def main():
    parser = argparse.ArgumentParser(
        description="Classify MapReader building/railspace symbols directly on our patch grid"
    )
    parser.add_argument(
        "--patches-dir", required=True, help="Dir containing patch_df.csv"
    )
    parser.add_argument(
        "--label",
        required=True,
        choices=sorted(HF_REPO),
        help="Symbol class to classify",
    )
    parser.add_argument(
        "--output",
        help="Output JSONL path (default: <patches-dir>/mapreader_<label>.jsonl)",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.8,
        help="Minimum predicted-class probability to keep a detection (default: 0.8)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of *patches* per batch (each contributes 16 sub-crops, so the "
        "effective GPU batch is 16x this) (default: 16)",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    patches_dir = Path(args.patches_dir)
    output_path = (
        Path(args.output)
        if args.output
        else patches_dir / f"mapreader_{args.label}.jsonl"
    )

    print(f"Loading patch_df from {patches_dir / 'patch_df.csv'}...")
    patch_df = pd.read_csv(patches_dir / "patch_df.csv")
    print(f"  {len(patch_df):,} patches -> {len(patch_df) * GRID * GRID:,} sub-crops")

    print(f"Loading {args.label} classifier from Hugging Face...")
    model = timm.create_model(HF_REPO[args.label], pretrained=True)
    model.eval().to(args.device)

    loader = DataLoader(
        PatchSubCropDataset(patch_df),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    n_written = 0
    n_patches_seen = 0
    with open(output_path, "w") as fout, torch.inference_mode():
        for crop_stack, image_ids, parent_ids in loader:
            b, n_crops, c, h, w = crop_stack.shape
            flat = crop_stack.view(b * n_crops, c, h, w).to(args.device)
            probs = torch.softmax(model(flat), dim=1).view(b, n_crops, -1)

            for i in range(b):
                detections = []
                for crop_idx, (tile_x, tile_y) in enumerate(_SUB_CROP_CENTRES):
                    conf = probs[i, crop_idx, POSITIVE_CLASS_INDEX].item()
                    if conf >= args.min_conf:
                        detections.append(
                            {
                                "label": args.label,
                                "conf": round(conf, 4),
                                "tile_x": tile_x,
                                "tile_y": tile_y,
                            }
                        )
                if detections:
                    fout.write(
                        json.dumps(
                            {
                                "patch_id": image_ids[i],
                                "parent_id": parent_ids[i],
                                "detections": detections,
                            }
                        )
                        + "\n"
                    )
                    n_written += 1

            n_patches_seen += b
            if n_patches_seen % (args.batch_size * 50) == 0:
                print(f"  {n_patches_seen:,} / {len(patch_df):,} patches classified...")

    print(
        f"Done. {n_written:,} patches with {args.label} detections written to {output_path}."
    )


if __name__ == "__main__":
    main()
