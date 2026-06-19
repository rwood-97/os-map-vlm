"""Linear probe evaluation for MAE encoder on MapReader annotated patches.

Assigns building/railspace labels to 512x512 patches via spatial join with the
MapReader georeferenced annotations, then trains a frozen-encoder linear probe
and reports F1.

Prerequisites:
    Run 4-patchify.py on the MapReader annotation parent maps first:
        python scripts/4-patchify.py \\
            --maps-dir path/to/MapReader_Annotations_2024/maps \\
            --patches-dir data/patches_mapreader_probe

Usage:
    python scripts/7-linear_probe_eval.py \\
        --patches-dir data/patches_mapreader_probe \\
        --annot-building path/to/annots_building_georeferenced.csv \\
        --annot-railspace path/to/annots_railspace_all_georeferenced.csv \\
        --checkpoint data/checkpoints/mae_full/mae_step350000.pt \\
        --pretrain-metadata path/to/6inch_2nd_ed/metadata.csv
"""

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ViT_B_16_Weights, vit_b_16

from os_map_vlm.data.dataloader import IMAGENET_MEAN, IMAGENET_STD
from os_map_vlm.model.mae import MAE

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_features_mae(model: MAE, imgs: torch.Tensor) -> torch.Tensor:
    """MAE encoder: mean-pool patch tokens (no masking, no CLS token)."""
    x = model.patch_embed(imgs) + model.pos_embed
    for blk in model.encoder_blocks:
        x = blk(x)
    return model.encoder_norm(x).mean(dim=1)  # (B, 768)


@torch.no_grad()
def extract_features_vit(model: nn.Module, imgs: torch.Tensor) -> torch.Tensor:
    """torchvision ViT-B/16: mean-pool patch tokens (exclude CLS) to match MAE."""
    x = model._process_input(imgs)  # (B, N, 768)
    cls = model.class_token.expand(x.shape[0], -1, -1)
    x = torch.cat([cls, x], dim=1)
    x = model.encoder(x)  # (B, 1+N, 768)
    return x[:, 1:, :].mean(dim=1)  # (B, 768) — patch tokens only


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def make_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class PatchDataset(Dataset):
    def __init__(self, image_paths: list[str], img_size: int):
        self.image_paths = image_paths
        self.transform = make_transform(img_size)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.image_paths[idx]).convert("RGB"))


def extract_all_embeddings(
    model: nn.Module,
    extract_fn,
    image_paths: list[str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    img_size: int = 512,
) -> torch.Tensor:
    loader = DataLoader(
        PatchDataset(image_paths, img_size),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    all_emb = []
    for i, imgs in enumerate(loader):
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            all_emb.append(
                extract_fn(model, imgs.to(device, non_blocking=True)).float().cpu()
            )
        if (i + 1) % 50 == 0:
            print(
                f"  {min((i + 1) * batch_size, len(image_paths))}/{len(image_paths)} embedded",
                flush=True,
            )
    return torch.cat(all_emb)


# ---------------------------------------------------------------------------
# Geo-space label assignment
# ---------------------------------------------------------------------------


def assign_labels(
    patch_df: pd.DataFrame, annot_csv: str, positive_label: str
) -> pd.Series:
    """For each 512x512 patch, True if any intersecting annotation has the positive label.

    Uses geopandas sjoin with an R-tree index over EPSG:4326 polygon geometries.
    """
    annot_df = pd.read_csv(annot_csv)
    annot_gdf = gpd.GeoDataFrame(
        annot_df, geometry=gpd.GeoSeries.from_wkt(annot_df["polygon"]), crs="EPSG:4326"
    )
    annot_gdf["is_positive"] = annot_gdf["label"] == positive_label

    patch_gdf = gpd.GeoDataFrame(
        patch_df[["image_id"]],
        geometry=gpd.GeoSeries.from_wkt(patch_df["geometry"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(
        patch_gdf,
        annot_gdf[["is_positive", "geometry"]],
        how="left",
        predicate="intersects",
    )
    agg = joined.groupby("image_id")["is_positive"].any()
    return patch_df["image_id"].map(agg).fillna(False).astype(bool)


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------


def train_probe(
    train_emb: torch.Tensor,
    train_label: torch.Tensor,
    val_emb: torch.Tensor,
    val_label: torch.Tensor,
    task_name: str,
    epochs: int,
    lr: float,
    device: torch.device,
) -> float:
    n_pos = train_label.sum().item()
    n_neg = len(train_label) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    print(
        f"  {task_name}: {int(n_pos)} pos / {int(n_neg)} neg  (pos_weight={pos_weight.item():.1f})"
    )

    probe = nn.Linear(train_emb.shape[1], 1).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    train_emb, train_label = train_emb.to(device), train_label.to(device)

    for _ in range(epochs):
        probe.train()
        optimizer.zero_grad()
        F.binary_cross_entropy_with_logits(
            probe(train_emb).squeeze(1), train_label, pos_weight=pos_weight
        ).backward()
        optimizer.step()

    probe.eval()
    with torch.no_grad():
        preds = (probe(val_emb.to(device)).squeeze(1) > 0).cpu().numpy()
    targets = val_label.numpy().astype(int)
    report = classification_report(
        targets,
        preds,
        target_names=["negative", "positive"],
        output_dict=True,
        zero_division=0,
    )
    print(f"  {task_name} val F1: {report['positive']['f1-score']:.4f}")
    print(
        classification_report(
            targets, preds, target_names=["negative", "positive"], zero_division=0
        )
    )
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--patches-dir",
        required=True,
        help="Directory containing 512x512 patches created from MapReader annotation maps",
    )
    p.add_argument(
        "--annot-building", required=True, help="annots_building_georeferenced.csv"
    )
    p.add_argument(
        "--annot-railspace",
        required=True,
        help="annots_railspace_all_georeferenced.csv",
    )
    p.add_argument("--checkpoint", required=True, help="MAE checkpoint .pt file")
    p.add_argument(
        "--pretrain-metadata",
        default=None,
        help="metadata.csv from 6-inch 2nd ed pretraining maps — used to exclude pretraining maps from eval",
    )
    p.add_argument("--output-dir", default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir or Path(args.checkpoint).parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    patch_df = pd.read_csv(Path(args.patches_dir) / "patch_df.csv")
    print(f"Loaded {len(patch_df):,} patches from {args.patches_dir}")

    if args.pretrain_metadata:
        pretrain_ids = set(pd.read_csv(args.pretrain_metadata)["name"].unique())
        before = len(patch_df)
        patch_df = patch_df[~patch_df["parent_id"].isin(pretrain_ids)]
        print(
            f"Excluded {before - len(patch_df):,} patches from {len(pretrain_ids):,} pretraining maps → {len(patch_df):,} remaining"
        )
    else:
        print(
            "WARNING: --pretrain-metadata not provided, not filtering pretraining maps"
        )

    print("Assigning labels via spatial join...")
    patch_df = patch_df.copy()
    patch_df["label_building"] = assign_labels(
        patch_df, args.annot_building, "building"
    ).values
    patch_df["label_railspace"] = assign_labels(
        patch_df, args.annot_railspace, "railspace"
    ).values
    print(
        f"  building:  {patch_df['label_building'].sum():,} pos / {(~patch_df['label_building']).sum():,} neg"
    )
    print(
        f"  railspace: {patch_df['label_railspace'].sum():,} pos / {(~patch_df['label_railspace']).sum():,} neg"
    )

    # Train/val split at parent map level to avoid leakage
    parent_ids = patch_df["parent_id"].unique()
    rng.shuffle(parent_ids)
    n_val = max(1, int(len(parent_ids) * args.val_split))
    train_df = patch_df[patch_df["parent_id"].isin(set(parent_ids[n_val:]))]
    val_df = patch_df[patch_df["parent_id"].isin(set(parent_ids[:n_val]))]
    print(
        f"Train: {len(train_df):,} patches ({len(parent_ids) - n_val} maps) | Val: {len(val_df):,} patches ({n_val} maps)"
    )

    def run_probe(model, extract_fn, cache_path, label, img_size=512):
        if cache_path.exists():
            print(f"Loading cached {label} embeddings from {cache_path}")
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            return cached["train_emb"], cached["val_emb"]
        print(f"Extracting {label} train embeddings...")
        train_emb = extract_all_embeddings(
            model,
            extract_fn,
            train_df["image_path"].tolist(),
            args.batch_size,
            args.num_workers,
            device,
            img_size,
        )
        print(f"Extracting {label} val embeddings...")
        val_emb = extract_all_embeddings(
            model,
            extract_fn,
            val_df["image_path"].tolist(),
            args.batch_size,
            args.num_workers,
            device,
            img_size,
        )
        torch.save({"train_emb": train_emb, "val_emb": val_emb}, cache_path)
        print(f"Cached to {cache_path}")
        return train_emb, val_emb

    # MAE encoder
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mae = MAE(mask_ratio=ckpt.get("args", {}).get("mask_ratio", 0.75))
    mae.load_state_dict(
        {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    )
    mae.eval().to(device)
    for p in mae.parameters():
        p.requires_grad_(False)
    print(f"Loaded MAE checkpoint (step {ckpt.get('step', '?')})")
    mae_train_emb, mae_val_emb = run_probe(
        mae, extract_features_mae, output_dir / "probe_embeddings_mae.pt", "MAE"
    )

    # ImageNet ViT-B/16 baseline (same architecture, different pretraining)
    # Uses 224x224 input — its native size (positional embeddings are fixed at 196 tokens)
    vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).eval().to(device)
    for p in vit.parameters():
        p.requires_grad_(False)
    print("Loaded ImageNet ViT-B/16 baseline")
    vit_train_emb, vit_val_emb = run_probe(
        vit,
        extract_features_vit,
        output_dir / "probe_embeddings_vit.pt",
        "ViT-B/16",
        img_size=224,
    )

    all_results = {}
    for encoder_label, train_emb, val_emb in [
        ("mae", mae_train_emb, mae_val_emb),
        ("imagenet_vit_b16", vit_train_emb, vit_val_emb),
    ]:
        all_results[encoder_label] = {}
        for task, col in [
            ("building", "label_building"),
            ("railspace", "label_railspace"),
        ]:
            print(f"\n--- {task} [{encoder_label}] ---")
            report = train_probe(
                train_emb,
                torch.tensor(train_df[col].values, dtype=torch.float32),
                val_emb,
                torch.tensor(val_df[col].values, dtype=torch.float32),
                f"{task} [{encoder_label}]",
                args.epochs,
                args.lr,
                device,
            )
            all_results[encoder_label][task] = report

    print("\n=== Results ===")
    for encoder_label, results in all_results.items():
        for task, r in results.items():
            print(f"  {encoder_label} {task}: F1={r['positive']['f1-score']:.4f}")

    results_path = output_dir / "probe_results.json"
    results_path.write_text(
        json.dumps({"checkpoint": args.checkpoint, "results": all_results}, indent=2)
    )
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
