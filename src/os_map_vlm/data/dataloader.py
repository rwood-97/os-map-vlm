"""Streaming WebDataset dataloader for MAE pretraining on OS map tiles.

Shards are produced by scripts/7-create_shards.py. Each shard sample has:
    .png  — raw image bytes (may be RGBA)
    .json — metadata dict (image_id, parent_id, coordinates, pixel_bounds, series, scale, edition, survey_date_start, survey_date_end, pub_date_start, pub_date_end)

The dataloader yields (image_tensor, coords) batches.

Typical usage in a training script
------------------------------------
    import glob
    from os_map_vlm.data.dataloader import build_mae_dataloader

    shards = sorted(glob.glob("data/shards_6inch_2nd_ed/shard-*.tar"))
    loader = build_mae_dataloader(shards, batch_size=64, num_workers=8)

    for images, coords in loader:
        # images: (B, 3, 512, 512) float32
        # coords: (B, 4) float32  [lon_min, lat_min, lon_max, lat_max]
        loss = model(images)
        ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import webdataset as wds
from torchvision import transforms

if TYPE_CHECKING:
    from collections.abc import Callable

# ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_mae_transform(img_size: int = 512) -> transforms.Compose:
    """MAE augmentation pipeline for 512x512 OS map tiles.

    Scale=(0.2, 1.0) trains the encoder to reconstruct from partial context regardless of zoom level.
    Rotations 0/90/180/270 degrees ensures good performance across different map orientations (e.g. text rotation).
    Normalised with ImageNet statistics.

    Note: scale 0.2 might be too cropped, worth to try scale=(0.5, 1.0) too.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                img_size,
                scale=(0.2, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.Lambda(
                lambda img: transforms.functional.rotate(
                    img, [0, 90, 180, 270][torch.randint(4, (1,)).item()]
                )
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _make_preprocess(transform: Callable) -> Callable:
    def preprocess(sample: tuple) -> tuple[torch.Tensor, torch.Tensor]:
        image, meta = sample
        image = image.convert("RGB")  # Patches are RGBA
        image = transform(image)
        # meta["coordinates"] comes back as a list from JSON: [lon_min, lat_min, lon_max, lat_max]
        coords = torch.tensor(meta["coordinates"], dtype=torch.float32)
        return image, coords

    return preprocess


def build_mae_dataloader(
    shard_patterns: list[str] | str,
    batch_size: int,
    num_workers: int = 4,
    shuffle_buffer: int = 1000,
    img_size: int = 512,
    distributed: bool = False,
    partial_batches: bool = False,
) -> wds.WebLoader:
    """Build a streaming WebDataset dataloader for MAE pretraining.

    Parameters
    ----------
    shard_patterns:
        Shard path(s) or braceexpand glob pattern(s). Can be a single string, a list of absolute paths, or a braceexpand pattern such as ``"data/shards_6inch_2nd_ed/shard-{000000..001999}.tar"``.
        To mix shards from multiple series pass a concatenated list such as ``["data/shards_series1/shard-{000000..000999}.tar", "data/shards_series2/shard-{000000..000999}.tar"]``.
    batch_size:
        N samples per batch.
    num_workers:
        DataLoader worker processes. Default 4.
    shuffle_buffer:
        Within-shard shuffle buffer (number of samples). Shards are also shuffled at the shard level (``shardshuffle=True``).
    img_size:
        Spatial size fed to the ViT encoder.
        512 for ViT-B with 16x16 patches. Default 512.
    distributed:
        If True, shards are split across nodes (``wds.split_by_node``) and across workers within a node.
        Set True for multi-node runs; Default is False for single-GPU runs.
    partial_batches:
        If True, the final incomplete batch is yielded.
        Default False so all batches are the same size.

    Returns
    -------
    wds.WebLoader
        Yields ``(images, coords)`` tuples:
        - ``images``: ``(B, 3, img_size, img_size)`` float32 on CPU
        - ``coords``: ``(B, 4)`` float32 — ``[lon_min, lat_min, lon_max, lat_max]``

        Move tensors to the GPU in the training loop with ``.to(device)``.
    """
    transform = build_mae_transform(img_size)
    preprocess = _make_preprocess(transform)

    nodesplitter = wds.split_by_node if distributed else wds.single_node_only

    dataset = (
        wds.WebDataset(
            shard_patterns,
            shardshuffle=True,
            nodesplitter=nodesplitter,
        )
        .shuffle(shuffle_buffer)
        .decode("pil")
        .to_tuple("png", "json")
        .map(preprocess)
        .batched(batch_size, partial=partial_batches)
    )

    # batch_size=None because .batched() above already groups samples;
    # passing a non-None value here would double-batch.
    return wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
    )
