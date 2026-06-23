"""Visualise MAE reconstructions from a saved checkpoint.

Usage:
    python scripts/visualise_mae.py \
        --checkpoint data/checkpoints/mae_full/mae_step350000.pt \
        --shard-dirs data/shards_6inch_2nd_ed \
        --output data/checkpoints/mae_full/reconstructions \
        --n-images 8
"""

import argparse
import glob
from pathlib import Path

import torch
import torchvision.utils as vutils
import webdataset as wds
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from os_map_vlm.data.dataloader import IMAGENET_MEAN, IMAGENET_STD
from os_map_vlm.model.mae import MAE

MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)


def unnormalise(t):
    return (t * STD + MEAN).clamp(0, 1)


def patchify_mask(imgs, mask, patch_size=16):
    """Grey out masked patches so we can see what the encoder saw."""
    B, _C, H, W = imgs.shape
    p = patch_size
    h, w = H // p, W // p
    mask_spatial = (
        mask.reshape(B, h, w)
        .unsqueeze(1)
        .repeat(1, 1, p, p)
        .reshape(B, 1, h * p, w * p)
    )
    masked = imgs.clone()
    masked[mask_spatial.expand_as(imgs).bool()] = 0.5
    return masked


@torch.no_grad()
def reconstruct(model, imgs, device):
    imgs = imgs.to(device)
    latent, mask, ids_restore = model.encode(imgs)
    pred = model.decode(latent, ids_restore)

    # The decoder predicts per-patch normalised values — denormalise per patch
    p = model.PATCH_SIZE
    h = w = model.IMG_SIZE // p
    B = imgs.shape[0]
    target = (
        imgs.reshape(B, 3, h, p, w, p)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(B, h * w, 3 * p * p)
    )
    patch_mean = target.mean(-1, keepdim=True)
    patch_std = target.var(-1, keepdim=True).add(1e-6).sqrt()
    # recon is normalised in patch space; undo that, then undo ImageNet norm
    pred_denorm = (
        (pred * patch_std + patch_mean)
        .reshape(B, h, w, 3, p, p)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(B, 3, model.IMG_SIZE, model.IMG_SIZE)
    )

    imgs_vis = unnormalise(imgs.cpu())
    masked_vis = patchify_mask(imgs_vis, mask.cpu())
    recon_vis = pred_denorm.cpu().clamp(0, 1)

    return imgs_vis, masked_vis, recon_vis


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--shard-dirs", nargs="+", required=True)
    p.add_argument(
        "--output",
        required=True,
        help="Output directory; saves one .png per image named 0.png, 1.png, ...",
    )
    p.add_argument("--n-images", type=int, default=8)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = MAE(mask_ratio=ckpt.get("args", {}).get("mask_ratio", 0.75))
    model.load_state_dict(
        {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    )
    model.eval().to(device)
    print(f"Loaded checkpoint (step {ckpt.get('step', '?')})")

    shards = []
    for d in args.shard_dirs:
        shards.extend(sorted(glob.glob(f"{d}/shard-*.tar")))
    if not shards:
        raise RuntimeError(f"No shards found in {args.shard_dirs}")

    transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    imgs = []
    dataset = wds.WebDataset(shards[:5], shardshuffle=100).decode("pil").to_tuple("png", "json")
    for img, _ in dataset:
        imgs.append(transform(img.convert("RGB")))
        if len(imgs) >= args.n_images:
            break

    batch = torch.stack(imgs)
    originals, masked, recons = reconstruct(model, batch, device)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(len(imgs)):
        row = vutils.make_grid(
            [originals[i], masked[i], recons[i]], nrow=3, padding=4, pad_value=0.8
        )
        to_pil_image(row).save(out_dir / f"{i}.png", format="PNG")

    print(f"Saved {len(imgs)} reconstructions → {out_dir}/")
    print("Column order: original | masked input | reconstruction")


if __name__ == "__main__":
    main()
