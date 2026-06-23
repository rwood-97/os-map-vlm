"""MAE pretraining on OS map tiles.

Masked Autoencoder (He et al. 2021, arXiv:2111.06377) with a ViT-B encoder.
Trains on WebDataset shards produced by scripts/5-create_shards.py.

Smoke-test (confirm loss decreases, ~5 min on a GH200):
    python scripts/6-train_mae.py \\
        --shard-dirs data/shards_25inch data/shards_6inch_1st_ed \\
                     data/shards_town_plans_1056 data/shards_town_plans_500 \\
        --output-dir data/checkpoints/mae_smoke \\
        --max-steps 200 --warmup-steps 20

Full training:
    python scripts/6-train_mae.py \\
        --shard-dirs data/shards_25inch data/shards_6inch_1st_ed \\
                     data/shards_town_plans_1056 data/shards_town_plans_500 \\
        --output-dir data/checkpoints/mae_full \\
        --epochs 20 --compile
"""

import argparse
import glob
import json
import math
import time
from pathlib import Path

import torch
import wandb

from os_map_vlm.data.dataloader import build_mae_dataloader
from os_map_vlm.model.mae import MAE

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def cosine_lr(step, total_steps, warmup_steps, base_lr, min_lr=0.0):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))


def parse_args():
    p = argparse.ArgumentParser(description="MAE pretraining on OS map tiles")
    p.add_argument(
        "--shard-dirs",
        nargs="+",
        required=True,
        help="Directories containing shard-*.tar files",
    )
    p.add_argument("--output-dir", required=True, help="Where to save checkpoints")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument(
        "--max-steps", type=int, default=None, help="Stop after N steps (smoke test)"
    )
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument(
        "--min-lr",
        type=float,
        default=1.5e-5,
        help="Cosine decay floor (default 10%% of peak)",
    )
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--mask-ratio", type=float, default=0.75)
    p.add_argument(
        "--reconstruction-target",
        default="pixel",
        choices=["pixel", "hog"],
        help="MAE reconstruction target: pixel (default) or hog (FG-MAE style)",
    )
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the model (slower start, faster throughput)",
    )
    p.add_argument(
        "--shardshuffle",
        type=int,
        help="Shuffling between shards, passed to dataloader",
        default=100,
    )
    p.add_argument("--wandb-project", type=str, default="os-map-vlm")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args),
        )

    shards = []
    for d in args.shard_dirs:
        shards.extend(sorted(glob.glob(f"{d}/shard-*.tar")))
    if not shards:
        raise RuntimeError(f"No shards found in: {args.shard_dirs}")
    print(f"Found {len(shards)} shards across {len(args.shard_dirs)} series")

    # Estimate total steps from shard manifests for LR schedule
    total_patches = 0
    for d in args.shard_dirs:
        manifest = Path(d) / "manifest.json"
        if manifest.exists():
            total_patches += json.loads(manifest.read_text())["total_patches"]
    if total_patches > 0:
        steps_per_epoch = total_patches // args.batch_size
        print(f"  {total_patches:,} patches → ~{steps_per_epoch:,} steps/epoch")
    else:
        steps_per_epoch = 100_000
        print("  No manifests found, using estimated steps_per_epoch=100,000")
    total_steps = (
        args.max_steps if args.max_steps is not None else args.epochs * steps_per_epoch
    )

    loader = build_mae_dataloader(
        shards,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shardshuffle=args.shardshuffle,
    )

    model = MAE(
        mask_ratio=args.mask_ratio, reconstruction=args.reconstruction_target
    ).to(device)
    print(f"Reconstruction target: {args.reconstruction_target}")
    if args.compile:
        print("Compiling model with torch.compile …")
        model = torch.compile(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MAE parameters: {n_params / 1e6:.1f}M  (encoder+decoder)")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    step = 0
    start_epoch = 0
    existing_ckpts = sorted(
        output_dir.glob("mae_step*.pt"),
        key=lambda p: int(p.stem.split("mae_step")[1]),
        reverse=True,
    )
    if existing_ckpts:
        ckpt = torch.load(existing_ckpts[0], map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step = ckpt["step"]
        start_epoch = step // steps_per_epoch
        print(f"Resumed from {existing_ckpts[0]} at step {step}, epoch {start_epoch}")
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        for imgs, _ in loader:
            if args.max_steps is not None and step >= args.max_steps:
                break

            imgs = imgs.to(device, non_blocking=True)

            lr = cosine_lr(step, total_steps, args.warmup_steps, args.lr, args.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(imgs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % args.log_every == 0:
                loss_val = loss.item()
                elapsed = time.time() - t0
                print(
                    f"step={step:5d}  loss={loss_val:.4f}  lr={lr:.2e}  t={elapsed:.0f}s",
                    flush=True,
                )
                if use_wandb:
                    wandb.log(
                        {"train/loss": loss_val, "train/lr": lr, "train/step": step},
                        step=step,
                    )
            step += 1

        if (epoch + 1) % args.checkpoint_every == 0:
            ckpt_path = output_dir / f"mae_step{step:06d}.pt"
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"Checkpoint saved: {ckpt_path}")
            (output_dir / "run_args.json").write_text(json.dumps(vars(args), indent=2))

        if args.max_steps is not None and step >= args.max_steps:
            break
        print(f"--- epoch {epoch + 1}/{args.epochs} ---", flush=True)
        if use_wandb:
            wandb.log({"epoch": epoch + 1}, step=step)

    ckpt_path = output_dir / f"mae_step{step:06d}.pt"
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        },
        ckpt_path,
    )
    print(f"Checkpoint saved: {ckpt_path}")
    (output_dir / "run_args.json").write_text(json.dumps(vars(args), indent=2))
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
