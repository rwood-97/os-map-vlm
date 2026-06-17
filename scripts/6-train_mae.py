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
import torch.nn as nn
import torch.nn.functional as F
import wandb

from os_map_vlm.data.dataloader import build_mae_dataloader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    def __init__(self, img_size=512, patch_size=16, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, D))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MAE(nn.Module):
    """Masked Autoencoder with ViT-B encoder (512x512 input, 16x16 patches)."""

    PATCH_SIZE = 16
    IMG_SIZE = 512
    NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 1024
    PIXELS_PER_PATCH = PATCH_SIZE**2 * 3  # 768

    def __init__(
        self,
        mask_ratio=0.75,
        encoder_dim=768,
        encoder_depth=12,
        encoder_heads=12,
        decoder_dim=512,
        decoder_depth=8,
        decoder_heads=16,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio

        # Encoder (ViT-B)
        self.patch_embed = PatchEmbed(self.IMG_SIZE, self.PATCH_SIZE, encoder_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.NUM_PATCHES, encoder_dim))
        self.encoder_blocks = nn.ModuleList(
            [Block(encoder_dim, encoder_heads, mlp_ratio) for _ in range(encoder_depth)]
        )
        self.encoder_norm = nn.LayerNorm(encoder_dim)

        # Decoder
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.NUM_PATCHES, decoder_dim))
        self.decoder_blocks = nn.ModuleList(
            [Block(decoder_dim, decoder_heads, mlp_ratio) for _ in range(decoder_depth)]
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, self.PIXELS_PER_PATCH)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _patchify(self, imgs):
        p = self.PATCH_SIZE
        B, C, H, W = imgs.shape
        h, w = H // p, W // p
        return imgs.reshape(B, C, h, p, w, p).permute(0, 2, 4, 1, 3, 5).reshape(B, h * w, C * p * p)

    def _random_masking(self, x):
        B, N, D = x.shape
        keep = int(N * (1 - self.mask_ratio))
        ids_shuffle = torch.rand(B, N, device=x.device).argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        ids_keep = ids_shuffle[:, :keep]
        x_vis = x.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(B, N, device=x.device)
        mask[:, :keep] = 0
        mask = mask.gather(1, ids_restore)  # 1 = masked, 0 = visible
        return x_vis, mask, ids_restore

    def encode(self, x):
        x = self.patch_embed(x) + self.pos_embed
        x, mask, ids_restore = self._random_masking(x)
        for blk in self.encoder_blocks:
            x = blk(x)
        return self.encoder_norm(x), mask, ids_restore

    def decode(self, x, ids_restore):
        x = self.decoder_embed(x)
        B, N_vis, D = x.shape
        N = ids_restore.shape[1]
        x_full = torch.cat([x, self.mask_token.expand(B, N - N_vis, -1)], dim=1)
        x_full = x_full.gather(1, ids_restore.unsqueeze(-1).expand(-1, -1, D))
        x_full = x_full + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x_full = blk(x_full)
        return self.decoder_pred(self.decoder_norm(x_full))

    def forward(self, imgs):
        latent, mask, ids_restore = self.encode(imgs)
        pred = self.decode(latent, ids_restore)

        # Per-patch normalised MSE on masked patches (MAE paper §3.1)
        target = self._patchify(imgs)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6).sqrt()
        loss = (F.mse_loss(pred, target, reduction="none").mean(dim=-1) * mask).sum() / mask.sum()
        return loss


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
    p.add_argument("--shard-dirs", nargs="+", required=True, help="Directories containing shard-*.tar files")
    p.add_argument("--output-dir", required=True, help="Where to save checkpoints")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=None, help="Stop after N steps (smoke test)")
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=400)
    p.add_argument("--mask-ratio", type=float, default=0.75)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--compile", action="store_true", help="torch.compile the model (slower start, faster throughput)")
    p.add_argument("--shardshuffle", type=int, help="Shuffling between shards, passed to dataloader", default=100)
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
    total_steps = args.max_steps if args.max_steps is not None else args.epochs * steps_per_epoch

    loader = build_mae_dataloader(shards, batch_size=args.batch_size, num_workers=args.num_workers, shardshuffle=args.shardshuffle)

    model = MAE(mask_ratio=args.mask_ratio).to(device)
    if args.compile:
        print("Compiling model with torch.compile …")
        model = torch.compile(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MAE parameters: {n_params / 1e6:.1f}M  (encoder+decoder)")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )

    step = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        for imgs, _ in loader:
            if args.max_steps is not None and step >= args.max_steps:
                break

            imgs = imgs.to(device, non_blocking=True)

            lr = cosine_lr(step, total_steps, args.warmup_steps, args.lr)
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
                print(f"step={step:5d}  loss={loss_val:.4f}  lr={lr:.2e}  t={elapsed:.0f}s", flush=True)
                if use_wandb:
                    wandb.log({"train/loss": loss_val, "train/lr": lr, "train/step": step}, step=step)

            step += 1

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
