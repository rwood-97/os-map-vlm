import torch
from torch import nn
from torch.nn import functional as F


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
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
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
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchHOG(nn.Module):
    """HOG descriptor per 16x16 patch, FG-MAE style reconstruction target.

    Each patch is divided into a 4x4 grid of 4x4-pixel cells.
    9 orientation bins over [0, π) with soft assignment weighted by gradient magnitude.
    Output is L2-normalised per patch: (B, N_patches, 144).
    """

    CELL_SIZE = 4
    N_BINS = 9

    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.cells_per_side = patch_size // self.CELL_SIZE  # 4
        self.hog_dim = self.cells_per_side**2 * self.N_BINS  # 144

        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    @torch.no_grad()
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        B, C, H, W = imgs.shape
        p = self.patch_size

        # Gradients per channel, pick the channel with max magnitude at each pixel
        imgs_flat = imgs.reshape(B * C, 1, H, W)
        gx = F.conv2d(imgs_flat, self.sobel_x, padding=1).reshape(B, C, H, W)
        gy = F.conv2d(imgs_flat, self.sobel_y, padding=1).reshape(B, C, H, W)

        mag = (gx**2 + gy**2).sqrt()  # (B, C, H, W)
        best = mag.argmax(dim=1, keepdim=True)  # (B, 1, H, W)
        mag_max = mag.gather(1, best).squeeze(1)  # (B, H, W)
        gx_best = gx.gather(1, best).squeeze(1)
        gy_best = gy.gather(1, best).squeeze(1)

        # Angles in [0, π) — unsigned orientation
        angle = torch.atan2(gy_best, gx_best) % torch.pi  # (B, H, W)

        # Soft bin assignment with triangle kernel
        bin_width = torch.pi / self.N_BINS
        centers = torch.arange(self.N_BINS, device=imgs.device, dtype=imgs.dtype)
        centers = centers * bin_width + bin_width / 2  # (N_BINS,)

        dist = (angle.unsqueeze(-1) - centers).abs()  # (B, H, W, N_BINS)
        dist = torch.minimum(dist, torch.pi - dist)  # circular distance
        votes = (1.0 - dist / bin_width).clamp(0)  # triangle kernel
        votes = votes * mag_max.unsqueeze(-1)  # weight by gradient magnitude

        # Average-pool into CELL_SIZE x CELL_SIZE cells
        # (B, H, W, N_BINS) → (B, N_BINS, H, W)
        votes = votes.permute(0, 3, 1, 2).contiguous()
        cell_hog = F.avg_pool2d(
            votes, kernel_size=self.CELL_SIZE, stride=self.CELL_SIZE
        )
        # cell_hog: (B, N_BINS, H//CELL_SIZE, W//CELL_SIZE)

        # Reshape cell grid into patches: group cells_per_side x cells_per_side cells per patch
        h_patches = H // p
        w_patches = W // p
        cps = self.cells_per_side  # cells per patch side

        cell_hog = cell_hog.reshape(B, self.N_BINS, h_patches, cps, w_patches, cps)
        # (B, N_BINS, h_patches, cps, w_patches, cps) → (B, h_patches, w_patches, cps, cps, N_BINS)
        cell_hog = cell_hog.permute(0, 2, 4, 3, 5, 1).reshape(
            B, h_patches * w_patches, self.hog_dim
        )

        return F.normalize(cell_hog, dim=-1)


class MAE(nn.Module):
    """Masked Autoencoder with ViT-B encoder (512x512 input, 16x16 patches).

    reconstruction: "pixel" (default) or "hog".
      - "pixel": reconstruct per-patch normalised RGB values (MAE paper §3.1).
      - "hog": reconstruct L2-normalised HOG descriptors (FG-MAE style).
    """

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
        reconstruction="pixel",
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.reconstruction = reconstruction

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
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.NUM_PATCHES, decoder_dim)
        )
        self.decoder_blocks = nn.ModuleList(
            [Block(decoder_dim, decoder_heads, mlp_ratio) for _ in range(decoder_depth)]
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim)

        if reconstruction == "hog":
            self.hog = PatchHOG(self.PATCH_SIZE)
            out_dim = self.hog.hog_dim  # 144
        else:
            self.hog = None
            out_dim = self.PIXELS_PER_PATCH  # 768

        self.decoder_pred = nn.Linear(decoder_dim, out_dim)

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
        return (
            imgs.reshape(B, C, h, p, w, p)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(B, h * w, C * p * p)
        )

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

        if self.reconstruction == "hog":
            # Target: L2-normalised HOG descriptors — no further normalisation needed
            target = self.hog(imgs)
            loss = (
                F.mse_loss(pred, target, reduction="none").mean(dim=-1) * mask
            ).sum() / mask.sum()
        else:
            # Per-patch normalised MSE on masked patches (MAE paper §3.1)
            target = self._patchify(imgs)
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()
            loss = (
                F.mse_loss(pred, target, reduction="none").mean(dim=-1) * mask
            ).sum() / mask.sum()

        return loss
