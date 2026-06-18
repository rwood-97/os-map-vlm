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
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.NUM_PATCHES, decoder_dim)
        )
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

        # Per-patch normalised MSE on masked patches (MAE paper §3.1)
        target = self._patchify(imgs)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6).sqrt()
        loss = (
            F.mse_loss(pred, target, reduction="none").mean(dim=-1) * mask
        ).sum() / mask.sum()
        return loss
