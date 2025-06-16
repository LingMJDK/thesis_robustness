import torch
from torch import nn
import numpy as np
class ViTPatchEmbedding(nn.Module):
    """
    Turn 2D input image into 1d learnable sequence embedding vector
    Params:
      in_channels (int): Number of color channels
      patch_size  (int): Size of each square patch
      emb_size    (int): Dimension of embedding vectors
    Returns:
      Tensor of shape (B, N, emb_size) where N = (H/patch_size)*(W/patch_size)
    """
    def __init__(self, in_channels=3, patch_size=16, emb_size=768):
        super().__init__()
        self.patch_size = patch_size

        self.patches = nn.Conv2d(
            in_channels=in_channels,
            out_channels=emb_size,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=True
        )

        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

        # paper-faithful init
        nn.init.xavier_uniform_(self.patches.weight)
        nn.init.normal_(self.patches.bias, std=1e-6)

    def forward(self, x):
        # Assert if image size is a multiple of patch size
        B, C, H, W = x.shape

        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"Image size ({H} x {W}) must be a multiple of patch_size {self.patch_size}"

        # (B, C, H, W) -> (B, emb_size, H/patch_size, W/patch_size)
        x = self.patches(x)

        # (B, emb_size, n_h, n_w) -> (B, emb_size, N)
        x = self.flatten(x)

        # (B, emb_size, N) -> (B, N, emb_size)
        x = x.transpose(1, 2)

        return x


class MultiHeadSelfAttentionModule(nn.Module):
    def __init__(self, emb_size, n_heads=4, mask=False):
        super().__init__()

        self.n_heads = n_heads
        self.k = emb_size // n_heads

        assert emb_size % n_heads == 0, (
            "Embedding size must be a multiple of the number of heads"
        )

        # linear projection for Q, K, V (all heads combined!)
        self.W_q = nn.Linear(emb_size, emb_size)
        self.W_k = nn.Linear(emb_size, emb_size)
        self.W_v = nn.Linear(emb_size, emb_size)

        # Final projection after heads are concatenated
        self.output_proj = nn.Linear(emb_size, emb_size)
        self.mask = mask

    def forward(self, x):
        batch_size, seq_len, emb_size = x.shape

        Q = self.W_q(x)  # (batch, seq, emb)
        K = self.W_k(x)
        V = self.W_v(x)

        # Split heads: (batch, seq, emb) → (batch, n_heads, seq, k)
        split_heads = lambda x: x.view(
            batch_size, seq_len, self.n_heads, self.k
        ).transpose(1, 2)

        Q = split_heads(Q).contiguous()
        K = split_heads(K).contiguous()
        V = split_heads(V).contiguous()

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (
            self.k**0.5
        )  # (batch, n_heads, seq, seq)

        # In case of autoregressive task, apply mask.
        if self.mask:
            mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
            mask = mask.to(x.device)
            scores = scores.masked_fill(
                ~mask, float("-inf")
            )  # `~` Flips True to False and vice versa

        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(
            attention_weights, V
        )  # (batch, n_heads, seq, k)

        # Concatenate heads (batch, n_heads, seq, d_k) ---> (batch, seq, emb)
        attention_output = (
            attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        )

        # Final projection
        output_vector = self.output_proj(attention_output)  # (batch, seq, emb)

        return output_vector
    
    
class TransformerBlock(nn.Module):
    def __init__(self, emb_size, n_heads=4, ff_hidden_mult=4, dropout=0.1, mask=False):
        super().__init__()
        self.attention = MultiHeadSelfAttentionModule(emb_size, n_heads, mask)
        self.layer_norm1 = nn.LayerNorm(emb_size)
        self.layer_norm2 = nn.LayerNorm(emb_size)
        self.ff = nn.Sequential(
            nn.Linear(emb_size, ff_hidden_mult * emb_size),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb_size, emb_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attended = self.attention(x)
        x = self.layer_norm1(attended + x)
        x = self.dropout(x)

        ff_out = self.ff(x)
        x = self.layer_norm2(ff_out + x)
        x = self.dropout(x)
        return x

class ViTEncoder(nn.Module):
    """
    ViT encoder:
      1) patch embedding
      2) prepend CLS token
      3) add learned positional embeddings
      4) stack of TransformerBlock
      5) final LayerNorm
    Returns: tensor of shape (B, N+1, emb_size)
    """
    def __init__(
        self,
        in_channels: int,
        emb_size: int,
        patch_size: int,
        image_size: int,  #  image size = H = W
        n_layers: int = 2,
        n_heads: int = 4,
        ff_hidden_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        num_patches = (image_size // patch_size) ** 2

        self.patch_emb = ViTPatchEmbedding(in_channels, patch_size, emb_size)

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.pos_emb   = nn.Parameter(torch.zeros(1, num_patches + 1, emb_size))
        self.dropout   = nn.Dropout(dropout)

        # TransformerBlock:
        self.blocks = nn.ModuleList([
            TransformerBlock(
                emb_size,
                n_heads=n_heads,
                ff_hidden_mult=ff_hidden_mult,
                dropout=dropout,
                mask=False
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(emb_size)

    def forward(self, x):
        B = x.size(0)

        # 1) patch embed → (B, N, D)
        x = self.patch_emb(x)

        # 2) prepend CLS → (B, N+1, D)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)

        # 3) add pos emb + dropout
        x = x + self.pos_emb
        x = self.dropout(x)

        # 4) Transformer blocks
        for block in self.blocks:
            x = block(x)

        # 5) final norm
        x = self.norm(x)
        return x
    
class ViTClassificationHead(nn.Module):
    """
    """
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        image_size: int = 32,
        patch_size: int = 4,
        emb_size: int = 192,
        n_layers: int = 9,
        n_heads: int = 12,
        ff_hidden_mult:int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = ViTEncoder(
            in_channels,
            emb_size,
            patch_size,
            image_size,
            n_layers,
            n_heads,
            ff_hidden_mult,
            dropout
        )
        self.classifier = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = self.encoder(x)         # (Batch, N+1, Dim)
        cls = x[:, 0]               # extract CLS token
        return self.classifier(cls)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    embed = np.concatenate([emb_sin, emb_cos], axis=1)
    return embed

def random_mask_with_cls(x_with_cls, mask_ratio: float = 0.75, generator: torch.Generator = None):
    assert 0 < mask_ratio < 1, "mask_ratio must be between 0 and 1 (exclusive)"

    B, N_seq, D = x_with_cls.shape
    N_patches = N_seq - 1
    N_selection_patches = int(N_patches * (1 - mask_ratio))

    cls_token = x_with_cls[:, 0:1]
    patches   = x_with_cls[:, 1:]

    patch_score = torch.rand(B, N_patches, device=x_with_cls.device, generator=generator)
    idx_shuffling_patches = torch.argsort(patch_score, dim=1)
    idx_selection_patches = idx_shuffling_patches[:, :N_selection_patches]
    idx_restore_patches = torch.argsort(idx_shuffling_patches, dim=1)

    patches_visible = torch.gather(
        patches, dim=1,
        index=idx_selection_patches.unsqueeze(-1).expand(-1, -1, D)
    )

    x_visible_sequence = torch.cat([cls_token, patches_visible], dim=1)

    bin_mask_patches = torch.ones(B, N_patches, dtype=torch.bool, device=x_with_cls.device)
    bin_mask_patches[:, :N_selection_patches] = False
    mask_tensor_patches = torch.gather(bin_mask_patches, dim=1, index=idx_restore_patches)

    return x_visible_sequence, mask_tensor_patches, idx_restore_patches, idx_selection_patches

class MAEDecoder(nn.Module):
    def __init__(self, emb_size, n_layers, n_heads, ff_hidden_mult, dropout, num_patches):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        grid_size = int(num_patches ** 0.5)
        pos_embed_dec = get_2d_sincos_pos_embed(emb_size, grid_size, cls_token=False)
        self.pos_emb_dec = nn.Parameter(torch.from_numpy(pos_embed_dec).float().unsqueeze(0), requires_grad=False)

        self.blocks = nn.ModuleList([
            TransformerBlock(emb_size, n_heads, ff_hidden_mult, dropout, mask=False)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(emb_size)
        self.num_patches = num_patches
        self.emb_size = emb_size

    def forward(self, x_enc_out_vis, idx_restore_patches):
        B, N_vis_seq, D = x_enc_out_vis.shape
        N_patches = self.num_patches
        N_vis_patches = N_vis_seq - 1

        x_vis_patches_out = x_enc_out_vis[:, 1:]

        mask_tokens = self.mask_token.expand(B, N_patches - N_vis_patches, D)

        x_shuffled_patches = torch.empty(B, N_patches, D, device=x_enc_out_vis.device)

        idx_shuffled_visible_patches = torch.arange(N_vis_patches, device=x_enc_out_vis.device).unsqueeze(0).expand(B, -1)
        x_shuffled_patches.scatter_(dim=1, index=idx_shuffled_visible_patches.unsqueeze(-1).expand(-1, -1, D), src=x_vis_patches_out)

        idx_shuffled_mask_patches = torch.arange(N_vis_patches, N_patches, device=x_enc_out_vis.device).unsqueeze(0).expand(B, -1)
        x_shuffled_patches.scatter_(dim=1, index=idx_shuffled_mask_patches.unsqueeze(-1).expand(-1, -1, D), src=mask_tokens)

        x_full_patches_ordered = torch.gather(
            x_shuffled_patches, dim=1,
            index=idx_restore_patches.unsqueeze(-1).expand(-1, -1, D)
        )

        x_full_patches_ordered = x_full_patches_ordered + self.pos_emb_dec

        decoded_patches = x_full_patches_ordered
        for blk in self.blocks:
            decoded_patches = blk(decoded_patches)

        decoded_patches = self.norm(decoded_patches)

        return decoded_patches


class MaskedAutoencoderViT(nn.Module):
    def __init__(
        self,
        in_channels: int=3,
        patch_size: int=4,
        image_size: int=32,
        emb_size: int=192,
        encoder_layers: int=9,
        decoder_layers: int=4,
        n_heads: int=12,
        ff_hidden_mult: int=4,
        dropout: float=0.1,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size"
        grid_size = image_size // patch_size
        num_patches = grid_size * grid_size
        patch_dim = patch_size * patch_size * in_channels

        self.patch_embed = ViTPatchEmbedding(in_channels, patch_size, emb_size)

        pos_embed_enc = get_2d_sincos_pos_embed(emb_size, grid_size, cls_token=True)
        self.pos_emb_enc = nn.Parameter(torch.from_numpy(pos_embed_enc).float().unsqueeze(0), requires_grad=False)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))

        self.enc_dropout = nn.Dropout(dropout)
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(emb_size, n_heads, ff_hidden_mult, dropout, mask=False)
            for _ in range(encoder_layers)
        ])
        self.enc_norm = nn.LayerNorm(emb_size)

        self.decoder = MAEDecoder(emb_size, decoder_layers, n_heads, ff_hidden_mult, dropout, num_patches=num_patches)

        self.reconstruction_head = nn.Linear(emb_size, patch_dim)

        self.loss_fn = nn.MSELoss()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = num_patches

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def patchify(self, imgs):
        p = self.patch_size
        B, C, H, W = imgs.shape
        assert H % p == 0 and W % p == 0
        imgs = imgs.reshape(B, C, H//p, p, W//p, p)
        imgs = imgs.permute(0, 2, 4, 3, 5, 1)
        return imgs.reshape(B, -1, p*p*C)

    def forward(self, imgs, mask_ratio: float = 0.75):
        B = imgs.size(0)

        patches = self.patchify(imgs)

        x = self.patch_embed(imgs)

        cls = self.cls_token.expand(B, -1, -1)
        x_with_cls = torch.cat([cls, x], dim=1)
        x_enc_in = x_with_cls + self.pos_emb_enc

        x_enc_in = self.enc_dropout(x_enc_in)

        x_visible_sequence, mask_tensor_patches, idx_restore_patches, _ = random_mask_with_cls(x_enc_in, mask_ratio)

        y_enc_out_vis = x_visible_sequence
        for blk in self.encoder_blocks:
            y_enc_out_vis = blk(y_enc_out_vis)
        y_enc_out_vis = self.enc_norm(y_enc_out_vis)

        decoded_patches = self.decoder(y_enc_out_vis, idx_restore_patches)

        pred_patches = self.reconstruction_head(decoded_patches)

        loss = self.loss_fn(pred_patches[mask_tensor_patches], patches[mask_tensor_patches])

        return loss, pred_patches, mask_tensor_patches


def build_finetune_vit_from_mae(mae_model: MaskedAutoencoderViT, num_classes, dropout=0.1, patch_size=4):

    first_blk = mae_model.encoder_blocks[0]
    emb_size = first_blk.attention.k * first_blk.attention.n_heads
    ff_hidden_mult = 4 
    n_heads = first_blk.attention.n_heads
    n_layers = len(mae_model.encoder_blocks)
    in_channels = mae_model.in_channels
    grid_size = int(mae_model.num_patches ** 0.5)
    image_size = grid_size * patch_size
    num_patches = mae_model.num_patches

    # Initialize ViT
    finetune_model = ViTClassificationHead(
        num_classes=num_classes,
        in_channels=in_channels,
        image_size=image_size,
        patch_size=patch_size,
        emb_size=emb_size,
        n_layers=n_layers,
        n_heads=n_heads,
        ff_hidden_mult=ff_hidden_mult, 
        dropout=dropout
    )

    finetune_model.encoder.patch_emb.patches.weight.data.copy_(
        mae_model.patch_embed.patches.weight.data)
    
    if mae_model.patch_embed.patches.bias is not None:
        finetune_model.encoder.patch_emb.patches.bias.data.copy_(
            mae_model.patch_embed.patches.bias.data)

    finetune_model.encoder.cls_token.data.copy_(
        mae_model.cls_token.data
        )

    with torch.no_grad():
         if finetune_model.encoder.pos_emb.shape == mae_model.pos_emb_enc.shape:
             finetune_model.encoder.pos_emb.copy_(mae_model.pos_emb_enc)
         else:
             print("Warning: Finetune model's pos_emb shape does not match MAE's. Cannot copy sine-cosine positional embeddings directly.")


    for mae_blk, vit_blk in zip(mae_model.encoder_blocks,
                                 finetune_model.encoder.blocks):
        vit_blk.load_state_dict(mae_blk.state_dict())

    finetune_model.encoder.norm.load_state_dict(
        mae_model.enc_norm.state_dict())

    return finetune_model

def unpatchify(patches, patch_size, in_channels, image_size):
    p = patch_size
    B, N, patch_dim = patches.shape
    H = W = image_size
    C = in_channels
    assert N == (H // p) * (W // p)
    assert patch_dim == p * p * C

    h = H // p
    w = W // p

    patches = patches.reshape(B, h, w, p, p, C)
    imgs = patches.permute(0, 5, 1, 3, 2, 4).contiguous()
    imgs = imgs.reshape(B, C, H, W)
    return imgs