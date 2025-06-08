import torch
from torch import nn

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