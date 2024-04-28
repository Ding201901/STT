import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, seq_len, dim, heads, dim_head, dropout, raduce_ratio):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        # self.reduce_len = nn.Linear(seq_len + 1, seq_len // raduce_ratio)
        # self.reduce_len = nn.Conv2d(seq_len + 1, seq_len // raduce_ratio, 1, bias = False)
        self.reduce_len = nn.Conv1d(dim, dim, 5, stride = raduce_ratio, bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.dwconv = nn.Conv2d(in_channels = heads, out_channels = heads, kernel_size = 3, padding = 1, groups = heads)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        # x:[b,n,dim]
        h = self.heads
        # x_kv = torch.transpose(x, -1, -2)
        # x_kv = self.reduce_len(x_kv)
        # x_kv = torch.transpose(x_kv, -1, -2)

        # get kv tuple:([b,ns,head_num*head_dim],[b,ns,head_num*head_dim])
        kv = self.to_kv(x).chunk(2, dim = -1)
        # get q:[b,n,head_num*head_dim]
        q = self.to_q(x)
        # split k,v from [b,ns,head_num*head_dim] -> [b,head_num,ns,head_dim]
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), kv)
        # split q from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # mask value: -inf
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # dots = self.dwconv(dots)

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim = -1)
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel, mode, reduce_ratio):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(num_channel, dim, heads = heads, dim_head = dim_head, dropout = dropout, raduce_ratio = reduce_ratio))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout = dropout)))
            ]))

        self.mode = mode
        self.skipcat = nn.ModuleList([])
        for _ in range(depth - 2):
            self.skipcat.append(nn.Conv2d(num_channel + 1, num_channel + 1, [1, 2], 1, 0))

    def forward(self, x, mask = None):
        if self.mode == 'ViT':
            for attn, ff in self.layers:
                x = attn(x, mask = mask)
                x = ff(x)
        elif self.mode == 'CAF':
            last_output = []
            nl = 0
            for attn, ff in self.layers:
                last_output.append(x)
                if nl > 1:
                    x = self.skipcat[nl - 2](
                        torch.cat([x.unsqueeze(3), last_output[nl - 2].unsqueeze(3)], dim = 3)).squeeze(3)
                x = attn(x, mask = mask)
                x = ff(x)
                nl += 1

        return x


class ViT(nn.Module):
    def __init__(self, image_size, near_band, num_patches, num_classes, dim, depth, heads, mlp_dim, pool = 'cls',
                 channels = 1, dim_head = 16, dropout = 0., emb_dropout = 0., mode = 'ViT', reduce_ratio = 1):
        super().__init__()

        patch_dim = image_size ** 2 * near_band

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 2, dim))
        self.tem_embedding = nn.Parameter(torch.randn(1, 2, dim))  # temporal
        self.seg_embedding = nn.Parameter(torch.randn(1, 1, dim))  # segmentation
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches, mode, reduce_ratio)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x, mask = None):
        # patchs[batch, patch_num, patch_size*patch_size*c]  [batch,200,145*145]
        # x = rearrange(x, 'b c h w -> b c (h w)')

        # embedding every patch vector to embedding size: [batch, patch_num, embedding_size]
        x = self.patch_to_embedding(x)  # [b,n,dim]
        b, n, _ = x.shape

        # add position embedding
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)  # [b,1,dim]
        seg_tokens = repeat(self.seg_embedding, '() n d -> b n d', b = b)  # [b,1,dim]
        x1, x2 = torch.chunk(x, 2, dim = 1)
        # x1 = torch.cat((torch.cat((cls_tokens, x1), dim = 1), seg_tokens), dim = 1)  # [b,n/2+2,dim]
        # x1 = x1 + self.pos_embedding[:, :int(n / 2 + 2)]
        x1 = torch.cat((cls_tokens, x1), dim = 1)
        x1 = x1 + self.pos_embedding[:, :int(n / 2 + 1)]  # + self.tem_embedding[:, 0, :]
        x2 = x2 + self.pos_embedding[:, :int(n / 2)]  # + self.tem_embedding[:, 1, :]
        x = torch.cat((x1, x2), dim = 1)  # [b,n+2,dim]
        x = self.dropout(x)

        # transformer: x[b,n + 1,dim] -> x[b,n + 1,dim]
        x = self.transformer(x, mask)

        # classification: using cls_token output
        x = self.to_latent(x[:, 0])

        # MLP classification layer
        return self.mlp_head(x)

# Example
# x = torch.rand([32, 340, 27])
# model = ViT(
#     image_size = 3,
#     near_band = 3,
#     num_patches = 2 * 170,
#     num_classes = 2,
#     dim = 64,
#     depth = 5,
#     heads = 4,
#     mlp_dim = 8,
#     dropout = 0.1,
#     emb_dropout = 0.1,
#     mode = 'CAF'
# )
#
# out = model(x)
