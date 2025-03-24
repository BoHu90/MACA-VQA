import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
# from CLIP import clip
from clip import clip
import torch.nn.functional as F
from einops import rearrange
from torch_geometric.nn import GCNConv, global_mean_pool
import torch_geometric
from . import kan

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, image_token):
        B, N, C = image_token.shape
        kv = (
            self.kv(image_token)
            .reshape(B, N, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        B, N, C = query.shape
        q = (
            self.q(query)
            .reshape(B, N, 1, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q = q[0]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ViTbCLIP_SpatialTemporalDistortion_loda(torch.nn.Module):
    def __init__(self, feat_len=8, sr=True, tr=True, dr=True, ar=True, dropout_sp=0.2, dropout_tp=0.2, dropout_dp=0.2, dropout_ap=0.2):
        super(ViTbCLIP_SpatialTemporalDistortion_loda, self).__init__()
        ViT_B_16, _ = clip.load("ViT-B/16")
        clip_vit_b_pretrained_features = ViT_B_16.visual
        self.feature_extraction = clip_vit_b_pretrained_features
        self.feat_len = feat_len

        self.base_quality = self.base_quality_regression(512, 128, 1)
        self.linear_param = self.base_quality_regression(512, 1024, 512)

        self.unify_videomae_rec = nn.Linear(1408, 512)

        self.unify_distortion_rec_loda = self.base_quality_regression(4096, 1024, 768)
        self.cross_atten_1 = CrossAttention(64)
        self.cross_atten_2 = CrossAttention(512)
        self.down_proj = self.base_quality_regression(768, 128, 64)
        self.up_proj = self.base_quality_regression(64, 256, 768)
        self.scale_factor = nn.Parameter(torch.randn(12, 197, 768) * 0.02)

    def base_quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.ReLU(),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

    def forward(self, x, x_3D_features, lp, dist, aes, videomae):
        x_size = x.shape
        x = x.view(-1, x_size[2], x_size[3], x_size[4])

        dist_size = dist.shape
        dist_loda = dist.view(-1, dist_size[2])
        dist_loda = self.unify_distortion_rec_loda(dist_loda)
        dist_loda = dist_loda.unsqueeze(1).repeat(1, 197, 1)
        x_loda = self.feature_extraction.conv1(x)
        x_loda = x_loda.flatten(2).transpose(1, 2)

        cls_token = self.feature_extraction.class_embedding.unsqueeze(0).repeat(x_loda.shape[0], 1, 1)
        x_loda = torch.cat([cls_token, x_loda], dim=1)

        x_loda = x_loda + self.feature_extraction.positional_embedding
        x_loda = self.feature_extraction.ln_pre(x_loda)

        del cls_token

        for i, block in enumerate(self.feature_extraction.transformer.resblocks):
            with torch.no_grad():
                x_loda_down = self.down_proj(x_loda).detach()
                dist_loda_down = self.down_proj(dist_loda).detach()
                x_down = x_loda_down + self.cross_atten_1(x_loda_down, dist_loda_down)
                x_up = self.up_proj(x_down)
                x_loda = x_loda + x_up * self.scale_factor[i]
                x_loda = block(x_loda)

        x_loda = self.feature_extraction.ln_post(x_loda[:, 0, :])

        if self.feature_extraction.proj is not None:
            x_loda = x_loda @ self.feature_extraction.proj
        x_loda = x_loda.view(x_size[0], x_size[1], 512)

        x = self.feature_extraction(x)
        x = x.contiguous().view(x_size[0], x_size[1], 512)
        x_videomae_features = self.unify_videomae_rec(videomae)
        x_loda = self.cross_atten_2(x_loda, x_videomae_features)
        x = x + x_loda
        x = self.linear_param(x)
        x = self.base_quality(x)
        x = torch.mean(x, dim=1).squeeze(1)

        return x
