import os
import time
from functools import partial

import timm
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# MLP 预归一化残差块：使用层归一化后输入到fn的结果 再与原始输入相加得到最后输出
class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


# MLPMixer：不使用vit、卷积的编码器，可提取跨patch信息
class MLPMixer(nn.Module):
    def __init__(self, image_size, channels, patch_size, dim, depth, expansion_factor=4, dropout=0., output='map'):
        """
        Args:
            image_size (int): 输入图像的尺寸 7
            channels (int): 输入图像的通道数 10*64*4 = 2560 (/4=640 *2 = 1280)
            patch_size (int): 每个 patch 的尺寸 1
            dim (int): 隐藏层的维度 64*4 = 256 (/4=64 *2 =128)
            depth (int): 模型的深度，即残差块的数量 9
            expansion_factor (int): 扩展因子，即 MLP 中扩展维度的倍数 4
            dropout (float): dropout 的概率 0
        """
        super().__init__()
        assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
        self.num_patches = (image_size // patch_size) ** 2  # 计算patch数量 7
        # partial作用是调用 nn.Conv1d 函数并将 kernel_size 参数设置为 1
        self.chan_first, self.chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear  # 分别用于空间混合和通道混合

        self.mlp = nn.Sequential(
            # (batch_size, num_patches, patch_size * patch_size * channels)
            # [b, 4*p*64, 7, 7]即[b, 2560, 7, 7] -> [b, h*w=49, 2560 or 640]
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            # [b, 49, 2560 or 640 or 1280]
            # 映射到隐藏层的维度,nn.Linear只影响最后一维度
            nn.Linear((patch_size ** 2) * channels, dim),  # [b, 49, 2560 or 640 or 1280] -> [b, 49, 64 or 128]
            # 根据深度 构建残差MLP块
            *[nn.Sequential(  # 不会改变维度[b, 49, 64]
                PreNormResidual(dim, self.FeedForward(self.num_patches, expansion_factor, dropout, self.chan_first)),
                PreNormResidual(dim, self.FeedForward(dim, expansion_factor, dropout, self.chan_last))
            ) for _ in range(depth)],
            # 归一化，不改变维度[b, 49, 64]
            nn.LayerNorm(dim),
            # TODO DAF使用Rearrange以满足nn.Cov2d输入的格式，Cat使用Reduce
            # Reduce('b n c -> b c', 'mean'),  # 池化 [b, 256] or [b, 64 or 128]
            # Rearrange('b (h w) c -> b c h w', h=7, w=7)  # [b, 64 or 128, 7, 7]
            # nn.Linear(dim, num_classes)
        )
        # print(self.mlp)

        self.output = Reduce('b n c -> b c', 'mean') if output == 'vector' else Rearrange('b (h w) c -> b c h w', h=7, w=7)

    def FeedForward(self, dim, expansion_factor=4, dropout=0., dense=nn.Linear):
        return nn.Sequential(
            dense(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            dense(dim * expansion_factor, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):  # 输入的是[b, 4p*64=256p, 7, 7]
        # [3, 256*self_patch_num, 7, 7]
        for mlp_single in self.mlp:
            x = mlp_single(x)
        return self.output(x)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLPAttentionMixer(nn.Module):
    def __init__(self, image_size, channels, patch_size, dim, depth, expansion_factor=4, dropout=0., output='vector', heads=8):
        super().__init__()
        assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
        self.num_patches = (image_size // patch_size) ** 2
        self.chan_first, self.chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear

        self.mlp = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear((patch_size ** 2) * channels, dim),
            *[nn.Sequential(
                PreNormResidual(dim, self.SelfAttention(dim, heads=heads, dropout=dropout)),
                PreNormResidual(dim, self.FeedForward(dim, expansion_factor, dropout, self.chan_last))
            ) for _ in range(depth)],
            nn.LayerNorm(dim)
        )

        self.output = Reduce('b n c -> b c', 'mean') if output == 'vector' else Rearrange('b (h w) c -> b c h w', h=7, w=7)

    def SelfAttention(self, dim, heads=8, dropout=0.):
        return SelfAttention(dim, heads, dropout)

    def FeedForward(self, dim, expansion_factor=4, dropout=0., dense=nn.Linear):
        return nn.Sequential(
            dense(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            dense(dim * expansion_factor, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        for mlp_single in self.mlp:
            x = mlp_single(x)
        return self.output(x)

if __name__ == "__main__":
    t1 = time.time()
    # model = MLPAttentionMixer(image_size=7, channels=2560, patch_size=1, dim=256, depth=4).to('cuda')
    model = timm.models.mlp_mixer.MlpMixer(
        img_size=7,
        patch_size=1,
        in_chans=2560,
        num_classes=0,
        embed_dim=256,
        num_blocks=4
    ).to('cuda')
    print(model)
    t2 = time.time()
    img = torch.randn(16, 2560, 7, 7).to('cuda')
    feature = model(img)  # (1, 1000)
    t3 = time.time()
    print(feature.shape)
    # 打印时间
    print(t2 - t1)
    print(t3 - t2)
    print(t3 - t1)