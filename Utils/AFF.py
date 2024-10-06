import torch
import torch.nn as nn

from einops.layers.torch import Rearrange, Reduce


class DAF(nn.Module):
    '''
    直接相加 DirectAddFuse
    '''

    def __init__(self):
        super(DAF, self).__init__()

    def forward(self, x, residual):
        return x + residual


class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 第二次本地注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo


class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()
        # TODO 融合模块，减少维度以符合回归器的输入
        self.reduce = Reduce('b c h w -> b c', 'mean')

    def forward(self, x, residual):
        xa = x + residual  # 不变  [b, 256, 7, 7]
        # 融合模块修改，cat，通道数加倍
        # xa = torch.cat((x, residual), 1)  # [b, 64*4*2 = 512, 7, 7]
        xl = self.local_att(xa)  # Output: [b, channels, 7, 7]
        xg = self.global_att(xa)  # Output: [b, channels, 1, 1]
        xlg = xl + xg  # Broadcasting: [b, channels, 7, 7] + [b, channels, 1, 1] -> [b, channels, 7, 7]
        wei = self.sigmoid(xlg)  # Output: [b, channels, 7, 7]

        xo = 2 * x * wei + 2 * residual * (1 - wei)  # No dimension change
        xo = self.reduce(xo)  # Output: [b, channels]
        return xo

# DPAFF:QK和V
class DPAFF(nn.Module):
    """
    Dot Product Attention Feature Fusion: 点乘注意力特征融合
    多头：关注各个子空间的注意力
    """
    def __init__(self,
                 channels=64,  # 输入token的dim
                 num_heads=8,  # 多头
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(DPAFF, self).__init__()
        self.rearrange = Rearrange('b c h w -> b (h w) c')  #
        self.num_heads = num_heads
        head_dim = channels // num_heads  # 每个头均分维度e
        self.scale = qk_scale or head_dim ** -0.5  # 传入 or 根号下head_dim
        self.qk = nn.Linear(channels, channels * 2, bias=qkv_bias)  # 参数矩阵 用于得到qkv(使用一个全连接层，加速运算)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(channels, channels)  # 多头合并投影层
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.reduce = Reduce('b n c -> b c', 'mean')

    def forward(self, x, y):  # x作为q和k，y作为v
        x = self.rearrange(x)  # [b, 256, 7, 7] -> [b, 49, 256]
        v = self.rearrange(y)
        # [batch_size, num_patches + 1, total_embed_dim]（[b, 49, 256]）
        B, N, C = x.shape

        # qk(): -> [b, 49, 2 * 256]
        # reshape: -> [b, 49, 2, 8, 32]
        # permute: -> [2, b, 8, 49, 32]
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        v = v.reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [b, 8, 49, 32] 拆分出qkv
        q, k = qk[0], qk[1]  # make torchscript happy (cannot use tensor as tuple)
        v = v[0]
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # 点乘注意力
        attn = attn.softmax(dim=-1)  # softmax转化为概率分布
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        f = (attn @ v).transpose(1, 2).reshape(B, N, C)  # 得到注意力特征图 [b, 49, 256]
        f = self.proj(f)
        f = self.proj_drop(f)
        f = self.reduce(f)  # [b, 256]
        return f


# DPAFF1：Q和KV
class DPAFF1(nn.Module):
    """
    Dot Product Attention Feature Fusion: 点乘注意力特征融合
    多头：关注各个子空间的注意力
    """
    def __init__(self,
                 channels=64,  # 输入token的dim
                 num_heads=8,  # 多头
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(DPAFF1, self).__init__()
        self.rearrange = Rearrange('b c h w -> b (h w) c')  #
        self.num_heads = num_heads
        head_dim = channels // num_heads  # 每个头均分维度e
        self.scale = qk_scale or head_dim ** -0.5  # 传入 or 根号下head_dim
        self.kv = nn.Linear(channels, channels * 2, bias=qkv_bias)  # 参数矩阵 用于得到qkv(使用一个全连接层，加速运算)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(channels, channels)  # 多头合并投影层
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.reduce = Reduce('b n c -> b c', 'mean')

    def forward(self, x, y):  # x作为q，y作为kv
        q = self.rearrange(x)  # [b, 256, 7, 7] -> [b, 49, 256]
        kv = self.rearrange(y)
        # [batch_size, num_patches + 1, total_embed_dim]（[b, 49, 256]）
        B, N, C = q.shape

        # qk(): -> [b, 49, 2 * 256]
        # reshape: -> [b, 49, 2, 8, 32]
        # permute: -> [2, b, 8, 49, 32]
        q = q.reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(kv).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [b, 8, 49, 32] 拆分出qkv
        q = q[0]
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # 点乘注意力
        attn = attn.softmax(dim=-1)  # softmax转化为概率分布
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        f = (attn @ v).transpose(1, 2).reshape(B, N, C)  # 得到注意力特征图 [b, 49, 256]
        f = self.proj(f)
        f = self.proj_drop(f)
        f = self.reduce(f)  # [b, 256]
        return f


class Cat(nn.Module):
    def __init__(self):
        super(Cat, self).__init__()

    def forward(self, x, y):
        f = torch.cat((x,y), dim=1)
        return f

class MS_CAM(nn.Module):
    '''
    单特征 进行通道加权,作用类似SE模块
    '''

    def __init__(self, channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei



# if __name__ == '__main__':
#     import os
#     os.environ['CUDA_VISIBLE_DEVICES'] = "0"
#     device = torch.device("cuda:0")
#
#     x, residual= torch.ones(8,64, 32, 32).to(device),torch.ones(8,64, 32, 32).to(device)
#     channels=x.shape[1]
#
#     model=AFF(channels=channels)
#     model=model.to(device).train()
#     output = model(x, residual)
#     print(output.shape)
