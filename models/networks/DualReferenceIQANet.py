import os
import time

import timm
import torch as torch
import torch.nn as nn
from einops.layers.torch import Reduce
from torch.nn import init

from Utils.AFF import DPAFF
from models.networks.MLPMixer import MLPMixer, MLPAttentionMixer
from models.networks.Regressors import RegressionFCNet, RegressionFCNet1, RegressionFCNet2
from models.networks.build_backbone import build_model
from models.networks.SwinT import swin_tiny_patch4_window7_224, swin_base_patch4_window7_224, \
    swin_small_patch4_window7_224
from options.option_train_In import set_args
from options.train_options import TrainOptions

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def initialize_weights(net_l, scale=0.1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)
            else:
                pass


# 学生模型：更换为SwinT
class DualReferenceIQANet(nn.Module):
    def __init__(self, my_config, my_args, self_patch_num=10, lda_channel=64, encode_decode_channel=64):
        super(DualReferenceIQANet, self).__init__()

        self.my_config = my_config
        self.MLP_depth = my_config.mlp_depth
        self.self_patch_num = self_patch_num
        self.lda_channel = lda_channel
        self.encode_decode_channel = encode_decode_channel
        # 内容特征提取器以及权重
        # self.model_content, _ = build_model(my_args)
        # self.model_content = torch.nn.DataParallel(self.model_content)
        # if my_config.content_model_path:  # 加载预训练权重
        #     checkpoint = torch.load(my_config.content_model_path, map_location='cpu')
        #     self.model_content.load_state_dict(checkpoint['model'])
        #     # 打印加载成功信息
        #     print(f'load {my_config.content_model_path} success')
        # TODO 考虑将内容特征提取器更换为swin
        if my_config.model == 'tiny':
            self.model_content = swin_tiny_patch4_window7_224()  # 多尺度特征提取器
        elif my_config.model == 'small':
            self.model_content = swin_small_patch4_window7_224()  # 多尺度特征提取器
        else:
            self.model_content = swin_base_patch4_window7_224()  # 多尺度特征提取器
        # 质量特征提取器以及权重
        self.model_quality, _ = build_model(my_args)
        self.model_quality = torch.nn.DataParallel(self.model_quality)
        if my_config.quality_model_path:  # 加载预训练权重
            checkpoint = torch.load(my_config.quality_model_path, map_location='cpu')
            self.model_quality.load_state_dict(checkpoint['model'])
            # 打印加载成功信息
            print(f'load {my_config.quality_model_path} success')

        # # 质量特征提取器以及权重
        # self.model_quality, _ = build_model(my_args)
        # # self.model_quality = torch.nn.DataParallel(self.model_quality)
        #  # 加载预训练权重
        # checkpoint = torch.load('/data/user/cwz/DRIQA/best_moco_model2.pth', map_location='cpu')
        # self.model_quality.load_state_dict(checkpoint)
        # # 打印加载成功信息
        # print(f'load /data/user/cwz/DRIQA/best_moco_model2.pth success')


        # 作为特征提取器无需学习
        for param in self.model_content.parameters():
            param.requires_grad = False
        for param in self.model_quality.parameters():
            param.requires_grad = False

        # TODO 更换backbone 2. 处理局部特征的卷积层维度要变
        # 对content提取的局部特征的处理：使得输出的特征图的大小都是 (7,7)，通道数都是 self.lda_channel = 64 即[_,64,7,7]
        self.lda1_content = nn.Sequential(nn.Conv2d(192, self.lda_channel, kernel_size=1, stride=1, padding=0),
                                          nn.AdaptiveAvgPool2d((7, 7)))  # 先用1*1卷积核统一通道数为64，再用自适应平均池化统一尺寸为7*7
        self.lda2_content = nn.Sequential(nn.Conv2d(384, self.lda_channel, kernel_size=1, stride=1, padding=0),
                                          nn.AdaptiveAvgPool2d((7, 7)))
        self.lda3_content = nn.Sequential(nn.Conv2d(768, self.lda_channel, kernel_size=1, stride=1, padding=0),
                                          nn.AdaptiveAvgPool2d((7, 7)))
        self.lda4_content = nn.Sequential(nn.Conv2d(768, self.lda_channel, kernel_size=1, stride=1, padding=0),
                                          nn.AdaptiveAvgPool2d((7, 7)))
        self.lda_content = [self.lda1_content, self.lda2_content, self.lda3_content, self.lda4_content]
        # 对quality提取的局部特征的处理
        self.lda1_quality = nn.Sequential(nn.Conv2d(256, self.lda_channel, kernel_size=1, stride=1, padding=0),
                                          nn.AdaptiveAvgPool2d((7, 7)))
        self.lda2_quality = nn.Sequential(nn.Conv2d(512, self.lda_channel, kernel_size=1, stride=1, padding=0),
                                          nn.AdaptiveAvgPool2d((7, 7)))
        self.lda3_quality = nn.Sequential(nn.Conv2d(1024, self.lda_channel, kernel_size=1, stride=1, padding=0),
                                          nn.AdaptiveAvgPool2d((7, 7)))
        self.lda4_quality = nn.Sequential(nn.Conv2d(2048, self.lda_channel, kernel_size=1, stride=1, padding=0),
                                          nn.AdaptiveAvgPool2d((7, 7)))
        self.lda_quality = [self.lda1_quality, self.lda2_quality, self.lda3_quality, self.lda4_quality]

        # 融合模块0 使用MLPMixer1(改为输出一个特征图)
        self.MLP_content_encoder = MLPAttentionMixer(image_size=7, channels=self.self_patch_num * self.lda_channel * 4,
                                            patch_size=1, dim=self.encode_decode_channel * 4, depth=self.MLP_depth, output='vector')
        # LQ特征编码器：9层MLP
        self.MLP_quality_encoder = MLPAttentionMixer(image_size=7, channels=self.self_patch_num * self.lda_channel * 4,
                                            patch_size=1, dim=self.encode_decode_channel * 4, depth=self.MLP_depth)

        # self.MLP_content_encoder = timm.models.mlp_mixer.MlpMixer(
        #     img_size=7,
        #     patch_size=1,
        #     in_chans=self.self_patch_num * self.lda_channel * 4,
        #     num_classes=0,
        #     embed_dim=self.encode_decode_channel * 4,
        #     num_blocks=self.MLP_depth
        # )
        # self.MLP_quality_encoder = timm.models.mlp_mixer.MlpMixer(
        #     img_size=7,
        #     patch_size=1,
        #     in_chans=self.self_patch_num * self.lda_channel * 4,
        #     num_classes=0,
        #     embed_dim=self.encode_decode_channel * 4,
        #     num_blocks=self.MLP_depth
        # )
        # enc_blks = [1, 1, 1, 1]
        # middle_blk_num = 1
        # dec_blks = [1, 1, 1, 1]
        # 融合模块1 定义
        self.ff = DPAFF(channels=lda_channel * 4)  # 4层64通道cat在一起
        self.reduce = Reduce('b c h w -> b c', 'mean')
        # TODO 回归器
        self.regressor = RegressionFCNet2(in_channel=512)

        # 权重初始化
        initialize_weights(self.MLP_content_encoder)
        initialize_weights(self.MLP_quality_encoder)
        initialize_weights(self.regressor)
        initialize_weights(self.ff)

        initialize_weights(self.lda1_quality)
        initialize_weights(self.lda2_quality)
        initialize_weights(self.lda3_quality)
        initialize_weights(self.lda4_quality)
        initialize_weights(self.lda1_content)
        initialize_weights(self.lda2_content)
        initialize_weights(self.lda3_content)
        initialize_weights(self.lda4_content)

    def forward(self, LQ_patches, HQ_patches, ZLQ_patches):
        # 形状处理
        b, p, c, h, w = LQ_patches.shape
        LQ_patches_reshape = LQ_patches.view(b * p, c, h, w)
        HQ_patches_reshape = HQ_patches.view(b * p, c, h, w)
        ZLQ_patches_reshape = ZLQ_patches.view(b * p, c, h, w)

        # 提取lq的内容特征 #  return [[b*p, 256, 56, 56], [b*p, 512, 28, 28], [b*p, 1024, 14, 14], [b*p, 2048, 7, 7]]
        LQ_content_feature = self.model_content(LQ_patches_reshape)
        # 提取lq/hq/zlq的质量特征
        LQ_quality_feature = self.model_quality(LQ_patches_reshape, mode=3)
        HQ_quality_feature = self.model_quality(HQ_patches_reshape, mode=3)
        ZLQ_quality_feature = self.model_quality(ZLQ_patches_reshape, mode=3)


        # 依次处理每一层特征, 计算差异, 并用一个列表存下
        multi_scale_LZ_diff_quality, multi_scale_LH_diff_quality, multi_scale_lq_content = [], [], []
        for LQ_content_feature, LQ_quality_feature,HQ_quality_feature, ZLQ_quality_feature, lda_quality, lda_content \
                in zip(LQ_content_feature, LQ_quality_feature, HQ_quality_feature, ZLQ_quality_feature, self.lda_quality, self.lda_content):
            # 先将四个特征统一为[b*p, 64, 7, 7]再view为[b ,p*64, 7, 7]，p=patch_num=10
            # 由于使用了.permute操作，使得内存不连续，这里改用.reshape方法不要求内存连续
            # 一个内容特征
            LQ_content_feature = lda_content(LQ_content_feature).reshape(b, -1, 7, 7)
            # 三个质量特征
            LQ_quality_feature = lda_quality(LQ_quality_feature).reshape(b, -1, 7, 7)  # [b ,p*64, 7, 7]
            HQ_quality_feature = lda_quality(HQ_quality_feature).reshape(b, -1, 7, 7)  # [b ,p*64, 7, 7]
            ZLQ_quality_feature = lda_quality(ZLQ_quality_feature).reshape(b, -1, 7, 7)  # [b ,p*64, 7, 7]
            # 两个差异特征
            LZ_diff_lda_feature = LQ_quality_feature - ZLQ_quality_feature  # 差异特征，高质量LQ减去低质量ZLQ
            LH_diff_lda_feature = HQ_quality_feature - LQ_quality_feature   # 差异特征，高质量HQ减去低质量LQ

            multi_scale_LZ_diff_quality.append(LZ_diff_lda_feature)  # 加入多尺度差异特征列表
            multi_scale_LH_diff_quality.append(LH_diff_lda_feature)  # 加入多尺度差异特征列表
            multi_scale_lq_content.append(LQ_content_feature)  # 加入多尺度lq特征列表
        # 拼接列表
        multi_scale_LZ_diff_quality = torch.cat(multi_scale_LZ_diff_quality, dim=1)  # [b, 4*64, 7, 7]
        multi_scale_LH_diff_quality = torch.cat(multi_scale_LH_diff_quality, dim=1)  # [b, 4*64, 7, 7]
        multi_scale_lq_content = torch.cat(multi_scale_lq_content, dim=1)  # [b, 4*64, 7, 7]
        # 打印3个向量的形状
        # print(multi_scale_LZ_diff_quality.shape, multi_scale_LH_diff_quality.shape, multi_scale_lq_content.shape)

        LZ_diff_mlp_quality_features = self.MLP_quality_encoder(multi_scale_LZ_diff_quality)
        LH_diff_mlp_quality_features = self.MLP_quality_encoder(multi_scale_LH_diff_quality)
        # print(LH_diff_mlp_quality_features.shape, multi_scale_LH_diff_quality.shape)
        lq_mlp_content_features = self.MLP_content_encoder(multi_scale_lq_content)
        # 打印3个向量的形状
        # print(LZ_diff_mlp_quality_features.shape, LH_diff_mlp_quality_features.shape, lq_mlp_content_features.shape)

        # # TODO 消融 只有LQ 256
        # print(f'lq_mlp_content_features.shape: {lq_mlp_content_features.shape}')
        feature = lq_mlp_content_features

        if self.my_config.ablation == 'lq_lz':
            # print('lq_lz')
            # # TODO 消融 LQ+LZ 512
            # print(f'feature.shap:{feature.shape}')
            # print(f'lz_diff_mlp_quality_features.shape: {LZ_diff_mlp_quality_features.shape}')
            # feature = torch.cat((feature, LZ_diff_mlp_quality_features), 1)
            feature = torch.cat((feature, LZ_diff_mlp_quality_features), 1)
        elif self.my_config.ablation == 'lq_lh':
            # print('lq_lh')
            # # TODO 消融 LQ+LH 512
            # print(f'lh_diff_mlp_quality_features.shape: {LH_diff_mlp_quality_features.shape}')
            # feature = torch.cat((feature, LH_diff_mlp_quality_features), 1)
            feature = torch.cat((feature, LH_diff_mlp_quality_features), 1)
        elif self.my_config.ablation == 'all':
            # print('all')
            # # TODO 融合模块 512
            feature_LZH = self.ff(LZ_diff_mlp_quality_features, LH_diff_mlp_quality_features)
            # print(feature_LZH.shape, lq_mlp_content_features.shape)
            feature = torch.cat((feature_LZH, lq_mlp_content_features), 1)

            # TODO 融合模块
            # feature_LZ = self.ff(lq_mlp_content_features, LZ_diff_mlp_quality_features)  # content作为qk，diff作为v [b, 256 or 512]
            # feature_LH = self.ff(lq_mlp_content_features, LH_diff_mlp_quality_features)  # content作为qk，diff作为v [b, 256 or 512]
            # # print(f'feature_LZ.shape: {feature_LZ.shape}, feature_LH.shape: {feature_LH.shape}')
            # feature = torch.cat((feature_LZ, feature_LH), dim=1)

            # TODO CAT LQ+LZ+LH 768
            # feature_LZH = torch.cat((LZ_diff_mlp_quality_features, LH_diff_mlp_quality_features), dim=1)
            # feature = torch.cat((feature_LZH, lq_mlp_content_features), 1)
        else:
            pass
            # print('lq')

        # feature = self.reduce(feature)
        # print(feature.shape)
        pred = self.regressor(feature)  # 分数回归
        return pred  # 返回两inner特征列表用于蒸馏L2损失 以及 分数预测用于L1约束

    def _load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


if __name__ == "__main__":
    config = set_args()
    args = TrainOptions().parse()
    # config.content_model_path = '../weights/content_aware_r50.pth'
    # config.quality_model_path = '../weights/quality_aware_r50.pth'
    t1 = time.time()
    m = DualReferenceIQANet(config, args).to('cuda')
    # m = DualReferenceIQANetNoMixer(config, args).to('cuda')
    t2 = time.time()
    lq = torch.rand((3, 10, 3, 224, 224)).to('cuda')
    hq = torch.rand((3, 10, 3, 224, 224)).to('cuda')
    zlq = torch.rand((3, 10, 3, 224, 224)).to('cuda')
    p = m(lq, hq, zlq)
    print(p.shape)
    t3 = time.time()
    print(f'Init时间：{t2 - t1}s, 前向传播时间:{t3 - t2}, 总运行时间：{t3 - t1}s')
