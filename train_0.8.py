import sys

import torch
import os
import random

from Utils.OutputSaver import Saver
from Utils.tools import convert_obj_score
from dataloaders.dataloader_LQ_HQ_diff_content_HQ import DataLoader
from models.networks.Regressors import RegressionFCNet
from models.networks.build_backbone import build_model
from options.option_train_In import set_args, check_args
from scipy import stats
import numpy as np
from options.train_options import TrainOptions

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

img_num = {
    'kadid10k': list(range(0, 10125)),
    'live': list(range(0, 29)),  # ref HR image
    'csiq': list(range(0, 30)),  # ref HR image
    'tid2013': list(range(0, 25)),
    'livec': list(range(0, 1162)),  # no-ref image
    'koniq-10k': list(range(0, 10073)),  # no-ref image
    'bid': list(range(0, 586)),  # no-ref imaged
    'flive': list(range(0, 39807)),  # no-ref imaged
    'led': list(range(0, 1000)),
}
folder_path = {
    'pipal': '/data/dataset/PIPAL',
    'live': '/data/dataset/LIVE/',
    'csiq': '/data/dataset/CSIQ/',
    'tid2013': '/data/dataset/tid2013/',
    'livec': '/data/dataset/ChallengeDB_release',
    'koniq-10k': '/data/dataset/koniq-10k/',
    'bid': '/data/dataset/BID/',
    'kadid10k': '/data/dataset/kadid10k/',
    'led':'/data/dataset/LEDataset',
    'flive': '/data/dataset/FLive/database'
}


class DRIQASolver(object):
    def __init__(self, config, args):
        # 加载配置、设置设备（GPU或CPU）、创建日志文件
        self.config = config
        # self.device = torch.device(f'cuda:{config.gpu_ids}' if config.gpu_ids is not None else 'cpu')
        self.device = torch.device('cuda' if config.gpu_ids is not None else 'cpu')
        # 内容特征提取器以及权重
        self.model_content, _ = build_model(args)
        self.model_content = torch.nn.DataParallel(self.model_content)
        if config.content_model_path:
            checkpoint = torch.load(config.content_model_path, map_location='cpu')
            self.model_content.load_state_dict(checkpoint['model'])
            # 打印加载成功信息
            print(f'load {config.content_model_path} success')
        self.model_content = self.model_content.to(self.device)
        # self.model_content.eval()  # 冻结模型
        self.model_content.train(True)  # 训练模型
        # 质量特征提取器以及权重
        self.model_quality, _ = build_model(args)
        self.model_quality = torch.nn.DataParallel(self.model_quality)
        if config.quality_model_path:
            checkpoint = torch.load(config.quality_model_path, map_location='cpu')
            self.model_quality.load_state_dict(checkpoint['model'])
            # 打印加载成功信息
            print(f'load {config.quality_model_path} success')
        self.model_quality = self.model_quality.to(self.device)
        # self.model_quality.eval()  # 冻结模型
        self.model_quality.train(True)  # 训练模型
        # 分数回归器
        self.regressor = RegressionFCNet().to(self.device)

        # lr,opt,loss,epoch
        self.lr = config.lr
        self.lr_ratio = 10
        # TODO 考虑改变蒸馏损失的权重
        self.feature_loss_ratio = 1
        print(f'lr_ratio:{self.lr_ratio},feature_loss_ratio:{self.feature_loss_ratio}')
        self.params = self.regressor.parameters()  # 特征提取器参数
        paras = [{'params': self.params, 'lr': self.lr * self.lr_ratio}
                 ]
        # 学习率\损失函数的定义
        self.optimizer = torch.optim.Adam(paras, weight_decay=config.weight_decay)
        self.mse_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.epochs = config.epochs

        # data：使用的是kadid10k
        # 获取index
        sel_num = img_num[config.train_dataset]
        random.shuffle(sel_num)
        #  做库内:
        config.train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
        test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]
        # 训练数据
        train_loader = DataLoader(config.train_dataset, folder_path[config.train_dataset],
                                  config.train_index, config.patch_size, config.train_patch_num,
                                  batch_size=config.batch_size, istrain=True, self_patch_num=config.self_patch_num)
        # 库内测试数据
        test_loader = DataLoader(config.train_dataset, folder_path[config.train_dataset],
                                 test_index, config.patch_size, config.test_patch_num, istrain=False,  # 测试模式
                                 self_patch_num=config.self_patch_num)

        self.train_data = train_loader.get_dataloader()
        # 库内测试数据
        self.test_data = test_loader.get_dataloader()

    def train(self):
        # 库内
        best_srcc = 0.0
        best_plcc = 0.0
        best_krcc = 0.0

        print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC\tTest_KRCC')

        scaler = torch.cuda.amp.GradScaler()

        for t in range(self.epochs):
            epoch_loss = []
            pred_scores = []
            gt_scores = []  # gt: ground truth LQ, HQ, ZLQ, target
            for LQ_patches, HQ_patches, ZLQ_patches, label in self.train_data:
                # 转设备
                LQ_patches, _, ZLQ_patches, label = LQ_patches.to(self.device), HQ_patches.to(
                    self.device), ZLQ_patches.to(self.device), label.to(self.device)
                # reshape
                # b, p, c, h, w = LQ_patches.shape
                # LQ_patches = LQ_patches.view(b * p, c, h, w)
                # b, p, c, h, w = ZLQ_patches.shape
                # ZLQ_patches = ZLQ_patches.view(b * p, c, h, w)
                # 提取清零
                self.optimizer.zero_grad()

                # 开启混合精度训练
                with torch.cuda.amp.autocast():
                    # 将 图片 输入到两个特征提取器
                    content_features = self.model_content(LQ_patches)
                    # 分别提取LQ和ZLQ的质量特征
                    quality_features_LQ = self.model_quality(LQ_patches)
                    quality_features_ZLQ = self.model_quality(ZLQ_patches)
                    # 获取质量差异特征
                    quality_features = quality_features_LQ - quality_features_ZLQ

                    # 计算分数
                    pred = self.regressor(content_features, quality_features)

                    # 存储预测分数与真实分数并计算损失
                    pred_scores = pred_scores + pred.cpu().tolist()
                    gt_scores = gt_scores + label.cpu().tolist()
                    pred_loss = self.l1_loss(pred.squeeze(), label.float().detach())

                    loss = pred_loss
                # 反向传播
                epoch_loss.append(loss.item())
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
            # 每个epoch进行一次测试
            # 库内测试
            test_srcc, test_plcc, test_krcc = self.test(self.test_data)
            # 库内
            if test_srcc + test_plcc + test_krcc > best_srcc + best_plcc + best_krcc:
                best_srcc, best_plcc, best_krcc = test_srcc, test_plcc, test_krcc
                # TODO 路径修改:模型
                torch.save(self.regressor.state_dict(), os.path.join(self.config.model_checkpoint_dir,
                                                                      f'{config.train_dataset}_in_saved_regressor.pth'))
                print("Regressor更新：")
            print('%d:%s\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f \n' %
                  (t, config.train_dataset, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc,
                   test_krcc))

            self.lr = self.lr / pow(10, (t // self.config.update_opt_epoch))
            # if t > 20:
            #     self.lr_ratio = 1
            # paras = [{'params': self.params, 'lr': self.lr * self.lr_ratio}
            #          ]
            # self.optimizer = torch.optim.Adam(paras, weight_decay=self.config.weight_decay)
        # 取所有epoch最佳的指标
        # 库内
        print('Best %s test SRCC %f, PLCC %f, KRCC %f\n' % (config.train_dataset, best_srcc, best_plcc, best_krcc))

    def test(self, test_data):
        # 测试模式
        self.regressor.train(False)
        self.model_content.train(False)
        self.model_quality.train(False)
        test_pred_scores, test_gt_scores = [], []

        for LQ_patches, _, ZLQ_patches, label in test_data:
            LQ_patches, ZLQ_patches, label = LQ_patches.to(self.device), ZLQ_patches.to(self.device), label.to(
                self.device)
            with torch.no_grad():
                # 将 图片 输入到两个特征提取器
                content_features = self.model_content(LQ_patches)
                quality_features_LQ = self.model_quality(LQ_patches)
                quality_features_ZLQ = self.model_quality(ZLQ_patches)
                # 获取质量差异特征
                quality_features = quality_features_LQ - quality_features_ZLQ
                # 计算分数
                pred = self.regressor(content_features, quality_features)
                test_pred_scores.append(float(pred.item()))
                test_gt_scores = test_gt_scores + label.cpu().tolist()
        if self.config.use_fitting_prcc_srcc:
            fitting_pred_scores = convert_obj_score(test_pred_scores, test_gt_scores)
        # 取平均
        test_pred_scores = np.mean(np.reshape(np.array(test_pred_scores), (-1, self.config.test_patch_num)), axis=1)
        test_gt_scores = np.mean(np.reshape(np.array(test_gt_scores), (-1, self.config.test_patch_num)), axis=1)
        test_srcc, _ = stats.spearmanr(test_pred_scores, test_gt_scores)
        if self.config.use_fitting_prcc_srcc:
            test_plcc, _ = stats.pearsonr(fitting_pred_scores, test_gt_scores)
        else:
            test_plcc, _ = stats.pearsonr(test_pred_scores, test_gt_scores)
        test_krcc, _ = stats.stats.kendalltau(test_pred_scores, test_gt_scores)
        test_srcc, test_plcc, test_krcc = abs(test_srcc), abs(test_plcc), abs(test_krcc)
        self.regressor.train(True)
        self.model_content.train(True)
        self.model_quality.train(True)
        return test_srcc, test_plcc, test_krcc


if __name__ == "__main__":
    config = set_args()
    args = TrainOptions().parse()
    print(args.jigsaw, args.modal, args.arch, args.head, args.head, args.feat_dim, args.mem)

    # 保存控制台输出
    # TODO 路径修改:日志
    saver = Saver(f'./logs/In/{config.train_dataset}_b{config.batch_size}_lr{config.lr}in_zlq_patches.log', sys.stdout)
    sys.stdout = saver
    config = check_args(config)
    solver = DRIQASolver(config=config, args=args)
    solver.train()
