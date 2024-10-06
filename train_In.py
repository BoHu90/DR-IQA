import sys
import time

import torch
import os
import random

from Utils.OutputSaver import Saver
from Utils.tools import convert_obj_score
from dataloaders.dataloader_LQ_HQ_diff_content_HQ import DataLoader
from models.networks.DualReferenceIQANet import DualReferenceIQANet
from options.option_train_In import set_args, check_args
from scipy import stats
import numpy as np
from options.train_options import TrainOptions

from torch.optim.lr_scheduler import CosineAnnealingLR


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

img_num = {
    'kadid10k': list(range(0, 10125)),
    'live': list(range(0, 29)),  # ref HR image
    'csiq': list(range(0, 30)),  # ref HR image
    'tid2013': list(range(0, 25)),
    'livec': list(range(0, 1162)),  # no-ref image
    'koniq10k': list(range(0, 10073)),  # no-ref image
    'bid': list(range(0, 586)),  # no-ref imaged
    'flive': list(range(0, 39807)),  # no-ref imaged
    'spaq': list(range(0, 11125))
}
folder_path = {
    'live': '/data/dataset/LIVE/',
    'csiq': '/data/dataset/CSIQ/',
    'tid2013': '/data/dataset/tid2013/',
    'livec': '/data/dataset/ChallengeDB_release',
    'koniq10k': '/data/dataset/koniq-10k/',
    'bid': '/data/dataset/BID/',
    'kadid10k': '/data/dataset/kadid10k/',
    'led':'/data/dataset/LEDataset',
    'flive': '/data/dataset/FLive/database',
    'spaq': '/data/dataset/SPAQ'
}

class DRIQASolver(object):
    def __init__(self, config, args):
        # 加载配置、设置设备（GPU或CPU）、创建日志文件
        self.config = config
        # self.device = torch.device(f'cuda:{config.gpu_ids}' if config.gpu_ids is not None else 'cpu')
        self.device = torch.device('cuda' if config.gpu_ids is not None else 'cpu')
        # 模型
        self.DRNet = DualReferenceIQANet(config, args)
        # 模型转设备
        self.DRNet = self.DRNet.to(self.device)

        # lr,opt,loss,epoch
        self.lr = config.lr
        self.lr_ratio = 10
        # 获取DRNet中model_content和model_quality的参数ID
        content_params = list(map(id, self.DRNet.model_content.parameters()))
        quality_params = list(map(id, self.DRNet.model_quality.parameters()))
        extract_params = content_params + quality_params
        # 获取DRNet中除了content_params和quality_params以外的参数other_params
        self.other_params = filter(lambda p: id(p) not in extract_params, self.DRNet.parameters())  # 其他参数
        paras = [{'params': self.other_params, 'lr': self.lr * self.lr_ratio}]
        # 学习率\损失函数的定义
        # self.optimizer = torch.optim.NAdam(paras, weight_decay=config.weight_decay)
        if config.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(paras, weight_decay=config.weight_decay)
        elif config.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(paras, weight_decay=0.01)
        elif config.optimizer == 'Adamax':
            self.optimizer = torch.optim.Adamax(paras, weight_decay=config.weight_decay)
        elif config.optimizer == 'NAdam':
            self.optimizer = torch.optim.NAdam(paras, weight_decay=config.weight_decay)

        # TODO 2 学习率调度器
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.epochs)

        # 损失函数
        self.mse_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.sml_loss = torch.nn.SmoothL1Loss()
        self.epochs = config.epochs

        # data：使用的是kadid10k
        # 获取index
        sel_num = img_num[config.train_dataset]
        random.shuffle(sel_num)
        #  做库内:
        config.train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
        test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]
        # 训练数据
        train_loader = DataLoader(config.train_dataset, folder_path[config.train_dataset], config.ref_train_dataset_path,
                                  config.train_index, config.patch_size, config.train_patch_num,
                                  batch_size=config.batch_size, istrain=True, self_patch_num=config.self_patch_num)
        # 库内测试数据
        test_loader = DataLoader(config.train_dataset, folder_path[config.train_dataset], config.ref_test_dataset_path,
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
                LQ_patches, HQ_patches, ZLQ_patches, label = LQ_patches.to(self.device), HQ_patches.to(self.device), ZLQ_patches.to(self.device), label.to(self.device)
                # 提取清零
                self.optimizer.zero_grad()
                # 开启混合精度训练
                with torch.cuda.amp.autocast():

                    # 计算分数
                    pred = self.DRNet(LQ_patches, HQ_patches, ZLQ_patches)

                    # 存储预测分数与真实分数并计算损失
                    pred_scores = pred_scores + pred.cpu().tolist()
                    gt_scores = gt_scores + label.cpu().tolist()
                    pred_loss = self.sml_loss(pred.squeeze(), label.float().detach())

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
                torch.save(self.DRNet.state_dict(), os.path.join(self.config.model_checkpoint_dir,
                                                                      # f'{config.train_dataset}_b{config.batch_size}_lr{config.lr}in_{config.mlp_depth}depth_{config.model}_{config.optimizer}_qf{config.weight_decay}.pth'))
                                                                      f'{config.train_dataset}_{config.ablation}_b{config.batch_size}_lr{config.lr}in_{config.mlp_depth}depth_{config.model}_{config.optimizer}.pth'))
                print("DRNet更新：")
            print('%d:%s\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f \n' %
                  (t, config.train_dataset, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc,
                   test_krcc))
            # TODO 3 更新学习率
            self.scheduler.step()
            # self.lr = self.lr / pow(10, (t // self.config.update_opt_epoch))
            # if t > 20:
            #     self.lr_ratio = 1
            #
            # # 获取DRNet中model_content和model_quality的参数ID
            # content_params = list(map(id, self.DRNet.model_content.parameters()))
            # quality_params = list(map(id, self.DRNet.model_quality.parameters()))
            # extract_params = content_params + quality_params
            # # 获取DRNet中除了content_params和quality_params以外的参数other_params
            # self.other_params = filter(lambda p: id(p) not in extract_params, self.DRNet.parameters())  # 其他参数
            # paras = [{'params': self.other_params, 'lr': self.lr * self.lr_ratio}]
            # self.optimizer = torch.optim.Adam(paras, weight_decay=self.config.weight_decay)
        # 取所有epoch最佳的指标
        # 库内
        print('Best %s test SRCC %f, PLCC %f, KRCC %f\n' % (config.train_dataset, best_srcc, best_plcc, best_krcc))

    def test(self, test_data):
        # 测试模式
        self.DRNet.train(False)
        test_pred_scores, test_gt_scores = [], []

        for LQ_patches, HQ_patches, ZLQ_patches, label in test_data:
            LQ_patches, HQ_patches, ZLQ_patches, label = LQ_patches.to(self.device), HQ_patches.to(self.device), ZLQ_patches.to(self.device), label.to(
                self.device)
            with torch.no_grad():
                # 计算分数
                pred = self.DRNet(LQ_patches,HQ_patches, ZLQ_patches)
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
        test_krcc, _ = stats.kendalltau(test_pred_scores, test_gt_scores)
        test_srcc, test_plcc, test_krcc = abs(test_srcc), abs(test_plcc), abs(test_krcc)
        self.DRNet.train(True)
        return test_srcc, test_plcc, test_krcc


if __name__ == "__main__":
    config = set_args()
    args = TrainOptions().parse()

    # 保存控制台输出
    # TODO 路径修改:日志
    saver = Saver(f'./logs/ablation/{config.train_dataset}_{config.ablation}_b{config.batch_size}_lr{config.lr}in_{config.mlp_depth}depth_{config.model}_{config.optimizer}.log', sys.stdout)
    # saver = Saver(f'./logs/AAIn/{config.train_dataset}_b{config.batch_size}_lr{config.lr}in_{config.mlp_depth}depth_{config.model}_{config.optimizer}_qf{config.weight_decay}.log', sys.stdout)
    sys.stdout = saver
    config = check_args(config)
    time1 = time.time()
    solver = DRIQASolver(config=config, args=args)
    solver.train()
    time2 = time.time()
    print('Running time: %f s' % (time2 - time1))
