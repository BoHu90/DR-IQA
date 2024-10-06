import sys
import torch
import os
import random
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
import numpy as np

from Utils.OutputSaver import Saver
from Utils.tools import convert_obj_score
from dataloaders.dataloader_LQ_HQ_diff_content_HQ import DataLoader
from models.networks.build_backbone import build_model
from options.option_train_In import set_args, check_args
from options.train_options import TrainOptions

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

img_num = {
    'kadid10k': list(range(0, 10125)),
    'live': list(range(0, 29)),
    'csiq': list(range(0, 30)),
    'tid2013': list(range(0, 25)),
    'livec': list(range(0, 1162)),
    'koniq-10k': list(range(0, 10073)),
    'bid': list(range(0, 586)),
    'flive': list(range(0, 39807)),
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
    'led': '/data/dataset/LEDataset',
    'flive': '/data/dataset/FLive/database',
}

class DRIQASolver(object):
    def __init__(self, config, args):
        self.config = config
        self.device = torch.device('cuda' if config.gpu_ids is not None else 'cpu')

        # 内容特征提取器以及权重
        self.model_content, _ = build_model(args)
        if config.content_model_path:
            checkpoint = torch.load(config.content_model_path, map_location='cpu')
            self.model_content.load_state_dict(checkpoint['model'])
            print(f'load {config.content_model_path} success')
        self.model_content = self.model_content.to(self.device)
        self.model_content.eval()

        # 质量特征提取器以及权重
        self.model_quality, _ = build_model(args)
        if config.quality_model_path:
            checkpoint = torch.load(config.quality_model_path, map_location='cpu')
            self.model_quality.load_state_dict(checkpoint['model'])
            print(f'load {config.quality_model_path} success')
        self.model_quality = self.model_quality.to(self.device)
        self.model_quality.eval()

        # TODO 回归模型
        self.regressor = linear_model.Ridge(alpha=2.0)  # 增大正则化强度

        # 特征标准化
        self.scaler = StandardScaler()

        # PCA降维
        self.pca = PCA(n_components=100)

        # 获取训练和测试数据集的索引
        sel_num = img_num[config.train_dataset]
        random.shuffle(sel_num)
        config.train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
        test_index = sel_num[int(round(0.8 * len(sel_num))):]

        # 数据加载器
        train_loader = DataLoader(config.train_dataset, folder_path[config.train_dataset],
                                  config.train_index, config.patch_size, config.train_patch_num,
                                  batch_size=config.batch_size, istrain=True, self_patch_num=config.self_patch_num)
        test_loader = DataLoader(config.train_dataset, folder_path[config.train_dataset],
                                 test_index, config.patch_size, config.test_patch_num, istrain=False,
                                 self_patch_num=config.self_patch_num)

        self.train_data = train_loader.get_dataloader()
        self.test_data = test_loader.get_dataloader()

        # 提取并预处理训练数据的特征
        self.train_features, self.train_labels = self.preprocess_data(self.train_data)
        # 提取并预处理测试数据的特征
        self.test_features, self.test_labels = self.preprocess_data(self.test_data, fit_scaler=False)

    def extract_features(self, patches):
        content_features = self.model_content(patches)
        quality_features = self.model_quality(patches)
        combined_features = torch.cat((content_features, quality_features), dim=1)
        return combined_features

    def preprocess_data(self, data_loader, fit_scaler=True):
        features_list = []
        labels_list = []

        for LQ_patches, _, _, label in data_loader:
            LQ_patches, label = LQ_patches.to(self.device), label.to(self.device)
            combined_features = self.extract_features(LQ_patches)
            combined_features_np = combined_features.cpu().detach().numpy()
            label_np = label.cpu().detach().numpy()

            features_list.append(combined_features_np)
            labels_list.append(label_np)

        features = np.vstack(features_list)
        labels = np.hstack(labels_list)

        # 标准化
        if fit_scaler:
            features = self.scaler.fit_transform(features)
        else:
            features = self.scaler.transform(features)

        # PCA降维
        if fit_scaler:
            features = self.pca.fit_transform(features)
        else:
            features = self.pca.transform(features)

        return features, labels

    def train(self):
        best_srcc, best_plcc, best_krcc = 0.0, 0.0, 0.0

        print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC\tTest_KRCC')

        for t in range(10):
            # 训练Ridge回归模型
            self.regressor.fit(self.train_features, self.train_labels)

            pred_train = self.regressor.predict(self.train_features)
            train_loss = mean_squared_error(self.train_labels, pred_train)
            train_srcc, _ = stats.spearmanr(pred_train, self.train_labels)

            pred_test = self.regressor.predict(self.test_features)
            test_srcc, test_plcc, test_krcc = self.evaluate(pred_test, self.test_labels)

            if test_srcc + test_plcc + test_krcc > best_srcc + best_plcc + best_krcc:
                best_srcc, best_plcc, best_krcc = test_srcc, test_plcc, test_krcc
                torch.save(self.regressor, os.path.join(self.config.model_checkpoint_dir,
                                                          f'{self.config.train_dataset}_in_saved_ridge_model.pth'))
                print("Ridge模型更新：")

            print('%d:%s\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f \n' %
                  (t, self.config.train_dataset, train_loss, train_srcc, test_srcc, test_plcc, test_krcc))

        print('Best %s test SRCC %f, PLCC %f, KRCC %f\n' % (self.config.train_dataset, best_srcc, best_plcc, best_krcc))

    def evaluate(self, pred, gt):
        test_srcc, _ = stats.spearmanr(pred, gt)
        test_plcc, _ = stats.pearsonr(pred, gt)
        test_krcc, _ = stats.kendalltau(pred, gt)

        return abs(test_srcc), abs(test_plcc), abs(test_krcc)

if __name__ == "__main__":
    config = set_args()
    args = TrainOptions().parse()
    saver = Saver(f'./logs/In/ridge_b{config.batch_size}_lr{config.lr}in_SGDRegressor.log', sys.stdout)
    sys.stdout = saver
    config = check_args(config)
    solver = DRIQASolver(config=config, args=args)
    solver.train()
