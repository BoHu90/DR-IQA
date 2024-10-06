import os
import time
import torch
import faiss
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from models.networks.SwinT import swin_tiny_patch4_window7_224, swin_small_patch4_window7_224
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
class ImageSelector:
    def __init__(self, model, ref_paths, feature_save_path='ref_features.pkl', batch_size=128):
        self.model_content = model
        self.model_content.eval()
        self.ref_paths = ref_paths
        self.feature_save_path = feature_save_path
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        if os.path.exists(self.feature_save_path):
            print('存在特征文件')
            self.ref_features, self.ref_feature_paths = self.load_features()
        else:
            self.ref_features, self.ref_feature_paths = self.extract_ref_features()
            self.save_features(self.ref_features, self.ref_feature_paths)
        self.index = self.build_index(self.ref_features)

    def extract_ref_features(self):
        ref_features = []
        ref_feature_paths = []
        with torch.no_grad():
            for i in range(0, len(self.ref_paths), self.batch_size):
                batch_paths = self.ref_paths[i:i + self.batch_size]
                batch_imgs = [self.transform(Image.open(p).convert('RGB')).unsqueeze(0) for p in batch_paths]
                batch_imgs = torch.cat(batch_imgs, dim=0).to('cuda')
                batch_features = self.model_content(batch_imgs)[-1]
                batch_features = batch_features.reshape(batch_features.size(0), -1).cpu().numpy()
                ref_features.append(batch_features)
                ref_feature_paths.extend(batch_paths)
        ref_features = np.vstack(ref_features)
        return ref_features, ref_feature_paths

    def save_features(self, features, paths):
        with open(self.feature_save_path, 'wb') as f:
            pickle.dump((features, paths), f)

    def load_features(self):
        with open(self.feature_save_path, 'rb') as f:
            features, paths = pickle.load(f)
        return features, paths

    def build_index(self, ref_features):
        dimension = ref_features.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(ref_features)
        return index

    def select_similar_image(self, LQ):
        LQ = self.transform(LQ).unsqueeze(0).to('cuda')
        with torch.no_grad():
            LQ_feature = self.model_content(LQ)[-1]
            LQ_feature = LQ_feature.reshape(LQ_feature.size(0), -1).cpu().numpy()

        D, I = self.index.search(LQ_feature, 1)  # 寻找最近邻
        best_path = self.ref_feature_paths[I[0][0]]
        return best_path


# 主函数测试
if __name__ == '__main__':
    root_paths = [
        '/data/dataset/DIV2K/train_HR',
        '/data/dataset/DIV2K/val_HR',
        # '/data/dataset/Flickr2K'
    ]
    ref_paths = []
    for ref_root in root_paths:
        for HQ_diff_content_img_path in os.listdir(ref_root):
            if HQ_diff_content_img_path[-3:] in ['png', 'jpg', 'bmp']:
                ref_paths.append(os.path.join(ref_root, HQ_diff_content_img_path))
    print(f'Total number of samples in the HQ imgs pool: {len(ref_paths)}')

    t1 = time.time()
    model = swin_tiny_patch4_window7_224().to('cuda')
    image_selector = ImageSelector(model,ref_paths)
    t2 = time.time()
    LQ_paths = [
        # '/data/dataset/LIVE/refimgs/lighthouse.bmp',
        # '/data/dataset/tid2013/reference_images/I09.BMP'

        '/data/dataset/BID/ImageDatabase/DatabaseImage0058.JPG',
        '/data/dataset/CSIQ/dst_imgs_all/aerial_city.fnoise.1.png',
        '/data/dataset/FLive/database/blur_dataset/motion0062.jpg',
        '/data/dataset/FLive/database/blur_dataset/motion0089.jpg',
        '/data/dataset/kadid10k/imgs/I22_07_02.png',
        '/data/dataset/FLive/database/voc_emotic_ava/AVA__100991.jpg',
        '/data/dataset/koniq-10k/1024x768/10047832035.jpg',
        '/data/dataset/kadid10k/imgs/I13_01_02.png',
        '/data/dataset/ChallengeDB_release/Images/103.bmp',
        '/data/dataset/kadid10k/imgs/I10_21_03.png'
    ]
    for i, LQ_path in enumerate(LQ_paths):
        LQ = Image.open(LQ_path).convert('RGB')
        best_match_path = image_selector.select_similar_image(LQ)
        print(f'Best match for {LQ_path} is {best_match_path}')

        # 展示LQ图像和最相似的图像
        LQ_img = Image.open(LQ_path)
        best_match_img = Image.open(best_match_path)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(LQ_img)
        plt.title(f'LQ Image {i + 1}')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(best_match_img)
        plt.title(f'Best Match Image {i + 1}')
        plt.axis('off')

        plt.show()

    t3 = time.time()
    print('提取特征时间', t2 - t1)
    print('找到最相似图像时间', t3 - t2)
    print(f'平均时间: {(t3 - t2) / len(LQ_paths)}')
    print('总时间', t3 - t1)


# # ImageSelector.py
# import os
# import time
# import torch
# import faiss
# import numpy as np
# from PIL import Image
# from torchvision import transforms
# from models.networks.SwinT import swin_tiny_patch4_window7_224, swin_small_patch4_window7_224
# import pickle
# import csv
#
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#
#
# class ImageSelector:
#     def __init__(self, model, ref_paths, feature_save_path='ref_features.pkl', batch_size=128):
#         self.model_content = model
#         self.model_content.eval()
#         self.ref_paths = ref_paths
#         self.feature_save_path = feature_save_path
#         self.batch_size = batch_size
#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor()
#         ])
#         if os.path.exists(self.feature_save_path):
#             print('存在特征文件')
#             self.ref_features, self.ref_feature_paths = self.load_features()
#         else:
#             self.ref_features, self.ref_feature_paths = self.extract_ref_features()
#             self.save_features(self.ref_features, self.ref_feature_paths)
#         self.index = self.build_index(self.ref_features)
#
#     def extract_ref_features(self):
#         ref_features = []
#         ref_feature_paths = []
#         with torch.no_grad():
#             for i in range(0, len(self.ref_paths), self.batch_size):
#                 batch_paths = self.ref_paths[i:i + self.batch_size]
#                 batch_imgs = [self.transform(Image.open(p).convert('RGB')).unsqueeze(0) for p in batch_paths]
#                 batch_imgs = torch.cat(batch_imgs, dim=0).to('cuda')
#                 batch_features = self.model_content(batch_imgs)[-1]
#                 batch_features = batch_features.reshape(batch_features.size(0), -1).cpu().numpy()
#                 ref_features.append(batch_features)
#                 ref_feature_paths.extend(batch_paths)
#         ref_features = np.vstack(ref_features)
#         return ref_features, ref_feature_paths
#
#     def save_features(self, features, paths):
#         with open(self.feature_save_path, 'wb') as f:
#             pickle.dump((features, paths), f)
#
#     def load_features(self):
#         with open(self.feature_save_path, 'rb') as f:
#             features, paths = pickle.load(f)
#         return features, paths
#
#     def build_index(self, ref_features):
#         dimension = ref_features.shape[1]
#         index = faiss.IndexFlatL2(dimension)
#         index.add(ref_features)
#         return index
#
#     def select_similar_image(self, LQ):
#         LQ = self.transform(LQ).unsqueeze(0).to('cuda')
#         with torch.no_grad():
#             LQ_feature = self.model_content(LQ)[-1]
#             LQ_feature = LQ_feature.reshape(LQ_feature.size(0), -1).cpu().numpy()
#
#         D, I = self.index.search(LQ_feature, 1)
#         best_path = self.ref_feature_paths[I[0][0]]
#         return best_path
#
#     def process_test_images(self, test_root, output_csv):
#         with open(output_csv, 'w', newline='') as csvfile:
#             fieldnames = ['LQ_image', 'HQ_image']
#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#             writer.writeheader()
#
#             for root, dirs, files in os.walk(test_root):
#                 for file in files:
#                     if file.lower().endswith(('.png', '.jpg', '.bmp')):
#                         LQ_path = os.path.join(root, file)
#                         LQ_image = Image.open(LQ_path).convert('RGB')
#                         best_match_path = self.select_similar_image(LQ_image)
#                         writer.writerow({'LQ_image': LQ_path, 'HQ_image': best_match_path})
#                         print(f'Processed {LQ_path} -> {best_match_path}')
#
#
# # 主函数测试
# if __name__ == '__main__':
#     root_paths = [
#         '/data/dataset/DIV2K/train_HR',
#         '/data/dataset/DIV2K/val_HR',
#         '/data/dataset/Flickr2K'
#     ]
#     ref_paths = []
#     for ref_root in root_paths:
#         for HQ_diff_content_img_path in os.listdir(ref_root):
#             if HQ_diff_content_img_path.lower().endswith(('png', 'jpg', 'bmp')):
#                 ref_paths.append(os.path.join(ref_root, HQ_diff_content_img_path))
#     print(f'Total number of samples in the HQ imgs pool: {len(ref_paths)}')
#
#     model = swin_small_patch4_window7_224().to('cuda')
#     t1 = time.time()
#     image_selector = ImageSelector(model, ref_paths)
#     t2 = time.time()
#     test_root = '/data/dataset/BID/ImageDatabase'
#     output_csv = 'LQ_HQ_pairs.csv'
#     image_selector.process_test_images(test_root, output_csv)
#     time3 = time.time()
#     print(f'初始化时间: {t2 - t1}, 测试时间: {time3 - t2}, 平均单张测试时间: {(time3 - t2) / len(ref_paths)}')


# 生成整个数据集的非对齐参考图像路径
# import os
# import time
# import torch
# import faiss
# import numpy as np
# from PIL import Image
# from torchvision import transforms
# from models.networks.SwinT import swin_small_patch4_window7_224
# import pickle
# import csv
#
# class ImageSelector:
#     def __init__(self, model, ref_paths, feature_save_path='ref_features.pkl', batch_size=512):
#         self.model_content = model
#         self.model_content.eval()
#         self.ref_paths = ref_paths
#         self.feature_save_path = feature_save_path
#         self.batch_size = batch_size
#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor()
#         ])
#
#         # 记录提取特征的时间
#         start_time = time.time()
#
#         if os.path.exists(self.feature_save_path):
#             print('存在特征文件')
#             self.ref_features, self.ref_feature_paths = self.load_features()
#         else:
#             self.ref_features, self.ref_feature_paths = self.extract_ref_features()
#             self.save_features(self.ref_features, self.ref_feature_paths)
#
#         self.index = self.build_index(self.ref_features)
#
#         # 计算并打印提取特征的时间开销
#         self.feature_extraction_time = time.time() - start_time
#         print(f'提取特征的时间开销: {self.feature_extraction_time:.2f} 秒')
#
#     def extract_ref_features(self):
#         ref_features = []
#         ref_feature_paths = []
#         with torch.no_grad():
#             for i in range(0, len(self.ref_paths), self.batch_size):
#                 batch_paths = self.ref_paths[i:i + self.batch_size]
#                 batch_imgs = [self.transform(Image.open(p).convert('RGB')).unsqueeze(0) for p in batch_paths]
#                 batch_imgs = torch.cat(batch_imgs, dim=0).to('cuda')
#                 batch_features = self.model_content(batch_imgs)[-1]
#                 batch_features = batch_features.reshape(batch_features.size(0), -1).cpu().numpy()
#                 ref_features.append(batch_features)
#                 ref_feature_paths.extend(batch_paths)
#         ref_features = np.vstack(ref_features)
#         return ref_features, ref_feature_paths
#
#     def save_features(self, features, paths):
#         with open(self.feature_save_path, 'wb') as f:
#             pickle.dump((features, paths), f)
#
#     def load_features(self):
#         with open(self.feature_save_path, 'rb') as f:
#             features, paths = pickle.load(f)
#         return features, paths
#
#     def build_index(self, ref_features):
#         dimension = ref_features.shape[1]
#         index = faiss.IndexFlatL2(dimension)
#         index.add(ref_features)
#         return index
#
#     def select_similar_image(self, LQ_path):
#         LQ_image = Image.open(LQ_path).convert('RGB')
#         LQ = self.transform(LQ_image).unsqueeze(0).to('cuda')
#         with torch.no_grad():
#             LQ_feature = self.model_content(LQ)[-1]
#             LQ_feature = LQ_feature.reshape(LQ_feature.size(0), -1).cpu().numpy()
#
#         D, I = self.index.search(LQ_feature, 1)
#         best_path = self.ref_feature_paths[I[0][0]]
#         return LQ_path, best_path
#
#     def process_test_images(self, test_root, output_csv):
#         # 记录开始时间
#         start_time = time.time()
#
#         # 获取测试集中所有图像文件路径
#         LQ_paths = []
#         for root, _, files in os.walk(test_root):
#             for file in files:
#                 if not file.startswith('.') and file.lower().endswith(('.png', '.jpg', '.bmp')):
#                     LQ_paths.append(os.path.join(root, file))
#
#         # 记录寻找最相似图像的时间
#         matching_start_time = time.time()
#
#         # 串行处理每一张图像
#         results = []
#         for LQ_path in LQ_paths:
#             result = self.select_similar_image(LQ_path)
#             results.append(result)
#
#         # 计算并打印寻找最相似图像的时间开销
#         matching_time = time.time() - matching_start_time
#         print(f'寻找最相似图像的时间开销: {matching_time:.2f} 秒')
#         print(f'平均每张图像的时间开销: {matching_time / len(LQ_paths):.2f} 秒')
#
#         # 保存结果到CSV
#         with open(output_csv, 'w', newline='') as csvfile:
#             fieldnames = ['LQ_image', 'HQ_image']
#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#             writer.writeheader()
#             for LQ_path, best_match_path in results:
#                 writer.writerow({'LQ_image': LQ_path, 'HQ_image': best_match_path})
#                 print(f'Processed {LQ_path} -> {best_match_path}')
#
#         # 计算并打印总时间开销
#         total_time = time.time() - start_time
#         print(f'总时间开销: {total_time:.2f} 秒')
#
# # 主函数测试
# if __name__ == '__main__':
#     root_paths = [
#         '/data/dataset/DIV2K/train_HR',
#         '/data/dataset/DIV2K/val_HR',
#         '/data/dataset/Flickr2K'
#     ]
#     ref_paths = []
#     for ref_root in root_paths:
#         for HQ_diff_content_img_path in os.listdir(ref_root):
#             if HQ_diff_content_img_path.lower().endswith(('png', 'jpg', 'bmp')):
#                 ref_paths.append(os.path.join(ref_root, HQ_diff_content_img_path))
#     print(f'Total number of samples in the HQ imgs pool: {len(ref_paths)}')
#
#     model = swin_small_patch4_window7_224().to('cuda')
#     image_selector = ImageSelector(model, ref_paths)
#
#     test_root = '/data/dataset/kadid10k/imgs'
#     output_csv = 'Kadid_refs.csv'
#     image_selector.process_test_images(test_root, output_csv)
