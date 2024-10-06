import os
import sys

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch.nn.functional as F

from Utils.OutputSaver import Saver
from models.networks.build_backbone import build_model
from options.train_options import TrainOptions

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# 保存控制台输出
# TODO 路径修改:日志
saver = Saver( f'moco_log.log',sys.stdout)
sys.stdout = saver
# 自定义数据集类以加载图像及其失真类型和等级
class KADISDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = self.load_data()
        self.pairs = self.generate_pairs()

    def load_data(self):
        data = []
        for filename in os.listdir(self.root_dir):
            if filename.endswith('.png'):
                parts = filename.split('_')
                if len(parts) == 3:
                    img_path = os.path.join(self.root_dir, filename)
                    dist_type = parts[1]  # 失真类型
                    dist_level = parts[2].split('.')[0]  # 失真等级
                    data.append((img_path, int(dist_type), int(dist_level)))
        return data

    def generate_pairs(self):
        pairs = []
        for i in range(len(self.data)):
            for j in range(i + 1, len(self.data)):
                img1, dist_type1, dist_level1 = self.data[i]
                img2, dist_type2, dist_level2 = self.data[j]
                pairs.append((img1, dist_type1, dist_level1, img2, dist_type2, dist_level2))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path1, dist_type1, dist_level1, img_path2, dist_type2, dist_level2 = self.pairs[idx]
        image1 = Image.open(img_path1).convert('RGB')
        image2 = Image.open(img_path2).convert('RGB')
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, dist_type1, dist_level1, image2, dist_type2, dist_level2

# 数据增强
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载数据集并选择子集
dataset = KADISDataset(root_dir='/data/dataset/kadid10k/imgs', transform=transform)
print(f"Total number of samples in the dataset: {len(dataset)}")
subset_indices = list(range(2000000))  # 选择一个子集
subset = torch.utils.data.Subset(dataset, subset_indices)
dataloader = DataLoader(subset, batch_size=128, shuffle=True, num_workers=4)

args = TrainOptions().parse()
args.mem = 'moco'

# MoCo v2 模型定义
class MoCo(nn.Module):
    def __init__(self, dim=2048, K=65536, m=0.999, T=0.07):  # 将dim修改为2048
        super(MoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T

        # 创建编码器和动量编码器
        self.encoder_q, self.encoder_k = build_model(args)

        # 动量编码器参数不参与梯度更新
        for param_k in self.encoder_k.parameters():
            param_k.requires_grad = False

        # 创建队列和标签
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        # 动量更新动量编码器的参数
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # 更新队列
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        # Update queue with new keys
        if ptr + batch_size <= self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            num_keys_in_first_section = self.K - ptr
            num_keys_in_second_section = batch_size - num_keys_in_first_section

            self.queue[:, ptr:] = keys[:num_keys_in_first_section].T
            self.queue[:, :num_keys_in_second_section] = keys[num_keys_in_first_section:].T

        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        # 计算查询编码
        q = self.encoder_q(im_q, mode=3)[-1]
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k, mode=3)[-1]
            k = nn.functional.normalize(k, dim=1)

        return q, k

# 初始化MoCo模型
model = MoCo().cuda()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)

best_loss = float('inf')

# 训练模型
for epoch in range(200):  # 200个训练轮次
    running_loss = 0.0
    for im_q, dist_type_q, dist_level_q, im_k, dist_type_k, dist_level_k in dataloader:
        im_q, im_k = im_q.cuda(), im_k.cuda()
        dist_type_q, dist_level_q = dist_type_q.cuda(), dist_level_q.cuda()
        dist_type_k, dist_level_k = dist_type_k.cuda(), dist_level_k.cuda()

        # 前向传播
        q, k = model(im_q, im_k)
        q = q.view(q.size(0), q.size(1), -1).mean(dim=2)
        k = k.view(k.size(0), k.size(1), -1).mean(dim=2)

        # 定义正样本和负样本
        pos_mask = (dist_type_q == dist_type_k) & (dist_level_q == dist_level_k)
        neg_mask = ~pos_mask

        # 计算对比损失
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1) * pos_mask.float()
        l_neg = torch.einsum('nc,ck->nk', [q, model.queue.clone().detach()])

        l_neg = l_neg * neg_mask.float().unsqueeze(1)

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= model.T

        # labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        labels = pos_mask.long().cuda()
        # 反向传播和优化
        optimizer.zero_grad()
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        model._dequeue_and_enqueue(k)
        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Loss: {epoch_loss}")

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.encoder_q.state_dict(), 'best_moco_model2.pth')
        print(f"Best model saved at epoch {epoch+1} with loss {epoch_loss}")

print("Training completed.")
