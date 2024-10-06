import torch as torch
import torch.nn as nn


# 分数预测回归器
class RegressionFCNet(nn.Module):
    """
    Target network for quality prediction.
    """

    def __init__(self, in_channel=512):
        super(RegressionFCNet, self).__init__()
        self.target_in_size = in_channel
        self.target_fc1_size = 256
        self.target_fc2_size = 256

        self.sigmoid = nn.Sigmoid()
        self.l1 = nn.Linear(self.target_in_size, self.target_fc1_size)
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(self.target_fc1_size)
        self.l2 = nn.Linear(self.target_fc1_size, self.target_fc2_size)
        self.relu2 = nn.PReLU()
        self.drop2 = nn.Dropout(0.5)
        self.bn2 = nn.BatchNorm1d(256)
        self.l3 = nn.Linear(self.target_fc2_size, 1)

    def forward(self,x):
        # l1
        q = self.l1(x)
        q = self.relu1(q)
        q = self.drop1(q)
        q = self.bn1(q)
        # l2
        q = self.l2(q)
        q = self.relu2(q)
        q = self.drop2(q)
        q = self.bn2(q)
        # l3
        q = self.l3(q).squeeze()
        return q



# 分数预测回归器
class RegressionFCNet1(nn.Module):
    """
    Target network for quality prediction.
    """

    def __init__(self, in_channel=256):
        super(RegressionFCNet1, self).__init__()
        self.in_channel = in_channel

        self.l1 = nn.Linear(self.in_channel, 1)

    def forward(self, x):
        q = self.l1(x).squeeze()
        return q


# 分数预测回归器
class RegressionFCNet2(nn.Module):
    """
    Target network for quality prediction.
    """

    def __init__(self, in_channel=512):
        super(RegressionFCNet2, self).__init__()
        self.target_in_size = in_channel
        self.target_fc1_size = 256

        self.sigmoid = nn.Sigmoid()
        self.l1 = nn.Linear(self.target_in_size, self.target_fc1_size)
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(self.target_fc1_size)

        self.l2 = nn.Linear(self.target_fc1_size, 1)

    def forward(self, x):
        q = self.l1(x)
        q = self.relu1(q)
        q = self.drop1(q)
        q = self.bn1(q)
        q = self.l2(q).squeeze()
        return q
