import torch.nn as nn


class HybridSN(nn.Module):
    def __init__(self, num_classes=16):
        super(HybridSN, self).__init__()

        # conv1：（1, 30, 25, 25）， 8个 7x3x3 的卷积核 ==>（8, 24, 23, 23）
        self.conv1 = nn.Conv3d(1, 8, (7, 3, 3))

        # conv2：（8, 24, 23, 23）， 16个 5x3x3 的卷积核 ==>（16, 20, 21, 21）
        self.conv2 = nn.Conv3d(8, 16, (5, 3, 3))

        # conv3：（16, 20, 21, 21），32个 3x3x3 的卷积核 ==>（32, 18, 19, 19）
        self.conv3 = nn.Conv3d(16, 32, (3, 3, 3))

        # conv3_2d （576, 19, 19），64个 3x3 的卷积核 ==>（（64, 17, 17）
        self.conv3_2d = nn.Conv2d(576, 64, (3, 3))

        # 全连接层（256个节点）
        self.dense1 = nn.Linear(18496, 256)
        # 全连接层（128个节点）
        self.dense2 = nn.Linear(256, 128)
        # 最终输出层(16个节点)
        self.out = nn.Linear(128, num_classes)
        #  Dropout（0.4)
        self.drop = nn.Dropout(p=0.4)
        # 激活函数ReLU
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        # 进行二维卷积，因此把前面的 32*18 reshape 一下，得到 （576, 19, 19）
        out = out.view(-1, out.shape[1] * out.shape[2], out.shape[3], out.shape[4])
        out = self.relu(self.conv3_2d(out))
        # flatten 操作，变为 18496 维的向量，
        out = out.view(out.size(0), -1)
        out = self.dense1(out)
        out = self.drop(out)
        out = self.dense2(out)
        out = self.drop(out)
        out = self.out(out)
        return out
