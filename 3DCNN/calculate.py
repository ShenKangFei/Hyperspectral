import scipy.io as sio
from processing_library import *
from sklearn.metrics import accuracy_score, recall_score, cohen_kappa_score
from CNN import HybridSN
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class_num = 16

X = sio.loadmat('data/PaviaU.mat')['paviaU']
y = sio.loadmat('data/PaviaU_gt.mat')['paviaU_gt']

# action_list = np.random.randint(2, size=200)
# X, idx = select(X, action_list)
# 用于测试样本的比例
test_ratio = 0.90
# 每个像素周围提取 patch 的尺寸
patch_size = 25
# 使用 PCA 降维，得到主成分的数量
pca_components = 30


X_pca = applyPCA(X, numComponents=pca_components)
X_pca, y = createImageCubes(X_pca, y, windowSize=patch_size)
X_train, X_test, y_train, y_test = splitTrainTestSet(X_pca, y, test_ratio, randomState=345)
# 改变 X_train, Y_train 的形状，以符合 keras 的要求
X_train = X_train.reshape(-1, patch_size, patch_size, pca_components, 1)
X_test = X_test.reshape(-1, patch_size, patch_size, pca_components, 1)
# 为了适应 pytorch 结构，数据要做 transpose
X_train = X_train.transpose(0, 4, 3, 1, 2)
X_test = X_test.transpose(0, 4, 3, 1, 2)
# 创建 train_loader 和 test_loader
train_set = TrainDS(X_train, y_train)
test_set = TestDS(X_test, y_test)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=128, shuffle=False)
# 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 网络放到GPU上
net = HybridSN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)
# 初始学习率【0.0005】，epoch【100】，精度98%
try:
    checkpoint = torch.load('./checkpoint/PaviaU.t7')
    # 从字典中依次读取
    start_epoch = checkpoint['epoch']
    net.load_state_dict(checkpoint['state'])
    print('===> Load last checkpoint data')
except FileNotFoundError:
    print('Can\'t found net.t7')
OA = []
AA = []
KAPPA = []
for i in range(30):
    count = 0
    y_pred_test = 0
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))

    # 生成分类报告
    oa = accuracy_score(y_test, y_pred_test)
    per_class_acc = recall_score(y_test, y_pred_test, average=None)
    if i == 0:
        per_class = per_class_acc
    else:
        per_class = np.vstack((per_class, per_class_acc))
    aa = np.mean(per_class_acc)
    kappa = cohen_kappa_score(y_test, y_pred_test)
    OA.append(oa)
    AA.append(aa)
    KAPPA.append(kappa)
c = 1
