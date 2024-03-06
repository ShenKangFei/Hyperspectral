import copy

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import torch
import torch.utils.data
import scipy.io as sio
import torch.utils.data
import os
import random


# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


# 在每个像素周围提取 patch ，然后创建成符合 keras 处理的格式
def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels


def random_cut_bands(data, cut_number):

    cut_idx = np.random.choice(list(range(0, 200)), size=cut_number, replace=False)
    new_data = torch.from_numpy(np.zeros((data.shape[0], 1, cut_number, 25, 25)))
    for idx, i in enumerate(cut_idx):
        new_data[:, :, idx, :, :] = data[:, :, i, :, :]
    return new_data.type(torch.FloatTensor)


def splitTrainTestSet(X, y, testRatio, randomState):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState, stratify=y)
    return X_train, X_test, y_train, y_test


def center_Loss(feature, label, num_classification):
    # 有监督的center_loss
    total_center = torch.mean(feature, dim=0, keepdim=True)
    class_center = torch.tensor([], dtype=torch.float32).cuda().view(
        [-1, feature.shape[1]])  # 类内中心

    for i in range(num_classification):
        index = np.where(label.cpu().numpy() == i)[0]
        if index.shape[0] != 0:
            result = torch.mean(feature[index, :], dim=0, keepdim=True)
        else:
            result = total_center
        class_center = torch.cat([class_center, result])
    class_center_gather = class_center[label]
    loss_wtl = torch.mean(torch.mean(torch.abs(
        feature - class_center_gather), dim=1, keepdim=True).view([feature.shape[0]]))
    return loss_wtl


def select(data, action_list):
    for i in action_list:
        if i == 1:
            data[:, :, i] = 0
    return data, i


def get_test(data, test_ratio, patch_size, pca_components):
    X = sio.loadmat('data/Indian_pines_corrected.mat')['indian_pines_corrected']
    y = sio.loadmat('data/Indian_pines_gt.mat')['indian_pines_gt']

    X_pca = applyPCA(X, numComponents=pca_components)

    X_pca, y = createImageCubes(X_pca, y, windowSize=patch_size)

    X_train, X_test, y_train, y_test = splitTrainTestSet(X_pca, y, test_ratio, randomState=345)
    X_test = X_test.reshape(-1, patch_size, patch_size, pca_components, 1)
    X_test = X_test.transpose(0, 4, 3, 1, 2)
    test_set = TestDS(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=128, shuffle=False)
    return test_loader, y_test


""" Training dataset"""


def plot(data_norm, y_pr):
    plot_max = np.zeros((data_norm.shape[0], data_norm.shape[1]))
    print("plot_max", plot_max.shape)
    # print(loc_full[0].shape)
    # print('y_pr', y_pr)
    for i in range(plot_max.shape[0]):
        for j in range(plot_max.shape[1]):
            plot_max[i, j] = y_pr[plot_max.shape[1] * i + j]
    return plot_max


class TrainDS(torch.utils.data.Dataset):
    def __init__(self, X_train, y_train):
        self.len = X_train.shape[0]
        self.x_data = torch.FloatTensor(X_train)
        self.y_data = torch.LongTensor(y_train)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


""" Testing dataset"""


class TestDS(torch.utils.data.Dataset):
    def __init__(self, X_test, y_test):
        self.len = X_test.shape[0]
        self.x_data = torch.FloatTensor(X_test)
        self.y_data = torch.LongTensor(y_test)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len
