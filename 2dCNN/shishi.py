import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import datetime
import time

import numpy as np

# import matplotlib.pyplot as plt
from pre_color import plot_label
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import cohen_kappa_score
from processing_library import load_data, one_hot, disorder, next_batch, pca_trans_extend
from processing_library import extend, contrary_one_hot
from processing_library import save_result

from CNN import CNN_2d

###############################################################################
# 加载数据，设置参数
data_norm, labels_ori, y_train, train_loc, y_test, test_loc, y_val, val_loc, _ = load_data('Indian_pines')  # 读取数据

y_test = np.hstack([y_test, y_val])
test_loc = np.vstack([test_loc, val_loc])

nrows, ncols, ndim = data_norm.shape
batch_size = 128
display_step = 100  # 每训练display_step个step，就显示一次
step = 1
index = batch_size
# mse_rate = 0
lr_init = 0.001  # 初始学习率【调参】
lr_decay_step = 100  # 每过lr_decay_step个step，学习率就下降为原来的0.9倍
num_classification = int(np.max(labels_ori))  # 类别数
w = 27  # 图像块大小【调参】
num_epoch = 10  # 训练循环次数
selected_bands = [10, 15, 16, 21, 30, 32, 34, 45, 47, 71, 72, 74, 82, 88, 90, 97, 101, 119, 139, 162]
data_norm = data_norm[:, :, selected_bands]
