import copy

import scipy.io as sio
from processing_library import *
from sklearn.metrics import accuracy_score, recall_score, cohen_kappa_score
from CNN_new import HybridSN
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class Trainer:
    def __init__(self, ):

        self.class_num = 16

        # 用于测试样本的比例
        self.test_ratio = 0.90
        # 每个像素周围提取 patch 的尺寸
        self.patch_size = 25
        # 使用 PCA 降维，得到主成分的数量
        self.pca_components = 30

        # 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 网络放到GPU上
        self.net = HybridSN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.0005)

        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)

    def test(self):
        # selected_bands = select_band
        # selected_bands = [i for i in range(
        #     selected_bands.shape[0]) if selected_bands[i] == 1]
        # data_norm_new = np.zeros_like(data_norm)
        # for i in selected_bands:  # .tolist():
        #     data_norm_new[:, :, i] = data_norm[:, :, i]
        # data_norm = data_norm_new
        data_norm = 0
        test_loader, y_test = get_test(data_norm, self.test_ratio, self.patch_size, self.pca_components)
        # 加载预训练好的模型
        checkpoint = torch.load('./checkpoint/Indian_pines_fb_pca30.t7')
        self.net.load_state_dict(checkpoint['state'])
        count = 0
        y_pred_test = 0
        for inputs, _ in test_loader:
            inputs = inputs.to(self.device)
            outputs = self.net(inputs)
            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            if count == 0:
                y_pred_test = outputs
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, outputs))

        # 生成分类结果
        oa = accuracy_score(y_test, y_pred_test)
        per_class_acc = recall_score(y_test, y_pred_test, average=None)
        aa = np.mean(per_class_acc)
        kappa = cohen_kappa_score(y_test, y_pred_test)
        print("OA: %.4f" % oa,
              " | AA:  %.4f" % aa,
              " | Kappa:  %.4f" % kappa)
        return oa, outputs


Evaluate = Trainer()
Evaluate.test()
