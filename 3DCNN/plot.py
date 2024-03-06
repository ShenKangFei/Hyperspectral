import scipy.io as sio
from processing_library import *
from sklearn.metrics import accuracy_score, recall_score, cohen_kappa_score
from CNN import HybridSN
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import os
from pre_color import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class_num = 16
X = sio.loadmat('data/Indian_pines_corrected.mat')['indian_pines_corrected']
y = sio.loadmat('data/Indian_pines_gt.mat')['indian_pines_gt']
# 每个像素周围提取 patch 的尺寸
patch_size = 25
# 使用 PCA 降维，得到主成分的数量
pca_components = 30

X_pca = applyPCA(X, numComponents=pca_components)

X_pca, y = createImageCubes(X_pca, y, windowSize=patch_size)

# 改变 X_train, Y_train 的形状，以符合 keras 的要求

X_test = X_pca.reshape(-1, patch_size, patch_size, pca_components, 1)

# 为了适应 pytorch 结构，数据要做 transpose
X_test = X_test.transpose(0, 4, 3, 1, 2)


# 创建 train_loader 和 test_loader

test_set = TestDS(X_test, y)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=128, shuffle=False)

# 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 网络放到GPU上
net = HybridSN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)
# 初始学习率【0.0005】，epoch【100】，精度98%

try:
    checkpoint = torch.load('./checkpoint/Indian_pines_fb_pca30.t7')
    # 从字典中依次读取
    start_epoch = checkpoint['epoch']
    net.load_state_dict(checkpoint['state'])
    print('===> Load last checkpoint data')
except FileNotFoundError:
    print('Can\'t found net.t7')

count = 0
y_pred_test = 0
for inputs, _ in test_loader:
    inputs = inputs.to(device)
    outputs, feature = net(inputs)
    outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
    if count == 0:
        y_pred_test = outputs
        count = 1
    else:
        y_pred_test = np.concatenate((y_pred_test, outputs))
plot_max = plot(X, y_pred_test)
data_list = ["Indian_pines"]
plot_model = plot_label(data_list[0])
img = plot_model.plot_color(plot_max)
oa = accuracy_score(y, y_pred_test)
plt.imsave(data_list[0] + "_" + str(oa) + ".tiff", img)
print("画图结束！")
