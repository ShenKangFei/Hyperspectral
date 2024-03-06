import scipy.io as sio
from processing_library import *
from sklearn.metrics import accuracy_score, recall_score, cohen_kappa_score
from CNN import HybridSN
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class_num = 16

X = sio.loadmat('data/Indian_pines_corrected.mat')['indian_pines_corrected']
y = sio.loadmat('data/Indian_pines_gt.mat')['indian_pines_gt']

# action_list = np.random.randint(2, size=200)
# X, idx = select(X, action_list)
# 用于测试样本的比例
test_ratio = 0.99
# 每个像素周围提取 patch 的尺寸
patch_size = 25
# 使用 PCA 降维，得到主成分的数量
pca_components = 30

print('Hyper_spectral data shape: ', X.shape)
print('Label shape: ', y.shape)

print('\n... ... PCA transformation ... ...')
X_pca = applyPCA(X, numComponents=pca_components)
print('Data shape after PCA: ', X_pca.shape)

print('\n... ... create data cubes ... ...')
X_pca, y = createImageCubes(X_pca, y, windowSize=patch_size)
print('Data cube X shape: ', X_pca.shape)
print('Data cube y shape: ', y.shape)

print('\n... ... create train & test data ... ...')
X_train, X_test, y_train, y_test = splitTrainTestSet(X_pca, y, test_ratio, randomState=345)
print('X_train shape: ', X_train.shape)
print('X_test  shape: ', X_test.shape)

# 改变 X_train, Y_train 的形状，以符合 keras 的要求
X_train = X_train.reshape(-1, patch_size, patch_size, pca_components, 1)
X_test = X_test.reshape(-1, patch_size, patch_size, pca_components, 1)
print('before transpose: X_train shape: ', X_train.shape)
print('before transpose: X_test  shape: ', X_test.shape)

# 为了适应 pytorch 结构，数据要做 transpose
X_train = X_train.transpose(0, 4, 3, 1, 2)
X_test = X_test.transpose(0, 4, 3, 1, 2)
print('after transpose: X_train shape: ', X_train.shape)
print('after transpose: X_test  shape: ', X_test.shape)

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

isTrain = False
if not isTrain:
    try:
        checkpoint = torch.load('./checkpoint/Indian_pines_fb_pca30.t7')
        # 从字典中依次读取
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state'])
        print('===> Load last checkpoint data')
    except FileNotFoundError:
        print('Can\'t found net.t7')
else:
    start_epoch = 0
    print('===> Start from scratch')

    # 开始训练
    loss = 0
    total_loss = 0
    epoch = 0
    for epoch in range(100):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 优化器梯度归零
            optimizer.zero_grad()
            # 正向传播 +　反向传播 + 优化
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('[Epoch: %d]   [loss avg: %.6f]   [current loss: %.6f]' % (
            epoch + 1, total_loss / (epoch + 1), loss.item()))

    print('Finished Training')

    state = {
        'state': net.state_dict(),
        'epoch': epoch  # 将epoch一并保存
    }
    if os.path.isdir('checkpoint'):
        torch.save(state, './checkpoint/PaviaU.t7')
    print('model saved')

# 模型测试

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
aa = np.mean(per_class_acc)
kappa = cohen_kappa_score(y_test, y_pred_test)

print("OA: %.4f" % oa,
      " | AA:  %.4f" % aa,
      " | Kappa:  %.4f" % kappa)
print(per_class_acc)
