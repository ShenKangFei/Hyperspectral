# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 18:17:28 2019extend

@author: ld
"""
import torch.optim as optim
import torch
import torch.nn as nn
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pre_color import plot_label
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import cohen_kappa_score
from processing_library import load_data, one_hot, disorder, next_batch, pca_trans_extend, contrary_one_hot
from CNN import CNN_2d

# 加载数据，设置参数
data_norm, labels_ori, y_train, train_loc, y_test, test_loc, y_val, val_loc, _ = load_data('Indian_pines')  # 读取数据

y_test = np.hstack([y_test, y_val])
test_loc = np.vstack([test_loc, val_loc])

nrows, ncols, ndim = data_norm.shape
batch_size = 128
display_step = 100  # 每训练display_step个step，就显示一次
step = 1
index = batch_size
lr_init = 0.001  # 初始学习率【调参】
lr_decay_step = 100  # 每过lr_decay_step个step，学习率就下降为原来的0.9倍
num_classification = int(np.max(labels_ori))  # 类别数
w = 27  # 图像块大小【调参】
num_epoch = 220  # 训练循环次数
selected_bands = []
for i in range(200):
    selected_bands.append(i)

data_norm = data_norm[:, :, selected_bands]
pca_dim_out = 5  # PCA输出维数【调参】
data_norm = pca_trans_extend(data_norm, pca_dim_out, w)  # 对数据做PCA和补零操作

###############################################################################
# 打乱数据
Y_train = one_hot(y_train, num_classification)  # 在tensorflow中有用，pytorch中不需要
Y_test = one_hot(y_test, num_classification)
Y_val = one_hot(y_val, num_classification)

train_loc, Y_train = disorder(train_loc, Y_train)
test_loc, Y_test = disorder(test_loc, Y_test)
val_loc, Y_val = disorder(val_loc, Y_val)

# 把数据预先放进显存中，提升速度
data_norm = torch.tensor(data_norm, dtype=torch.float32).cuda()
train_loc = torch.tensor(train_loc, dtype=torch.int32).cuda()  # 形如（n,x,y）的array,其中为样本数，x,y为样本在原图中的坐标
Y_train = torch.tensor(Y_train, dtype=torch.long).cuda()


##############################################################################
def get_oa(data, X_valid_loc, Y_valid):
    """
    用于Evaluate
    参数：
        data：经过归一化，PCA，扩展边界之后的数据，（N，w，w，C）的形式
        X_valid_loc：验证数据的位置 ，（N，2）
        Y_valid： 验证数据的类标，（N,）
    """
    size = np.shape(X_valid_loc)
    num = size[0]
    index_all = 0
    step_ = 256
    y_pred = []
    while index_all < num:
        if index_all + step_ > num:
            input_loc = X_valid_loc[index_all:, :]
        else:
            input_loc = X_valid_loc[index_all:(index_all + step_), :]
        input = _windowFeature_torch(data, input_loc, w).permute(
            0, 3, 1, 2)
        index_all += step_
        output = model(input)
        temp1 = output
        y_pred1 = contrary_one_hot(temp1.cpu()).astype('int32')
        y_pred.extend(y_pred1)
    y = contrary_one_hot(Y_valid).astype('int32')
    return y_pred, y


def _contrary_one_hot_torch(label):
    """
    将onehot形式的标签转化为数字形式
    这部分在显卡中完成
    参数：
        label：（N，）
    """
    label_ori = torch.zeros([label.shape[0], ], dtype=torch.long).cuda()
    for i in range(label.shape[0]):
        label_ori[i] = torch.argmax(label[i])  # argmax就是得到最大值的序号索引
    return label_ori


def _windowFeature_torch(data_extend, loc, w):  # 不是很懂
    """
    输入扩展后的数据，根据所给窗大小按patch取出数据
    参数：
        data_extend（N，H+w，W+w，C）
    输出：
        （N，w，w，C）
    """
    new_data = torch.zeros(
        [loc.shape[0], w, w, data_extend.shape[2]]).cuda()
    for i in range(loc.shape[0]):
        x1 = loc[i, 0]  # 这里没错！做完扩展以后坐标实际上已经发生了偏移
        y1 = loc[i, 1]
        x2 = loc[i, 0] + w
        y2 = loc[i, 1] + w
        c = data_extend[x1:x2, y1:y2, :]
        new_data[i, :, :, :] = c
    return new_data


model = CNN_2d(input_bands=5, num_classification=16).cuda()
criterion = nn.CrossEntropyLoss()  # 定义损失函数
optimizer = optim.RMSprop(model.parameters(), lr=lr_init, momentum=0.8)  # 定义优化器【调参】

# training
torch.cuda.set_device(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)
model.to(device)
# model = torch.nn.DataParallel(model, [0,1]) #实现多显卡计算，如果不需要可以注释掉

epoch = 0
step = 1
start = time.perf_counter()
running_loss = 0.0
index = batch_size

# 下面这部分代码用于加载训练好的模型
print('===> Try resume from checkpoint')
if os.path.isdir('checkpoint'):
    try:
        checkpoint = torch.load('./checkpoint/Indian_pines.t7')
        #        CNN_2d.load_state_dict(checkpoint['state'])        # 从字典中依次读取
        start_epoch = checkpoint['epoch']
        print('===> Load last checkpoint data')
    except FileNotFoundError:
        print('Can\'t found net.t7')
else:
    start_epoch = 0
    print('===> Start from scratch')
print("Start training...")
#
train_start = time.perf_counter()
# 下面开始训练

while epoch < num_epoch:
    with torch.no_grad():
        batch_train_loc, batch_y_one_hot = next_batch(
            train_loc, Y_train, index, batch_size)  # 先取一个batch的数据
        batch_x = _windowFeature_torch(data_norm, batch_train_loc, w).permute(
            0, 3, 1, 2)  # pytorch默认的数据格式是（N，C，H，W），也就是（batchsize，通道数，长，宽）的形式，这里做一下转置，把C换到前面
        batch_y = _contrary_one_hot_torch(
            batch_y_one_hot)
    optimizer.zero_grad()  # reset一下optimizer，不然它会把以前的梯度和现在的梯度相加
    output = model(batch_x)  # 前向推导
    loss = criterion(output, batch_y)  # 算一下loss
    print(loss)
    loss.backward()  # 反向更新梯度
    optimizer.step()  # 更新分类层权值
    running_loss += loss.item()
    if step % display_step == 0:  # print every display_step mini-batches
        print('[%d, %5d] loss: %.7f' %
              (epoch + 1, step + 1, running_loss / display_step),
              " | learning_rate: %.5f" % (optimizer.param_groups[0]["lr"]))

        with torch.no_grad():
            y_pr, y_real = get_oa(data_norm, val_loc, Y_val)
            oa = accuracy_score(y_real, y_pr)
            per_class_acc = recall_score(y_real, y_pr, average=None)
            aa = np.mean(per_class_acc)
            kappa = cohen_kappa_score(y_real, y_pr)
            now = time.perf_counter()
            # print(per_class_acc)
            print("OA: %.4f" % oa,
                  " | AA:  %.4f" % aa,
                  " | Kappa:  %.4f" % kappa,
                  " | Time: %.1fs" % (now - start))
        running_loss = 0.0
        start = now

    if step % lr_decay_step == 0:
        # 学习率衰减
        for p in optimizer.param_groups:
            p['lr'] *= 0.9  # 【调参】

    index = index + batch_size
    step += 1

    if index > train_loc.shape[0]:
        index = batch_size
        epoch = epoch + 1

        # ReShuffle the data
        perm = np.arange(len(train_loc))
        np.random.shuffle(perm)
        train_loc = train_loc[perm]
        Y_train = Y_train[perm]
        # Start next epoch
time_train_end = time.perf_counter()
print('Finished Training')

train_end = time.perf_counter()
print("training time: %.1fs" % (train_end - train_start))
# 保存模型
state = {
    'state': model.state_dict(),
    'epoch': epoch  # 将epoch一并保存
}
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
    torch.save(state, './checkpoint/Indian_pines.t7')
print('model saved')

##############################################################################
test_start = time.perf_counter()
with torch.no_grad():
    y_pr, y_real = get_oa(data_norm, test_loc, Y_test)
    oa = accuracy_score(y_real, y_pr)
    per_class_acc = recall_score(y_real, y_pr, average=None)
    aa = np.mean(per_class_acc)
    kappa = cohen_kappa_score(y_real, y_pr)
    time_test_end = time.perf_counter()
    print(per_class_acc)
    print(oa, aa, kappa)
    test_end = time.perf_counter()
    #    #保存一下结果
    #    with open("result.txt", "a") as f:
    #        t = datetime.datetime.now()
    #        f.write("\n 当前时间为%s："%t+"\n" )
    #        f.write("score:\n"+"lr_init:"+str(lr_init)+"w:"+str(w)+"num_epoch:"+str(num_epoch)+"batch_size:"+str(batch_size)+"pca_dim_out:"+str(pca_dim_out)+"\n OA: "+str(oa)+" AA"+str(aa)+" kappa:"+str(kappa)+"test_time:"+str(test_end - test_start)+"\n per_class_acc:"+str(per_class_acc))
    #        f.write("\n\n")
    # plot
    data_name = "Indian_pines"
    plot_loc = np.array([[i, j] for i in range(nrows) for j in range(ncols)])
    print(plot_loc.shape)
    order = np.random.permutation(range(plot_loc.shape[0]))
    plot_loc = plot_loc[order]

    plot_labela = np.zeros([plot_loc.shape[0]])
    y_plot, _ = get_oa(data_norm, plot_loc, plot_labela)
    plot = np.zeros([nrows, ncols])
    for idx, item in enumerate(order.tolist()):
        plot[item // ncols, item % ncols] = y_plot[idx]
    #            y_plot = np.array(y_plot).reshape([nrows,ncols])
    # 去除多余背景操作
    for i in range(labels_ori.shape[0]):
        for j in range(labels_ori.shape[1]):
            if labels_ori[i, j] == 0:
                if data_name != "Houston":
                    plot[i, j] = 0
            else:
                plot[i, j] += 1

    plot_model = plot_label(data_name)
    img = plot_model.plot_color(plot)
    plt.imsave(data_name + "_" + str(oa) + ".png", img)
