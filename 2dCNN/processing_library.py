# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 20:38:39 2018

@author: Jiantong Chen
"""
import os
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA


def load_data(data_name):
    """读取数据"""
    path = os.getcwd()
    pre = sio.loadmat(path + '/data/' + data_name +
                      '/' + data_name + '_pre.mat')

    data_norm = pre['data_norm']
    labels_ori = pre['labels_ori']
    x_train = pre['train_x']
    y_train = pre['train_y'][0]
    train_loc = pre['train_loc']
    x_test = pre['test_x']
    y_test = pre['test_y'][0]
    test_loc = pre['test_loc']
    y_val = pre["val_y"][0]
    val_loc = pre["val_loc"]
    ulab_loc = pre['Ulab_loc']

    return data_norm, labels_ori, y_train, train_loc, y_test, test_loc, y_val, val_loc, ulab_loc


def one_hot(lable, class_number):
    """转变标签形式"""
    one_hot_array = np.zeros([len(lable), class_number])
    for i in range(len(lable)):
        one_hot_array[i, int(lable[i]-1)] = 1
    return one_hot_array


def disorder(X, Y):
    """打乱顺序"""
    index_train = np.arange(X.shape[0])
    np.random.shuffle(index_train)
    X = X[index_train, :]
    Y = Y[index_train, :]
    return X, Y


def next_batch(image, lable, index, batch_size):
    """数据分批"""
    start = index-batch_size
    end = index
    return image[start:end, :], lable[start:end]


def contrary_one_hot(label):
    """将onehot标签转化为真实标签"""
    size = len(label)
    label_ori = np.empty(size)
    for i in range(size):
        label_ori[i] = np.argmax(label[i])
    return label_ori


def save_result(data_name, oa, aa, kappa, num_band_seclection_now, band_loction, per_class_acc, train_time, test_time):
    """将实验结果保存在txt文件中"""
    write_content = '\n'+data_name+'\n'+'oa:'+str(oa)+' aa:'+str(aa)+' kappa:'+str(kappa)+'\n'+'num_band_seclection:'+str(num_band_seclection_now)+'\n'+'band_loction:'+str(
        band_loction)+'\n'+'per_class_acc:'+str(per_class_acc)+'\n'+'train_time:'+str(train_time)+' test_time:'+str(test_time)+'\n'
    f = open(os.getcwd()+'/实验结果.txt', 'a')
    f.write(write_content)
    f.close()
    return


def extend(data, w):
    size = data.shape
    data_extend = np.zeros((int(size[0]+w-1), int(size[1]+w-1), size[2]))
    for j in range(size[2]):
        data_extend[:, :, j] = np.lib.pad(data[:, :, j], ((
            int(w / 2), int(w / 2)), (int(w / 2), int(w / 2))), 'symmetric')
    return data_extend


def pca_trans_extend(data, n, w):
    """PCA + extend
    Args:
        data: input data, size like (W,H,b)
        n : n_components of PCA, a integer number
        w : width of patch_size, a odd number
    """
    data_reshape = data.reshape((-1, data.shape[2]))
    pca = PCA(n_components=n)
    data_pca = pca.fit_transform(data_reshape)
    data_reshape_2 = data_pca.reshape([data.shape[0], data.shape[1], -1])
    data_ex = extend(data_reshape_2, w)
    return data_ex
