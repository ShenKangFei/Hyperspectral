# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 20:38:39 2018

@author: Jiantong Chen
"""

import copy
import os

import numpy as np
import scipy.io as sio

import tensorflow as tf
import matplotlib.pyplot as plt

###############################################################################
def load_data(data_name):
    '''读取数据'''
    path = os.getcwd()
    pre = sio.loadmat(path + '/data/' + data_name + '/' + data_name + '_pre.mat')
    
    data_norm = pre['data_norm']
    labels_ori = pre['labels_ori']
    x_train = pre['train_x']
    y_train = pre['train_y'][0]
    train_loc = pre['train_loc']
    x_test = pre['test_x']
    y_test = pre['test_y'][0]
    test_loc = pre['test_loc']
    
    return data_norm,labels_ori,x_train,y_train,train_loc,x_test,y_test,test_loc
###############################################################################
def windowFeature(data, loc, w ):
    '''从扩展矩阵中得到窗口特征'''
    size = np.shape(data)
    print(size)
    data_expand = np.zeros((int(size[0]+w-1),int(size[1]+w-1),size[2]))
    newdata = np.zeros((len(loc[0]), w, w,size[2]))
    for j in range(size[2]):    
        data_expand[:,:,j] = np.lib.pad(data[:,:,j], ((int(w / 2), int(w / 2)), (int(w / 2),int(w / 2))), 'symmetric')
        newdata[:,:,:,j] = np.zeros((len(loc[0]), w, w))
        for i in range(len(loc[0])):
            loc1 = loc[0][i]
            loc2 = loc[1][i]
            f = data_expand[loc1:loc1 + w, loc2:loc2 + w,j]
            newdata[i, :, :,j] = f
    return newdata
###############################################################################
def one_hot(lable,class_number):
    '''转变标签形式'''
    one_hot_array = np.zeros([len(lable),class_number])
    for i in range(len(lable)):
        one_hot_array[i,int(lable[i]-1)] = 1
    return one_hot_array
###############################################################################
def disorder(X,Y):
    '''打乱顺序'''
    index_train = np.arange(X.shape[0])
    np.random.shuffle(index_train)
    X = X[index_train, :]
    Y = Y[index_train, :]
    return X,Y
###############################################################################
def next_batch(image,lable,index,batch_size):
    '''数据分批'''
    start = index-batch_size
    end = index
    return image[start:end,:,:,:],lable[start:end]
###############################################################################
def first_layer(x,W,B,stride):
    '''第一层卷积操作'''
    x=tf.nn.depthwise_conv2d(x,W,stride,padding='VALID',name='CONV')
    h = tf.nn.bias_add(x,B)
    return h
###############################################################################
def conv_layer_same(x,W,B,stride):
    '''不改变特征图尺寸的卷积'''
    x = tf.nn.conv2d(x,W,stride,padding='SAME',name='CONV')
    h = tf.nn.bias_add(x,B)
    bn = tf.contrib.layers.batch_norm(h,decay=0.9,epsilon=1e-5,scale=True,is_training=True)
    convout = tf.nn.relu(bn)
    return convout
###############################################################################
def conv_layer_valid(x,W,B,stride):
    '''改变特征图尺寸的卷积'''
    x = tf.nn.conv2d(x,W,stride,padding='VALID',name='CONV')
    h = tf.nn.bias_add(x,B)
    bn = tf.contrib.layers.batch_norm(h,decay=0.9,epsilon=1e-5,scale=True,is_training=True)
    convout = tf.nn.relu(bn)
    return convout
###############################################################################
def conv1d_layer_valid(x,W,B,stride):
    '''改变特征图尺寸的卷积'''
    x = tf.nn.conv1d(x,W,stride, padding='VALID',name='CONV')
    h = tf.nn.bias_add(x,B)
    bn = tf.contrib.layers.batch_norm(h,decay=0.9,epsilon=1e-5,scale=True,is_training=True)
    convout = tf.nn.relu(bn)
    return convout
###############################################################################
def contrary_one_hot(label):
    '''将onehot标签转化为真实标签'''
    size=len(label)
    label_ori=np.empty(size)
    for i in range(size):
        label_ori[i]=np.argmax(label[i])+1
    return label_ori
###############################################################################
#def order_weight_fixed(w,num_band_seclection):
#    '''选择权重值降序后前num_band_seclection个权重值'''
#    a=w.eval()
#    b=abs(copy.deepcopy(a))
#    b.sort(axis=2)
#    c=np.where(abs(a)>b[0,0,-(num_band_seclection+1),0],a,0)
#    return c


def order_weight_fixed(w,num_band_seclection):
    '''选择权重值降序后前num_band_seclection个权重值'''
    a=w.eval()    
#    a=np.array(list(map(binaryzation,a)))
#    b=abs(copy.deepcopy(a))  
#    b.sort(axis=2)
    a=a.flatten().tolist()
    c=a
#    th=b[0,0,-(num_band_seclection+1),0]
    for i in range(len(a)):
        c[i]=ternaryzation(a[i],0.5)
#    c=np.where(abs(a)>b[0,0,-(num_band_seclection+1),0],a,0)
    c=np.array(c)
    c=c.reshape(1,1,-1,1)
    return c
def ternaryzation(x,th):
    if x>th:
        return 1
    elif x<-th:
        return -1
    else:
        return 0
###############################################################################
def index_band_selection(w):
    '''找到所选波段位置'''
    c=np.where(w!=0)[2].tolist()
    return c
###############################################################################
def save_result(data_name,oa,aa,kappa,num_band_seclection_now,band_loction,per_class_acc,train_time,test_time):
    '''将实验结果保存在txt文件中'''
    write_content='\n'+data_name+'\n'+'oa:'+str(oa)+' aa:'+str(aa)+' kappa:'+str(kappa)+'\n'+'num_band_seclection:'+str(num_band_seclection_now)+'\n'+'band_loction:'+str(band_loction)+'\n'+'per_class_acc:'+str(per_class_acc)+'\n'+'train_time:'+str(train_time)+' test_time:'+str(test_time)+'\n'
    f = open(os.getcwd()+'/实验结果.txt','a')
    f.writelines(write_content)
    f.close()
    return       
###############################################################################
def DrawResult(labels,imageID):
    #ID=1:Pavia University
    #ID=2:Indian Pines
    #ID=6:KSC
    num_class = labels.max()
    if imageID == 1:
        row = 610
        col = 340
        palette = np.array([[216,191,216],
                            [0,255,0],
                            [0,255,255],
                            [45,138,86],
                            [255,0,255],
                            [255,165,0],
                            [159,31,239],
                            [255,0,0],
                            [255,255,0]])
        palette = palette*1.0/255
    elif imageID == 2:
        row = 145
        col = 145
        palette = np.array([[255,0,0],
                            [0,255,0],
                            [0,0,255],
                            [255,255,0],
                            [0,255,255],
                            [255,0,255],
                            [176,48,96],
                            [46,139,87],
                            [160,32,240],
                            [255,127,80],
                            [127,255,212],
                            [218,112,214],
                            [160,82,45],
                            [127,255,0],
                            [216,191,216],
                            [238,0,0]])
        palette = palette*1.0/255
    elif imageID == 6:
        row = 512
        col = 614
        palette = np.array([[94, 203, 55],
                            [255, 0, 255],
                            [217, 115, 0],
                            [179, 30, 0],
                            [0, 52, 0],
                            [72, 0, 0],
                            [255, 255, 255],
                            [145, 132, 135],
                            [255, 255, 172],
                            [255, 197, 80],
                            [60, 201, 255],
                            [11, 63, 124],
                            [0, 0, 255]])
        palette = palette*1.0/255
    
    X_result = np.zeros((labels.shape[0],3))
    for i in range(0,num_class):
        X_result[np.where(labels==i),0] = palette[i,0]
        X_result[np.where(labels==i),1] = palette[i,1]
        X_result[np.where(labels==i),2] = palette[i,2]
    
    X_result = np.reshape(X_result,(row,col,3))
    plt.axis ( "off" ) 
    plt.imshow(X_result)    
    return X_result
    
def CalAccuracy(predict,label):
    n = label.shape[0]
    OA = np.sum(predict==label)*1.0/n
    correct_sum = np.zeros((max(label)+1))
    reali = np.zeros((max(label)+1))
    predicti = np.zeros((max(label)+1))
    producerA = np.zeros((max(label)+1))
    
    for i in range(0,max(label)+1):
        correct_sum[i] = np.sum(label[np.where(predict==i)]==i)
        reali[i] = np.sum(label==i)
        predicti[i] = np.sum(predict==i)
        producerA[i] = correct_sum[i] / reali[i]
   
    Kappa = (n*np.sum(correct_sum) - np.sum(reali * predicti)) *1.0/ (n*n - np.sum(reali * predicti))
    return OA,Kappa,producerA
   