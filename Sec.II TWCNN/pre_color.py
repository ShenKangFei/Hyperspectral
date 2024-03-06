import numpy as np
import scipy.io as sio
import os
import copy
import matplotlib.pyplot as plt
from sklearn import preprocessing

'''
exmple

plot_model = plot_label('Indian_pines')

temp1=y_.eval(feed_dict={x: data_all})
y_pred=contrary_one_hot(temp1).astype('int32')

img = plot_model.plot_color(y_pred)

plt.imsave(img_name,img)
'''

class plot_label(object):
    def __init__(self,data_name='Indian_pines'):
        self.data_name = data_name
        self.set_default()

    def set_default(self):
        self.color = np.zeros([18, 3])
        if self.data_name == 'Indian_pines':
            self.dim = 17
            self.shape = [145, 145, 3]
            self.color[0, :]=[0,         0,    0.5156]
            self.color[1, :]=[0,         0,    0.7656]
            self.color[2, :]=[0,    0.0156,    1.0000]
            self.color[3, :]=[0,    0.2656,    1.0000]
            self.color[4, :]=[0,    0.5156,    1.0000]
            self.color[5, :]=[0,    0.7656,    1.0000]
            self.color[6, :]=[0.0156 ,   1.0000,    0.9844]
            self.color[7, :]=[0.2656,    1.0000,    0.7344]
            self.color[8, :]=[0.5156,    1.0000,    0.4844]
            self.color[9, :]=[0.7656,    1.0000,    0.2344]
            self.color[10, :]=[1.0000,    0.9844,         0]
            self.color[11, :]=[1.0000,    0.7344,         0]
            self.color[12, :]=[1.0000,    0.4844,         0]
            self.color[13, :]=[1.0000,    0.2344,         0]
            self.color[14, :]=[0.5451,        0.2706,        0.0745]
            self.color[15, :]=[0.7344,         0,         0]
            self.color[16, :]=[0.5,         0,         0]
        elif self.data_name == 'PaviaU':
            self.dim = 10
            self.shape = [610, 340 ,3]
            self.color[0, :]=[0, 0, 0]
            self.color[1, :]=[1, 1, 1]
            self.color[2, :]=[0, 1, 0]
            self.color[3, :]=[0, 1, 1]
            self.color[4, :]=[0, 0, 1]
            self.color[5, :]=[1, 0, 1]
            self.color[6, :]=[0.7412, 0.4196, 0.0353]
            self.color[7, :]=[0.3647, 0.0471, 0.4824]
            self.color[8, :]=[1, 0, 0]
            self.color[9, :]=[1, 1, 0]
        elif self.data_name == 'Houston':
            self.dim = 16
            self.shape = [1905, 349, 3]
            # self.color[0, :]=[0,         0,    0.5156]
            # self.color[1, :]=[1, 1, 1]
            # self.color[2, :]=[0,    1,    0]
            # self.color[3, :]=[0,    1,         1]
            # self.color[4, :]=[1,  1,        0]
            # self.color[5, :]=[0.1333,  0.5451, 0.1333]
            # self.color[6, :]=[1, 0, 1]
            # self.color[7, :]=[1,  0.4980,  0.3137]
            self.color[0, :]=[0,         0,    0.5156]
            self.color[1, :]=[0,         0,    0.7656]
            self.color[2, :]=[0,    0.0156,    1.0000]
            self.color[3, :]=[0,    0.2656,    1.0000]
            self.color[4, :]=[0,    0.5156,    1.0000]
            self.color[5, :]=[0,    0.7656,    1.0000]
            self.color[6, :]=[0.0156 ,   1.0000,    0.9844]
            self.color[7, :]=[0.2656,    1.0000,    0.7344]
            self.color[8, :]=[0.5156,    1.0000,    0.4844]
            self.color[9, :]=[0.7656,    1.0000,    0.2344]
            self.color[10, :]=[1.0000,    0.9844,         0]
            self.color[11, :]=[1.0000,    0.7344,         0]
            self.color[12, :]=[1.0000,    0.4844,         0]
            self.color[13, :]=[1.0000,    0.2344,         0]
            self.color[14, :]=[0.5451,        0.2706,        0.0745]
            self.color[15, :]=[0.7344,         0,         0]
            self.color[16, :]=[0.5,         0,         0]
        elif self.data_name == 'Salinas':
            self.dim = 16
            self.shape = [512, 217, 3]
            # self.color[0, :]=[0,         0,    0.5156]
            # self.color[1, :]=[1, 1, 1]
            # self.color[2, :]=[0,    1,    0]
            # self.color[3, :]=[0,    1,         1]
            # self.color[4, :]=[1,  1,        0]
            # self.color[5, :]=[0.1333,  0.5451, 0.1333]
            # self.color[6, :]=[1, 0, 1]
            # self.color[7, :]=[1,  0.4980,  0.3137]
            self.color[0, :] = [0, 0, 0.5156]
            self.color[1, :] = [0, 0, 0.7656]
            self.color[2, :] = [0, 0.0156, 1.0000]
            self.color[3, :] = [0, 0.2656, 1.0000]
            self.color[4, :] = [0, 0.5156, 1.0000]
            self.color[5, :] = [0, 0.7656, 1.0000]
            self.color[6, :] = [0.0156, 1.0000, 0.9844]
            self.color[7, :] = [0.2656, 1.0000, 0.7344]
            self.color[8, :] = [0.5156, 1.0000, 0.4844]
            self.color[9, :] = [0.7656, 1.0000, 0.2344]
            self.color[10, :] = [1.0000, 0.9844, 0]
            self.color[11, :] = [1.0000, 0.7344, 0]
            self.color[12, :] = [1.0000, 0.4844, 0]
            self.color[13, :] = [1.0000, 0.2344, 0]
            self.color[14, :] = [0.5451, 0.2706, 0.0745]
            self.color[15, :] = [0.7344, 0, 0]
            self.color[16, :] = [0.5, 0, 0]
        # self.color = np.array(self.color*255,dtype=np.uint8)


    def change_data(self,pre,img,d):
        c = np.argwhere(pre == d)
        for i in range(len(c)):
            img[c[i, 0], c[i, 1], :] = self.color[d, :]
        return img


    def plot_color(self,pre):
        pre=np.reshape(pre,[self.shape[0],self.shape[1]])
        img=np.zeros(self.shape)
        for i in range(self.dim):
            img=self.change_data(pre,img,i)
        return img

def normalizeData(data):
    ''' 原始数据归一化处理（每条） '''
    data_norm = np.zeros(np.shape(data))
    for i in range(np.shape(data)[0]):
        for j in range(np.shape(data)[1]):
            data_norm[i,j,:] = preprocessing.normalize(data[i,j,:].reshape(1,-1))[0]
    return data_norm

data_name = 'Salinas'
path = os.getcwd()+'/data/'+data_name
data = sio.loadmat(path+'/Salinas_corrected.mat')['salinas_corrected']
data = np.array(data).astype(float)
data_norm = normalizeData(data)
data_norm = data_norm[:,:,[69,27,11]]

label = sio.loadmat(path+'/Salinas_gt.mat')['salinas_gt']
label = np.array(label).astype(float)

#训练样本的位置

# c = int(label.max())
# for i in range(1, c+1):
#     loc1, loc2 = np.where(label == i)
#     num = len(loc1)
#     order = np.random.permutation(range(num))
#     loc1 = loc1[order]
#     loc2 = loc2[order]
#     num1 = int(np.round(num*p))
#     x_loc1=(loc1[:num1])
#     x_loc2=(loc2[:num1])
#     y_loc1=(loc1[num1:])
#     y_loc2=(loc2[num1:])
#     label_train[y_loc1, y_loc2]=0
#     label_test[x_loc1, x_loc2]=0


plot_model = plot_label(data_name)
img = plot_model.plot_color(label)
# img_test = plot_model.plot_color(label_test)
# img_train = plot_model.plot_color(label_train)


plt.imshow(img)
plt.show()
# plt.imshow(img_train)
# plt.show()


