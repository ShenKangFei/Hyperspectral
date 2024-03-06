from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import cohen_kappa_score
from processing_library import load_data, one_hot, disorder, next_batch, next_batch_unl
from processing_library import save_result, random_cut_bands
import copy
from processing_library import index_band_selection
from sklearn.svm import SVC
import numpy as np
from pre_color import plot_label
import matplotlib.pyplot as plt
import dit

###############################################################################
# data_norm,labels_ori,x_train,y_train,train_loc,x_test,y_test,test_loc=load_data('Indian_pines')

data_name = "PaviaU"

data_norm, labels_ori, y_train, x_train, train_loc, y_test, x_test, test_loc, y_val, x_val, val_loc, _ = load_data(
    data_name)
# ulab_loc = val_loc  # Caution！
nrows, ncols, ndim = data_norm.shape
x = data_norm.reshape([nrows * ncols, ndim])
x = np.transpose(x)
hx = np.zeros([ndim, ])
for i in range(x.shape[0]):
    hist = np.histogram(x[i], bins=255)[0]
    hist = hist / np.sum(hist)
    dist = dit.Distribution.from_ndarray(hist)
    hx[i] = dit.shannon.entropy(dist)

bands = {
    "mRMR": [1, 2, 4, 5, 9, 10, 13, 14, 15, 22, 24, 26, 32, 35, 37, 41, 45, 46, 49, 53, 65, 66, 67, 74, 83, 87, 89, 100,
             101, 102],
    "TWCNN": [3, 8, 9, 22, 24, 25, 31, 37, 41, 48, 65, 66, 72, 73, 77, 83, 84, 87, 99, 100],
    "MR-SVM": [1, 2, 3, 4, 5, 7, 9, 38, 37, 45, 39, 36, 34, 41, 42, 46, 55, 63, 69, 73, 78, 83, 85, 88, 91, 94, 97, 100,
               102, 103],
    "SICNN": [1, 3, 5, 7, 10, 12, 15, 24, 26, 27, 30, 31, 32, 37, 38, 42, 47, 61, 62, 65, 67, 71, 73, 75, 81, 84, 85,
              98, 99, 100],
    "DDCNN": [1, 2, 3, 4, 10, 11, 14, 19, 20, 23, 27, 28, 31, 35, 37, 40, 42, 43, 39, 55, 58, 59, 62, 72, 73, 76, 82,
              96, 98, 102],
    "ABCNN": [0, 2, 3, 4, 5, 6, 71, 75, 78, 79, 80, 83, 85, 87, 90, 92, 93, 95, 97, 98],
    "BS-Nets": [33, 32, 35, 36, 37, 38, 40, 41, 60, 61, 62, 63, 64, 67, 68, 69, 70, 100, 101, 102],
    "DRLBS": [1, 6, 10, 20, 39, 45, 55, 56, 65, 82, 83, 84, 75, 76, 58, 59, 67, 68, 91, 92],
    "MH-DRL": [5, 8, 11, 25, 27, 32, 34, 44, 46, 48, 51, 61, 74, 79, 83, 85, 90, 92, 93, 98]
    }
# x = np.array(list(range(200)))
# plt.plot(x,hx)
# plt.xlabel("bands")
# plt.ylabel("entropy")
# plt.rcParams['figure.figsize'] = (8, 4.0) 
# plt.savefig("Indian_entropy.tiff")
# plt.show()

band_num = 20

fontsize = 18
import random

bands["mRMR"] = random.sample(bands["mRMR"], band_num)
bands["MR-SVM"] = random.sample(bands["MR-SVM"], band_num)

bands["SICNN"] = random.sample(bands["SICNN"], band_num)
bands["DDCNN"] = random.sample(bands["DDCNN"], band_num)
for i in bands.keys():
    bands[i] = bands[i][:band_num]
bands_data = np.array(list(bands.values()))
plot = np.zeros([bands_data.shape[0] * bands_data.shape[1], 2])
for i in range(bands_data.shape[0]):
    for j in range(bands_data.shape[1]):
        plot[i * bands_data.shape[1] + j] = [bands_data[i, j], i + 1]
plot = plot.reshape([len(bands.keys()), band_num, -1])
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(14, 8))
x = np.array(list(range(103)))
ax[1].plot(x, hx)
ax[1].set_xlabel("Spectral bands", fontsize=fontsize)
ax[1].set_ylabel("Entropy value", fontsize=fontsize)
ax[1].grid(axis="x")
for i in range(plot.shape[0]):
    ax[0].scatter(plot[i, :, 0] + 1, plot[i, :, 1], marker="o")
    ax[0].plot([min(plot[i, :, 0]) + 1, max(plot[i, :, 0]) + 1], [min(plot[i, :, 1]), max(plot[i, :, 1])])
ax[0].set_yticks(list(range(1, len(bands.keys()) + 1)))
ax[0].set_yticklabels(list(bands.keys()), fontsize=fontsize)
ax[0].grid(axis="x")
# ax[0].grid(axis = "y")
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.show()
fig.savefig(data_name + ".tif", format='tif', pad_inches=0, bbox_inches='tight')
