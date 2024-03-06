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

data_name = "Houston"

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
    "mRMR": [1, 2, 3, 9, 16, 17, 35, 36, 46, 53, 56, 58, 65, 74, 75, 78, 81, 84, 85, 87, 89, 102, 120, 121, 124, 125,
             126, 138, 140, 143],
    "TWCNN": [3, 8, 9, 10, 12, 18, 24, 33, 35, 43, 46, 49, 50, 85, 93, 94, 107, 127, 132, 141],
    "MR-SVM": [1, 9, 17, 35, 30, 31, 37, 41, 59, 57, 64, 68, 69, 74, 76, 84, 85, 88, 98, 101, 110, 131, 134, 137, 139,
               140, 141, 142, 143, 144],
    "SICNN": [1, 2, 3, 9, 16, 17, 35, 36, 46, 53, 56, 58, 65, 74, 75, 78, 81, 84, 85, 87, 89, 102, 110, 121, 124, 130,
              138, 140, 141, 19],
    "DDCNN": [5, 10, 13, 14, 15, 33, 39, 40, 41, 56, 59, 70, 73, 78, 80, 84, 85, 96, 108, 109, 115, 120, 121, 129, 132,
              133, 137, 139, 140, 141],
    "ABCNN": [3, 5, 6, 7, 11, 22, 29, 37, 40, 62, 64, 67, 68, 72, 76, 77, 101, 118, 121, 122],
    "BS-Nets": [5, 6, 7, 10, 11, 15, 118, 119, 120, 124, 125, 126, 130, 131, 132, 133, 134, 136, 137, 139],
    "DRLBS": [2, 5, 10, 15, 16, 28, 30, 31, 65, 75, 80, 81, 91, 99, 101, 114, 128, 130, 132, 135],
    "MH-DRL": [3, 8, 9, 17, 21, 27, 38, 57, 58, 62, 65, 66, 79, 85, 87, 95, 98, 116, 131, 139]
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
x = np.array(list(range(144)))
ax[1].plot(x, hx)
ax[1].set_xlabel("Spectral bands", fontsize=fontsize)
ax[1].set_ylabel("Entropy value", fontsize=fontsize)
ax[1].grid(axis="x")
for i in range(plot.shape[0]):
    ax[0].scatter(plot[i, :, 0], plot[i, :, 1], marker="o")
    ax[0].plot([min(plot[i, :, 0]), max(plot[i, :, 0])], [min(plot[i, :, 1]), max(plot[i, :, 1])])
ax[0].set_yticks(list(range(1, len(bands.keys()) + 1)))
ax[0].set_yticklabels(list(bands.keys()), fontsize=fontsize)
ax[0].grid(axis="x")
# ax[0].grid(axis = "y")
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.show()
fig.savefig(data_name + ".tiff", format='tiff', pad_inches=0, bbox_inches='tight')
