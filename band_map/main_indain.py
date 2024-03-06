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

data_name = "Indian_pines"

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

# bands = {"mRMR":[11, 14, 16, 34, 41, 45, 45, 61, 78, 81, 84, 90, 93, 103, 105, 107, 132, 139, 145, 149,           151, 158, 163, 177, 179, 182, 185, 187, 191, 196],
#          "TWCNN":[10, 15, 16, 21, 30, 32, 34, 45, 47, 71, 72, 74, 82, 88, 90, 97, 101, 119, 139, 162],
#          "MR-SVM": [1, 6, 10, 20, 22, 30, 36, 39, 45, 55, 56, 65, 70, 74, 81, 82, 83, 84, 85, 86,                  87, 121, 123, 124, 164, 175, 177, 180, 184, 187],
#          "SICNN": [0,1,2, 6, 10, 20, 22, 30, 36, 39, 45, 55, 56, 65, 100, 104, 111, 113, 85, 87,                    91, 96, 97, 121, 123, 124, 164, 175, 177, 180, 184, 187],
#          "DDCNN":[1,2,6,8,10,14,16,26,32,33,39,41,47,50,51,52,53,55,70,81,                            83,84,85,88,91,110,130,160,181,192],
#          "ABCNN":[1,3,5,8,9,11,12,13,14,15,41,42,47,48,60,121,122,123,126,139],
#          "HM":[125,128,175,131,191,178,169,100,196,199,76,189,105,181,122,157,142,190,133,103,130,148,155,33,177,123,179,138,173,124,150,136,195,132,193,185,172,192,119,129,134,187,164,126,198,174,194,19,183,180,186,166,167,151,101,141,158,143,154,104,170,127,188,162,118,168,176,184,146,171,163,98,149,144,159,152,117,108,197,120,137,106,102,147,135,20,121,107,18,145,160,161,80,153,165,139,182,78,22,156,140,60,77,109,116,99,113,110,93,79,3,114,6,115,5,112,0,23,4,85,111,36,58,13,97,57,1,2,35,56,24,34,81,84,91,38,9,14,7,95,8,25,30,92,21,41,48,37,94,62,15,10,63,43,16,49,51,50,47,52,42,87,31,44,46,96,65,11,53,45,40,70,89,27,39,88,64,75,69,12,66,29,67,28,68,59,26,17,71,54,61,55,32,72,83,86,73,74,82,90],
#          "DCS":[1, 6, 10, 20, 22, 30, 36, 39, 45, 55, 56, 65, 70, 74, 81, 82, 83, 84, 85, 86, 87, 121, 123, 124, 164, 175, 177, 180, 184, 187],
#          "BSD-GCN":[14,26,29,36,39,45,48,51,55,65,78,95,100,118,121,125,129,160,174,188]
#          }
bands = {"mRMR": [14, 34, 41, 45, 61, 78, 84, 90, 103, 105, 107, 132, 139, 149, 158, 163, 182, 187, 191, 196],
         "TWCNN": [10, 15, 16, 21, 30, 32, 34, 45, 47, 71, 72, 74, 82, 88, 90, 97, 101, 119, 139, 162],
         "MR-SVM": [1, 6, 22, 30, 36, 39, 55, 56, 65, 70, 81, 83, 85, 86, 87, 121, 123, 124, 164, 184],
         "SICNN": [0, 6, 10, 20, 22, 39, 45, 55, 56, 87, 91, 96, 97, 104, 111, 113, 121, 123, 175, 177],
         "DDCNN": [2, 6, 8, 14, 16, 32, 33, 39, 47, 50, 51, 52, 53, 55, 81, 83, 85, 91, 110, 192],
         "ABCNN": [1, 3, 5, 8, 9, 11, 12, 13, 14, 15, 41, 42, 47, 48, 60, 121, 122, 123, 126, 139],
         "DRLBS": [1, 6, 10, 20, 39, 45, 55, 56, 65, 82, 83, 84, 121, 123, 124, 164, 175, 177, 186, 192],
         "BS-Nets": [12, 18, 19, 20, 25, 26, 27, 28, 65, 74, 78, 135, 138, 139, 165, 166, 167, 173, 174, 175],
         "MH-DRL": [156, 28, 134, 7, 144, 67, 148, 194, 153, 43, 193, 31, 100, 169, 118, 196, 6, 64, 65, 160]
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


# bands["mRMR"]  = random.sample(bands["mRMR"],band_num)
# bands["MR-SVM"]  = random.sample(bands["MR-SVM"],band_num)
# # bands["DCS"]  = random.sample(bands["DCS"],band_num)
# bands["SICNN"]  = random.sample(bands["SICNN"],band_num)
# bands["DDCNN"]  = random.sample(bands["DDCNN"],band_num)
# # bands["HM"]  = random.sample(bands["HM"],band_num)

for i in bands.keys():
    bands[i] = bands[i][:band_num]
bands_data = np.array(list(bands.values()))
plot = np.zeros([bands_data.shape[0] * bands_data.shape[1], 2])
for i in range(bands_data.shape[0]):
    for j in range(bands_data.shape[1]):
        plot[i * bands_data.shape[1] + j] = [bands_data[i, j], i + 1]
plot = plot.reshape([len(bands.keys()), band_num, -1])
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(14, 8))
x = np.array(list(range(200))) + 1
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
# ax[0].margins(0,0)
# ax[1].margins(0,0)
# ax[0].grid(axis = "y")
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.show()
fig.savefig(data_name + ".tif", format='tif', pad_inches=0, bbox_inches='tight')
