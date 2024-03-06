import scipy.io as sio
import numpy as np
import cv2
import os

data_name = 'Salinas'
path = os.getcwd()+'/data/'+data_name
data = sio.loadmat(path+'/Salinas_corrected.mat')['salinas_corrected']

print(data.shape)
rgb_datas = data[:, :, (69, 27, 11)]
rgb_datas = np.array(rgb_datas).astype(float)
# scio.savemat("F:/Projects/Data/reflect_norm_sg_visualization.mat", {'reflect_norm_sg_visualization':rgb_datas})

bgr_datas = rgb_datas[:,:,(2,1,0)]
print(np.max(bgr_datas))

cv2.imshow('output', bgr_datas)
cv2.waitKey(3000)
min_ = np.min(bgr_datas)
max_ = np.max(bgr_datas)
bgr_datas -= min_
bgr_datas /= max_
bgr_datas *= 255
cv2.imwrite('D:/yzw_duibisuanfa/Sec.II TWCNN/rgb.png', bgr_datas)
cv2.destroyAllWindows()
