
import os
from tqdm import tqdm
import cv2 
import numpy as np
import matplotlib.pyplot as plt
# import torch 
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from torchvision import datasets, transforms
# import torchvision.transforms.functional as TF
# import torch.optim as optim
from sklearn.model_selection import train_test_split

TestSize = 0.5

############# FFT PART #############

import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt,exp
'''
def distance(point1,point2):
    return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def gaussianLP(imgShape, D0=5):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

def gaussianHP(imgShape, D0=5):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1 - exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

def Filter_low_high(img):

    original = np.fft.fft2(img)
    # print(original.shape)
    # plt.imshow(original)
    # plt.show()

    center = np.fft.fftshift(original)
    # print("Center : ", center.transpose(2,0,1).shape)

    # print("gaussianLP(img.shape)", gaussianLP(img.shape).shape)

    LowPassCenter = center * gaussianLP(img.shape)
    LowPass = np.fft.ifftshift(LowPassCenter)
    inverse_LowPass = np.fft.ifft2(LowPass)
    

    HighPassCenter = center*gaussianHP(img.shape)
    HighPass = np.fft.ifftshift(HighPassCenter)
    inverse_HighPass = np.fft.ifft2(HighPass)
    # plt.imshow(np.abs(inverse_HighPass), "gray"), plt.title("Gaussian High Pass")

    # plt.show()
    return np.abs(inverse_LowPass), np.abs(inverse_HighPass)


####################################


print(" In Prepare FFT")


path_benign = '../../../../srv/share/kgeorge37/BreaKHis_v1/histology_slides/breast/benign/SOB/'
path_malignant = '../../../../srv/share/kgeorge37/BreaKHis_v1/histology_slides/breast/malignant/SOB'  

print("Benign")

c = 0
img_benign_list = []
img_benign_list_lowF = []
img_benign_list_HighF = []
Y_data_benign = []   
for i in tqdm(os.listdir(path_benign)):
  for j in os.listdir(path_benign + '/'+ i):
    p = path_benign + '/'+ i
    for k in os.listdir(p + '/'+ j):
      q = p + '/'+ j
      for l in os.listdir(q + '/'+ k):
        img_path = q + '/'+ k + '/'+ l
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.pyrDown(cv2.pyrDown(img))
        low_F , high_F = Filter_low_high(img)
        img_benign_list.append(img)
        img_benign_list_lowF.append(low_F)
        img_benign_list_HighF.append(high_F)
        Y_data_benign.append(0)
        # for rot in range(3):
        #   img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        #   img_benign_list.append(img)
        #   Y_data_benign.append(0)
        # break
        

X_benign_og = np.array(img_benign_list)
X_benign_lowF = np.array(img_benign_list_lowF)
X_benign_highF = np.array(img_benign_list_HighF)
Y_data_benign = np.array(Y_data_benign)

np.save("X_benign_og",X_benign_og)
np.save("X_benign_lowF",X_benign_lowF)
np.save("X_benign_highF",X_benign_highF)
np.save("Y_data_benign",Y_data_benign)
'''
X_benign_og = np.load("X_benign_og.npy")
X_benign_lowF = np.load("X_benign_lowF.npy")
X_benign_highF = np.load("X_benign_highF.npy")
Y_data_benign = np.load("Y_data_benign.npy")


print("Benign Shapes : ",X_benign_og.shape, Y_data_benign.shape, X_benign_og.max(), X_benign_lowF.min())
'''
print("Malignant")


c = 0
img_malignant_list = []
img_malignant_list_lowF = []
img_malignant_list_highF = []
Y_data_malignant = []
for i in tqdm(os.listdir(path_malignant)):
  for j in os.listdir(path_malignant + '/'+ i):
    p = path_malignant + '/'+ i
    for k in os.listdir(p + '/'+ j):
      q = p + '/'+ j
      for l in os.listdir(q + '/'+ k):
        img_path = q + '/'+ k + '/'+ l
        img = cv2.resize(cv2.imread(img_path), (700,460))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.pyrDown(cv2.pyrDown(img))
        low_F , high_F = Filter_low_high(img)
        img_malignant_list.append(img)
        # print(low_F.shape)
        img_malignant_list_lowF.append(low_F)
        img_malignant_list_highF.append(high_F)
        Y_data_malignant.append(1)
        # break
        # for rot in range(3):
        #   img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        #   img_malignant_list.append(img)
        #   Y_data_malignant.append(1)
    
        

#print(set(img_malignant_list))
# idx = np.random.choice(len(img_malignant_list), 2480, replace=False)


X_malignant_og = np.array(img_malignant_list)
X_malignant_lowF = np.array(img_malignant_list_lowF)
X_malignant_highF = np.array(img_malignant_list_highF)
Y_data_malignant = np.array(Y_data_malignant)

np.save("X_malignant_og.npy",X_malignant_og)
np.save("X_malignant_lowF.npy",X_malignant_lowF)
np.save("X_malignant_highF.npy",X_malignant_highF)
np.save("Y_data_malignant.npy",Y_data_malignant)
'''
X_malignant_og = np.load("X_malignant_og.npy")
X_malignant_lowF = np.load("X_malignant_lowF.npy")
X_malignant_highF = np.load("X_malignant_highF.npy")
Y_data_malignant = np.load("Y_data_malignant.npy")


######### Split now ###########

print( "\nSplit ..")

X_train_mal_idx, X_val_mal_idx , Y_train_mal, Y_val_mal = train_test_split(np.arange(X_malignant_og.shape[0]), Y_data_malignant, test_size=TestSize)
X_train_benigh_idx, X_val_benigh_idx , Y_train_benigh, Y_val_benigh = train_test_split(np.arange( X_benign_og.shape[0]), Y_data_benign, test_size=TestSize)

print( "\nConcatinate and save Y ..")
Y_train = np.concatenate((Y_train_benigh , Y_train_mal), axis=0)
Y_val = np.concatenate((Y_val_benigh , Y_val_mal), axis=0)

np.save("Y_train.npy",Y_train)
np.save("Y_val.npy",Y_val)



print("\n Concatenate and Save Low ...")
# print("X_benign_og[ X_train_benigh_idx] : ",X_benign_lowF[ X_train_benigh_idx].shape)
# print("X_malignant_og[ X_train_mal_idx ] : ", X_malignant_lowF[ X_train_mal_idx ].shape)
# print("X_train_benigh_idx : ", X_train_benigh_idx.shape)
# print("X_train_mal_idx : ",X_train_mal_idx )
X_train = np.concatenate((X_benign_lowF[ X_train_benigh_idx]  , X_malignant_lowF[ X_train_mal_idx ]), axis=0)
X_val = np.concatenate((X_benign_lowF[X_val_benigh_idx] ,X_malignant_lowF[ X_val_mal_idx]), axis=0)

print("lowF Shapes : ",X_train.shape, X_val.shape)

np.save("X_train_lowF.npy",X_train)
np.save("X_val_lowF.npy",X_val)


print("\n Concatenate and Save Og ...")
# print("X_benign_og[ X_train_benigh_idx] : ",X_benign_og[ X_train_benigh_idx].shape)
# print("X_malignant_og[ X_train_mal_idx ] : ", X_malignant_og[ X_train_mal_idx ].shape)
# print("X_train_benigh_idx : ", X_train_benigh_idx.shape)
# print("X_train_mal_idx : ",X_train_mal_idx )
X_train = np.concatenate((X_benign_og[ X_train_benigh_idx]  , X_malignant_og[ X_train_mal_idx ]), axis=0)
X_val = np.concatenate((X_benign_og[X_val_benigh_idx] ,X_malignant_og[ X_val_mal_idx]), axis=0)


print("Og Shapes : ",X_train.shape, X_val.shape)

np.save("X_train_og.npy",X_train)
np.save("X_val_og.npy",X_val)


print("\n Concatenate and Save High ...")
X_train = np.concatenate((X_benign_highF[ X_train_benigh_idx]  , X_malignant_highF[ X_train_mal_idx ]), axis=0)
X_val = np.concatenate((X_benign_highF[X_val_benigh_idx] ,X_malignant_highF[ X_val_mal_idx]), axis=0)

print("HighF Shapes : ",X_train.shape, X_val.shape)

np.save("X_train_highF.npy",X_train)
np.save("X_val_highF.npy",X_val)




