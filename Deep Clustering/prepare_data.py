
import os
from tqdm import tqdm
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import torch.optim as optim
from sklearn.model_selection import train_test_split

Train_size = 0.1
Test_size = 0.45
Val_size = 0.45

print(" In Prepare data Deep Cluster")


path_benign = '../../../../srv/share/kgeorge37/BreaKHis_v1/histology_slides/breast/benign/SOB/'
path_malignant = '../../../../srv/share/kgeorge37/BreaKHis_v1/histology_slides/breast/malignant/SOB'  
img_size = (64, 64)

print("Benign")

c = 0
img_benign_list = []
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
        img_benign_list.append(img)
        Y_data_benign.append(0)
        
        # for rot in range(3):
        #   img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        #   img_benign_list.append(img)
        #   Y_data_benign.append(0)


X_benign = np.array(img_benign_list)
Y_data_benign = np.array(Y_data_benign)

print("Benign Shapes : ",X_benign.shape, Y_data_benign.shape, X_benign.max(), X_benign.min())


print("Malignant")


c = 0
img_malignant_list = []
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
        img_malignant_list.append(img)
        Y_data_malignant.append(1)
        
        # for rot in range(3):
        #   img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        #   img_malignant_list.append(img)
        #   Y_data_malignant.append(1)
        

#print(set(img_malignant_list))
# idx = np.random.choice(len(img_malignant_list), 2480, replace=False)


X_malignant = np.array(img_malignant_list)
Y_data_malignant = np.array(Y_data_malignant)

######### Split now ###########

print( "Split ..")

X_train_mal, X_val_test_mal , Y_train_mal, Y_val_test_mal = train_test_split(X_malignant, Y_data_malignant, test_size=Test_size+Val_size)
X_train_benigh, X_val_test_benigh , Y_train_benigh, Y_val_test_benigh = train_test_split(X_benign, Y_data_benign, test_size=Test_size+Val_size)

X_test_mal, X_val_mal , Y_test_mal, Y_val_mal = train_test_split(X_val_test_mal, Y_val_test_mal, test_size=Val_size/(Val_size + Test_size))
X_test_benigh, X_val_benigh , Y_test_benigh, Y_val_benigh = train_test_split(X_val_test_benigh, Y_val_test_benigh, test_size=Val_size/(Val_size + Test_size))


print( "Concatinate ..")
X_train = np.concatenate((X_train_mal , X_train_benigh), axis=0)
X_val = np.concatenate((X_val_mal , X_val_benigh), axis=0)
X_test = np.concatenate((X_test_mal , X_test_benigh), axis=0)

Y_train = np.concatenate((Y_train_mal , Y_train_benigh), axis=0)
Y_val = np.concatenate((Y_val_mal , Y_val_benigh), axis=0)
Y_test = np.concatenate((Y_test_mal , Y_test_benigh), axis=0)

#print("Save X_data")

#np.save("X_data.npy", X_data)

# Y_data = np.concatenate((Y_data_malignant , Y_data_benign), axis=0)
#np.save("Y_data.npy", Y_data)

# X_data = np.load("X_data.npy")
# Y_data = np.load("Y_data.npy")


# X_train, X_val , Y_train, Y_val = train_test_split(X_data, Y_data, test_size=0.975)

print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

np.save("X_train.npy",X_train)
np.save("X_val.npy",X_val)
np.save("X_test.npy",X_test)

np.save("Y_train.npy",Y_train)
np.save("Y_val.npy",Y_val)
np.save("Y_test.npy",Y_test)