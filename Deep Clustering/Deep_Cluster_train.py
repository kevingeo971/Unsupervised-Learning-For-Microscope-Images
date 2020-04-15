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
import torchvision.models as models
from unet import UNet
from sklearn.cluster import KMeans

print("\n\n Deep Cluster Train ")

print("\n\nGPU Available : ",torch.cuda.is_available())

print("\n\nNumber of GPU's : ", torch.cuda.device_count())

X1 = np.load("X_train.npy")
# X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X2 = np.load("X_val.npy")
# X_val = X_val.reshape(X_val.shape[0],X_val.shape[1],X_val.shape[2],1)
Y1 = np.load("Y_train.npy")
Y2 = np.load("Y_val.npy")

X_train = np.concatenate( (X1, X2)) 
Y_train = np.concatenate( (Y1, Y2)) 

print(" Train Shapes : ",X_train.shape, Y_train.shape)

print("\n\nLoaded Data")

X_train_processed = ( X_train.copy() - 127.5 )/ 127.5
X_train_tensor = torch.Tensor( X_train_processed ).unsqueeze(1)
print(" X_train_tensor : ",X_train_tensor.shape , type( X_train_tensor),  X_train_tensor.cpu().numpy().max(),X_train_tensor.cpu().numpy().min())

# plt.imshow(X_train[100])

class CancerData(Dataset):
    def __init__(self, data, label):
        self.X_data = data
        self.Y_data = label

    def __len__(self):
        return self.X_data.shape[0]

    def transform_image(self, image):
        #img_resized_tensor = TF.to_tensor(image)
        normalize_img = (image - 127.5) / 127.5
        return TF.to_tensor(normalize_img).type(torch.FloatTensor)

    def __getitem__(self, idx):
        # print(s\n\nelf.images[idx].shape)
        img_shape = self.X_data.shape
        img_batch = self.X_data[idx]
        lable_batch = self.Y_data[idx]
        return self.transform_image(img_batch), lable_batch.astype(np.long)

Data_train = CancerData(X_train, Y_train)

BATCH_SIZE = 32
LEARNING_RATE = 0.05
LR_DECAY = 0.95
WEIGHT_DECAY = 0
Epochs = 200

print("HyperParameters : ")
print("\nBATCH_SIZE : ",BATCH_SIZE)
print("LEARNING_RATE : ", LEARNING_RATE)
print("LR_DECAY : ",LR_DECAY)
print("WEIGHT_DECAY : ",WEIGHT_DECAY)
print("Epochs : ",Epochs)

data_loader_train = DataLoader(Data_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=50)
X_batch , Y_batch = next(iter(data_loader_train))
print("\n\nOne Batch : ",type(X_batch), X_batch.shape, X_batch.numpy().max(),X_batch.numpy().min(), Y_batch.shape)
print(Y_batch)

class Network(nn.Module):

    def __init__  (self):

        super(Network, self).__init__()

        self.net = UNet(in_channels=1, out_channels=3, only_encode=True)
        self.features = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512,2)
        
    def forward(self, img, features_only=False):
        x = self.net(img)
        x = self.features(x)
        x = x.squeeze()
        if features_only:
            return x

        scores = self.fc(x)
        return scores

model = Network().cuda()
# model = nn.DataParallel(model)

best_ep = 0
best_val_acc = -1

# prev_lables = np.random.rand(X_train_tensor.shape[0])

def train(epochs):
    
    #Train
    train_acc_sum = 0
    total_loss = 0 
    model.train()

    ### Feature Extraction ###
    
    feat = torch.empty( 0,512)

    for i in range( X_train_tensor.shape[0]//500):

        with torch.no_grad():
            f = model(X_train_tensor[i*500:(i+1)*500].cuda(), features_only=True)        

        feat = torch.cat( (feat, f.cpu()),0) 

    with torch.no_grad():
        f = model(X_train_tensor[(i+1)*500 : ].cuda(), features_only=True)        

    feat = torch.cat( (feat, f.cpu()),0) 

    ##############################
    
    cluster_lables = KMeans(n_clusters=2, random_state=0).fit(feat).labels_

    print("\n\n Epoch : ",epochs," Cluster :", cluster_lables.sum())

    Data_train = CancerData(X_train, cluster_lables)
    data_loader_train = DataLoader(Data_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=50)

    
    model.fc.weight.data.normal_(0, 0.01)
    model.fc.bias.data.zero_()

    for batch_idx, batch in enumerate(tqdm(data_loader_train)):
        
        images, targets = batch[0], batch[1]
        images, targets = images.cuda(), targets.cuda()
        
          
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output,targets)
        loss.backward()
        optimizer.step()
        # print("Loss : ",loss.item(),loss.item())
        total_loss+=loss.item()
        # print(b\n\natch_idx)
        # if batch_idx == 3:
        #   break
        output = output.cpu().argmax(axis=1)
        diff = torch.abs(output - targets.cpu())
        train_acc_sum += diff.sum().item()

    # print(batch_idx)
    op_train_loss = total_loss / (batch_idx + 1)
    train_total_accuracy = ( X_train.shape[0] - train_acc_sum ) / X_train.shape[0]

    print("Loss : ",op_train_loss, "Accuracy : ", train_total_accuracy)

    return train_total_accuracy

optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = LR_DECAY)
criterion = F.cross_entropy

best_ep = 0
best_val_acc = -1

for i in range(Epochs+1):
    accuracy = train(i)
    # scheduler.step()
    
    if accuracy>best_val_acc:
        best_val_acc = accuracy
        best_ep = i
        torch.save(model.state_dict(), "Best_Model.pwf")
        print( "\n\n ---- Best Saved ----\n\n")

    torch.save(model.state_dict(), "Last_Model.pwf")




####################
###### RESULTS #####
####################

'''
----- Balance Dataset  

Batch Size 96 lr = 1e-5 Wights Decay = 1E-5 - accuracy = 89 something  

----- Full Dataset ( 7118 Train, 790 Test)------- 

Batch Size 96 lr = 1e-5 Wights Decay = 1E-5 - accuracy = 90.26% 
Batch Size 96 lr = 1e-6 Wights Decay = 1E-5 - accuracy = 91.02% 

----- Bigger Network ( 7118 Train, 790 Test)------- 

Batch Size 25 (Max) lr = 1e-6 Wights Decay = 0 - accuracy = 92% 

----- Smaller img size Bigger Network ( 7513 Train, 396 Test)------- 

Batch Size 48 (Max) lr = 1e-6 Wights Decay = 0 - accuracy = 93.4%

Batch Size 64 Skynet lr = 1e-5 Wights Decay = 0 - accuracy = 91.4%

Batch Size 96 Skynet lr = 1e-6 Wights Decay = 0 - accuracy = 92.4%
'''


