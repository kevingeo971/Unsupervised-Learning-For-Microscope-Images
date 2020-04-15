import os
from tqdm import tqdm
import cv2 
import numpy as np
import matplotlib.image as mim
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import torch.optim as optim
from sklearn.model_selection import train_test_split
from unet import UNet

print("\n\n Very Few with No grad")

print("\n\nGPU Available : ",torch.cuda.is_available())

print("\n\nNumber of GPU's : ", torch.cuda.device_count())

print("\n Name of 1st device : ", torch.cuda.get_device_name())

X_train = np.load("X_train_og.npy")
# X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_val = np.load("X_val_og.npy")
# X_val = X_val.reshape(X_val.shape[0],X_val.shape[1],X_val.shape[2],1)
Y_train = np.load("Y_train.npy")
Y_val = np.load("Y_val.npy")
print(X_train.shape, X_val.shape, Y_train.shape, Y_val.shape)

print("\n\nLoaded Data")

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
        #img_shape = self.X_data.shape
        img_batch = self.X_data[idx]
        lable_batch = self.Y_data[idx]
        return self.transform_image(img_batch), lable_batch

Data_train = CancerData(X_train, Y_train)
Data_val = CancerData(X_val, Y_val)

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
LR_DECAY = 0.9
WEIGHT_DECAY = 0.01
Epochs = 200

print("\n - - HyperParameters : ",BATCH_SIZE,LEARNING_RATE,WEIGHT_DECAY,Epochs)

data_loader_train = DataLoader(Data_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)
data_loader_val = DataLoader(Data_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)
X_batch , Y_batch = next(iter(data_loader_val))
print("\n\nOne Batch : ",type(X_batch), X_batch.shape, X_batch.numpy().max(),X_batch.numpy().min(), Y_batch.shape)


cae_model = UNet(in_channels=1, out_channels=1, only_encode=True)
cae_model = nn.DataParallel(cae_model)
cae_model.load_state_dict(torch.load("Best_Cancer_CAE.pwf"))
# model = nn.DataParallel(model)
# model.cuda()

class Network(nn.Module):

    def __init__  (self):

        super(Network, self).__init__()
        self.cae_model = cae_model
        self.features = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512,2)
        
    def forward(self, img):
        # with torch.no_grad():
        x = self.cae_model(img)
        x = self.features(x)
        x = x.squeeze()
        scores = self.fc(x)
        return scores

model = Network().cuda()

def train(epoch):
    
    #Train
    train_acc_sum = 0
    total_loss = 0 
    model.train()
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

    op_train_loss = total_loss / (batch_idx + 1)
    train_total_accuracy = ( X_train.shape[0] - train_acc_sum ) / X_train.shape[0]

    #Validate
    acc_sum = 0
    total_loss = 0 
    model.eval()
    for batch_idx, batch in enumerate(data_loader_val):
        images, targets = batch[0], batch[1]
        # images, targets = images, targets
        with torch.no_grad():
            output = model(images.cuda())
        # val_loss = criterion(output.cpu(),targets)
        # total_loss+=val_loss
        # targets = targets
        output = output.cpu().argmax(axis=1)
        diff = torch.abs(output - targets)
        acc_sum += diff.sum().item()

    # op_val_loss = total_loss / (batch_idx + 1)

    total_accuracy = ( X_val.shape[0] - acc_sum ) / X_val.shape[0]
    # print("\n\nEpoch - ",epoch, " Training Loss = ", op_train_loss.item(), " Val loss : ",op_val_loss.item(), " Val Accuracy : ", total_accuracy)
    print("\n\nTest Epoch - ",epoch, " Train Accuracy : ", train_total_accuracy, " Val Accuracy : ", total_accuracy,"\n\n")

    return total_accuracy
    


optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = LR_DECAY)
criterion = F.cross_entropy

best_ep = 0
best_val_acc = -1

for i in range(Epochs+1):
    
    accuracy = train(i)
    scheduler.step()
    if accuracy>best_val_acc:
        best_val_acc = accuracy
        best_ep = i
        #torch.save(model.state_dict(), "Model.pwf")

    if i%10 == 0:
        print("\n\n ----------- Best Values so far : Epoch - ",best_ep, "Val Accuracy : ", best_val_acc,"\n\n\n")
