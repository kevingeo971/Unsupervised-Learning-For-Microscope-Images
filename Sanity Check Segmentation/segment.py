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
from sklearn.metrics import accuracy_score as acc


print("\n\n New Sanity Check Seg")

print("\n\nGPU Available : ",torch.cuda.is_available())

print("\n\nNumber of GPU's : ", torch.cuda.device_count())

X_train = np.load("seg_X_train.npy",allow_pickle=True)
# X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_val = np.load("seg_X_val.npy",allow_pickle=True)
# X_val = X_val.reshape(X_val.shape[0],X_val.shape[1],X_val.shape[2],1)
Y_train = np.load("seg_Y_train.npy",allow_pickle=True).astype(int)
Y_val = np.load("seg_Y_val.npy",allow_pickle=True).astype(int)

print(Y_train.dtype)
print("Shapes : ",X_train.shape, X_val.shape, Y_train.shape, Y_val.shape, )

n_classes = 57

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
        img_batch = self.X_data[idx]
        lable_batch = self.Y_data[idx]
        return self.transform_image(img_batch), TF.to_tensor(lable_batch).type(torch.LongTensor)

Data_train = CancerData(X_train, Y_train)
Data_val = CancerData(X_val, Y_val)

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
LR_DECAY = 0.95
WEIGHT_DECAY = 0.0001
Epochs = 500

print("HyperParameters : ")
print("\nBATCH_SIZE : ",BATCH_SIZE)
print("LEARNING_RATE : ", LEARNING_RATE)
print("LR_DECAY : ",LR_DECAY)
print("WEIGHT_DECAY : ",WEIGHT_DECAY)
print("Epochs : ",Epochs)

''' 
#For 3 channel BRG
BATCH_SIZE = 96
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.4
Epochs = 50
'''

data_loader_train = DataLoader(Data_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=50)
data_loader_val = DataLoader(Data_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=50)
X_batch , Y_batch = next(iter(data_loader_val))
print("\n\nOne Batch : ",type(X_batch), X_batch.shape, X_batch.numpy().max(),X_batch.numpy().min(), Y_batch.shape)

class Network(nn.Module):

    def __init__  (self):

        super(Network, self).__init__()

        self.net = UNet(in_channels=1, out_channels=n_classes)
        self.features = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512,256)
        self.fc2 = nn.Linear(256,20)
        
    def forward(self, img):
        x = self.net(img)
        x = x.squeeze()
        return x

model = Network().cuda()
model = nn.DataParallel(model)
# model.load_state_dict(torch.load("Model.pwf"))


best_ep = 0
best_val_acc = -1

def train(epoch):
    
    #Train
    train_acc_sum = 0
    total_loss = 0 
    model.train()
    pred_y = torch.empty(0).long()
    act_y = torch.empty(0).long()
    for batch_idx, batch in enumerate(tqdm(data_loader_train)):
        
        images, targets = batch[0], batch[1]
        images, targets = images.cuda(), targets.cuda().squeeze()
        
        
        optimizer.zero_grad()
        output = model(images)
        # print(output.shape, targets.shape)
        loss = criterion(output,targets)
        loss.backward()
        optimizer.step()
        # print("Loss : ",loss.item(),loss.item())
        total_loss+=loss.item()
        # print(b\n\natch_idx)
        # if batch_idx == 3:
        #   break
        output = output.cpu().argmax(axis=1)
        pred_y = torch.cat((pred_y,output))
        act_y = torch.cat((act_y,targets.cpu()))
        

    # print(batch_idx)
    op_train_loss = total_loss / (batch_idx + 1)
    # train_accuracy = acc(act_y, pred_y)[]
    # train_total_accuracy = ( X_train.shape[0] - train_acc_sum ) / X_train.shape[0]

    #Validate
    acc_sum = 0
    total_loss = 0 
    model.eval()
    pred_y = torch.empty(0).long()
    act_y = torch.empty(0).long()
    for batch_idx, batch in enumerate(data_loader_val):
        
        images, targets = batch[0], batch[1]
        # print(images.shape)
        #images, targets = images, targets
        with torch.no_grad():
            output = model(images.cuda())
        val_loss = criterion(output,targets.cuda().squeeze())
        total_loss+=val_loss
        # targets = targets
        output = output.cpu()
        

    # print(batch_idx)
    op_val_loss = total_loss / (batch_idx + 1)
    # val_accuracy = acc(act_y, pred_y)
    # print("\n\nEpoch - ",epoch, " Training Loss = ", op_train_loss.item(), " Val loss : ",op_val_loss.item(), " Val Accuracy : ", total_accuracy)
    print("\n\nTrain Epoch - ",epoch, " Train Accuracy : ", op_train_loss, "op_val_loss : ", op_val_loss,"\n\n")

    return output[0].numpy().argmax(axis=0)*4,targets[0].cpu().numpy().squeeze()*4,op_val_loss
    


optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = LR_DECAY)
criterion = nn.CrossEntropyLoss()

best_ep = 0
best_val_acc = 10000

for i in range(Epochs+1):
    
    output,target,accuracy = train(i)
    # print("output.shape, target.shape : ",output.shape, target.shape)
    scheduler.step()
    if accuracy<best_val_acc:
        best_val_acc = accuracy
        best_ep = i
        print("Saved")
        torch.save(model.state_dict(), "Seg_Model.pwf")
        plt.imsave("Outputs/best_op.png", output)
        plt.imsave("Outputs/best_target.png",target )
    else:
        plt.imsave("Outputs/curr_op.png", output)
        plt.imsave("Outputs/curr_target.png",target )

    if i%10 == 0:
        print("\n\n ----------- Best Values so far : Epoch - ",best_ep, "Val Accuracy : ", best_val_acc,"\n\n\n")
        
