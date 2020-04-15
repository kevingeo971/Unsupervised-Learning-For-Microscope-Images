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

print("\n\n Small No SM Also saving 1 No compress Stride ")

print("\n\nGPU Available : ",torch.cuda.is_available())

print("\n\nNumber of GPU's : ", torch.cuda.device_count())

X_train = np.load("X_train.npy")
X_val = np.load("X_val.npy")
Y_train = np.load("Y_train.npy")
Y_val = np.load("Y_val.npy")
print(X_train.shape, X_val.shape, Y_train.shape, Y_val.shape)

print("\n\nLoaded Data")

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
        return self.transform_image(img_batch), lable_batch

Data_train = CancerData(X_train, Y_train)
Data_val = CancerData(X_val, Y_val)

BATCH_SIZE = 96
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0
Epochs = 13


data_loader_train = DataLoader(Data_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)
data_loader_val = DataLoader(Data_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)
X_batch , Y_batch = next(iter(data_loader_val))
print("\n\nOne Batch : ",type(X_batch), X_batch.shape, X_batch.numpy().max(),X_batch.numpy().min(), Y_batch.shape)

class Network(nn.Module):

    def __init__  (self):

        super(Network, self).__init__()

        self.res = models.resnet18(pretrained=True)
        self.fc1 = nn.Linear(1000,512)
        #self.bn1 = nn.Batchnorm1d(512)
        self.fc2 = nn.Linear(512,2)
        

    def forward(self, img):
        
        x = F.relu(self.res(img))
        #print(x.shape)
        x = F.relu(self.fc1(x))
        scores = self.fc2(x)
        return scores

model = Network().cuda()
model = nn.DataParallel(model)
# model = model.cuda()
# min_loss = 10

# x = torch.FloatTensor(X_batch)
# op = model(x.cuda())


best_ep = 0
best_val_acc = -1

def train(epoch):
    
    #Train
    train_acc_sum = 0
    total_loss = 0 
    for batch_idx, batch in enumerate(tqdm(data_loader_train)):
        model.train()
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
    for batch_idx, batch in enumerate(data_loader_val):
        model.eval()
        images, targets = batch[0], batch[1]
        # images, targets = images, targets
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
    print("\n\nEpoch - ",epoch, " Train Accuracy : ", train_total_accuracy, " Val Accuracy : ", total_accuracy,"\n\n")

    return total_accuracy
    


optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = F.cross_entropy

best_ep = 0
best_val_acc = -1

for i in range(Epochs):
  accuracy = train(i)

  if accuracy>best_val_acc:
        best_val_acc = accuracy
        best_ep = i
        torch.save(model.state_dict(), "Model.pwf")


print("\n\nBest Values \n\n Epoch - ",best_ep, "Val Accuracy : ", best_val_acc,"\n\n")

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


