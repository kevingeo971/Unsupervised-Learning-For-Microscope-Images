print("\n\nTrain CAE Unet")

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

print("\n\n  Best Res CAE pr")

print("\n\nGPU Available : ",torch.cuda.is_available())

print("\n\nNumber of GPU's : ", torch.cuda.device_count())

X_train = np.load("../X_train.npy")
# X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_val = np.load("../X_val.npy")
# X_val = X_val.reshape(X_val.shape[0],X_val.shape[1],X_val.shape[2],1)
Y_train = np.load("../Y_train.npy")
Y_val = np.load("../Y_val.npy")
print(X_train.shape, X_val.shape, Y_train.shape, Y_val.shape)

print("\n\nLoaded Data")

# mim.imsave("Outputs/test.png",X_train[0])
# mim.imsave("Outputs/test_flip.png",X_train[0][::-1,:,:])

# print(a)

class CancerData(Dataset):
    def __init__(self, data, label):
        self.X_data = data
        self.Y_data = label

    def __len__(self):
        return self.X_data.shape[0]

    def transform_image(self, image):
        img_resized_tensor = TF.to_tensor(image)
        normalize_img = (image - 127.5) / 127.5
        return TF.to_tensor(normalize_img).type(torch.FloatTensor)

    def __getitem__(self, idx):
        # print(s\n\nelf.images[idx].shape)
        # img_shape = self.X_data.shape
        img_batch = self.X_data[idx]
        lable_batch = self.Y_data[idx]
        # final_img = self.transform_image(img_batch)
        return self.transform_image(img_batch),self.transform_image(lable_batch) 

rev_train = X_train[:,::-1,:,:].copy()
rev_val = X_val[:,::-1,:,:].copy()
Data_train = CancerData(X_train, rev_train)
Data_val = CancerData(X_val, rev_val)

BATCH_SIZE = 16
LEARNING_RATE = 1e-3
LR_DECAY = 0.95
WEIGHT_DECAY = 0.000
Epochs = 1000

print("\n - - HyperParameters : ",BATCH_SIZE,LEARNING_RATE,LR_DECAY,WEIGHT_DECAY,Epochs)

'''
#Best Gray
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.0
Epochs = 120 

'''

data_loader_train = DataLoader(Data_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=50)
data_loader_val = DataLoader(Data_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=50)
X_batch , Y_batch = next(iter(data_loader_val))
print("\n\nOne Batch : ",type(X_batch), X_batch.shape, X_batch.numpy().max(),X_batch.numpy().min(), Y_batch.shape)

# mim.imsave("Outputs/test.png",X_batch)
# mim.imsave("Outputs/test_flip.png",Y_batch)

# print(a)

model = UNet(in_channels=3, out_channels=3, only_encode=False)
     
#def weights_init(m):
#    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
#        nn.init.xavier_uniform_(m.weight.data)
        #nn.init.xavier_uniform_(m.bias.data)

#model.apply(weights_init)
'''
#
#Data Parallel ---------------------------------------------
def get_ifname():
    return ifcfg.default_interface()["device"]

if "GLOO_SOCKET_IFNAME" not in os.environ:
        os.environ["GLOO_SOCKET_IFNAME"] = get_ifname()

if "NCCL_SOCKET_IFNAME" not in os.environ:
    os.environ["NCCL_SOCKET_IFNAME"] = get_ifname()

master_port = int(os.environ.get("MASTER_PORT", 8738))
master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
world_rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))

tcp_store = torch.distributed.TCPStore(master_addr, master_port, world_size, world_rank == 0)
torch.distributed.init_process_group(
    "nccl", store=tcp_store, rank=world_rank, world_size=world_size
)
#-----------------------------------------------------------
'''

# model = nn.parallel.DistributedDataParallel(model)

model = nn.DataParallel(model).cuda()

#model.cuda()
#model = model.cuda()
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
        # model.train()
        images, targets = batch[0], batch[1]
        images, targets = images.cuda(), targets.cuda()
        
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output,targets)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
        
    op_train_loss = total_loss / (batch_idx + 1)

    return op_train_loss

def validate(epoch):

    #Validate
    acc_sum = 0
    total_loss = 0 
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader_val):
            model.eval()
            images, targets = batch[0], batch[1]
            images, targets = images.cuda(), targets.cuda()
            with torch.no_grad():
                output = model(images)
            # print(images.cpu().numpy().shape,images.cpu().numpy().shape)
            val_loss = criterion(output,targets)
            total_loss+=val_loss.item()

    image = (images[0].cpu().numpy().transpose(1,2,0)*127.5 + 127.5).astype(np.int32)
    output = (output[0].cpu().numpy().transpose(1,2,0)*127.5 + 127.5).astype(np.int32)
    # mim.imsave("Outputs/image.png",image)
    # mim.imsave("Outputs/output.png",output)
    op_val_loss = total_loss / (batch_idx + 1)

    H,W,_ = image.shape
    return op_val_loss, image, output


optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = LR_DECAY)
criterion = nn.MSELoss().cuda()

best_ep = 0
best_loss = 100

for i in range(Epochs):
    op_train_loss = train(i)
    op_val_loss,image, output = validate(i)
    scheduler.step()
    print("\n\nEpoch - ",i, " Train Loss : ", op_train_loss," Val Loss : ", op_val_loss, "\n\n")
    if op_val_loss<best_loss:
        best_loss = op_val_loss
        best_ep = i
        torch.save(model.state_dict(), "Best_Cancer_CAE.pwf")
        print("Model saved")
        mim.imsave("Outputs/Best_image.png",image)
        mim.imsave("Outputs/Best_output.png",output)

    else:
        mim.imsave("Outputs/Current_image.png",image)
        mim.imsave("Outputs/Current_output.png",output)

    if i%10==0:
        print("\n\n--------------- Best Values \n\n Epoch - ",best_ep, "Val Accuracy : ", best_loss,"\n\n")

####################
###### RESULTS #####
####################

'''

DDL Network 

'''

